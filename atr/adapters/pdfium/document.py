"""Otevřený dokument pro adaptéry `pdfium`: PDF přes PDFium, obrázky přes Pillow.

Souřadnice: PDFium převádí body PDF na zobrazenou stranu (CropBox, /Rotate,
počátek vlevo nahoře) funkcí FPDF_PageToDevice. /UserUnit PDFium neuplatňuje,
čte se proto zvlášť (pypdf) a násobí se jím. Obrázek je strana o rozměru
pixely × 72 / DPI (DPI z metadat souboru, jinak `default_image_dpi`).

Paměť: při otevření se čte jen geometrie stran. Strana PDF i snímek obrázku se
načítají až na vyžádání a drží se vždy jen jedna (potrubí jde po stranách),
takže stostránkový TIFF nezabere paměť za sto dekódovaných stran.
"""

from __future__ import annotations

import ctypes
import io
import threading
from dataclasses import dataclass

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageOps, UnidentifiedImageError

from atr.ports.errors import InputError
from atr.ports.types import DocumentInput, PageGeometry

# PDFium není bezpečné pro souběžná vlákna.
PDFIUM_LOCK = threading.RLock()

_PRECISION = 100  # FPDF_PageToDevice pracuje s celými čísly; 1/100 bodu stačí
_EXIF_ORIENTATION = 0x0112
_EXIF_SWAPS_AXES = {5, 6, 7, 8}


@dataclass
class _PdfPage:
    handle: pdfium.PdfPage
    width_raw: float  # body PDF bez UserUnit, po CropBox a /Rotate
    height_raw: float
    user_unit: float

    def to_display(self, x: float, y: float) -> tuple[float, float]:
        """Bod v prostoru PDF → bod na zobrazené straně (s UserUnit)."""
        dx, dy = ctypes.c_int(), ctypes.c_int()
        pdfium_c.FPDF_PageToDevice(
            self.handle,
            0,
            0,
            round(self.width_raw * _PRECISION),
            round(self.height_raw * _PRECISION),
            0,
            x,
            y,
            dx,
            dy,
        )
        return dx.value / _PRECISION * self.user_unit, dy.value / _PRECISION * self.user_unit


class PdfDocument:
    kind = "pdf"

    def __init__(self, data: bytes, password: str | None = None) -> None:
        with PDFIUM_LOCK:
            try:
                self._doc = pdfium.PdfDocument(data, password=password)
            except pdfium.PdfiumError as exc:
                raise InputError("unreadable_document", str(exc)) from exc
            count = len(self._doc)
        user_units = _user_units(data, count)
        self._sizes: list[tuple[float, float, float]] = []  # šířka, výška, UserUnit
        self.pages: list[PageGeometry] = []
        self._current: tuple[int, _PdfPage] | None = None
        try:
            for i in range(count):
                # Zámek po stranách, aby otevření dlouhého PDF neblokovalo jiný převod.
                with PDFIUM_LOCK:
                    handle = self._doc[i]
                    try:
                        w, h = handle.get_size()
                        rotation = handle.get_rotation()
                    finally:
                        handle.close()
                uu = user_units[i]
                self._sizes.append((w, h, uu))
                self.pages.append(PageGeometry(i + 1, w * uu, h * uu, rotation, uu))
        except pdfium.PdfiumError as exc:
            self.close()
            raise InputError("unreadable_document", f"strana {len(self.pages) + 1}: {exc}") from exc

    def page(self, number: int) -> _PdfPage:
        """Načte stranu na vyžádání; drží se jen naposledy použitá."""
        if self._current is not None and self._current[0] == number:
            return self._current[1]
        with PDFIUM_LOCK:
            if self._current is not None:
                self._current[1].handle.close()
                self._current = None
            try:
                handle = self._doc[number - 1]
            except pdfium.PdfiumError as exc:
                raise InputError("unreadable_document", f"strana {number}: {exc}") from exc
            w, h, uu = self._sizes[number - 1]
            page = _PdfPage(handle, w, h, uu)
            self._current = (number, page)
            return page

    def close(self) -> None:
        with PDFIUM_LOCK:
            if self._current is not None:
                self._current[1].handle.close()
                self._current = None
            self._doc.close()


class ImageDocument:
    kind = "image"

    def __init__(self, data: bytes, default_dpi: float) -> None:
        self._lock = threading.Lock()
        try:
            self._img = Image.open(io.BytesIO(data))
            count = getattr(self._img, "n_frames", 1)
        except Image.DecompressionBombError as exc:
            raise InputError("image_too_large", str(exc)) from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InputError("unsupported_format", f"soubor není PDF ani čitelný obrázek: {exc}") from exc
        # U každého snímku jen metadata (rozměr, DPI, orientace); pixely se nedekódují.
        self._frames: list[tuple[float, float, int]] = []  # DPI x, DPI y, orientace EXIF
        self.pages: list[PageGeometry] = []
        try:
            for i in range(count):
                self._img.seek(i)
                orientation = int(self._img.getexif().get(_EXIF_ORIENTATION, 1) or 1)
                w, h = self._img.size
                dpi_x, dpi_y = _image_dpi(self._img, default_dpi)
                if orientation in _EXIF_SWAPS_AXES:
                    # Strana odpovídá tomu, co ukazují prohlížeče (orientace z EXIF uplatněná).
                    w, h, dpi_x, dpi_y = h, w, dpi_y, dpi_x
                self._frames.append((dpi_x, dpi_y, orientation))
                self.pages.append(PageGeometry(i + 1, w * 72 / dpi_x, h * 72 / dpi_y, 0, 1.0, (w, h)))
        except Image.DecompressionBombError as exc:
            self.close()
            raise InputError("image_too_large", str(exc)) from exc
        except (OSError, ValueError, EOFError) as exc:
            self.close()
            raise InputError("unreadable_document", f"snímek {len(self.pages) + 1}: {exc}") from exc

    def frame(self, number: int) -> tuple[Image.Image, float, float]:
        """Dekóduje snímek na vyžádání: (obrázek RGB nebo L s uplatněnou orientací, DPI x, DPI y)."""
        dpi_x, dpi_y, _ = self._frames[number - 1]
        with self._lock:
            try:
                self._img.seek(number - 1)
                image = ImageOps.exif_transpose(self._img.copy())
            except Image.DecompressionBombError as exc:
                raise InputError("image_too_large", str(exc)) from exc
            except (OSError, ValueError, EOFError) as exc:
                raise InputError("unreadable_document", f"snímek {number}: {exc}") from exc
        return to_display_mode(image), dpi_x, dpi_y

    def close(self) -> None:
        self._img.close()


def to_display_mode(image: Image.Image) -> Image.Image:
    """Převede obrázek na RGB nebo L tak, aby se obsah neztratil.

    Holé `convert("RGB")` ořízne 16bitovou šeď na bílou a průhledné pixely vykreslí černě.
    """
    if image.mode in ("RGB", "L"):
        return image
    if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        lo, hi = image.getextrema()
        if hi <= 255 and lo >= 0:
            return image.convert("L")
        # Vyšší bitovou hloubku přeškálovat na 0–255 podle rozsahu formátu (16 bitů),
        # u jiných rozsahů podle skutečných hodnot.
        top = 65535.0 if hi <= 65535 else float(hi)
        scale = 255.0 / top
        return image.point(lambda v: v * scale).convert("L")
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    if image.mode == "1":
        return image.convert("L")
    return image.convert("RGB")


def open_document(document: DocumentInput, default_image_dpi: float) -> PdfDocument | ImageDocument:
    head = document.data[:1024]
    if b"%PDF-" in head:
        return PdfDocument(document.data)
    return ImageDocument(document.data, default_image_dpi)


def _image_dpi(frame: Image.Image, default: float) -> tuple[float, float]:
    """Vodorovné a svislé DPI z metadat; bez věrohodného údaje `default` pro obě osy."""
    dpi = frame.info.get("dpi")
    try:
        if isinstance(dpi, (tuple, list)):
            x, y = float(dpi[0]), float(dpi[1] if len(dpi) > 1 else dpi[0])
        else:
            x = y = float(dpi)
    except (TypeError, ValueError, IndexError):
        return default, default
    # Mnoho souborů nese 72 nebo 1 jako „nevím“; skeny mají aspoň 100 v delší ose
    # (fax 204 × 98 je platný údaj).
    if max(x, y) < 100 or min(x, y) <= 0:
        return default, default
    return x, y


def _user_units(data: bytes, count: int) -> list[float]:
    """/UserUnit každé strany. PDFium ho nečte, pypdf ano.

    Když soubor klíč neobsahuje ani v čitelné podobě a nemá komprimované objekty
    (`/ObjStm`), ve kterých by se mohl skrývat, pypdf se nevolá.
    """
    if b"/UserUnit" not in data and b"/ObjStm" not in data:
        return [1.0] * count
    from pypdf import PdfReader

    out = []
    try:
        for page in PdfReader(io.BytesIO(data)).pages:
            try:
                out.append(float(page.get("/UserUnit", 1.0)))
            except (TypeError, ValueError):
                out.append(1.0)
    except Exception:  # noqa: BLE001 - PDF, které pypdf nepřečte, ale PDFium ano: bez UserUnit
        return [1.0] * count
    return out if len(out) == count else [1.0] * count
