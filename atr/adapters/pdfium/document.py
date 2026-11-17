"""Otevřený dokument pro adaptéry `pdfium`: PDF přes PDFium, obrázky přes Pillow.

Souřadnice: PDFium převádí body PDF na zobrazenou stranu (CropBox, /Rotate,
počátek vlevo nahoře) funkcí FPDF_PageToDevice. /UserUnit PDFium neuplatňuje,
čte se proto zvlášť (pypdf) a násobí se jím. Obrázek je strana o rozměru
pixely × 72 / DPI (DPI z metadat souboru, jinak `default_image_dpi`).
"""

from __future__ import annotations

import ctypes
import io
import threading
from dataclasses import dataclass

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageSequence, UnidentifiedImageError

from atr.ports.errors import InputError
from atr.ports.types import DocumentInput, PageGeometry

# PDFium není bezpečné pro souběžná vlákna.
PDFIUM_LOCK = threading.RLock()

_PRECISION = 100  # FPDF_PageToDevice pracuje s celými čísly; 1/100 bodu stačí


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
            user_units = _user_units(data, len(self._doc))
            self._pages: list[_PdfPage] = []
            self.pages: list[PageGeometry] = []
            for i in range(len(self._doc)):
                handle = self._doc[i]
                w, h = handle.get_size()
                uu = user_units[i]
                self._pages.append(_PdfPage(handle, w, h, uu))
                self.pages.append(PageGeometry(i + 1, w * uu, h * uu, handle.get_rotation(), uu))

    def page(self, number: int) -> _PdfPage:
        return self._pages[number - 1]

    def close(self) -> None:
        with PDFIUM_LOCK:
            for p in self._pages:
                p.handle.close()
            self._doc.close()


class ImageDocument:
    kind = "image"

    def __init__(self, data: bytes, default_dpi: float) -> None:
        try:
            img = Image.open(io.BytesIO(data))
            frames = [f.copy() for f in ImageSequence.Iterator(img)]
        except (UnidentifiedImageError, OSError) as exc:
            raise InputError("unsupported_format", f"soubor není PDF ani čitelný obrázek: {exc}") from exc
        self.frames: list[tuple[Image.Image, float]] = []
        self.pages: list[PageGeometry] = []
        for i, frame in enumerate(frames):
            dpi = _image_dpi(frame, img, default_dpi)
            self.frames.append((frame, dpi))
            w, h = frame.size
            self.pages.append(PageGeometry(i + 1, w * 72 / dpi, h * 72 / dpi, 0, 1.0, (w, h)))

    def close(self) -> None:
        for frame, _ in self.frames:
            frame.close()


def open_document(document: DocumentInput, default_image_dpi: float) -> PdfDocument | ImageDocument:
    head = document.data[:1024]
    if b"%PDF-" in head:
        return PdfDocument(document.data)
    return ImageDocument(document.data, default_image_dpi)


def _image_dpi(frame: Image.Image, img: Image.Image, default: float) -> float:
    dpi = frame.info.get("dpi") or img.info.get("dpi")
    try:
        value = float(dpi[0] if isinstance(dpi, (tuple, list)) else dpi)
    except (TypeError, ValueError):
        return default
    # Mnoho souborů nese 72 nebo 1 jako „nevím“; skeny mají aspoň 100.
    return value if value >= 100 else default


def _user_units(data: bytes, count: int) -> list[float]:
    """/UserUnit každé strany. PDFium ho nečte, pypdf ano."""
    if b"/UserUnit" not in data:
        return [1.0] * count
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    out = []
    for page in reader.pages:
        try:
            out.append(float(page.get("/UserUnit", 1.0)))
        except (TypeError, ValueError):
            out.append(1.0)
    return out if len(out) == count else [1.0] * count
