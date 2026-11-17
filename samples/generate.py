"""Syntetická testovací sada: PDF a obrázky s přesně známou geometrií.

Každý vzorek nese v `expected` to, co musí adaptéry vrátit: rozměry zobrazené
strany, otočení a polohu značkovacího slova v bodech zobrazené strany
(počátek vlevo nahoře). Skutečné dokumenty z AMČR doplní složka
`samples/corpus/` (mimo git), viz README.

Použití:  python samples/generate.py <cílová složka>
Závislosti (jen pro vývoj): reportlab, pypdf, pillow.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, NumberObject, RectangleObject
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

W, H = 595.0, 842.0  # A4 v bodech
MARKER = "ZNACKA"
CZECH = "Žluťoučký kůň úpěl ďábelské ódy. Nálezová zpráva č. 123/87."

_FONT_CANDIDATES = [
    os.environ.get("ATR_TEST_FONT", ""),
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font() -> tuple[str, bool]:
    """Vrátí (jméno písma, umí češtinu)."""
    for path in _FONT_CANDIDATES:
        if path and Path(path).is_file():
            pdfmetrics.registerFont(TTFont("TestSans", path))
            return "TestSans", True
    return "Helvetica", False


def _base_page(font: str, czech: bool, marker_xy: tuple[float, float]) -> bytes:
    """Jedna strana A4: nadpis, odstavec, značka na dané pozici (v souřadnicích PDF)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H))
    c.setFont(font, 18)
    c.drawString(72, H - 90, "Nalezova zprava" if not czech else "Nálezová zpráva")
    c.setFont(font, 11)
    text = CZECH if czech else "Archeologicky vyzkum odkryl sidliste z doby bronzove."
    y = H - 130
    for _ in range(4):
        c.drawString(72, y, text)
        y -= 16
    c.setFont(font, 12)
    c.drawString(*marker_xy, MARKER)
    c.showPage()
    c.save()
    return buf.getvalue()


def _marker_box(x: float, y_baseline: float, size: float = 12) -> list[float]:
    """Přibližný bbox značky v souřadnicích PDF (počátek vlevo dole)."""
    width = pdfmetrics.stringWidth(MARKER, "Helvetica", size)
    return [x, y_baseline - 0.25 * size, x + width, y_baseline + 0.8 * size]


def _to_display(box: list[float], *, crop: list[float], rotation: int, user_unit: float = 1.0) -> list[float]:
    """bbox v souřadnicích PDF → zobrazená strana (CropBox, /Rotate, počátek vlevo nahoře)."""
    cx0, cy0, cx1, cy1 = crop
    cw, ch = cx1 - cx0, cy1 - cy0
    x0, y0, x1, y1 = box[0] - cx0, box[1] - cy0, box[2] - cx0, box[3] - cy0
    if rotation == 0:
        out = [x0, ch - y1, x1, ch - y0]
    elif rotation == 90:
        out = [y0, x0, y1, x1]
    elif rotation == 180:
        out = [cw - x1, y0, cw - x0, y1]
    elif rotation == 270:
        out = [ch - y1, cw - x1, ch - y0, cw - x0]
    else:
        raise ValueError(rotation)
    return [v * user_unit for v in out]


def _write(writer: PdfWriter, path: Path) -> None:
    with open(path, "wb") as fh:
        writer.write(fh)


def _pdf_from(data: bytes, *, rotation: int = 0, crop: list[float] | None = None,
              user_unit: float | None = None) -> PdfWriter:
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(data)).pages:
        if rotation:
            page.rotate(rotation)
        if crop:
            page.cropbox = RectangleObject(crop)
        if user_unit:
            page[NameObject("/UserUnit")] = NumberObject(user_unit)
        writer.add_page(page)
    return writer


def _scan_image(text_lines: list[str], dpi: int = 150) -> Image.Image:
    img = Image.new("L", (int(W / 72 * dpi), int(H / 72 * dpi)), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(12, dpi // 6))  # zhruba 12 pt, čitelné pro rozvržení
    y = dpi
    for line in text_lines:
        draw.text((dpi, y), line, fill=0, font=font)
        y += dpi // 3
    return img


def generate(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    font, czech = _font()
    marker_xy = (300.0, 200.0)
    page = _base_page(font, czech, marker_xy)
    marker = _marker_box(*marker_xy)
    full = [0.0, 0.0, W, H]
    expected: dict[str, dict] = {"_font": {"name": font, "czech": czech}}

    def add(name: str, *, rotation: int = 0, crop: list[float] | None = None, user_unit: float = 1.0):
        _write(_pdf_from(page, rotation=rotation, crop=crop, user_unit=user_unit if user_unit != 1 else None),
               out / f"{name}.pdf")
        c = crop or full
        cw, ch = (c[2] - c[0]) * user_unit, (c[3] - c[1]) * user_unit
        size = (ch, cw) if rotation in (90, 270) else (cw, ch)
        expected[name] = {
            "pages": 1,
            "page": {"width": size[0], "height": size[1], "rotation": rotation, "user_unit": user_unit},
            "marker_bbox": _to_display(marker, crop=c, rotation=rotation, user_unit=user_unit),
            "has_text": True,
        }

    add("born_digital")
    add("rotated_90", rotation=90)
    add("rotated_180", rotation=180)
    add("rotated_270", rotation=270)
    add("cropped", crop=[50.0, 100.0, 545.0, 742.0])
    add("cropped_rotated_90", rotation=90, crop=[50.0, 100.0, 545.0, 742.0])
    add("user_unit_2", user_unit=2.0)

    # Sken: strana jen s obrázkem, bez textové vrstvy.
    scan = _scan_image(["Nalezova zprava", "Sonda S2 zachytila kulturni vrstvu o mocnosti 40 cm.",
                        "Ve vyplni objektu 12 se nachazely zlomky mazanice a zvireci kosti.",
                        "Dokumentace byla porizena v meritku 1:20 a ulozena v archivu."] * 3)
    img_buf = io.BytesIO()
    scan.save(img_buf, format="PNG")
    c = canvas.Canvas(str(out / "scan.pdf"), pagesize=(W, H))

    c.drawImage(ImageReader(io.BytesIO(img_buf.getvalue())), 0, 0, W, H)
    c.showPage()
    c.save()
    expected["scan"] = {"pages": 1, "page": {"width": W, "height": H, "rotation": 0, "user_unit": 1.0},
                        "has_text": False}

    # Smíšený dokument: textová strana, sken, prázdná strana.
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(page)))
    writer.append(PdfReader(str(out / "scan.pdf")))
    writer.add_blank_page(W, H)
    _write(writer, out / "mixed.pdf")
    expected["mixed"] = {"pages": 3, "has_text_by_page": [True, False, False]}

    # Obrázky jako vstup.
    scan.save(out / "scan_300dpi.png", dpi=(300, 300))
    expected["scan_300dpi.png"] = {"pages": 1, "source_pixels": list(scan.size),
                                   "page": {"width": scan.size[0] * 72 / 300, "height": scan.size[1] * 72 / 300,
                                            "rotation": 0, "user_unit": 1.0}}
    frames = [scan, scan.rotate(0)]
    frames[0].save(out / "two_pages.tiff", save_all=True, append_images=frames[1:], dpi=(150, 150))
    expected["two_pages.tiff"] = {"pages": 2, "source_pixels": list(scan.size)}

    # Starší OCR vrstva: sken a pod ním neviditelný text (PDF režim vykreslení 3).
    c = canvas.Canvas(str(out / "old_ocr.pdf"), pagesize=(W, H))
    c.drawImage(ImageReader(io.BytesIO(img_buf.getvalue())), 0, 0, W, H)
    t = c.beginText(72, H - 120)
    t.setTextRenderMode(3)
    t.setFont(font, 11)
    for _ in range(12):
        t.textLine(CZECH if czech else "Nalezova zprava o vyzkumu sidliste z doby bronzove.")
    c.drawText(t)
    c.showPage()
    c.save()
    expected["old_ocr"] = {"pages": 1, "verdict": "ocr"}

    # Neúplná textová vrstva: text je jen v horní části, zbytek strany je sken textu.
    c = canvas.Canvas(str(out / "partial_text.pdf"), pagesize=(W, H))
    c.drawImage(ImageReader(io.BytesIO(img_buf.getvalue())), 0, 0, W, H * 0.8)
    c.setFont(font, 11)
    c.drawString(72, H - 60, CZECH if czech else "Nalezova zprava c. 123/87.")
    c.showPage()
    c.save()
    expected["partial_text"] = {"pages": 1}

    # Nekvalitní digitální vrstvy (text je v PDF, ale nepoužitelný).
    good = CZECH if czech else "Archeologicky vyzkum odkryl sidliste z mladsi doby bronzove a nalezy keramiky."
    variants = {
        "garbled_mojibake": good.encode("utf-8").decode("latin-1") if czech else None,
        "garbled_spaced": " ".join(good.replace(" ", "")),
        "garbled_glued": good.replace(" ", ""),
        # Znaky ze soukromé oblasti Unicode reportlab do PDF nezapíše; ty testuje přímo test verdiktu.
        "garbled_nonsense": "xkcd grfl tzpq wvbn hjkl qrst mnbv zxcv plkm frtg bvcx nmlk qwrt zzxc vbnm",
    }
    for name, line in variants.items():
        if line is None:
            continue
        c = canvas.Canvas(str(out / f"{name}.pdf"), pagesize=(W, H))
        c.setFont(font, 11)
        y = H - 100
        for _ in range(10):
            c.drawString(72, y, line[:95])
            y -= 16
        c.showPage()
        c.save()
        expected[name] = {"pages": 1, "verdict": "garbled"}

    (out / "expected.json").write_text(json.dumps(expected, indent=1, ensure_ascii=False), encoding="utf-8")
    return expected


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/generated")
    print(json.dumps(generate(target), indent=1, ensure_ascii=False))
