"""Adaptéry `pdfium` pro porty page_source, renderer a text_layer.

Sdílejí jeden otevřený dokument (`document.py`), proto jsou v jedné složce.
"""

from __future__ import annotations

import ctypes
import io

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from atr.adapters.pdfium.document import PDFIUM_LOCK, ImageDocument, PdfDocument, open_document
from atr.ports.types import BBox, ComponentInfo, DocumentInput, Line, PageImage, TextLayer, Word
from atr.registry import register

_VERSION = f"pypdfium2 {pdfium.version.PYPDFIUM_INFO}, pdfium {pdfium.version.PDFIUM_INFO}"


@register("page_source", "pdfium")
class PdfiumPageSource:
    def __init__(self, default_image_dpi: float = 300.0) -> None:
        self.default_image_dpi = default_image_dpi

    def open(self, document: DocumentInput) -> PdfDocument | ImageDocument:
        return open_document(document, self.default_image_dpi)

    def info(self) -> ComponentInfo:
        return ComponentInfo("page_source", "pdfium", _VERSION,
                             {"default_image_dpi": self.default_image_dpi}, "BSD-3-Clause")


@register("renderer", "pdfium")
class PdfiumRenderer:
    def __init__(self, grayscale: bool = False) -> None:
        self.grayscale = grayscale

    def render(self, doc: PdfDocument | ImageDocument, page: int, dpi: float) -> PageImage:
        if isinstance(doc, ImageDocument):
            img = _render_image(doc, page, dpi)
        else:
            p = doc.page(page)
            with PDFIUM_LOCK:
                bitmap = p.handle.render(scale=dpi / 72.0 * p.user_unit, grayscale=self.grayscale)
                img = bitmap.to_pil()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return PageImage(page, dpi, img.width, img.height, buf.getvalue())

    def info(self) -> ComponentInfo:
        return ComponentInfo("renderer", "pdfium", _VERSION, {"grayscale": self.grayscale}, "BSD-3-Clause")


def _render_image(doc: ImageDocument, page: int, dpi: float) -> Image.Image:
    frame, dpi_x, dpi_y = doc.frames[page - 1]
    img = frame.convert("RGB") if frame.mode not in ("RGB", "L") else frame
    sx, sy = dpi / dpi_x, dpi / dpi_y  # různé DPI v osách se tím srovná na čtvercové pixely
    if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
        size = (max(1, round(img.width * sx)), max(1, round(img.height * sy)))
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


@register("text_layer", "pdfium")
class PdfiumTextLayerReader:
    """Znaky textové vrstvy → slova a řádky s polohou na zobrazené straně.

    Řádky odděluje PDFium (generované konce řádků), slova mezery. Poloha slova
    je sjednocení volných rámečků znaků (loose char box: výška písma, ne glyfu),
    takže sousední řádky mají souvislou výšku.
    """

    def read(self, doc: PdfDocument | ImageDocument, page: int) -> TextLayer:
        if isinstance(doc, ImageDocument):
            return TextLayer(page, ())
        p = doc.page(page)
        with PDFIUM_LOCK:
            textpage = p.handle.get_textpage()
            try:
                return self._read(textpage, p, page)
            finally:
                textpage.close()

    def _read(self, textpage, p, page: int) -> TextLayer:
        lines: list[Line] = []
        words: list[Word] = []
        chars: list[tuple[str, BBox]] = []
        unmapped = 0
        invisible = 0
        page_w, page_h = p.width_raw * p.user_unit, p.height_raw * p.user_unit
        render_mode: dict[int, int] = {}  # adresa textového objektu → režim vykreslení

        def flush_word() -> None:
            if chars:
                text = "".join(c for c, _ in chars)
                box = BBox(min(b.x0 for _, b in chars), min(b.y0 for _, b in chars),
                           max(b.x1 for _, b in chars), max(b.y1 for _, b in chars))
                words.append(Word(text, box))
                chars.clear()

        def flush_line() -> None:
            flush_word()
            if words:
                box = BBox(min(w.bbox.x0 for w in words), min(w.bbox.y0 for w in words),
                           max(w.bbox.x1 for w in words), max(w.bbox.y1 for w in words))
                lines.append(Line(" ".join(w.text for w in words), box, tuple(words)))
                words.clear()

        for i in range(textpage.count_chars()):
            code = pdfium_c.FPDFText_GetUnicode(textpage, i)
            ch = chr(code) if code else ""
            if ch in ("\r", "\n"):
                flush_line()
                continue
            if not ch or ch.isspace() or pdfium_c.FPDFText_IsGenerated(textpage, i):
                flush_word()
                continue
            if pdfium_c.FPDFText_HasUnicodeMapError(textpage, i):
                unmapped += 1
            obj = pdfium_c.FPDFText_GetTextObject(textpage, i)
            if obj:
                key = ctypes.cast(obj, ctypes.c_void_p).value or 0
                if key not in render_mode:
                    render_mode[key] = pdfium_c.FPDFTextObj_GetTextRenderMode(obj)
                if render_mode[key] == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE:
                    invisible += 1
            if pdfium_c.FPDFText_IsHyphen(textpage, i):
                ch = "-"
            left, bottom, right, top = textpage.get_charbox(i, loose=True)
            corners = [p.to_display(left, bottom), p.to_display(right, top)]
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            # Text mimo zobrazenou stranu (za hranicí CropBox) do výstupu nepatří;
            # znak přesahující okraj se ořízne na stranu.
            x0, x1 = max(0.0, min(xs)), min(page_w, max(xs))
            y0, y1 = max(0.0, min(ys)), min(page_h, max(ys))
            if x1 <= x0 or y1 <= y0:
                flush_word()
                continue
            chars.append((ch, BBox(x0, y0, x1, y1)))
        flush_line()
        return TextLayer(page, tuple(lines), unmapped, invisible, _image_coverage(p))

    def info(self) -> ComponentInfo:
        return ComponentInfo("text_layer", "pdfium", _VERSION, {}, "BSD-3-Clause")


def _image_coverage(p) -> float:
    """Podíl plochy strany pokrytý rastrovými obrázky (součet, nejvýš 1)."""
    page = p.handle
    width, height = p.width_raw, p.height_raw
    if width <= 0 or height <= 0:
        return 0.0
    left, bottom, right, top = page.get_cropbox()
    covered = 0.0
    for i in range(pdfium_c.FPDFPage_CountObjects(page)):
        obj = pdfium_c.FPDFPage_GetObject(page, i)
        if pdfium_c.FPDFPageObj_GetType(obj) != pdfium_c.FPDF_PAGEOBJ_IMAGE:
            continue
        x0, y0, x1, y1 = (ctypes.c_float() for _ in range(4))
        if not pdfium_c.FPDFPageObj_GetBounds(obj, x0, y0, x1, y1):
            continue
        w = min(x1.value, right) - max(x0.value, left)
        h = min(y1.value, top) - max(y0.value, bottom)
        if w > 0 and h > 0:
            covered += w * h
    return min(1.0, covered / ((right - left) * (top - bottom)))
