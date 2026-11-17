"""Adaptéry `pdfium` pro porty page_source, renderer a text_layer.

Sdílejí jeden otevřený dokument (`document.py`), proto jsou v jedné složce.
"""

from __future__ import annotations

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
    frame, source_dpi = doc.frames[page - 1]
    img = frame.convert("RGB") if frame.mode not in ("RGB", "L") else frame
    scale = dpi / source_dpi
    if abs(scale - 1.0) > 1e-3:
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
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
            if pdfium_c.FPDFText_IsHyphen(textpage, i):
                ch = "-"
            left, bottom, right, top = textpage.get_charbox(i, loose=True)
            corners = [p.to_display(left, bottom), p.to_display(right, top)]
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            chars.append((ch, BBox(min(xs), min(ys), max(xs), max(ys))))
        flush_line()
        return TextLayer(page, tuple(lines), unmapped)

    def info(self) -> ComponentInfo:
        return ComponentInfo("text_layer", "pdfium", _VERSION, {}, "BSD-3-Clause")
