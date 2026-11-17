"""Testovací adaptéry `fake` pro všechny porty.

Slouží k testům potrubí a k vývoji bez GPU. Dokument je popsaný jako JSON
(viz `FakeDocumentSpec`), ne jako PDF.
"""

from __future__ import annotations

import io
import json

from PIL import Image

from atr.ports.types import (
    BBox,
    BlockType,
    ComponentInfo,
    DocumentInput,
    LayoutBlock,
    Line,
    PageClass,
    PageGeometry,
    PageImage,
    RecognizedContent,
    TextLayer,
    Word,
)
from atr.registry import register

# Formát vstupu:
# {"pages": [{"width": 595, "height": 842, "rotation": 0,
#             "text_lines": [{"text": "...", "bbox": [x0,y0,x1,y1]}],
#             "blocks": [{"type": "text", "bbox": [...], "confidence": 0.9}],
#             "class": "text"}]}


def _info(port: str) -> ComponentInfo:
    return ComponentInfo(port, "fake", "0")


class _FakeDoc:
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.pages = [
            PageGeometry(i + 1, p.get("width", 595), p.get("height", 842), p.get("rotation", 0))
            for i, p in enumerate(spec["pages"])
        ]
        self.closed = False

    def page(self, number: int) -> dict:
        return self.spec["pages"][number - 1]

    def close(self) -> None:
        self.closed = True


@register("page_source", "fake")
class FakePageSource:
    def open(self, document: DocumentInput) -> _FakeDoc:
        return _FakeDoc(json.loads(document.data))

    def info(self) -> ComponentInfo:
        return _info("page_source")


@register("renderer", "fake")
class FakeRenderer:
    def render(self, doc: _FakeDoc, page: int, dpi: float) -> PageImage:
        g = doc.pages[page - 1]
        w, h = int(g.width * dpi / 72), int(g.height * dpi / 72)
        buf = io.BytesIO()
        Image.new("RGB", (w, h), "white").save(buf, format="PNG")
        return PageImage(page, dpi, w, h, buf.getvalue())

    def info(self) -> ComponentInfo:
        return _info("renderer")


@register("text_layer", "fake")
class FakeTextLayerReader:
    def read(self, doc: _FakeDoc, page: int) -> TextLayer:
        lines = []
        for raw in doc.page(page).get("text_lines", []):
            x0, y0, x1, y1 = raw["bbox"]
            tokens = raw["text"].split()
            step = (x1 - x0) / max(len(tokens), 1)
            words = tuple(
                Word(t, BBox(x0 + i * step, y0, x0 + (i + 1) * step, y1)) for i, t in enumerate(tokens)
            )
            lines.append(Line(raw["text"], BBox(x0, y0, x1, y1), words))
        return TextLayer(page, tuple(lines))

    def info(self) -> ComponentInfo:
        return _info("text_layer")


@register("page_classifier", "fake")
class FakePageClassifier:
    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = labels or []
        self._calls = 0

    def classify(self, image: PageImage) -> PageClass:
        self._calls += 1
        label = self.labels[image.page - 1] if image.page <= len(self.labels) else "text"
        return PageClass(label, 1.0)

    def info(self) -> ComponentInfo:
        return _info("page_classifier")


@register("layout", "fake")
class FakeLayoutDetector:
    """Bloky bere ze specifikace dokumentu, kterou si pamatuje page_source."""

    def __init__(self, blocks_by_page: dict[int, list[dict]] | None = None) -> None:
        self.blocks_by_page = blocks_by_page or {}

    def detect(self, image: PageImage, geometry: PageGeometry) -> list[LayoutBlock]:
        return [
            LayoutBlock(BlockType(b["type"]), BBox(*b["bbox"]), b.get("confidence", 0.9), b.get("order"))
            for b in self.blocks_by_page.get(image.page, [])
        ]

    def info(self) -> ComponentInfo:
        return _info("layout")


@register("recognizer", "fake")
class FakeRecognizer:
    def __init__(self, text: str = "rozpoznaný text", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[tuple[int, BlockType]] = []

    def recognize(self, crop: PageImage, block: LayoutBlock) -> RecognizedContent:
        self.calls.append((crop.page, block.type))
        if self.fail:
            raise RuntimeError("fake recognizer failure")
        return RecognizedContent(self.text)

    def info(self) -> ComponentInfo:
        return _info("recognizer")


@register("language", "fake")
class FakeLanguageDetector:
    def __init__(self, language: str = "cs") -> None:
        self.language = language

    def detect(self, text: str) -> tuple[str, float] | None:
        return (self.language, 1.0) if text.strip() else None

    def info(self) -> ComponentInfo:
        return _info("language")
