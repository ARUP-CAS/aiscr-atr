"""Porty potrubí: pevná rozhraní, která naplňují adaptéry.

Potrubí (`atr.pipeline`) volá jen tyto protokoly. Konkrétní nástroj nebo model
je adaptér v `atr.adapters`, vybraný konfigurací. Nový model = nový adaptér
a změna `config/pipeline.yaml`, tok potrubí se nemění.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from atr.ports.types import (
    Block,
    ComponentInfo,
    DocumentInput,
    DocumentResult,
    LayoutBlock,
    PageClass,
    PageGeometry,
    PageImage,
    RecognizedContent,
    RouteDecision,
    TextLayer,
    TextLayerJudgement,
)


@runtime_checkable
class Component(Protocol):
    """Každý adaptér se umí popsat pro paradata."""

    def info(self) -> ComponentInfo: ...


class OpenedDocument(Protocol):
    """Otevřený dokument, který vrací PageSource. Zavírá se po běhu."""

    @property
    def pages(self) -> list[PageGeometry]: ...

    def close(self) -> None: ...


class PageSource(Component, Protocol):
    """Soubor → strany s geometrií."""

    def open(self, document: DocumentInput) -> OpenedDocument: ...


class PageRenderer(Component, Protocol):
    """Strana → obrázek při daném DPI."""

    def render(self, doc: OpenedDocument, page: int, dpi: float) -> PageImage: ...


class TextLayerReader(Component, Protocol):
    """Strana → řádky a slova textové vrstvy s polohou (prázdné, pokud vrstva není)."""

    def read(self, doc: OpenedDocument, page: int) -> TextLayer: ...


class TextLayerJudge(Component, Protocol):
    """Textová vrstva strany → verdikt, zda je použitelná."""

    def judge(self, layer: TextLayer, geometry: PageGeometry) -> TextLayerJudgement: ...


class PageClassifier(Component, Protocol):
    """Obrázek strany → kategorie (nepovinné vodítko, ne směrování)."""

    def classify(self, image: PageImage) -> PageClass: ...


class Router(Component, Protocol):
    """Rozhodne trasu strany."""

    def decide(
        self,
        geometry: PageGeometry,
        judgement: TextLayerJudgement,
        forced_reason: str | None,
        page_class: PageClass | None,
    ) -> RouteDecision: ...


class LayoutDetector(Component, Protocol):
    """Obrázek strany → bloky v bodech zobrazené strany."""

    def detect(self, image: PageImage, geometry: PageGeometry) -> list[LayoutBlock]: ...


class BlockRecognizer(Component, Protocol):
    """Výřez bloku → rozpoznaný obsah."""

    def recognize(self, crop: PageImage, block: LayoutBlock) -> RecognizedContent: ...


class TextAssigner(Component, Protocol):
    """Slova textové vrstvy + bloky rozvržení → bloky s textem a změřenými řádky."""

    def assign(self, layer: TextLayer, blocks: list[LayoutBlock]) -> list[Block]: ...


class LanguageDetector(Component, Protocol):
    """Text → jazyk (ISO 639-1) a jistota, nebo None."""

    def detect(self, text: str) -> tuple[str, float] | None: ...


class Assembler(Component, Protocol):
    """Výsledek dokumentu → výstup `atr/docproc-json` (slovník připravený k serializaci)."""

    def assemble(self, result: DocumentResult) -> dict: ...


__all__ = [
    "Assembler",
    "BlockRecognizer",
    "Component",
    "LanguageDetector",
    "LayoutDetector",
    "OpenedDocument",
    "PageClassifier",
    "PageRenderer",
    "PageSource",
    "Router",
    "TextAssigner",
    "TextLayerJudge",
    "TextLayerReader",
]
