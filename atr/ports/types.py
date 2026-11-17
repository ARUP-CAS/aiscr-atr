"""Datové typy, které si předávají porty potrubí.

Souřadnice: všechny polohy jsou v bodech (1/72 palce × UserUnit) na zobrazené
straně, tj. v CropBox po uplatnění /Rotate, s počátkem vlevo nahoře
(definice z aiscr-docs-pipeline#17, bod 10). Adaptéry, které pracují v pixelech
nebo v jiném rámu, převádějí na tento rám samy, potrubí jiný rám nezná.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True)
class BBox:
    """Obdélník v bodech na zobrazené straně, počátek vlevo nahoře."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"neplatný bbox: {self}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection(self, other: BBox) -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return w * h if w > 0 and h > 0 else 0.0

    def scaled(self, factor: float) -> BBox:
        return BBox(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)


# --- Vstup ---------------------------------------------------------------


@dataclass(frozen=True)
class DocumentInput:
    """Dokument, jak ho předává orchestrace."""

    data: bytes
    filename: str
    mime_type: str | None = None
    sha512: str | None = None
    # Čísla stran (od 1), které digital-convert označil k ATR, s důvodem.
    forced_pages: dict[int, str] = field(default_factory=dict)


# --- Strany ----------------------------------------------------------------


@dataclass(frozen=True)
class PageGeometry:
    """Geometrie strany originálu, jediný zdroj souřadnic pro celé potrubí."""

    number: int  # od 1
    width: float  # body, zobrazená strana (po /Rotate)
    height: float
    rotation: int  # /Rotate ve stupních po směru hodin: 0, 90, 180, 270
    user_unit: float = 1.0
    # Rozměry originálu v pixelech, pokud je vstupem obrázek.
    source_pixels: tuple[int, int] | None = None


@dataclass(frozen=True)
class PageImage:
    """Vykreslená strana."""

    page: int
    dpi: float
    width_px: int
    height_px: int
    png: bytes

    @property
    def px_to_pt(self) -> float:
        return 72.0 / self.dpi


# --- Textová vrstva -------------------------------------------------------


@dataclass(frozen=True)
class Word:
    text: str
    bbox: BBox


@dataclass(frozen=True)
class Line:
    text: str
    bbox: BBox
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class TextLayer:
    page: int
    lines: tuple[Line, ...]
    # Znaky, které čtečka nedokázala převést na Unicode (vadné mapování písma).
    unmapped_chars: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def words(self) -> tuple[Word, ...]:
        return tuple(w for line in self.lines for w in line.words)


class TextLayerVerdict(StrEnum):
    """Verdikt textové vrstvy (kontrakt #17, bod 13)."""

    DIGITAL = "digital"  # použitelná
    GARBLED = "garbled"  # vadná (nečitelné znaky, chybné mapování)
    OCR = "ocr"  # starší OCR vrstva, nevěříme jí
    NONE = "none"  # žádná
    BLANK = "blank"  # strana bez obsahu


@dataclass(frozen=True)
class TextLayerJudgement:
    verdict: TextLayerVerdict
    reason: str
    score: float | None = None


# --- Klasifikace a směrování ---------------------------------------------


@dataclass(frozen=True)
class PageClass:
    label: str
    confidence: float


class Route(StrEnum):
    DIGITAL = "digital"  # text z textové vrstvy
    OCR = "ocr"  # text z rozpoznání bloků
    FORCED = "forced"  # rozpoznání vynucené volajícím
    EMPTY = "empty"  # prázdná strana


@dataclass(frozen=True)
class RouteDecision:
    page: int
    route: Route
    reason: str


# --- Rozvržení a bloky -----------------------------------------------------


class BlockType(StrEnum):
    """Typy bloků, na které adaptéry rozvržení mapují třídy svých modelů."""

    TEXT = "text"
    TITLE = "title"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FORMULA = "formula"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    FOOTNOTE = "footnote"
    HANDWRITING = "handwriting"
    OTHER = "other"


class PositionAccuracy(StrEnum):
    """Přesnost polohy řádku (kontrakt #17, bod 13)."""

    MEASURED = "measured"
    DERIVED = "derived"
    ABSENT = "absent"


@dataclass(frozen=True)
class LayoutBlock:
    """Blok z detekce rozvržení, ještě bez textu."""

    type: BlockType
    bbox: BBox
    confidence: float
    order: int | None = None  # pořadí čtení, pokud ho model dává
    source_label: str | None = None  # původní třída modelu


@dataclass(frozen=True)
class BlockLine:
    text: str
    bbox: BBox | None
    accuracy: PositionAccuracy


@dataclass(frozen=True)
class RecognizedContent:
    """Výsledek rozpoznání jednoho bloku."""

    text: str
    format: str = "text"  # text | html | markdown | latex
    confidence: float | None = None


@dataclass
class Block:
    """Blok výstupu: rozvržení + text z textové vrstvy nebo z rozpoznání."""

    type: BlockType
    bbox: BBox
    confidence: float
    order: int
    text: str = ""
    format: str = "text"
    lines: list[BlockLine] = field(default_factory=list)
    language: str | None = None
    text_source: Route | None = None


@dataclass
class PageResult:
    geometry: PageGeometry
    decision: RouteDecision
    judgement: TextLayerJudgement | None = None
    page_class: PageClass | None = None
    blocks: list[Block] = field(default_factory=list)
    language: str | None = None


@dataclass
class DocumentResult:
    document: DocumentInput
    pages: list[PageResult]
    components: list[ComponentInfo]


# --- Popis komponent pro paradata ------------------------------------------


@dataclass(frozen=True)
class ComponentInfo:
    port: str
    name: str
    version: str
    params: dict[str, object] = field(default_factory=dict)
    licence: str | None = None
