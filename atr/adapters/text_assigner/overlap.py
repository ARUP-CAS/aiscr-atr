"""Přiřazení slov textové vrstvy do bloků rozvržení podle překryvu."""

from __future__ import annotations

from atr import __version__
from atr.ports.types import (
    NON_TEXT_TYPES,
    BBox,
    Block,
    BlockLine,
    BlockType,
    ComponentInfo,
    LayoutBlock,
    Line,
    PositionAccuracy,
    TextLayer,
    Word,
)
from atr.registry import register

# Typy, do kterých se text z textové vrstvy nepřiřazuje (ve výstupu jen polohou).
_NO_TEXT = NON_TEXT_TYPES


@register("text_assigner", "overlap")
class OverlapTextAssigner:
    """Slovo patří bloku, se kterým má největší překryv, pokud ten pokrývá aspoň
    `min_overlap` plochy slova. Slova mimo bloky skončí v doplňkových textových
    blocích po řádcích, aby se žádný text neztratil.
    """

    def __init__(self, min_overlap: float = 0.5) -> None:
        self.min_overlap = min_overlap

    def assign(self, layer: TextLayer, blocks: list[LayoutBlock]) -> list[Block]:
        assigned: dict[int, list[tuple[int, Word]]] = {i: [] for i in range(len(blocks))}
        orphans: dict[int, list[Word]] = {}

        for line_idx, line in enumerate(layer.lines):
            for word in line.words:
                best, best_ratio = None, 0.0
                for i, block in enumerate(blocks):
                    if block.type in _NO_TEXT or word.bbox.area <= 0:
                        continue
                    ratio = word.bbox.intersection(block.bbox) / word.bbox.area
                    if ratio > best_ratio:
                        best, best_ratio = i, ratio
                if best is not None and best_ratio >= self.min_overlap:
                    assigned[best].append((line_idx, word))
                else:
                    orphans.setdefault(line_idx, []).append(word)

        out: list[Block] = []
        for i, block in enumerate(blocks):
            lines = _group_lines(assigned[i])
            out.append(
                Block(
                    type=block.type,
                    bbox=block.bbox,
                    confidence=block.confidence,
                    order=block.order if block.order is not None else i,
                    text="\n".join(l.text for l in lines),
                    lines=lines,
                )
            )
        for line_idx in sorted(orphans):
            words = orphans[line_idx]
            line = _line(words)
            out.append(
                Block(
                    type=BlockType.TEXT,
                    bbox=line.bbox,
                    confidence=0.0,
                    order=len(out),
                    text=line.text,
                    lines=[BlockLine(line.text, line.bbox, PositionAccuracy.MEASURED)],
                )
            )
        return out

    def info(self) -> ComponentInfo:
        return ComponentInfo("text_assigner", "overlap", __version__, {"min_overlap": self.min_overlap})


def _line(words: list[Word]) -> Line:
    words = sorted(words, key=lambda w: w.bbox.x0)
    box = BBox(
        min(w.bbox.x0 for w in words),
        min(w.bbox.y0 for w in words),
        max(w.bbox.x1 for w in words),
        max(w.bbox.y1 for w in words),
    )
    return Line(" ".join(w.text for w in words), box, tuple(words))


def _group_lines(items: list[tuple[int, Word]]) -> list[BlockLine]:
    by_line: dict[int, list[Word]] = {}
    for line_idx, word in items:
        by_line.setdefault(line_idx, []).append(word)
    lines = [_line(ws) for _, ws in sorted(by_line.items())]
    return [BlockLine(l.text, l.bbox, PositionAccuracy.MEASURED) for l in lines]
