"""Pořadí čtení.

`model`  – ponechá pořadí, které vrátil model rozvržení (PP-DocLayoutV3 ho předpovídá);
           bloky bez pořadí zařadí podle polohy.
`xycut`  – rekurzivní XY-cut: stranu dělí podle mezer mezi bloky střídavě vodorovně
           a svisle, pak čte shora dolů a zleva doprava. Záhlaví na začátek,
           poznámky pod čarou a zápatí na konec.

Oba algoritmy jsou obecné, nezávislé na typu dokumentu.
"""

from __future__ import annotations

from dataclasses import replace

from atr import __version__
from atr.ports.types import BlockType, ComponentInfo, LayoutBlock, PageGeometry
from atr.registry import register

_HEAD = {BlockType.PAGE_HEADER}
_TAIL = {BlockType.FOOTNOTE, BlockType.PAGE_FOOTER}


def _numbered(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    return [replace(b, order=i) for i, b in enumerate(blocks)]


@register("reading_order", "model")
class ModelReadingOrder:
    def order(self, blocks: list[LayoutBlock], geometry: PageGeometry) -> list[LayoutBlock]:
        ranked = sorted(enumerate(blocks), key=lambda ib: (
            (0, ib[1].order, ib[0]) if ib[1].order is not None else (1, ib[1].bbox.y0, ib[1].bbox.x0)))
        return _numbered([b for _, b in ranked])

    def info(self) -> ComponentInfo:
        return ComponentInfo("reading_order", "model", __version__)


@register("reading_order", "xycut")
class XYCutReadingOrder:
    def __init__(self, min_gap_pt: float = 3.0, sections: bool = True) -> None:
        self.min_gap_pt = min_gap_pt
        self.sections = sections  # záhlaví první, poznámky a zápatí poslední

    def order(self, blocks: list[LayoutBlock], geometry: PageGeometry) -> list[LayoutBlock]:
        if self.sections:
            head = [b for b in blocks if b.type in _HEAD]
            tail = [b for b in blocks if b.type in _TAIL]
            body = [b for b in blocks if b.type not in _HEAD | _TAIL]
            ordered = self._cut(head) + self._cut(body) + self._cut(tail)
        else:
            ordered = self._cut(blocks)
        return _numbered(ordered)

    def _cut(self, blocks: list[LayoutBlock], horizontal_first: bool = True) -> list[LayoutBlock]:
        if len(blocks) <= 1:
            return list(blocks)
        for horizontal in (horizontal_first, not horizontal_first):
            groups = self._split(blocks, horizontal)
            if len(groups) > 1:
                return [b for g in groups for b in self._cut(g, not horizontal)]
        # Nedělitelné (bloky se v obou osách překrývají): shora dolů, zleva doprava.
        return sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

    def _split(self, blocks: list[LayoutBlock], horizontal: bool) -> list[list[LayoutBlock]]:
        """Rozdělí bloky podle mezer v projekci na osu y (horizontal) nebo x."""
        lo = (lambda b: b.bbox.y0) if horizontal else (lambda b: b.bbox.x0)
        hi = (lambda b: b.bbox.y1) if horizontal else (lambda b: b.bbox.x1)
        groups: list[list[LayoutBlock]] = []
        end = None
        for block in sorted(blocks, key=lo):
            if end is None or lo(block) > end + self.min_gap_pt:
                groups.append([block])
                end = hi(block)
            else:
                groups[-1].append(block)
                end = max(end, hi(block))
        return groups

    def info(self) -> ComponentInfo:
        return ComponentInfo("reading_order", "xycut", __version__,
                             {"min_gap_pt": self.min_gap_pt, "sections": self.sections})
