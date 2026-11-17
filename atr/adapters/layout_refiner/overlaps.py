"""Obecné řešení překryvů mezi bloky rozvržení.

Nic specifického pro typ dokumentu, jen geometrie:

1. Vnořený duplikát: čitelný blok, který leží z ≥ `contained` uvnitř jiného
   čitelného bloku, se vypustí (jeho text se přečte s větším blokem).
   Bloky uvnitř obrázku zůstávají, obrázek se nečte.
2. Výrazný překryv dvou bloků stejného typu (překryv ≥ `merge` výšky, resp.
   šířky menšího z nich): bloky se sloučí do jednoho. Typicky dva řádkové
   bloky, které model posadil přes sebe; řez by přesekl písmena.
3. Malý překryv: společný pás se rozdělí napůl po té ose, kde je překryv
   menší. Výřez jednoho bloku pak nezasahuje do sousedního.
"""

from __future__ import annotations

from dataclasses import replace

from atr import __version__
from atr.ports.types import BBox, BlockType, ComponentInfo, LayoutBlock, PageGeometry
from atr.registry import register

# Bloky, které se nečtou a nemají se s ostatními dělit o plochu.
_NOT_READ = {BlockType.FIGURE}


def _inside(inner: BBox, outer: BBox) -> float:
    return inner.intersection(outer) / inner.area if inner.area > 0 else 0.0


@register("layout_refiner", "overlaps")
class OverlapRefiner:
    def __init__(self, contained: float = 0.85, merge: float = 0.25, min_size_pt: float = 1.0) -> None:
        self.contained = contained
        self.merge = merge
        self.min_size_pt = min_size_pt

    def refine(self, blocks: list[LayoutBlock], geometry: PageGeometry) -> list[LayoutBlock]:
        readable = [i for i, b in enumerate(blocks) if b.type not in _NOT_READ]

        # 1. vnořené duplikáty: projít od největších, menší uvnitř vypustit
        kept: list[int] = []
        for i in sorted(readable, key=lambda i: -blocks[i].bbox.area):
            if any(_inside(blocks[i].bbox, blocks[k].bbox) >= self.contained for k in kept):
                continue
            kept.append(i)

        # 2. výrazné překryvy bloků stejného typu: sloučit (opakovat, dokud se něco slučuje)
        boxes = {i: blocks[i].bbox for i in kept}
        merged = True
        while merged:
            merged = False
            for a in range(len(kept)):
                for b in range(a + 1, len(kept)):
                    i, j = kept[a], kept[b]
                    if blocks[i].type is blocks[j].type and self._strong(boxes[i], boxes[j]):
                        boxes[i] = _union(boxes[i], boxes[j])
                        kept.pop(b)
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break

        # 3. malé překryvy: dělit napůl
        for a in range(len(kept)):
            for b in range(a + 1, len(kept)):
                i, j = kept[a], kept[b]
                boxes[i], boxes[j] = _split(boxes[i], boxes[j])

        # Výsledek v původním pořadí; pořadí čtení řeší jiná komponenta.
        out = []
        for i, block in enumerate(blocks):
            if block.type in _NOT_READ:
                out.append(block)
            elif i in boxes and boxes[i].width >= self.min_size_pt and boxes[i].height >= self.min_size_pt:
                out.append(replace(block, bbox=boxes[i]))
        return out

    def _strong(self, a: BBox, b: BBox) -> bool:
        ow = min(a.x1, b.x1) - max(a.x0, b.x0)
        oh = min(a.y1, b.y1) - max(a.y0, b.y0)
        if ow <= 0 or oh <= 0:
            return False
        # překryv přes menší z rozměrů (nad sebou → výška, vedle sebe → šířka)
        if oh <= ow:
            return oh >= self.merge * min(a.height, b.height)
        return ow >= self.merge * min(a.width, b.width)

    def info(self) -> ComponentInfo:
        return ComponentInfo("layout_refiner", "overlaps", __version__,
                             {"contained": self.contained, "merge": self.merge, "min_size_pt": self.min_size_pt})


def _union(a: BBox, b: BBox) -> BBox:
    return BBox(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def _split(a: BBox, b: BBox) -> tuple[BBox, BBox]:
    ow = min(a.x1, b.x1) - max(a.x0, b.x0)
    oh = min(a.y1, b.y1) - max(a.y0, b.y0)
    if ow <= 0 or oh <= 0:
        return a, b
    # Řez vedeme přes menší z obou rozměrů překryvu.
    if oh <= ow:
        upper, lower = (a, b) if (a.y0 + a.y1) <= (b.y0 + b.y1) else (b, a)
        cut = (max(a.y0, b.y0) + min(a.y1, b.y1)) / 2
        new_upper = BBox(upper.x0, upper.y0, upper.x1, max(upper.y0, min(upper.y1, cut)))
        new_lower = BBox(lower.x0, min(lower.y1, max(lower.y0, cut)), lower.x1, lower.y1)
        return (new_upper, new_lower) if upper is a else (new_lower, new_upper)
    left, right = (a, b) if (a.x0 + a.x1) <= (b.x0 + b.x1) else (b, a)
    cut = (max(a.x0, b.x0) + min(a.x1, b.x1)) / 2
    new_left = BBox(left.x0, left.y0, max(left.x0, min(left.x1, cut)), left.y1)
    new_right = BBox(min(right.x1, max(right.x0, cut)), right.y0, right.x1, right.y1)
    return (new_left, new_right) if left is a else (new_right, new_left)
