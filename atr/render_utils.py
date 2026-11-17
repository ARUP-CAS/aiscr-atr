"""Pomocné funkce nad vykreslenou stranou."""

from __future__ import annotations

import io

from PIL import Image

from atr.ports.types import BBox, PageImage


def decode(image: PageImage) -> Image.Image:
    """Dekóduje PNG strany jednou; výsledek jde předat `crop` pro všechny bloky strany."""
    img = Image.open(io.BytesIO(image.png))
    img.load()
    return img


def crop(image: PageImage, bbox: BBox, padding_pt: float = 0.0, mask: list[BBox] | None = None,
         decoded: Image.Image | None = None) -> PageImage:
    """Výřez bloku (bbox v bodech) z vykreslené strany jako nový PageImage.

    `mask`: obdélníky sousedních bloků; jejich části mimo vlastní blok se ve
    výřezu zabílí, aby model nečetl kusy sousedního textu z okraje výřezu.
    `decoded`: už dekódovaná strana (`decode`), aby se PNG nečetlo pro každý blok znovu.
    """
    scale = image.dpi / 72.0
    x0 = max(0, int((bbox.x0 - padding_pt) * scale))
    y0 = max(0, int((bbox.y0 - padding_pt) * scale))
    x1 = min(image.width_px, round((bbox.x1 + padding_pt) * scale))
    y1 = min(image.height_px, round((bbox.y1 + padding_pt) * scale))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"prázdný výřez pro {bbox} na straně {image.page}")
    part = (decoded if decoded is not None else decode(image)).crop((x0, y0, x1, y1))
    if mask:
        part = _mask(part, image.dpi / 72.0, (x0, y0), bbox, mask)
    buf = io.BytesIO()
    part.save(buf, format="PNG")
    return PageImage(page=image.page, dpi=image.dpi, width_px=x1 - x0, height_px=y1 - y0, png=buf.getvalue())


def _mask(part: Image.Image, scale: float, origin: tuple[int, int], own: BBox, others: list[BBox]) -> Image.Image:
    from PIL import ImageDraw

    ox, oy = origin
    own_px = (own.x0 * scale - ox, own.y0 * scale - oy, own.x1 * scale - ox, own.y1 * scale - oy)
    part = part.copy()
    draw = ImageDraw.Draw(part)
    white = 255 if part.mode == "L" else (255, 255, 255)
    for other in others:
        x0, y0 = other.x0 * scale - ox, other.y0 * scale - oy
        x1, y1 = other.x1 * scale - ox, other.y1 * scale - oy
        if x1 <= 0 or y1 <= 0 or x0 >= part.width or y0 >= part.height:
            continue
        # Zabílit jen pruhy souseda, které leží mimo vlastní blok (překryv s ním zůstává).
        for rect in _subtract((x0, y0, x1, y1), own_px):
            draw.rectangle(rect, fill=white)
    return part


def _subtract(r: tuple, keep: tuple) -> list[tuple]:
    """Obdélník r bez obdélníku keep, jako až čtyři obdélníky."""
    x0, y0, x1, y1 = r
    kx0, ky0, kx1, ky1 = keep
    if kx1 <= x0 or kx0 >= x1 or ky1 <= y0 or ky0 >= y1:
        return [r]
    out = []
    if y0 < ky0:
        out.append((x0, y0, x1, ky0))
    if ky1 < y1:
        out.append((x0, ky1, x1, y1))
    my0, my1 = max(y0, ky0), min(y1, ky1)
    if x0 < kx0:
        out.append((x0, my0, kx0, my1))
    if kx1 < x1:
        out.append((kx1, my0, x1, my1))
    return [o for o in out if o[2] > o[0] and o[3] > o[1]]
