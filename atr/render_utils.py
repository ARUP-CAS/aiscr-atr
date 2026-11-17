"""Pomocné funkce nad vykreslenou stranou."""

from __future__ import annotations

import io

from PIL import Image

from atr.ports.types import BBox, PageImage


def crop(image: PageImage, bbox: BBox, padding_pt: float = 0.0) -> PageImage:
    """Výřez bloku (bbox v bodech) z vykreslené strany jako nový PageImage."""
    scale = image.dpi / 72.0
    x0 = max(0, int((bbox.x0 - padding_pt) * scale))
    y0 = max(0, int((bbox.y0 - padding_pt) * scale))
    x1 = min(image.width_px, round((bbox.x1 + padding_pt) * scale))
    y1 = min(image.height_px, round((bbox.y1 + padding_pt) * scale))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"prázdný výřez pro {bbox} na straně {image.page}")
    with Image.open(io.BytesIO(image.png)) as img:
        part = img.crop((x0, y0, x1, y1))
        buf = io.BytesIO()
        part.save(buf, format="PNG")
    return PageImage(page=image.page, dpi=image.dpi, width_px=x1 - x0, height_px=y1 - y0, png=buf.getvalue())
