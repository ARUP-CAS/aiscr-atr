"""Společné části služeb rozvržení (server.py pro PyTorch, server_onnx.py pro ONNX Runtime)."""

from __future__ import annotations

import base64
import binascii
import io

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel


class PredictRequest(BaseModel):
    image_png_base64: str


def decode_image(req: PredictRequest, max_pixels: int) -> Image.Image:
    """Base64 → obrázek RGB; 400 pro neplatná data, 413 pro příliš velký obrázek."""
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image_png_base64)))
    except (UnidentifiedImageError, ValueError, binascii.Error) as exc:
        raise HTTPException(400, f"neplatný obrázek: {exc}") from exc
    if image.width * image.height > max_pixels:
        raise HTTPException(413, f"obrázek má víc než {max_pixels} pixelů")
    return image.convert("RGB")


def iou(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe(blocks: list[dict], threshold: float) -> list[dict]:
    """Detekce téže oblasti s různou třídou (IoU ≥ threshold) sloučí na tu s vyšší jistotou."""
    keep: list[dict] = []
    for block in blocks:
        twin = next((k for k in keep if iou(k["bbox_px"], block["bbox_px"]) >= threshold), None)
        if twin is None:
            keep.append(block)
        elif block["confidence"] > twin["confidence"]:
            keep[keep.index(twin)] = {**block, "order": twin["order"]}
    for i, block in enumerate(keep):
        block["order"] = i
    return keep
