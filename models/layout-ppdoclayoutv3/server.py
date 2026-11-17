"""Služba rozvržení: PP-DocLayoutV3 (Apache-2.0) přes transformers.

Rozhraní podle docs/model-services.md:
  GET  /health   {"status": "ok"}, až je model načtený
  GET  /info     model, revize, licence, třídy
  POST /predict  {"image_png_base64": "..."} → {"blocks": [{label, bbox_px, confidence, order, polygon_px}]}

Bloky jsou v pořadí čtení, které předpovídá model. Překrývající se detekce
téže oblasti s různou třídou (IoU ≥ DEDUPE_IOU) se sloučí na tu s vyšší jistotou.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = os.environ.get("MODEL_ID", "PaddlePaddle/PP-DocLayoutV3_safetensors")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "e4f489dc536556fc1be02b973ba06756e0d4a2ac")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.3"))
DEDUPE_IOU = float(os.environ.get("DEDUPE_IOU", "0.9"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(60_000_000)))

app = FastAPI(title="layout-ppdoclayoutv3")
_lock = threading.Lock()
_state: dict = {}


def _load() -> None:
    t = time.time()
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(DEVICE).eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    _state.update(model=model, processor=processor, load_s=round(time.time() - t, 2))


@app.on_event("startup")
def startup() -> None:
    _load()


class PredictRequest(BaseModel):
    image_png_base64: str


def _iou(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(blocks: list[dict]) -> list[dict]:
    keep: list[dict] = []
    for block in blocks:
        twin = next((k for k in keep if _iou(k["bbox_px"], block["bbox_px"]) >= DEDUPE_IOU), None)
        if twin is None:
            keep.append(block)
        elif block["confidence"] > twin["confidence"]:
            keep[keep.index(twin)] = {**block, "order": twin["order"]}
    for i, block in enumerate(keep):
        block["order"] = i
    return keep


@app.get("/health")
def health() -> dict:
    if "model" not in _state:
        raise HTTPException(503, "model se načítá")
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    model = _state.get("model")
    labels = sorted(set(model.config.id2label.values())) if model else []
    return {
        "model": "PP-DocLayoutV3",
        "version": f"{MODEL_ID}@{MODEL_REVISION[:12]}",
        "licence": "Apache-2.0",
        "labels": labels,
        "device": DEVICE,
        "params": {"threshold": THRESHOLD, "dedupe_iou": DEDUPE_IOU},
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image_png_base64)))
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(413, f"obrázek má víc než {MAX_PIXELS} pixelů")
        image = image.convert("RGB")
    except (UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(400, f"neplatný obrázek: {exc}") from exc

    model, processor = _state["model"], _state["processor"]
    with _lock, torch.inference_mode():
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        result = processor.post_process_object_detection(
            outputs, target_sizes=[image.size[::-1]], threshold=THRESHOLD)[0]

    blocks = []
    polygons = result.get("polygon_points", [None] * len(result["scores"]))
    for i, (score, label, box, polygon) in enumerate(
            zip(result["scores"], result["labels"], result["boxes"], polygons)):
        x0, y0, x1, y1 = (max(0.0, float(v)) for v in box.tolist())
        blocks.append({
            "label": model.config.id2label[int(label)],
            "bbox_px": [round(x0, 1), round(y0, 1), round(min(x1, image.width), 1),
                        round(min(y1, image.height), 1)],
            "confidence": round(float(score), 4),
            "order": i,  # výsledky jsou seřazené podle pořadí čtení z modelu
            "polygon_px": polygon.tolist() if polygon is not None else None,
        })
    return {"width_px": image.width, "height_px": image.height, "blocks": _dedupe(blocks)}
