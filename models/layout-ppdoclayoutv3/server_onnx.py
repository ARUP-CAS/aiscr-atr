"""Služba rozvržení nad ONNX Runtime: stejné rozhraní jako server.py, bez PyTorch.

Model: výstup export_onnx.py (model.onnx + model.json s třídami). Následné zpracování
odpovídá PPDocLayoutV3ImageProcessor.post_process_object_detection (bez polygonů).
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import onnxruntime as ort
from common import PredictRequest, decode_image, dedupe
from fastapi import FastAPI, HTTPException
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/model.onnx")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.3"))
DEDUPE_IOU = float(os.environ.get("DEDUPE_IOU", "0.9"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(60_000_000)))
THREADS = int(os.environ.get("ORT_THREADS", "0"))  # 0 = podle počtu jader

app = FastAPI(title="layout-ppdoclayoutv3-onnx")
_state: dict = {}
_lock = threading.Lock()


@app.on_event("startup")
def startup() -> None:
    options = ort.SessionOptions()
    if THREADS:
        options.intra_op_num_threads = THREADS
    providers = [p for p in ("CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    _state["session"] = ort.InferenceSession(MODEL_PATH, options, providers=providers)
    meta_path = os.path.splitext(MODEL_PATH)[0] + ".json"
    with open(meta_path, encoding="utf-8") as fh:
        _state["meta"] = json.load(fh)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def order_sequence(order_logits: np.ndarray) -> np.ndarray:
    """Pořadí čtení pro každý dotaz (stejně jako PPDocLayoutV3ImageProcessor._get_order_seqs)."""
    scores = _sigmoid(order_logits)
    votes = np.triu(scores, k=1).sum(axis=0) + np.tril(1.0 - scores.T, k=-1).sum(axis=0)
    pointers = np.argsort(votes, kind="stable")
    seq = np.empty_like(pointers)
    seq[pointers] = np.arange(len(pointers))
    return seq


def postprocess(logits, boxes, order_logits, width: int, height: int, threshold: float) -> list[dict]:
    logits, boxes, order_logits = logits[0], boxes[0], order_logits[0]
    num_queries, num_classes = logits.shape
    seq = order_sequence(order_logits)
    scores = _sigmoid(logits).reshape(-1)
    top = np.argsort(-scores, kind="stable")[:num_queries]
    keep = top[scores[top] >= threshold]
    query, label = keep // num_classes, keep % num_classes
    cx, cy, w, h = boxes[query].T
    xyxy = np.stack([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height], 1)
    order = np.argsort(seq[query], kind="stable")
    return [{"label": int(label[i]), "score": float(scores[keep[i]]), "box": xyxy[i].tolist()} for i in order]


@app.get("/health")
def health() -> dict:
    if "session" not in _state:
        raise HTTPException(503, "model se načítá")
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    meta = _state.get("meta", {})
    return {
        "model": "PP-DocLayoutV3",
        "version": f"{meta.get('model_id')}@{str(meta.get('revision'))[:12]} (onnx)",
        "licence": "Apache-2.0",
        "labels": sorted(set(meta.get("labels", []))),
        "device": ",".join(_state["session"].get_providers()) if "session" in _state else None,
        "params": {"threshold": THRESHOLD, "dedupe_iou": DEDUPE_IOU},
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    image = decode_image(req, MAX_PIXELS)

    size = _state["meta"]["size"]
    pixels = np.asarray(image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
    pixels = (pixels / 255.0).transpose(2, 0, 1)[None]
    with _lock:
        logits, boxes, order_logits = _state["session"].run(None, {"pixel_values": pixels})
    labels = _state["meta"]["labels"]
    blocks = []
    for i, det in enumerate(postprocess(logits, boxes, order_logits, image.width, image.height, THRESHOLD)):
        x0, y0, x1, y1 = det["box"]
        blocks.append({
            "label": labels[det["label"]],
            "bbox_px": [round(max(0.0, x0), 1), round(max(0.0, y0), 1),
                        round(min(float(image.width), x1), 1), round(min(float(image.height), y1), 1)],
            "confidence": round(det["score"], 4),
            "order": i,
            "polygon_px": None,
        })
    return {"width_px": image.width, "height_px": image.height, "blocks": dedupe(blocks, DEDUPE_IOU)}
