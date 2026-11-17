"""Převod PP-DocLayoutV3 (transformers) do ONNX a ověření shody s PyTorch.

Výstupy grafu: logits [1, 300, C], pred_boxes [1, 300, 4] (cx, cy, w, h; 0–1),
order_logits [1, 300, 300]. Masky (polygony) se nevyvážejí, potrubí je nepoužívá.

  python export_onnx.py --out model.onnx [--model-id ID --revision REV] [--check obrázek.png ...]

Funguje i pro dotrénovaný model (--model-id /cesta/k/runs/hw-v1/best --revision main).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForObjectDetection

SIZE = 800


class Wrapper(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes, out.order_logits


def _constant_position_embedding() -> None:
    """Poziční embedding se počítá ve float64, který ONNX Runtime pro Cos/Sin nemá.

    Závisí jen na tvaru vstupu (pevně 800 × 800), takže se při exportu spočítá mimo
    trasování a do grafu se vloží jako konstanta.
    """
    import transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 as modeling

    original = modeling.build_2d_sinusoidal_position_embedding

    def constant(*args, **kwargs):
        state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)
        try:
            return original(*args, **kwargs).detach().clone()
        finally:
            torch._C._set_tracing_state(state)

    modeling.build_2d_sinusoidal_position_embedding = constant
    for cls in vars(modeling).values():
        if isinstance(cls, type) and hasattr(cls, "_cached_build_2d_sinusoidal_position_embedding"):
            cls._cached_build_2d_sinusoidal_position_embedding = staticmethod(constant)


def preprocess(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC), dtype=np.float32)
    return (arr / 255.0).transpose(2, 0, 1)[None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="PaddlePaddle/PP-DocLayoutV3_safetensors")
    ap.add_argument("--revision", default="e4f489dc536556fc1be02b973ba06756e0d4a2ac")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", type=Path, nargs="*", default=[], help="obrázky pro ověření shody")
    args = ap.parse_args()

    model = AutoModelForObjectDetection.from_pretrained(args.model_id, revision=args.revision).eval()
    wrapper = Wrapper(model).eval()
    dummy = torch.rand(1, 3, SIZE, SIZE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _constant_position_embedding()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (dummy,), str(args.out), opset_version=args.opset, dynamo=False,
            input_names=["pixel_values"], output_names=["logits", "pred_boxes", "order_logits"],
        )
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    meta = {"model_id": args.model_id, "revision": args.revision, "labels": labels, "size": SIZE,
            "opset": args.opset}
    args.out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"uloženo {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")

    if args.check:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        for path in args.check:
            x = preprocess(Image.open(path))
            t = time.time()
            with torch.inference_mode():
                ref = [o.numpy() for o in wrapper(torch.from_numpy(x))]
            t_torch = time.time() - t
            t = time.time()
            got = session.run(None, {"pixel_values": x})
            t_onnx = time.time() - t
            diffs = [float(np.abs(a - b).max()) for a, b in zip(ref, got)]
            print(f"{path.name}: max rozdíl logits {diffs[0]:.2e}, boxy {diffs[1]:.2e}, pořadí {diffs[2]:.2e}; "
                  f"torch {t_torch:.2f} s, onnx {t_onnx:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
