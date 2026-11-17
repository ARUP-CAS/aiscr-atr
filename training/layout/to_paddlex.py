"""Převod datové sady COCO (build_dataset.py) do formátu PaddleX pro PP-DocLayoutV3.

PaddleX (modul layout_analysis, dataset COCOInstSegDataset) chce u každého objektu
polygon v `segmentation` a pořadí čtení `read_order` (0, 1, 2, … v rámci strany).
LADaS pořadí čtení nemá a nepokrývá všechny třídy V3. Obojí se doplní z předpovědí
původního PP-DocLayoutV3 (služba rozvržení, `--layout-url`):

* read_order: anotovaný blok převezme pořadí bloku modelu, se kterým se nejvíc
  překrývá; bloky bez protějšku se zařadí podle XY-cut mezi nejbližší spárované;
* pseudo-popisky: předpovědi modelu ve třídách, které datová sada nemá
  (`--pseudo-classes`), s jistotou ≥ --pseudo-min-conf a bez překryvu s anotací,
  se přidají jako anotace s příznakem `pseudo: true`. Model tak tyto třídy
  při dotrénování nezapomene.

Bez `--layout-url` se pořadí určí jen XY-cut a pseudo-popisky se nepřidají.

Výstup (OUT): images/ (odkazy na obrázky), annotations/instance_{train,val,test}.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import httpx
from PIL import Image

from atr.adapters.reading_order.adapters import XYCutReadingOrder
from atr.ports.types import BBox, BlockType, LayoutBlock, PageGeometry

# Třídy V3, které LADaS ani anotace AMČR nemají (bez nejednoznačných jmen ze služby:
# formula/header/footer/text tam slučují dvojice tříd).
DEFAULT_PSEUDO = ["abstract", "algorithm", "chart", "content", "formula_number", "reference",
                  "reference_content", "vision_footnote"]


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def xyxy(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def predict(client: httpx.Client, url: str, path: Path) -> list[dict]:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = client.post(f"{url}/predict", json={"image_png_base64": base64.b64encode(buf.getvalue()).decode()})
    resp.raise_for_status()
    return resp.json()["blocks"]


def order_annotations(anns: list[dict], preds: list[dict], width: int, height: int, min_iou: float) -> None:
    """Doplní ann['read_order'] (0..n-1)."""
    geometry = PageGeometry(1, width, height, 0)
    blocks = [LayoutBlock(BlockType.TEXT, BBox(*xyxy(a["bbox"])), 1.0, i) for i, a in enumerate(anns)]
    geo = [b.order for b in XYCutReadingOrder(sections=False).order(blocks, geometry)]
    # geo[k] = index anotace na k-tém místě podle XY-cut
    matched: dict[int, float] = {}
    for i, a in enumerate(anns):
        box = xyxy(a["bbox"])
        best = max(((iou(box, p["bbox_px"]), p["order"]) for p in preds), default=(0.0, None))
        if best[0] >= min_iou:
            matched[i] = float(best[1])
    keys: dict[int, float] = {}
    last = -1.0
    for rank, i in enumerate(geo):
        if i in matched:
            last = matched[i]
            keys[i] = matched[i]
        else:
            keys[i] = last + 0.5 + rank * 1e-6  # za nejbližší předchozí spárovaný blok
    for n, i in enumerate(sorted(range(len(anns)), key=lambda i: keys[i])):
        anns[i]["read_order"] = n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="výstup build_dataset.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout-url", help="služba PP-DocLayoutV3 pro pořadí čtení a pseudo-popisky")
    ap.add_argument("--pseudo-classes", nargs="*", default=DEFAULT_PSEUDO)
    ap.add_argument("--pseudo-min-conf", type=float, default=0.5)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="nejvýš tolik stran na část (na zkoušku)")
    args = ap.parse_args()

    (args.out / "annotations").mkdir(parents=True, exist_ok=True)
    client = httpx.Client(timeout=300) if args.layout_url else None
    summary = {}
    for split in ("train", "val", "test"):
        src = json.loads((args.dataset / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in src["categories"]}
        ids = {v: k for k, v in names.items()}
        by_image: dict[int, list[dict]] = {}
        for a in src["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        images = src["images"][: args.limit or None]
        out_anns, next_id, pseudo = [], 1, 0
        img_dir = args.out / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        for k, image in enumerate(images):
            path = args.dataset / "images" / split / image["file_name"]
            link = img_dir / image["file_name"]
            if not link.exists():
                os.symlink(path.resolve(), link)
            anns = [dict(a) for a in by_image.get(image["id"], [])]
            preds = predict(client, args.layout_url, path) if client else []
            for p in preds:
                if (p["label"] in args.pseudo_classes and p["confidence"] >= args.pseudo_min_conf
                        and all(iou(p["bbox_px"], xyxy(a["bbox"])) < 0.3 for a in anns)):
                    x0, y0, x1, y1 = p["bbox_px"]
                    anns.append({"image_id": image["id"], "category_id": ids[p["label"]],
                                 "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                                 "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]], "iscrowd": 0,
                                 "pseudo": True})
                    pseudo += 1
            if anns:
                order_annotations(anns, preds, image["width"], image["height"], args.match_iou)
            for a in anns:
                a["id"] = next_id
                next_id += 1
                out_anns.append(a)
            if client and (k + 1) % 100 == 0:
                print(f"{split}: {k + 1}/{len(images)}", flush=True)
        out = {"images": images, "annotations": out_anns, "categories": src["categories"]}
        (args.out / "annotations" / f"instance_{split}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8")
        summary[split] = {"images": len(images), "annotations": len(out_anns), "pseudo": pseudo}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
