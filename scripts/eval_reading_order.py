"""Vyhodnocení pořadí čtení proti pořadí oblastí v exportu eScriptorium.

Pro každou stranu: bloky ze služby rozvržení → úprava překryvů → každý
adaptér pořadí čtení. Anotované oblasti se spárují s bloky (blok, který oblast
nejlépe pokrývá) a pro spárované dvojice se spočítá Kendallovo tau mezi
anotovaným a předpovězeným pořadím (1 = stejné, 0 = náhodné, −1 = obrácené).
Pro srovnání je uvedena i naivní varianta „shora dolů, zleva doprava“.

Pozor: pořadí oblastí v eScriptoriu je spolehlivé jen tam, kde ho anotátoři
opravdu řadili; výsledek je proto orientační.

  python scripts/eval_reading_order.py EXPORT_DIR [...] --layout-url http://127.0.0.1:18002
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
from pathlib import Path

import httpx
from PIL import Image

from atr import registry
from atr.ports.types import BBox, BlockType, LayoutBlock, PageGeometry

LABELS = {"header": "page_header", "footer": "page_footer", "number": "page_footer", "footnote": "footnote",
          "vision_footnote": "footnote", "paragraph_title": "heading", "doc_title": "title", "table": "table",
          "image": "figure", "chart": "figure", "seal": "figure", "figure_title": "caption",
          "formula": "formula"}


def kendall(seq: list[int]) -> float | None:
    n = len(seq)
    if n < 3:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                concordant += 1
            elif seq[i] > seq[j]:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


def overlap(a: BBox, b: BBox) -> float:
    return a.intersection(b) / min(a.area, b.area) if min(a.area, b.area) > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+", type=Path)
    ap.add_argument("--layout-url", default="http://127.0.0.1:18002")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    args = ap.parse_args()

    registry.load_builtin_adapters()
    refiner = registry.create("layout_refiner", "overlaps")
    orders = {"model": registry.create("reading_order", "model"),
              "xycut": registry.create("reading_order", "xycut")}
    client = httpx.Client(timeout=120)
    scores: dict[str, list[float]] = {name: [] for name in [*orders, "shora-dolů"]}
    gt_vs_topdown: list[float] = []

    for export in args.exports:
        for line in (export / "parts.jsonl").open(encoding="utf-8"):
            part = json.loads(line)
            path = export / part["image"]["filename"]
            if not path.is_file():
                continue
            img = Image.open(path)
            img.seek(0)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            resp = client.post(f"{args.layout_url}/predict",
                               json={"image_png_base64": base64.b64encode(buf.getvalue()).decode()})
            resp.raise_for_status()
            geometry = PageGeometry(1, img.width, img.height, 0)  # px stačí, poměry se nemění
            blocks = [LayoutBlock(BlockType(LABELS.get(b["label"], "text")), BBox(*b["bbox_px"]),
                                  b["confidence"], b["order"]) for b in resp.json()["blocks"]]
            blocks = refiner.refine(blocks, geometry)

            regions = sorted(part["regions"], key=lambda r: r["order"])
            gt = [(r["order"], BBox(*r["bbox"])) for r in regions if r["bbox"][2] > r["bbox"][0]]
            topdown = sorted(range(len(gt)), key=lambda i: (gt[i][1].y0, gt[i][1].x0))
            tau = kendall([gt[i][0] for i in topdown])
            if tau is not None:
                gt_vs_topdown.append(tau)

            variants = {name: o.order(blocks, geometry) for name, o in orders.items()}
            variants["shora-dolů"] = sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
            for name, ordered in variants.items():
                seq, used = [], set()
                for block in ordered:
                    best = max(((overlap(block.bbox, box), o) for o, box in gt), default=(0, None))
                    if best[0] >= args.min_overlap and best[1] not in used:
                        used.add(best[1])
                        seq.append(best[1])
                tau = kendall(seq)
                if tau is not None:
                    scores[name].append(tau)

    print(f"Stran s aspoň 3 spárovanými oblastmi: {len(scores['model'])}")
    print(f"Anotované pořadí vs. shora-dolů (jak moc anotace kopíruje polohu): "
          f"{statistics.mean(gt_vs_topdown):.3f}")
    print(f"\n{'pořadí čtení':14s} {'průměr τ':>9s} {'medián τ':>9s} {'stran τ<0,9':>12s}")
    for name, values in scores.items():
        print(f"{name:14s} {statistics.mean(values):9.3f} {statistics.median(values):9.3f} "
              f"{sum(v < 0.9 for v in values):12d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
