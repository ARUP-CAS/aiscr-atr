"""Vyhodnocení služby rozvržení proti anotacím SegmOnto z exportu eScriptorium.

Pro každou stranu exportu (parts.jsonl + obrázky) pošle originální obrázek na
`POST {layout_url}/predict` a porovná bloky s ručně anotovanými oblastmi.
Výsledek pro každou anotovanou oblast:
  shoda     – některý blok ji pokrývá s IoU ≥ --iou;
  rozděleno – bloky ležící uvnitř ní ji dohromady pokryjí (model dělí jemněji);
  v širším  – leží uvnitř většího bloku (např. pole formuláře v tabulce); text se přečte s ním;
  chybí     – nic z toho; text oblasti by se při OCR ztratil.

Použití:
  python scripts/eval_layout_segmonto.py EXPORT_DIR [EXPORT_DIR ...] --layout-url http://127.0.0.1:18002
"""

from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

# SegmOnto zóna → skupina pro vyhodnocení (co od rozvržení čekáme)
GROUPS = {
    "MainZone-Head": "nadpis",
    "MainZone-HeadStructured": "nadpis",
    "RunningTitleZone": "záhlaví/zápatí",
    "NumberingZone": "záhlaví/zápatí",
    "MainZone-P": "odstavec",
    "MainZone-PStructured": "odstavec",
    "MainZone-PStyled": "odstavec",
    "MainZone-Ab": "odstavec",
    "MainZone-Item": "odstavec",
    "MainZone-Signed": "podpis/datum/adresa",
    "MainZone-Dateline": "podpis/datum/adresa",
    "MainZone-Address": "podpis/datum/adresa",
    "MarginTextZone-ManuscriptAddendum": "rukopisný doplněk",
    "MarginTextZone-PStructured": "okraj",
    "MarginTextZone-Ab": "okraj",
    "MarginTextZone-PStyled": "okraj",
    "FormZone": "formulář",
    "FormZone-Head": "formulář",
    "FormZone-Part": "formulář",
    "FormZone-Field": "pole formuláře",
    "TableZone": "tabulka",
    "TableZone-Head": "tabulka",
    "GraphicZone": "obrázek/razítko",
    "GraphicZone-Decoration": "obrázek/razítko",
    "StampZone": "obrázek/razítko",
}


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def inside(inner, outer) -> float:
    """Podíl plochy `inner`, který leží v `outer`."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (w * h / area) if w > 0 and h > 0 and area > 0 else 0.0


def covered(gt, preds, cover: float) -> float:
    """Kolik plochy oblasti pokrývají bloky, které v ní leží (hrubý odhad bez odečtu překryvů)."""
    area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    if area <= 0:
        return 0.0
    total = 0.0
    for p in preds:
        if inside(p, gt) >= cover:
            w = min(p[2], gt[2]) - max(p[0], gt[0])
            h = min(p[3], gt[3]) - max(p[1], gt[1])
            total += max(w, 0) * max(h, 0)
    return min(total / area, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+", type=Path)
    ap.add_argument("--layout-url", default="http://127.0.0.1:18002")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--cover", type=float, default=0.8)
    ap.add_argument("--min-covered", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0, help="nejvýš tolik stran z každého exportu")
    ap.add_argument("--out", type=Path, help="uložit podrobnosti po stranách (JSON)")
    ap.add_argument("--only", type=Path,
                    help="soubor se jmény obrázků (bez přípony, jedno na řádek); ostatní strany přeskočit, "
                         "např. strany, které model neviděl při tréninku")
    args = ap.parse_args()

    client = httpx.Client(timeout=120)
    stats = collections.defaultdict(lambda: collections.Counter())
    pred_labels = collections.Counter()
    matched_labels = collections.defaultdict(collections.Counter)
    details, times = [], []

    only = ({line.strip() for line in args.only.read_text(encoding="utf-8").splitlines() if line.strip()}
            if args.only else None)
    for export in args.exports:
        parts = [json.loads(line) for line in (export / "parts.jsonl").open(encoding="utf-8")]
        if only is not None:
            parts = [p for p in parts if p["image"]["filename"].rsplit(".", 1)[0] in only]
        for part in parts[: args.limit or None]:
            path = export / part["image"]["filename"]
            if not path.is_file():
                print(f"chybí obrázek {path.name}", file=sys.stderr)
                continue
            img = Image.open(path)
            img.seek(0)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            t = time.time()
            resp = client.post(f"{args.layout_url}/predict",
                               json={"image_png_base64": base64.b64encode(buf.getvalue()).decode()})
            resp.raise_for_status()
            times.append(time.time() - t)
            blocks = resp.json()["blocks"]
            preds = [b["bbox_px"] for b in blocks]
            pred_labels.update(b["label"] for b in blocks)

            page = {"file": path.name, "export": export.name, "regions": []}
            for region in part["regions"]:
                zone = (region.get("typology") or {}).get("name") or "Undefined"
                group = GROUPS.get(zone, "ostatní")
                gt = region["bbox"]
                best, best_label = max(((iou(gt, b["bbox_px"]), b["label"]) for b in blocks), default=(0.0, None))
                cov = covered(gt, preds, args.cover)
                if best >= args.iou:
                    result = "shoda"
                elif cov >= args.min_covered:
                    result = "rozděleno"
                elif any(inside(gt, p) >= args.cover for p in preds):
                    result = "v širším"  # text se přečte jako součást většího bloku (tabulka, formulář)
                else:
                    result = "chybí"
                stats[group][result] += 1
                if result == "shoda":
                    matched_labels[group][best_label] += 1
                page["regions"].append({"zone": zone, "bbox": gt, "best_iou": round(best, 3),
                                        "covered": round(cov, 3), "result": result})
            details.append(page)
            print(f"{export.name[:22]} {path.name}: {len(part['regions'])} oblastí, {len(preds)} bloků",
                  file=sys.stderr)

    print(f"\nStran: {len(details)}, průměrná doba rozvržení: {sum(times) / max(len(times), 1):.2f} s\n")
    print(f"{'skupina':24s} {'oblastí':>8s} {'shoda':>7s} {'rozděl.':>8s} {'v širším':>9s} {'chybí':>7s} {'pokryto':>8s}")
    total = collections.Counter()
    for group, c in sorted(stats.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        total.update(c)
        found = 1 - c["chybí"] / n
        print(f"{group:24s} {n:8d} {c['shoda']:7d} {c['rozděleno']:8d} {c['v širším']:9d} {c['chybí']:7d} "
              f"{found:8.0%}")
    n = sum(total.values())
    print(f"{'celkem':24s} {n:8d} {total['shoda']:7d} {total['rozděleno']:8d} {total['v širším']:9d} "
          f"{total['chybí']:7d} {1 - total['chybí'] / n:8.0%}")
    print("\nTřída bloku u shod (nejčastější):")
    for group, c in sorted(matched_labels.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {group:24s} " + ", ".join(f"{k} {v}" for k, v in c.most_common(4)))
    print("\nTřídy modelu:", ", ".join(f"{k} {v}" for k, v in pred_labels.most_common()))
    if args.out:
        args.out.write_text(json.dumps(details, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
