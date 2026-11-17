"""Datová sada pro dotrénování PP-DocLayoutV3 se třídou `handwriting` (formát COCO).

Zdroje:
  * LADaS (Inria, CC-BY-4.0), parquet z Hugging Face `almanach/LADaS`, rozdělení train/validation/test
    převzaté beze změny;
  * volitelně exporty z eScriptoria (JSON, SegmOnto) – anotace AMČR. Strany se rozdělí podle
    dokumentu do train/val/test (`--escriptorium-split`), stejný soubor nikdy ve dvou částech.

Třídy a převod zón: `mapping.yaml`. Výstup:

  OUT/
    images/{train,val,test}/<id>.<ext>
    annotations/instances_{train,val,test}.json      COCO: bbox [x, y, w, h] v pixelech,
                                                      segmentation (polygon; u LADaS obdélník)
    stats.json                                        počty tříd po částech a zdrojích
    MANIFEST.json                                     zdroje, revize, licence, parametry

Použití:
  python training/layout/build_dataset.py --ladas ~/.cache/aiscr-atr/datasets/ladas \\
      --escriptorium EXPORT_DIR [EXPORT_DIR ...] --out ~/.cache/aiscr-atr/datasets/ppdoclayoutv3-hw
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
LADAS_SPLIT = {"train": "train", "validation": "val", "test": "test"}


class Mapping:
    def __init__(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.classes: list[str] = raw["classes"]
        self.ids = {name: i for i, name in enumerate(self.classes)}
        self.zones: dict[str, str] = raw["zones"]
        self.unknown: collections.Counter = collections.Counter()

    def resolve(self, zone: str | None, box: list[float], width: int, height: int) -> str | None:
        """Vrátí cílovou třídu, nebo None (vypustit)."""
        target = self.zones.get(zone or "Undefined")
        if target is None:
            self.unknown[zone] += 1
            return None
        if target == "drop":
            return None
        if target == "rule:notes":
            # Poznámka dole přes velkou část šířky je poznámka pod čarou, jinak marginálie.
            _, y, w, h = box
            return "footnote" if (y + h / 2) > 0.6 * height and w > 0.4 * width else "aside_text"
        if target not in self.ids:
            raise ValueError(f"zóna {zone} → neznámá třída {target}")
        return target


class Coco:
    def __init__(self, classes: list[str]) -> None:
        self.data = {s: {"images": [], "annotations": [],
                         "categories": [{"id": i, "name": n} for i, n in enumerate(classes)]} for s in SPLITS}
        self.next_ann = 1
        self.stats: dict = collections.defaultdict(collections.Counter)

    def add(self, split: str, source: str, file_name: str, width: int, height: int,
            objects: list[tuple[int, list[float], list[float] | None]]) -> None:
        image_id = len(self.data[split]["images"]) + 1
        self.data[split]["images"].append(
            {"id": image_id, "file_name": file_name, "width": width, "height": height, "source": source})
        self.stats[(split, source)]["_images"] += 1
        for category, (x, y, w, h), polygon in objects:
            seg = polygon or [x, y, x + w, y, x + w, y + h, x, y + h]
            self.data[split]["annotations"].append({
                "id": self.next_ann, "image_id": image_id, "category_id": category,
                "bbox": [round(v, 2) for v in (x, y, w, h)], "area": round(w * h, 2),
                "segmentation": [[round(v, 2) for v in seg]], "iscrowd": 0,
            })
            self.next_ann += 1
            self.stats[(split, source)][category] += 1


def _clip(x: float, y: float, w: float, h: float, width: int, height: int):
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    return (x0, y0, x1 - x0, y1 - y0) if x1 - x0 >= 2 and y1 - y0 >= 2 else None


def add_ladas(root: Path, mapping: Mapping, coco: Coco, out: Path) -> None:
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(str(root / "data" / "*.parquet"))):
        split = LADAS_SPLIT[Path(path).name.split("-")[0]]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                width, height = row["width"], row["height"]
                data = row["image"]["bytes"]
                ext = (Image.open(io.BytesIO(data)).format or "JPEG").lower().replace("jpeg", "jpg")
                stem = hashlib.sha1(row["image_path"].encode()).hexdigest()[:16]
                file_name = f"ladas_{stem}.{ext}"
                (out / "images" / split / file_name).write_bytes(data)
                objects = []
                for (cx, cy, w, h), zone in zip(row["objects"]["bbox"], row["objects"]["category"]):
                    box = _clip((cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height, width, height)
                    if box is None:
                        continue
                    target = mapping.resolve(zone, list(box), width, height)
                    if target is not None:
                        objects.append((mapping.ids[target], list(box), None))
                coco.add(split, f"ladas/{row['set']}", file_name, width, height, objects)


def add_escriptorium(exports: list[Path], mapping: Mapping, coco: Coco, out: Path,
                     split_ratio: tuple[float, float, float], seed: int) -> None:
    pages = []
    for export in exports:
        for line in (export / "parts.jsonl").open(encoding="utf-8"):
            part = json.loads(line)
            if (export / part["image"]["filename"]).is_file():
                pages.append((export, part))
    # Rozdělení podle souboru (jedna strana = jeden dokument AMČR); deterministické.
    keys = sorted({p["image"]["filename"].rsplit(".", 1)[0] for _, p in pages})
    random.Random(seed).shuffle(keys)
    n_train = round(len(keys) * split_ratio[0])
    n_val = round(len(keys) * split_ratio[1])
    assign = {k: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
              for i, k in enumerate(keys)}

    for export, part in pages:
        key = part["image"]["filename"].rsplit(".", 1)[0]
        split = assign[key]
        src = export / part["image"]["filename"]
        img = Image.open(src)
        img.seek(0)
        width, height = img.size
        file_name = f"amcr_{key}.png"
        if not (out / "images" / split / file_name).exists():
            img.convert("RGB").save(out / "images" / split / file_name, format="PNG", optimize=True)
        objects = []
        for region in part["regions"]:
            zone = (region.get("typology") or {}).get("name")
            x0, y0, x1, y1 = region["bbox"]
            box = _clip(x0, y0, x1 - x0, y1 - y0, width, height)
            if box is None:
                continue
            target = mapping.resolve(zone, list(box), width, height)
            if target is None:
                continue
            polygon = [float(v) for pt in region.get("box") or [] for v in pt] or None
            objects.append((mapping.ids[target], list(box), polygon if polygon and len(polygon) >= 6 else None))
        coco.add(split, f"amcr/{export.name.split('_')[1]}", file_name, width, height, objects)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladas", type=Path, required=True, help="složka s data/*.parquet z almanach/LADaS")
    ap.add_argument("--ladas-revision", default="2c1f39f34bd05d15a2f384621ce0e00ac6824623")
    ap.add_argument("--escriptorium", type=Path, nargs="*", default=[], help="exporty eScriptoria (JSON)")
    ap.add_argument("--escriptorium-split", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--mapping", type=Path, default=HERE / "mapping.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in SPLITS:
        (args.out / "images" / split).mkdir(parents=True)
    (args.out / "annotations").mkdir()

    mapping = Mapping(args.mapping)
    coco = Coco(mapping.classes)
    add_ladas(args.ladas, mapping, coco, args.out)
    if args.escriptorium:
        add_escriptorium(args.escriptorium, mapping, coco, args.out, tuple(args.escriptorium_split), args.seed)

    for split in SPLITS:
        (args.out / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(coco.data[split], ensure_ascii=False), encoding="utf-8")

    stats = {}
    for (split, source), counter in sorted(coco.stats.items()):
        stats.setdefault(split, {})[source] = {
            "images": counter.pop("_images"),
            "objects": {mapping.classes[k]: v for k, v in sorted(counter.items())},
        }
    totals = collections.Counter()
    for split in SPLITS:
        for a in coco.data[split]["annotations"]:
            totals[(split, mapping.classes[a["category_id"]])] += 1
    stats["by_class"] = {c: {s: totals[(s, c)] for s in SPLITS} for c in mapping.classes}
    stats["unmapped_zones"] = dict(mapping.unknown)
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    manifest = {
        "classes": mapping.classes,
        "sources": [{"name": "LADaS", "url": "https://huggingface.co/datasets/almanach/LADaS",
                     "revision": args.ladas_revision, "licence": "CC-BY-4.0",
                     "credit": "Clérice, Janès et al., ALMANaCH, Inria"}]
                   + ([{"name": "AMČR (eScriptorium, SegmOnto)", "exports": [p.name for p in args.escriptorium],
                        "licence": "interní, nezveřejňovat", "split": list(args.escriptorium_split),
                        "seed": args.seed}] if args.escriptorium else []),
        "mapping": args.mapping.name,
        "mapping_sha1": hashlib.sha1(args.mapping.read_bytes()).hexdigest(),
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'třída':18s} {'train':>7s} {'val':>6s} {'test':>6s}")
    for c in mapping.classes:
        row = stats["by_class"][c]
        print(f"{c:18s} {row['train']:7d} {row['val']:6d} {row['test']:6d}")
    print("obrázky:", {s: len(coco.data[s]["images"]) for s in SPLITS})
    if mapping.unknown:
        print("zóny bez mapy (vypuštěny):", dict(mapping.unknown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
