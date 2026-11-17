"""Dotrénování PP-DocLayoutV3 v PyTorch (transformers) s rozšířeným slovníkem tříd.

PaddleX (oficiální cesta) neběží na AMD (ROCm) ani na ARM (DGX Spark). Tento skript
trénuje přímo implementaci z transformers, takže běží všude, kde běží PyTorch:
NVIDIA (CUDA, i aarch64), AMD (ROCm, např. RX 7900 XTX) i CPU (jen na zkoušku).

Co se trénuje:
  * detekce bloků (třídy a polohy) se ztrátou RT-DETR z transformers (Hungarian
    matching, VFL, L1, GIoU, pomocné ztráty dekodéru a denoising), stejná rodina
    modelů jako PP-DocLayoutV3;
  * hlava tříd se rozšíří z 25 na počet tříd datové sady (26 s `handwriting`);
    původních 25 tříd převezme váhy, nová třída startuje z vah třídy `text`.
Co se netrénuje (zmrazeno): hlava pořadí čtení a hlava masek (data je nemají).

Vstup: datová sada COCO z build_dataset.py, nebo výstup to_paddlex.py (s pseudo-popisky).
Výstup: složka ve formátu transformers (save_pretrained), použitelná ve službě
models/layout-ppdoclayoutv3 přes MODEL_ID=<cesta>.

  python training/layout/train_torch.py --data DATASET --out runs/hw-v1 --epochs 30 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForObjectDetection
from transformers.loss.loss_rt_detr import RTDetrForObjectDetectionLoss

BASE = "PaddlePaddle/PP-DocLayoutV3_safetensors"
BASE_REVISION = "e4f489dc536556fc1be02b973ba06756e0d4a2ac"
SIZE = 800  # vstup modelu (preprocessor_config: 800 × 800, bez zachování poměru)

# Výchozí parametry ztráty RT-DETR (konfigurace PP-DocLayoutV3 je neobsahuje).
LOSS_CONFIG = {
    "auxiliary_loss": True, "use_focal_loss": True,
    "matcher_class_cost": 2.0, "matcher_bbox_cost": 5.0, "matcher_giou_cost": 2.0,
    "matcher_alpha": 0.25, "matcher_gamma": 2.0,
    "weight_loss_vfl": 1.0, "weight_loss_bbox": 5.0, "weight_loss_giou": 2.0,
    "eos_coefficient": 1e-4, "focal_loss_alpha": 0.75, "focal_loss_gamma": 2.0,
}


def _annotation_file(root: Path, split: str) -> Path:
    for name in (f"instance_{split}.json", f"instances_{split}.json"):
        if (root / "annotations" / name).is_file():
            return root / "annotations" / name
    raise FileNotFoundError(f"chybí anotace pro {split} v {root}/annotations")


class CocoLayout(Dataset):
    def __init__(self, root: Path, split: str, train: bool, limit: int = 0) -> None:
        data = json.loads(_annotation_file(root, split).read_text(encoding="utf-8"))
        self.root, self.split, self.train = root, split, train
        self.categories = sorted(data["categories"], key=lambda c: c["id"])
        anns: dict[int, list] = {}
        for a in data["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        self.items = [(img, anns.get(img["id"], [])) for img in data["images"]][: limit or None]
        # Třídy s ručními anotacemi (bez pseudo-popisků z původního modelu): podle nich
        # se hodnotí a vybírá model; pseudo třídy jsou vzácné a jejich AP je šum.
        self.annotated = sorted({a["category_id"] for _, items in self.items for a in items
                                 if not a.get("pseudo")})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img, anns = self.items[i]
        image = Image.open(self.root / "images" / self.split / img["file_name"]).convert("RGB")
        w, h = image.size
        if self.train and random.random() < 0.5:
            # lehká augmentace: jas/kontrast, bez geometrie (pořadí a polohy zůstávají)
            from PIL import ImageEnhance

            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        pixels = torch.from_numpy(
            __import__("numpy").asarray(image.resize((SIZE, SIZE), Image.Resampling.BICUBIC), dtype="float32")
        ).permute(2, 0, 1) / 255.0
        boxes, labels = [], []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw < 1 or bh < 1:
                continue
            boxes.append([(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])
            labels.append(a["category_id"])
        target = {"class_labels": torch.tensor(labels, dtype=torch.long),
                  "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp(0, 1)}
        return pixels, target, {"image_id": img["id"], "size": (h, w)}


def collate(batch):
    pixels = torch.stack([b[0] for b in batch])
    return pixels, [b[1] for b in batch], [b[2] for b in batch]


def build_model(categories: list[dict], device: torch.device):
    """Načte PP-DocLayoutV3 a rozšíří hlavu tříd na `categories`."""
    base = AutoModelForObjectDetection.from_pretrained(BASE, revision=BASE_REVISION)
    old_n = base.config.num_labels
    new_n = len(categories)
    if new_n < old_n:
        raise ValueError("datová sada má méně tříd než PP-DocLayoutV3; očekává se rozšíření slovníku")
    names = [c["name"] for c in categories]
    seed_row = names.index("text") if "text" in names else 0

    config = base.config
    config.num_labels = new_n
    config.id2label = {i: n for i, n in enumerate(names)}
    config.label2id = {n: i for i, n in enumerate(names)}
    for k, v in LOSS_CONFIG.items():
        setattr(config, k, v)
    model = AutoModelForObjectDetection.from_config(config)
    model.load_state_dict(
        {k: v for k, v in base.state_dict().items() if v.shape == model.state_dict()[k].shape}, strict=False)

    old = base.state_dict()
    new = model.state_dict()
    with torch.no_grad():
        for key in new:
            if key in old and old[key].shape != new[key].shape:
                src = old[key]
                dst = new[key]
                # řádky (třídy) jsou v první ose: Linear.weight [n, d], Linear.bias [n], Embedding [n(+1), d]
                n = min(src.shape[0], dst.shape[0])
                dst[:n] = src[:n]
                for extra in range(old_n, new_n):
                    dst[extra] = src[seed_row]
                    if key.endswith("bias"):
                        dst[extra] -= 1.0  # nová třída začíná opatrněji než text
                new[key].copy_(dst)
    model.load_state_dict(new)
    resized = [k for k in old if k in new and old[k].shape != new[k].shape]

    # transformers vytváří denoising_class_embed jen s num_labels řádky, ale při tréninku
    # doplňuje chybějící dotazy indexem num_labels (padding). Pro trénink přidat řádek;
    # při ukládání se odebere (save_model), aby model šel načíst bez úprav.
    emb = model.model.denoising_class_embed
    padded = torch.nn.Embedding(new_n + 1, emb.embedding_dim, padding_idx=new_n)
    with torch.no_grad():
        padded.weight[:new_n] = emb.weight
        padded.weight[new_n] = 0
    model.model.denoising_class_embed = padded

    # Zmrazit hlavy, pro které nejsou data (pořadí čtení, masky).
    for name, param in model.named_parameters():
        if any(part in name for part in ("decoder_order_head", "decoder_global_pointer", "mask_")):
            param.requires_grad = False
    return model.to(device), resized


def save_model(model, path: Path) -> None:
    """Uloží ve formátu transformers; vrátí denoising_class_embed na num_labels řádků."""
    padded = model.model.denoising_class_embed
    n = model.config.num_labels
    plain = torch.nn.Embedding(n, padded.embedding_dim).to(padded.weight.device)
    with torch.no_grad():
        plain.weight.copy_(padded.weight[:n])
    model.model.denoising_class_embed = plain
    try:
        model.save_pretrained(path)
        from transformers import AutoImageProcessor

        AutoImageProcessor.from_pretrained(BASE, revision=BASE_REVISION).save_pretrained(path)
    finally:
        model.model.denoising_class_embed = padded


def loss_step(model, pixels, targets):
    out = model.model(pixel_values=pixels, labels=targets)
    logits = out.intermediate_logits[:, -1]
    boxes = out.intermediate_reference_points[:, -1]
    loss, parts, _ = RTDetrForObjectDetectionLoss(
        logits=logits, labels=targets, device=pixels.device, pred_boxes=boxes, config=model.config,
        outputs_class=out.intermediate_logits, outputs_coord=out.intermediate_reference_points,
        enc_topk_logits=out.enc_topk_logits, enc_topk_bboxes=out.enc_topk_bboxes,
        denoising_meta_values=out.denoising_meta_values)
    return loss, {k: float(v) for k, v in parts.items()}


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, annotated: list[int] | None = None) -> dict:
    """mAP@0.5:0.95 a AP@0.5 po třídách (torchmetrics).

    `map_annotated` a `map50_annotated` jsou průměry přes třídy s ručními anotacemi
    (`annotated`); podle `map50_annotated` se vybírá nejlepší model.
    """
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="cxcywh", class_metrics=True)
    metric50 = MeanAveragePrecision(box_format="cxcywh", class_metrics=True, iou_thresholds=[0.5])
    model.eval()
    for pixels, targets, _ in loader:
        out = model(pixel_values=pixels.to(device))
        probs = out.logits.sigmoid()
        preds = []
        for p, b in zip(probs, out.pred_boxes):
            scores, idx = p.flatten().topk(min(300, p.numel()))
            preds.append({"boxes": b[idx // num_classes].float().cpu(), "scores": scores.float().cpu(),
                          "labels": (idx % num_classes).cpu()})
        gt = [{"boxes": t["boxes"], "labels": t["class_labels"]} for t in targets]
        metric.update(preds, gt)
        metric50.update(preds, gt)
    model.train()

    def per_class(res) -> dict[int, float]:
        if res["classes"].numel() and res["map_per_class"].numel() == res["classes"].numel():
            return {int(c): round(float(m), 4) for c, m in zip(res["classes"], res["map_per_class"])}
        return {}

    res, res50 = metric.compute(), metric50.compute()
    pc, pc50 = per_class(res), per_class(res50)

    def mean(values: dict[int, float]) -> float | None:
        picked = [values[c] for c in (annotated or []) if values.get(c, -1) >= 0]
        return round(sum(picked) / len(picked), 4) if picked else None

    return {"map": round(float(res["map"]), 4), "map_50": round(float(res["map_50"]), 4),
            "map_annotated": mean(pc), "map50_annotated": mean(pc50),
            "per_class": pc, "per_class_ap50": pc50}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5, help="hlavy a dekodér")
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="nejvýš tolik stran (zkouška)")
    ap.add_argument("--max-steps", type=int, default=0, help="ukončit po tolika krocích (zkouška)")
    ap.add_argument("--eval-every", type=int, default=1, help="vyhodnotit každých N epoch")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--eval-only", action="store_true", help="jen vyhodnotit výchozí model (srovnání)")
    args = ap.parse_args()

    # Výchozí sdílení tenzorů mezi procesy přes deskriptory narazí na limit otevřených
    # souborů (RuntimeError: received 0 items of ancdata); souborový systém ho nemá.
    torch.multiprocessing.set_sharing_strategy("file_system")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ROCm se hlásí jako cuda
    train = CocoLayout(args.data, "train", train=True, limit=args.limit)
    val = CocoLayout(args.data, "val", train=False, limit=args.limit)
    model, resized = build_model(train.categories, device)
    print(f"zařízení {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}), "
          f"tříd {model.config.num_labels}, rozšířené vrstvy: {resized}", flush=True)

    backbone = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone}, {"params": rest, "lr": args.lr}],
                            weight_decay=args.weight_decay)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            collate_fn=collate)
    total = args.epochs * len(loader)
    warmup = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    use_amp = dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp and dtype is torch.float16)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        metrics = evaluate(model, val_loader, device, model.config.num_labels, val.annotated)
        names = model.config.id2label
        for key in ("per_class", "per_class_ap50"):
            metrics[key] = {names[k]: v for k, v in metrics[key].items()}
        print(json.dumps(metrics, ensure_ascii=False, indent=1))
        (args.out / "baseline.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        return 0
    log = (args.out / "log.jsonl").open("a", encoding="utf-8")
    best, step = -1.0, 0
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        for pixels, targets, _ in loader:
            pixels = pixels.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                loss, parts = loss_step(model, pixels, targets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 20 == 0 or step <= 3:
                print(f"epoch {epoch} step {step}/{total} loss {float(loss):.4f}", flush=True)
                log.write(json.dumps({"step": step, "epoch": epoch, "loss": float(loss), **parts}) + "\n")
                log.flush()
            if args.max_steps and step >= args.max_steps:
                break
        print(f"epoch {epoch} hotová za {time.time() - t0:.0f} s", flush=True)
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1 or args.max_steps:
            metrics = evaluate(model, val_loader, device, model.config.num_labels, val.annotated)
            names = model.config.id2label
            for key in ("per_class", "per_class_ap50"):
                metrics[key] = {names[k]: v for k, v in metrics[key].items()}
            print(f"val: AP50 anotovaných tříd {metrics['map50_annotated']}, mAP anotovaných "
                  f"{metrics['map_annotated']}, handwriting AP50 {metrics['per_class_ap50'].get('handwriting')}, "
                  f"mAP všech {metrics['map']}", flush=True)
            log.write(json.dumps({"epoch": epoch, "val": metrics}) + "\n")
            log.flush()
            score = metrics["map50_annotated"] if metrics["map50_annotated"] is not None else metrics["map"]
            if score > best:
                best = score
                save_model(model, args.out / "best")
                (args.out / "best" / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        if args.max_steps and step >= args.max_steps:
            break
    save_model(model, args.out / "last")
    print(f"hotovo, nejlepší AP50 anotovaných tříd {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
