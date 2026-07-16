"""Evaluate a trained ConvNeXt-Tiny checkpoint on val/test/all splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import RGBImageFolder, build_transforms, load_json
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate linzklar ConvNeXt checkpoint")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/best.pt"),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data_images/png/initial_dot_captured_or_augmented"),
    )
    p.add_argument(
        "--split-file",
        type=Path,
        default=Path("outputs/split.json"),
        help="Split indices written by train.py",
    )
    p.add_argument(
        "--split",
        choices=("test", "val", "train", "all"),
        default="test",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--top-confused",
        type=int,
        default=20,
        help="Print this many most frequent confusion pairs (pred != true)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    idx_to_class: dict[int, str],
    top_confused: int,
) -> None:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    confusion_counts: dict[tuple[str, str], int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)

    for images, targets in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)

        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == targets).sum().item()

        k = min(5, logits.size(1))
        topk = logits.topk(k, dim=1).indices
        correct5 += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()

        for t, p in zip(targets.tolist(), pred1.tolist()):
            true_name = idx_to_class[t]
            pred_name = idx_to_class[p]
            per_class_total[true_name] += 1
            if t == p:
                per_class_correct[true_name] += 1
            else:
                confusion_counts[(true_name, pred_name)] += 1

    print(
        f"loss={total_loss / max(total, 1):.4f}  "
        f"acc1={correct1 / max(total, 1):.4f}  "
        f"acc5={correct5 / max(total, 1):.4f}  "
        f"n={total}"
    )

    if per_class_total:
        worst = sorted(
            (
                (
                    name,
                    per_class_correct[name] / per_class_total[name],
                    per_class_total[name],
                )
                for name in per_class_total
            ),
            key=lambda x: x[1],
        )[:15]
        print("\nLowest per-class top-1 accuracy:")
        for name, acc, n in worst:
            print(f"  {name}: {acc:.3f} (n={n})")

    if top_confused > 0 and confusion_counts:
        print(f"\nTop {top_confused} confusion pairs (true -> pred):")
        ranked = sorted(confusion_counts.items(), key=lambda x: -x[1])[:top_confused]
        for (true_name, pred_name), count in ranked:
            print(f"  {true_name} -> {pred_name}: {count}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    use_amp = device.type == "cuda"

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, ckpt = load_checkpoint(str(args.checkpoint), device=device)
    image_size = int(ckpt.get("image_size", 128))
    class_to_idx: dict[str, int] = ckpt["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    print(f"Checkpoint: {args.checkpoint} (epoch={ckpt.get('epoch')}, "
          f"val_acc={ckpt.get('val_acc')})")
    print(f"Device: {device} | split={args.split} | image_size={image_size}")

    ds = RGBImageFolder(str(args.data_dir), transform=build_transforms(image_size, train=False))

    if args.split == "all":
        subset: torch.utils.data.Dataset = ds
        n = len(ds)
    else:
        if not args.split_file.is_file():
            raise FileNotFoundError(
                f"Split file not found: {args.split_file}. "
                "Run train.py first, or pass --split all."
            )
        split_indices = load_json(args.split_file)
        indices = split_indices[args.split]
        subset = Subset(ds, indices)
        n = len(indices)

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Evaluating {n} samples...")
    run_eval(model, loader, device, use_amp, idx_to_class, args.top_confused)


if __name__ == "__main__":
    main()
