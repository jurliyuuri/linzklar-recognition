"""Fine-tune ConvNeXt-Tiny on linzklar character images."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import build_dataloaders, save_json
from model import create_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ConvNeXt-Tiny for linzklar")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data_images/png/initial_dot_captured_or_augmented"),
        help="Class-folder image root (ImageFolder layout)",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=5, help="Early-stop patience on val top-1")
    p.add_argument("--no-pretrained", action="store_true", help="Train from scratch")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)

        # top-1
        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == targets).sum().item()

        # top-5
        k = min(5, logits.size(1))
        topk = logits.topk(k, dim=1).indices
        correct5 += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "loss": total_loss / max(total, 1),
        "acc1": correct1 / max(total, 1),
        "acc5": correct5 / max(total, 1),
        "n": float(total),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")

    return {
        "loss": total_loss / max(total, 1),
        "acc1": correct / max(total, 1),
        "n": float(total),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    val_acc: float,
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "num_classes": len(class_to_idx),
            "class_to_idx": class_to_idx,
            "image_size": args.image_size,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    output_dir = args.output_dir
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} (AMP={use_amp})")
    print(f"Data:   {args.data_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")

    train_loader, val_loader, test_loader, base, split_indices = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    num_classes = len(base.classes)
    print(
        f"Classes: {num_classes} | "
        f"train={len(split_indices['train'])} "
        f"val={len(split_indices['val'])} "
        f"test={len(split_indices['test'])}"
    )

    save_json(output_dir / "class_to_idx.json", base.class_to_idx)
    save_json(output_dir / "idx_to_class.json", {str(v): k for k, v in base.class_to_idx.items()})
    save_json(output_dir / "split.json", split_indices)

    model = create_model(num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history: list[dict] = []
    best_val_acc = -1.0
    epochs_without_improve = 0
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
            args.epochs,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_acc1": train_metrics["acc1"],
            "val_loss": val_metrics["loss"],
            "val_acc1": val_metrics["acc1"],
            "val_acc5": val_metrics["acc5"],
            "seconds": elapsed,
        }
        history.append(row)
        save_json(output_dir / "history.json", history)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} acc1={train_metrics['acc1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc1={val_metrics['acc1']:.4f} "
            f"acc5={val_metrics['acc5']:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            val_metrics["acc1"],
            base.class_to_idx,
            args,
        )

        if val_metrics["acc1"] > best_val_acc:
            best_val_acc = val_metrics["acc1"]
            epochs_without_improve = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                base.class_to_idx,
                args,
            )
            print(f"  -> new best val acc1={best_val_acc:.4f} saved to {best_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(
                    f"Early stopping: no val improvement for {args.patience} epochs "
                    f"(best acc1={best_val_acc:.4f})"
                )
                break

    # Final test with best checkpoint
    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(
            f"\nLoaded best checkpoint (epoch={ckpt['epoch']}, "
            f"val_acc1={ckpt['val_acc']:.4f})"
        )

    test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
    print(
        f"Test  loss={test_metrics['loss']:.4f} "
        f"acc1={test_metrics['acc1']:.4f} "
        f"acc5={test_metrics['acc5']:.4f} "
        f"(n={int(test_metrics['n'])})"
    )
    save_json(
        output_dir / "test_metrics.json",
        {
            "loss": test_metrics["loss"],
            "acc1": test_metrics["acc1"],
            "acc5": test_metrics["acc5"],
            "n": int(test_metrics["n"]),
            "best_val_acc1": best_val_acc,
        },
    )
    print(f"Done. Artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
