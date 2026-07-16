"""Run inference on a single image or a directory of images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import load_image_for_predict
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict linzklar character class")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/best.pt"),
    )
    p.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an image file, or a directory of images",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)
        if not files:
            raise FileNotFoundError(f"No images found under {path}")
        return files
    raise FileNotFoundError(f"Image path not found: {path}")


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    image_path: Path,
    image_size: int,
    device: torch.device,
    idx_to_class: dict[int, str],
    top_k: int,
) -> list[tuple[str, float]]:
    tensor = load_image_for_predict(image_path, image_size=image_size)
    batch = tensor.unsqueeze(0).to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)[0]
    k = min(top_k, probs.numel())
    values, indices = probs.topk(k)
    return [(idx_to_class[int(i)], float(v)) for v, i in zip(values, indices)]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, ckpt = load_checkpoint(str(args.checkpoint), device=device)
    image_size = int(ckpt.get("image_size", 128))
    class_to_idx: dict[str, int] = ckpt["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    images = collect_images(args.image)
    print(f"Checkpoint: {args.checkpoint} | device={device} | n_images={len(images)}")

    for image_path in images:
        results = predict_one(
            model, image_path, image_size, device, idx_to_class, args.top_k
        )
        print(f"\n{image_path}:")
        for rank, (name, prob) in enumerate(results, start=1):
            print(f"  {rank}. {name}  {prob:.4f}")


if __name__ == "__main__":
    main()
