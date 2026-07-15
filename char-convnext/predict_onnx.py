"""Run inference with an ONNX Runtime session (no PyTorch weights required at runtime)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import load_image_for_predict
from ort_utils import ensure_nvidia_lib_path, make_session

ensure_nvidia_lib_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict linzklar class via ONNX Runtime")
    p.add_argument(
        "--onnx",
        type=Path,
        default=Path("outputs/onnx/convnext_tiny_linzklar.onnx"),
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=Path("outputs/onnx/model_meta.json"),
        help="Metadata JSON written by export_onnx.py",
    )
    p.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an image file, or a directory of images",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--providers",
        type=str,
        default="auto",
        help="Comma-separated ORT providers, or 'auto'",
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


def load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def predict_one(
    session,
    image_path: Path,
    image_size: int,
    idx_to_class: dict[int, str],
    top_k: int,
    input_name: str,
    output_name: str,
) -> list[tuple[str, float]]:
    tensor = load_image_for_predict(image_path, image_size=image_size)
    batch = tensor.unsqueeze(0).numpy()
    logits = session.run([output_name], {input_name: batch})[0][0]
    probs = softmax(logits)
    k = min(top_k, probs.shape[0])
    top_idx = np.argsort(-probs)[:k]
    return [(idx_to_class[int(i)], float(probs[int(i)])) for i in top_idx]


def main() -> None:
    args = parse_args()

    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    if not args.meta.is_file():
        raise FileNotFoundError(
            f"Metadata not found: {args.meta}. Run export_onnx.py first."
        )

    meta = load_meta(args.meta)
    image_size = int(meta.get("image_size", 128))
    class_to_idx: dict[str, int] = meta["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    input_name = meta.get("input_name", "input")
    output_name = meta.get("output_name", "logits")

    session = make_session(args.onnx, providers=args.providers)
    images = collect_images(args.image)

    print(
        f"ONNX: {args.onnx} | providers={session.get_providers()} | "
        f"n_images={len(images)} | image_size={image_size}"
    )

    for image_path in images:
        results = predict_one(
            session,
            image_path,
            image_size,
            idx_to_class,
            args.top_k,
            input_name,
            output_name,
        )
        print(f"\n{image_path}:")
        for rank, (name, prob) in enumerate(results, start=1):
            print(f"  {rank}. {name}  {prob:.4f}")


if __name__ == "__main__":
    main()
