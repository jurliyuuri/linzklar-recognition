"""Export a trained ConvNeXt checkpoint to optimized ONNX."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD, load_image_for_predict
from model import load_checkpoint
from ort_utils import ensure_nvidia_lib_path, make_session

# Load CUDA libs before onnxruntime is first imported elsewhere.
ensure_nvidia_lib_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ConvNeXt-Tiny linzklar model to ONNX")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/best.pt"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/onnx"))
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used only during export (runtime batch is dynamic)",
    )
    p.add_argument("--fp16", action="store_true", help="Also emit a float16 ONNX model")
    p.add_argument("--skip-simplify", action="store_true")
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip PyTorch vs ONNX Runtime parity checks",
    )
    p.add_argument(
        "--verify-images",
        type=Path,
        nargs="*",
        default=None,
        help="Optional real images for parity check (defaults to a few from data dir)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data_images/png/initial_dot_captured_or_augmented"),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for PyTorch export/verify (CPU is most portable for export)",
    )
    return p.parse_args()


def export_onnx(
    model: torch.nn.Module,
    path: Path,
    image_size: int,
    batch_size: int,
    opset: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(
        batch_size, 3, image_size, image_size, device=next(model.parameters()).device
    )
    model.eval()

    export_kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **export_kwargs)


def simplify_onnx(path: Path) -> None:
    import onnxsim

    model = onnx.load(str(path))
    simplified, ok = onnxsim.simplify(model)
    if not ok:
        raise RuntimeError(f"onnxsim failed to simplify {path}")
    onnx.save(simplified, str(path))


def convert_to_fp16(src: Path, dst: Path) -> None:
    """Convert FP32 ONNX to FP16 while keeping float32 I/O types when possible."""
    model = onnx.load(str(src))
    try:
        from onnxconverter_common import float16

        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    except ImportError as exc:
        raise ImportError(
            "FP16 export requires onnxconverter-common. "
            "Install with: uv add onnxconverter-common"
        ) from exc
    onnx.save(model_fp16, str(dst))


def verify_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    image_size: int,
    device: torch.device,
    image_paths: list[Path],
) -> None:
    session = make_session(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ORT providers: {session.get_providers()}")

    for batch in (1, 4):
        x = torch.randn(batch, 3, image_size, image_size, device=device)
        with torch.no_grad():
            pt = model(x).detach().cpu().numpy()
        ort_out = session.run([output_name], {input_name: x.cpu().numpy()})[0]
        max_abs = float(np.max(np.abs(pt - ort_out)))
        cos = float(
            np.sum(pt * ort_out)
            / (np.linalg.norm(pt) * np.linalg.norm(ort_out) + 1e-12)
        )
        print(f"  dummy batch={batch}: max|Δ|={max_abs:.6e}  cosine={cos:.8f}")

    if not image_paths:
        return

    print("Real images:")
    for path in image_paths:
        tensor = load_image_for_predict(path, image_size=image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            pt = model(tensor).detach().cpu().numpy()
        ort_out = session.run([output_name], {input_name: tensor.cpu().numpy()})[0]
        max_abs = float(np.max(np.abs(pt - ort_out)))
        pt_top = int(pt.argmax(axis=1)[0])
        ort_top = int(ort_out.argmax(axis=1)[0])
        match = "OK" if pt_top == ort_top else "MISMATCH"
        print(
            f"  {path.name}: max|Δ|={max_abs:.6e}  top1 pt={pt_top} ort={ort_top} [{match}]"
        )


def pick_default_images(data_dir: Path, n: int = 3) -> list[Path]:
    if not data_dir.is_dir():
        return []
    images: list[Path] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        pngs = sorted(class_dir.glob("*.png"))
        if pngs:
            images.append(pngs[0])
        if len(images) >= n:
            break
    return images


def write_meta(
    path: Path,
    *,
    onnx_name: str,
    ckpt: dict,
    image_size: int,
    opset: int,
    fp16: bool,
    simplified: bool,
) -> None:
    meta = {
        "onnx_file": onnx_name,
        "num_classes": int(ckpt["num_classes"]),
        "image_size": image_size,
        "class_to_idx": ckpt["class_to_idx"],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "input_name": "input",
        "output_name": "logits",
        "input_layout": "NCHW",
        "opset": opset,
        "fp16": fp16,
        "simplified": simplified,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_acc": ckpt.get("val_acc"),
        "preprocess": (
            "RGB convert -> resize(image_size) -> ToTensor -> "
            "Normalize(ImageNet mean/std). Softmax is NOT in the graph."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, ckpt = load_checkpoint(str(args.checkpoint), device=device)
    image_size = int(ckpt.get("image_size", 128))
    num_classes = int(ckpt["num_classes"])
    print(
        f"Loaded {args.checkpoint} | classes={num_classes} "
        f"image_size={image_size} epoch={ckpt.get('epoch')} val_acc={ckpt.get('val_acc')}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "convnext_tiny_linzklar"
    onnx_path = args.output_dir / f"{stem}.onnx"
    raw_path = args.output_dir / f"{stem}.raw.onnx"

    print(f"Exporting ONNX (opset={args.opset}) -> {raw_path}")
    export_onnx(model, raw_path, image_size, args.batch_size, args.opset)
    onnx.checker.check_model(str(raw_path))

    simplified = False
    if args.skip_simplify:
        raw_path.replace(onnx_path)
        print(f"Saved (unsimplified): {onnx_path}")
    else:
        print("Simplifying with onnxsim...")
        shutil.copy2(raw_path, onnx_path)
        simplify_onnx(onnx_path)
        onnx.checker.check_model(str(onnx_path))
        simplified = True
        raw_path.unlink(missing_ok=True)
        print(f"Saved simplified: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    write_meta(
        args.output_dir / "model_meta.json",
        onnx_name=onnx_path.name,
        ckpt=ckpt,
        image_size=image_size,
        opset=args.opset,
        fp16=False,
        simplified=simplified,
    )

    if args.fp16:
        fp16_path = args.output_dir / f"{stem}_fp16.onnx"
        print(f"Converting to FP16 -> {fp16_path}")
        convert_to_fp16(onnx_path, fp16_path)
        onnx.checker.check_model(str(fp16_path))
        write_meta(
            args.output_dir / "model_meta_fp16.json",
            onnx_name=fp16_path.name,
            ckpt=ckpt,
            image_size=image_size,
            opset=args.opset,
            fp16=True,
            simplified=simplified,
        )
        print(f"Saved FP16: {fp16_path} ({fp16_path.stat().st_size / 1e6:.1f} MB)")

    if not args.no_verify:
        print("Verifying PyTorch vs ONNX Runtime...")
        if args.verify_images is not None:
            images = list(args.verify_images)
        else:
            images = pick_default_images(args.data_dir, n=3)
        verify_parity(model, onnx_path, image_size, device, images)

    print(f"Done. Artifacts in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
