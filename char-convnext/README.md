# char-convnext

Fine-tune **ConvNeXt-Tiny** on handwritten [Linzklar](https://github.com/jurliyuuri/linzklar-recognition) character images.

## Setup

```bash
# From this directory
uv sync
```

Dependencies: `torch`, `torchvision` (CUDA 12.8 wheels by default), `pillow`, `tqdm`, `numpy`, plus `onnx` / `onnxsim` / `onnxruntime-gpu` for export and ORT inference.

Confirm GPU:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Data

Default path (relative to this project):

```text
../data_images/png/initial_dot_captured_or_augmented/
  一/
    *.png
  二/
    *.png
  ...
```

- Layout is **one folder per class** (not pre-split into train/val/test).
- Images are 256×256, 16-bit grayscale+alpha; scripts convert to RGB and resize to **128×128**.
- A stratified **80% / 10% / 10%** train/val/test split is created in memory (seed 42) and saved to `outputs/split.json`.

## Train

```bash
uv run python train.py
```

Useful options:

```bash
uv run python train.py \
  --data-dir ../data_images/png/initial_dot_captured_or_augmented \
  --output-dir outputs \
  --image-size 128 \
  --batch-size 32 \
  --epochs 20 \
  --lr 1e-4 \
  --patience 5
```

Smoke test (1 epoch):

```bash
uv run python train.py --epochs 1
```

### Outputs

| Path | Description |
|------|-------------|
| `outputs/checkpoints/best.pt` | Best validation top-1 model |
| `outputs/checkpoints/last.pt` | Last epoch |
| `outputs/class_to_idx.json` | Class name → index |
| `outputs/split.json` | Train/val/test indices |
| `outputs/history.json` | Per-epoch metrics |
| `outputs/test_metrics.json` | Final test metrics |

## Evaluate

```bash
uv run python evaluate.py --checkpoint outputs/checkpoints/best.pt --split test
uv run python evaluate.py --split val
uv run python evaluate.py --split all   # entire dataset (no split file needed for subset choice, still uses data dir)
```

Prints top-1 / top-5 accuracy, weakest classes, and common confusions.

## Predict (PyTorch)

```bash
uv run python predict.py \
  --checkpoint outputs/checkpoints/best.pt \
  --image ../data_images/png/initial_dot_captured_or_augmented/一/some.png \
  --top-k 5
```

`--image` may be a single file or a directory.

## Export ONNX

Export the best checkpoint to a simplified ONNX graph (dynamic batch, logits output):

```bash
uv run python export_onnx.py
# optional FP16 (needs: uv add onnxconverter-common)
uv run python export_onnx.py --fp16
```

| Artifact | Description |
|----------|-------------|
| `outputs/onnx/convnext_tiny_linzklar.onnx` | Optimized FP32 model |
| `outputs/onnx/convnext_tiny_linzklar_fp16.onnx` | Optional FP16 model |
| `outputs/onnx/model_meta.json` | Class map, image size, mean/std, I/O names |

**Graph contract**

- Input `input`: `N×3×128×128` float32 (NCHW), dynamic batch
- Output `logits`: `N×412` (no softmax in the graph)
- Preprocess outside the model: RGB → resize 128 → ImageNet normalize

The exporter runs **onnxsim** and verifies PyTorch vs ONNX Runtime parity by default.

### Predict (ONNX Runtime)

```bash
uv run python predict_onnx.py \
  --onnx outputs/onnx/convnext_tiny_linzklar.onnx \
  --image ../data_images/png/initial_dot_captured_or_augmented/一/some.png \
  --top-k 5
```

Uses `CUDAExecutionProvider` when available, otherwise CPU.

## Notes

- Horizontal flips are **disabled** (character glyphs are not left-right invariant).
- Light `RandomAffine` is applied only on the training split.
- Mixed precision (AMP) is on by default when CUDA is available; pass `--no-amp` to disable.
- RTX 4070 SUPER (12GB) has ample headroom at batch 32 / 128×128; raise `--batch-size` if you want faster epochs.
