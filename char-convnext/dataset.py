"""Dataset loading, transforms, and stratified train/val/test splits."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

# ImageNet normalization (ConvNeXt pretrained weights)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RGBImageFolder(ImageFolder):
    """ImageFolder that always converts images to RGB.

    Handles 16-bit grayscale+alpha PNGs used by this dataset so that
    pretrained ImageNet models receive 3-channel input.
    """

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, target = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def build_transforms(image_size: int = 128, train: bool = False) -> transforms.Compose:
    """Build train or eval transforms for character images."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_stratified_splits(
    targets: list[int],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified index split into train / val / test.

    Ratios are applied per class. Remaining samples after val/test go to train.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios: val={val_ratio}, test={test_ratio} "
            "(must be non-negative and sum to less than 1)"
        )

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(targets):
        by_class[int(y)].append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_indices in by_class.values():
        indices = list(class_indices)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 and n > 2 else 0
        n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 and n > 2 else 0
        # Keep at least one training sample when possible
        if n_test + n_val >= n:
            n_test = min(n_test, max(0, n - 2))
            n_val = min(n_val, max(0, n - n_test - 1))

        test_part = indices[:n_test]
        val_part = indices[n_test : n_test + n_val]
        train_part = indices[n_test + n_val :]

        test_idx.extend(test_part)
        val_idx.extend(val_part)
        train_idx.extend(train_part)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def load_imagefolder(
    data_dir: str | Path,
    transform: transforms.Compose | None = None,
) -> RGBImageFolder:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return RGBImageFolder(str(data_dir), transform=transform)


def build_dataloaders(
    data_dir: str | Path,
    image_size: int = 128,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    split_indices: dict[str, list[int]] | None = None,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    RGBImageFolder,
    dict[str, list[int]],
]:
    """Create train/val/test loaders. Returns loaders, base dataset (no transform), splits."""
    # Base dataset without transform is used only for metadata + index lists.
    base = load_imagefolder(data_dir, transform=None)
    targets = [s[1] for s in base.samples]

    if split_indices is None:
        train_idx, val_idx, test_idx = make_stratified_splits(
            targets, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
    else:
        train_idx = list(split_indices["train"])
        val_idx = list(split_indices["val"])
        test_idx = list(split_indices["test"])

    train_ds = RGBImageFolder(str(data_dir), transform=build_transforms(image_size, train=True))
    eval_ds = RGBImageFolder(str(data_dir), transform=build_transforms(image_size, train=False))

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, base, split_indices


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image_for_predict(path: str | Path, image_size: int = 128) -> Any:
    """Load a single image tensor (C,H,W) ready for the model."""
    transform = build_transforms(image_size, train=False)
    with Image.open(path) as img:
        image = img.convert("RGB")
    return transform(image)
