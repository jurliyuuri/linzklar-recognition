"""ConvNeXt-Tiny model factory for fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create ConvNeXt-Tiny with a fresh classification head.

    Args:
        num_classes: Number of output classes (e.g. 412 for linzklar).
        pretrained: If True, load ImageNet-1K weights for the backbone.
    """
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)

    # classifier is Sequential(LayerNorm2d, Flatten, Linear)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    pretrained: bool = False,
) -> tuple[nn.Module, dict]:
    """Load a training checkpoint and return (model, checkpoint_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])
    model = create_model(num_classes=num_classes, pretrained=pretrained)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt
