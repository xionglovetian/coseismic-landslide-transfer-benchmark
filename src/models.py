from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .unet_plus_plus import (
    ASKUNetPlusPlus,
    BottleneckLiteASKUNetPlusPlus,
    DeepLiteASKUNetPlusPlus,
    LiteASKUNetPlusPlus,
    UNetPlusPlus,
)

ASUNET_REPO = Path(r"D:\landslide_unet_project\repos\AS-UNet")
if str(ASUNET_REPO) not in sys.path:
    sys.path.insert(0, str(ASUNET_REPO))
import archs
import losses


class BoundaryAwarePolyGHMDiceLoss(nn.Module):
    """PolyGHMDice plus BCE focused on a narrow ground-truth boundary band."""

    def __init__(self, focus_weight: float = 0.5):
        super().__init__()
        self.base = losses.PolyGHMDiceLoss()
        self.focus_weight = focus_weight

    def forward(self, logits, target):
        base_loss = self.base(logits, target)
        dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-target, kernel_size=3, stride=1, padding=1)
        boundary_band = (dilated - eroded).clamp(0.0, 1.0)
        bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        boundary_weight = boundary_band.sum()
        if boundary_weight.item() == 0:
            return base_loss
        boundary_loss = (bce_map * boundary_band).sum() / (boundary_weight + 1e-7)
        return base_loss + self.focus_weight * boundary_loss

def build_model(name: str, input_channels: int = 3, num_classes: int = 1):
    if name in {"UNetPlusPlus", "U-Net++"}:
        return UNetPlusPlus(
            num_classes=num_classes,
            input_channels=input_channels,
            deep_supervision=False,
        )
    if name in {"ASKUNetPlusPlus", "ASK-UNet++"}:
        return ASKUNetPlusPlus(num_classes=num_classes, input_channels=input_channels)
    if name == "LiteASKUNetPlusPlus":
        return LiteASKUNetPlusPlus(num_classes=num_classes, input_channels=input_channels)
    if name == "DeepLiteASKUNetPlusPlus":
        return DeepLiteASKUNetPlusPlus(num_classes=num_classes, input_channels=input_channels)
    if name == "BottleneckLiteASKUNetPlusPlus":
        return BottleneckLiteASKUNetPlusPlus(num_classes=num_classes, input_channels=input_channels)
    if name == "UNet":
        return archs.UNet(num_classes=num_classes, input_channels=input_channels)
    if name == "NestedUNet":
        return archs.NestedUNet(num_classes=num_classes, input_channels=input_channels, deep_supervision=False)
    if name == "AS_UNet":
        return archs.AS_UNet(num_classes=num_classes, input_channels=input_channels)
    if name in {"ResUNet", "ResUNet34"}:
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=input_channels,
            classes=num_classes,
        )
    if name in {"DeepLabV3Plus", "DeepLabV3+"}:
        return smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=input_channels,
            classes=num_classes,
        )
    if name in {"SegFormerB0", "SegFormer"}:
        return smp.Segformer(
            encoder_name="mit_b0",
            encoder_weights=None,
            in_channels=input_channels,
            classes=num_classes,
        )
    raise ValueError(f"Unknown model: {name}")


def build_loss(name: str):
    if name == "BCEDiceLoss":
        return losses.BCEDiceLoss()
    if name == "PolyGHMDiceLoss":
        return losses.PolyGHMDiceLoss()
    if name == "BoundaryAwarePolyGHMDiceLoss":
        return BoundaryAwarePolyGHMDiceLoss()
    raise ValueError(f"Unknown loss: {name}")
