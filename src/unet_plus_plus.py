"""U-Net++, ASK-UNet++, and parameter-efficient SK variants."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and SiLU, matching AS-UNet."""

    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.silu = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.silu(self.bn2(self.conv2(x)))
        return x


class DepthwiseSeparableBranch(nn.Module):
    """Depthwise convolution followed by pointwise projection."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SKAttention(nn.Module):
    """Selective Kernel attention with parallel 3x3 and 5x5 branches."""

    def __init__(
        self,
        channels: int,
        branches: int = 2,
        reduction: int = 16,
        min_hidden: int = 32,
        lightweight: bool = False,
    ):
        super().__init__()
        if branches != 2:
            raise ValueError("This SKAttention implementation expects exactly two branches")
        hidden = max(channels // reduction, min_hidden)
        kernel_sizes = (3, 5)
        if lightweight:
            self.branches = nn.ModuleList(
                [DepthwiseSeparableBranch(channels, kernel_size) for kernel_size in kernel_sizes]
            )
        else:
            self.branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * branches, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_features = torch.stack([branch(x) for branch in self.branches], dim=1)
        fused = branch_features.sum(dim=1)
        descriptor = F.adaptive_avg_pool2d(fused, output_size=1).flatten(1)
        attention = self.fc(descriptor)
        attention = F.softmax(
            attention.view(x.shape[0], len(self.branches), x.shape[1], 1, 1),
            dim=1,
        )
        return (branch_features * attention).sum(dim=1)


class UNetPlusPlus(nn.Module):
    """Standard U-Net++ with a five-stage VGG-style encoder/decoder."""

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        deep_supervision: bool = False,
        use_sk_attention: bool = False,
        lightweight_sk: bool = False,
        sk_levels: tuple[int, ...] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.filters = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, self.filters[0], self.filters[0])
        self.conv1_0 = VGGBlock(self.filters[0], self.filters[1], self.filters[1])
        self.conv2_0 = VGGBlock(self.filters[1], self.filters[2], self.filters[2])
        self.conv3_0 = VGGBlock(self.filters[2], self.filters[3], self.filters[3])
        self.conv4_0 = VGGBlock(self.filters[3], self.filters[4], self.filters[4])

        self.conv0_1 = VGGBlock(self.filters[0] + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_1 = VGGBlock(self.filters[1] + self.filters[2], self.filters[1], self.filters[1])
        self.conv2_1 = VGGBlock(self.filters[2] + self.filters[3], self.filters[2], self.filters[2])
        self.conv3_1 = VGGBlock(self.filters[3] + self.filters[4], self.filters[3], self.filters[3])

        self.conv0_2 = VGGBlock(self.filters[0] * 2 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_2 = VGGBlock(self.filters[1] * 2 + self.filters[2], self.filters[1], self.filters[1])
        self.conv2_2 = VGGBlock(self.filters[2] * 2 + self.filters[3], self.filters[2], self.filters[2])

        self.conv0_3 = VGGBlock(self.filters[0] * 3 + self.filters[1], self.filters[0], self.filters[0])
        self.conv1_3 = VGGBlock(self.filters[1] * 3 + self.filters[2], self.filters[1], self.filters[1])

        self.conv0_4 = VGGBlock(self.filters[0] * 4 + self.filters[1], self.filters[0], self.filters[0])

        if use_sk_attention:
            levels = tuple(range(len(self.filters))) if sk_levels is None else tuple(sorted(set(sk_levels)))
            if any(level < 0 or level >= len(self.filters) for level in levels):
                raise ValueError(f"Invalid SK levels: {levels}")
            self.sk_levels = levels
            self.sk_modules = nn.ModuleList(
                [
                    SKAttention(channels, lightweight=lightweight_sk) if level in levels else nn.Identity()
                    for level, channels in enumerate(self.filters)
                ]
            )
        else:
            self.sk_levels = ()
            self.sk_modules = None

        if self.deep_supervision:
            self.final1 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(self.filters[0], num_classes, kernel_size=1)

    def _encode(self, level: int, x: torch.Tensor) -> torch.Tensor:
        x = getattr(self, f"conv{level}_0")(x)
        if self.sk_modules is not None and level in self.sk_levels:
            x = self.sk_modules[level](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x0_0 = self._encode(0, x)
        x1_0 = self._encode(1, self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self._encode(2, self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self._encode(3, self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self._encode(4, self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        return self.final(x0_4)


class ASKUNetPlusPlus(UNetPlusPlus):
    """U-Net++ with full 3x3/5x5 SK attention on every encoder stage."""

    def __init__(self, num_classes: int = 1, input_channels: int = 3, **kwargs):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            deep_supervision=False,
            use_sk_attention=True,
            lightweight_sk=False,
            sk_levels=(0, 1, 2, 3, 4),
            **kwargs,
        )


class LiteASKUNetPlusPlus(UNetPlusPlus):
    """U-Net++ with depthwise-separable SK on every encoder stage."""

    def __init__(self, num_classes: int = 1, input_channels: int = 3, **kwargs):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            deep_supervision=False,
            use_sk_attention=True,
            lightweight_sk=True,
            sk_levels=(0, 1, 2, 3, 4),
            **kwargs,
        )


class DeepLiteASKUNetPlusPlus(UNetPlusPlus):
    """Parameter-efficient SK on the three deepest encoder stages."""

    def __init__(self, num_classes: int = 1, input_channels: int = 3, **kwargs):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            deep_supervision=False,
            use_sk_attention=True,
            lightweight_sk=True,
            sk_levels=(2, 3, 4),
            **kwargs,
        )


class BottleneckLiteASKUNetPlusPlus(UNetPlusPlus):
    """Parameter-efficient SK only on the encoder bottleneck."""

    def __init__(self, num_classes: int = 1, input_channels: int = 3, **kwargs):
        super().__init__(
            num_classes=num_classes,
            input_channels=input_channels,
            deep_supervision=False,
            use_sk_attention=True,
            lightweight_sk=True,
            sk_levels=(4,),
            **kwargs,
        )
