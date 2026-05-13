# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weight = self.avg_pool(x).view(b, c)
        weight = self.fc(weight).view(b, c, 1, 1)
        return x * weight


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(weight)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        weight = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weight


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionWrapper(nn.Module):
    """
    Wrap a torchvision ResNet BasicBlock and apply attention after its output.

    This keeps the pretrained residual block unchanged, while adding a manually
    written attention module on top of each residual block.
    """
    def __init__(self, block: nn.Module, attention: nn.Module):
        super().__init__()
        self.block = block
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.block(x))


def _resnet_constructor(arch: str, pretrained: bool):
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet18(weights=weights)
    if arch == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet34(weights=weights)
    raise ValueError(f"Unsupported arch: {arch}")


def _add_attention_to_resnet(model: nn.Module, attention_type: str) -> nn.Module:
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for i, block in enumerate(layer):
            channels = block.conv2.out_channels
            if attention_type == "se":
                attention = SEBlock(channels)
            elif attention_type == "cbam":
                attention = CBAMBlock(channels)
            else:
                raise ValueError(f"Unsupported attention_type={attention_type}")
            layer[i] = AttentionWrapper(block, attention)
    return model


def build_model(model_name: str, num_classes: int = 102, pretrained: bool = True) -> nn.Module:
    """
    model_name choices:
      - resnet18: baseline CNN
      - resnet34: stronger baseline CNN
      - se_resnet18: ResNet-18 + SE blocks
      - cbam_resnet18: ResNet-18 + CBAM blocks
    """
    if model_name in ["resnet18", "resnet34"]:
        model = _resnet_constructor(model_name, pretrained)
    elif model_name == "se_resnet18":
        model = _resnet_constructor("resnet18", pretrained)
        model = _add_attention_to_resnet(model, "se")
    elif model_name == "cbam_resnet18":
        model = _resnet_constructor("resnet18", pretrained)
        model = _add_attention_to_resnet(model, "cbam")
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def split_backbone_head_params(model: nn.Module):
    """Return parameter groups for small backbone LR and larger new-head LR."""
    head_params = list(model.fc.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return backbone_params, head_params
