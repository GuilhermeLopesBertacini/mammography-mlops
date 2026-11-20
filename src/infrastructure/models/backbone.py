"""Backbone timm (ConvNeXt) acoplado a FeaturePyramidNetwork com GroupNorm."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import timm
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


def _make_norm_layer(norm_type: str, num_groups: int) -> Callable[[int], nn.Module] | None:
    if norm_type == "group_norm":
        return lambda channels: nn.GroupNorm(num_groups, channels)
    if norm_type == "layer_norm":
        return lambda channels: nn.GroupNorm(1, channels)
    return None


class TimmFPNBackbone(nn.Module):
    """Extrai 4 escalas do backbone timm e projeta em FPN sem BatchNorm."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        out_channels: int = 256,
        norm_type: str = "group_norm",
        num_groups: int = 32,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.body = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
        )
        in_channels_list: list[int] = self.body.feature_info.channels()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
            norm_layer=_make_norm_layer(norm_type, num_groups),
        )
        self.out_channels = out_channels
        self.featmap_names = ["0", "1", "2", "3"]

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.body(x)
        feature_dict = OrderedDict(
            (str(idx), feat) for idx, feat in enumerate(features)
        )
        return self.fpn(feature_dict)
