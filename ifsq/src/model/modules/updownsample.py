from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import cast_tuple, video_to_image
from .conv import Conv2d
from einops import rearrange
from collections import deque


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.with_conv = True
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.with_conv = True
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

