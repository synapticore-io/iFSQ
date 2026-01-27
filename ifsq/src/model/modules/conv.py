import torch.nn as nn
from typing import Union, Tuple
import torch
from .ops import cast_tuple
from .ops import video_to_image
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from collections import deque

class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[str, int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    @video_to_image
    def forward(self, x):
        return super().forward(x)

