import torch.nn as nn
from .normalize import Normalize
import torch
import torch.nn.functional as F
from einops import rearrange

class AttnBlock2D(nn.Module):
    def __init__(self, in_channels, norm_type="groupnorm"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = torch.nn.Linear(
            in_channels, in_channels
        )
        self.k = torch.nn.Linear(
            in_channels, in_channels
        )
        self.v = torch.nn.Linear(
            in_channels, in_channels
        )
        self.proj_out = torch.nn.Linear(
            in_channels, in_channels
        )

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = x
        h_ = self.norm(h_)

        h_ = rearrange(h_, 'b c h w -> b 1 (h w) c')
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        h_ = F.scaled_dot_product_attention(q, k, v)

        h_ = self.proj_out(h_)
        h_ = rearrange(h_, 'b 1 (h w) c -> b c h w', h=h, w=w)

        return x + h_

