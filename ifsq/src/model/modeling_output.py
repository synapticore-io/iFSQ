from dataclasses import dataclass
from typing import Optional
import torch
from diffusers.utils import BaseOutput

from .utils.distrib_utils import DiagonalGaussianDistribution


@dataclass
class AutoencoderKLOutput(BaseOutput):
    latent_dist: DiagonalGaussianDistribution
    extra_output: Optional[tuple] = None


@dataclass
class EncoderOutput(BaseOutput):
    sample: torch.Tensor
    indices: Optional[torch.LongTensor] = None
    lfq_extra_output: Optional[tuple] = None
    extra_output: Optional[tuple] = None
    vq_extra_output: Optional[tuple] = None


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None
    extra_output: Optional[tuple] = None


@dataclass
class ForwardOutput(BaseOutput):
    sample: torch.Tensor
    latent_dist: Optional[DiagonalGaussianDistribution] = None
    lfq_extra_output: Optional[tuple] = None
    extra_output: Optional[tuple] = None
    vq_extra_output: Optional[tuple] = None
