import os
from typing import Tuple, List, Optional

import torch
import torch.nn as nn

from diffusers.configuration_utils import register_to_config

from ..modeling_imagebase import ImageBaseAE
from ..modules import Normalize
from ..modules.ops import nonlinearity
from ..utils.module_utils import resolve_str_to_obj, Module
from ..utils.distrib_utils import DiagonalGaussianDistribution
from ..registry import ModelRegistry
from ..modeling_output import EncoderOutput, DecoderOutput, ForwardOutput
from ..modules.quant import RunningVQ


class Encoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: Module = "Conv2d",
        conv_out: Module = "Conv2d",
        attention: Module = "AttnBlock2D",
        resnet_blocks: Tuple[Module] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
        ),
        spatial_downsample: Tuple[Module] = (
            "Downsample",
            "Downsample",
            "Downsample",
            "",
        ),
        mid_resnet: Module = "ResnetBlock2D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
        norm_type: str = "groupnorm",
    ) -> None:
        super().__init__()
        assert len(resnet_blocks) == len(hidden_size_mult), print(
            hidden_size_mult, resnet_blocks
        )
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        # ---- In ----
        self.conv_in = resolve_str_to_obj(conv_in)(
            3, hidden_size, kernel_size=3, stride=1, padding=1
        )

        # ---- Downsample ----
        curr_res = resolution
        in_ch_mult = (1,) + tuple(hidden_size_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_size * in_ch_mult[i_level]
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    resolve_str_to_obj(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        norm_type=norm_type,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(resolve_str_to_obj(attention)(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if spatial_downsample[i_level]:
                down.downsample = resolve_str_to_obj(spatial_downsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # ---- Mid ----
        self.mid_block = nn.ModuleList()
        for i_block in range(self.num_res_blocks):
            self.mid_block.append(
                resolve_str_to_obj(mid_resnet)(
                    in_channels=block_in,
                    out_channels=block_in,
                    dropout=dropout,
                    norm_type=norm_type,
                )
            )
        # ---- Out ----
        self.norm_out = Normalize(block_in, norm_type=norm_type)
        self.conv_out = resolve_str_to_obj(conv_out)(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        for res in range(self.num_res_blocks):
            h = self.mid_block[res](h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: Module = "Conv2d",
        conv_out: Module = "Conv2d",
        attention: Module = "AttnBlock2D",
        resnet_blocks: Tuple[Module] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
        ),
        spatial_upsample: Tuple[Module] = (
            "",
            "Upsample",
            "Upsample",
            "Upsample",
        ),
        mid_resnet: Module = "ResnetBlock2D",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        norm_type: str = "groupnorm",
    ):
        super().__init__()
        # ---- Config ----
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks

        # ---- In ----
        block_in = hidden_size * hidden_size_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = resolve_str_to_obj(conv_in)(
            z_channels, block_in, kernel_size=3, padding=1
        )

        # ---- Mid ----
        self.mid_block = nn.ModuleList()

        for i_level in range(self.num_res_blocks):
            self.mid_block.append(
                resolve_str_to_obj(mid_resnet)(
                    in_channels=block_in,
                    out_channels=block_in,
                    dropout=dropout,
                    norm_type=norm_type,
                )
            )

        # ---- Upsample ----
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resolve_str_to_obj(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        norm_type=norm_type,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        resolve_str_to_obj(attention)(block_in, norm_type=norm_type)
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if spatial_upsample[i_level]:
                up.upsample = resolve_str_to_obj(spatial_upsample[i_level])(
                    block_in, block_in
                )
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # ---- Out ----
        self.norm_out = Normalize(block_in, norm_type=norm_type)
        self.conv_out = resolve_str_to_obj(conv_out)(
            block_in, 3, kernel_size=3, padding=1
        )

    def forward(self, z):
        h = self.conv_in(z)
        for res in range(self.num_res_blocks):
            h = self.mid_block[res](h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


@ModelRegistry.register("ImageVQVAE")
class ImageVQVAEModel(ImageBaseAE):
    @register_to_config
    def __init__(
        self,
        hidden_size: int = 128,
        z_channels: int = 4,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = [],
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        q_conv: str = "Conv2d",
        encoder_conv_in: Module = "Conv2d",
        encoder_conv_out: Module = "Conv2d",
        encoder_attention: Module = "AttnBlock2D",
        encoder_resnet_blocks: Tuple[Module] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
        ),
        encoder_spatial_downsample: Tuple[Module] = (
            "Downsample",
            "Downsample",
            "Downsample",
            "",
        ),
        encoder_mid_resnet: Module = "ResnetBlock2D",
        decoder_conv_in: Module = "Conv2d",
        decoder_conv_out: Module = "Conv2d",
        decoder_attention: Module = "AttnBlock2D",
        decoder_resnet_blocks: Tuple[Module] = (
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
            "ResnetBlock2D",
        ),
        decoder_spatial_upsample: Tuple[Module] = (
            "",
            "Upsample",
            "Upsample",
            "Upsample",
        ),
        decoder_mid_resnet: Module = "ResnetBlock2D",
        scale: List[float] = [0.18215, 0.18215, 0.18215, 0.18215],
        shift: List[float] = [0, 0, 0, 0],
        use_quant_layer: bool = True,
        norm_type: str = "groupnorm",
        n_e: int = 16384,
        beta: float = 0.5,
    ) -> None:
        super().__init__()

        self.use_quant_layer = use_quant_layer

        self.encoder = Encoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=encoder_conv_in,
            conv_out=encoder_conv_out,
            attention=encoder_attention,
            resnet_blocks=encoder_resnet_blocks,
            spatial_downsample=encoder_spatial_downsample,
            mid_resnet=encoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            double_z=double_z,
            norm_type=norm_type,
        )

        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
        )
        if self.use_quant_layer:
            quant_conv_cls = resolve_str_to_obj(q_conv)
            self.quant_conv = quant_conv_cls(2 * z_channels, embed_dim, 1)
            self.post_quant_conv = quant_conv_cls(embed_dim, z_channels, 1)

        self.quantize = RunningVQ(
            n_e=n_e,
            e_dim=embed_dim,
            beta=beta,
        )

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder, self.quantize]
        return [self.encoder, self.quantize]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def encode(self, x):
        h = self.encoder(x)
        if self.use_quant_layer:
            h = self.quant_conv(h)
        out, indices, vq_extra_output = self.quantize(h)
        return EncoderOutput(
            sample=out, indices=indices, vq_extra_output=vq_extra_output
        )

    def decode(self, z):
        if self.use_quant_layer:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return DecoderOutput(sample=dec)

    def forward(self, input, sample_posterior=True):
        enc_out = self.encode(input)
        z = enc_out.sample
        dec = self.decode(z).sample
        return ForwardOutput(sample=dec, vq_extra_output=enc_out.vq_extra_output)

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        print("init from " + path)

        if (
            "ema_state_dict" in sd
            and len(sd["ema_state_dict"]) > 0
            and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0
        ):
            print("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            print("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]

        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
