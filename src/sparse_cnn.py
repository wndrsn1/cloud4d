# Code based of https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_flow.py
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import numpy as np

import sparse as sp
from sparse.transformer import SparseTransformerBlock, SparseTransformerCrossBlock

BACKEND = 'spconv'
DEBUG = False

class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        # self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        # self.emb_layers = nn.Sequential(
            # nn.SiLU(),
            # nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        # )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            # There's a bug with the caching. If there are 2x downsamples and 2x upsamples then second upsampling doesn't work
            # print('Should be doing updown here')
            x = self.updown(x)
        return x

    # def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        # emb_out = self.emb_layers(emb).type(x.dtype)
        # scale, shift = torch.chunk(emb_out, 2, dim=1)

        # print(f"Before updown {x.feats.shape}")
        x = self._updown(x)
        # print(f"After updown {x.feats.shape}")
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)

        # h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h
    
    
class Sparse3DCNN(nn.Module):
    def __init__(
        self,
        # resolution: int,
        in_channels: int,
        model_channels: int,
        # cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        # num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        # patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        # share_mod: bool = False,
        qk_rms_norm: bool = False,
        # qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        # self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        # self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        # self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        # self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        # self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # if self.io_block_channels is not None:
            # assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            # assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        # self.t_embedder = TimestepEmbedder(model_channels)
        # if share_mod:
            # self.adaLN_modulation = nn.Sequential(
                # nn.SiLU(),
                # nn.Linear(model_channels, 6 * model_channels, bias=True)
            # )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        
        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                self.input_blocks.extend([
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )
        
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=self.qk_rms_norm,
            )
            for _ in range(num_blocks)
        ])
                
        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                self.out_blocks.append(
                    SparseResBlock3d(
                        prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels,
                        out_channels=chs,
                        upsample=True,
                    )
                )
                self.out_blocks.extend([
                    SparseResBlock3d(
                        chs * 2 if self.use_skip_connection else chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
                
        self.out_layer = sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)
        # self.out_non_linear = sp.SparseActivation(nn.LeakyReLU())

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        # if self.share_mod:
            # nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        # else:
            # for block in self.blocks:
                # nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        # h = self.input_layer(x)
        # t_emb = self.t_embedder(t)
        # if self.share_mod:
            # t_emb = self.adaLN_modulation(t_emb)
        # t_emb = t_emb.type(self.dtype)
        # cond = cond.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h)
            # h = block(h, t_emb)
            skips.append(h.feats)
        
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
            # h = h + self.pos_embedder(h.coords[:, 1:])
        
        for block in self.blocks:
            # h = block(h, t_emb, cond)
            h = block(h)
        
        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                # h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
                # print(f"h feat shape: {h.feats.shape}")
                # print(f"Skip shape: {skip.shape}")

                h = block(h.replace(torch.cat([h.feats, skip], dim=1)))
            else:
                # h = block(h, t_emb)
                h = block(h)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))

        # Apply leaky ReLU
        # h = self.out_non_linear(h)
        
        # h = self.out_layer(h.type(x))

        return h

FP16_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    sp.SparseConv3d,
    sp.SparseInverseConv3d,
    sp.SparseLinear,
)

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.float()


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    
class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """
    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed