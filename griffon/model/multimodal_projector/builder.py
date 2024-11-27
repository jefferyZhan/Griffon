import torch
import torch.nn as nn
import re
import math
import numpy as np

from torch.nn.init import trunc_normal_
from functools import partial
from torch.nn import functional as F

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class ConvProjector(nn.Module):
    # default to the convolution in resnet50
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, stride=2, kernel_size=3, padding=1)
        self.linear = nn.Linear(outchannel, outchannel)
        self.hidden = outchannel
        self.gelu = nn.GELU()
        self.out = outchannel
    
    def forward(self, x):
        B, N, C = x.shape
        HW = int(math.sqrt(N))
        if not (HW**2 == N):
            raise ValueError(f'Wrong size of feature: {B}, {N}, {C}')
        x = x.permute(0, 2, 1).reshape(-1, C, HW, HW)
        out = self.conv(x)
        out = self.gelu(out)
        out = out.reshape(B, self.hidden, -1).permute(0, 2, 1)
        out = self.linear(out)
        return out

class ConvWoPadProjector(nn.Module):
    # default to the convolution in resnet50
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, stride=2, kernel_size=3)
        self.linear = nn.Linear(outchannel, outchannel)
        self.hidden = outchannel
        self.gelu = nn.GELU()
        self.out = outchannel
    
    def forward(self, x):
        B, N, C = x.shape
        HW = int(math.sqrt(N))
        if not (HW**2 == N):
            raise ValueError(f'Wrong size of feature: {B}, {N}, {C}')
        x = x.permute(0, 2, 1).reshape(-1, C, HW, HW)
        out = self.conv(x)
        out = self.gelu(out)
        out = out.reshape(B, self.hidden, -1).permute(0, 2, 1)
        out = self.linear(out)
        return out

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos
    

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
 
class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm,
            delay_load=False,
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        
        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        if delay_load:
            trunc_normal_(self.query, std=.02)
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        pos_embed = get_abs_pos(self.pos_embed.type(x.dtype), x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1).type(x.dtype),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        out = out.permute(1, 0, 2)

        out = self.ln_post(out)
        out = self.proj(out)
        return out

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
    
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    if projector_type == "conv_reduce":
        return ConvProjector(config.mm_hidden_size, config.hidden_size)
    
    if projector_type == "conv_reduce_wo_pad":
        print("Begin wo padding version")
        return ConvWoPadProjector(config.mm_hidden_size, config.hidden_size)
    
    if projector_type == "resampler":
        #when training in llava procedure, make delay load true
        return Resampler(grid_size=int(math.sqrt(24*24)), embed_dim=config.hidden_size, num_heads=config.hidden_size//128, \
                         kv_dim=config.mm_hidden_size, norm_layer=partial(nn.LayerNorm,eps=1e-6))

    raise ValueError(f'Unknown projector type: {projector_type}')
