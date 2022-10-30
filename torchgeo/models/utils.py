"""Common model utilities."""
from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Linear, Module, init


# TODO: Attribution (Facebook)
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Get 2D sine-cosine position embedding.
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_h = torch.arange(grid_size, dtype=torch.float, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first

    grid_array = torch.stack(grid, dim=0)
    grid_array = grid_array.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_array, device)

    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([1, embed_dim], dtype=torch.float, device=device), pos_embed],
            dim=0,
        )

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    """Get 2D sine-cosine position embedding from grid."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], device=device
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], device=device
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    """Get 1D sine-cosine position embedding from grid.

    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # type: ignore[no-untyped-call]

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb


def init_weights(m: Module) -> None:
    """Initialize the weights."""
    if isinstance(m, Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, LayerNorm):
        init.constant_(m.bias, 0)
        init.constant_(m.weight, 1.0)
    elif isinstance(m, Conv2d):
        w = m.weight.data
        init.xavier_uniform_(w.view([w.shape[0], -1]))


def get_positional_encodings(
    embed_dim: int,
    num_patches: int,
    channel_wise: bool,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Initialize the positional embeddings."""
    positional_embeddings = get_2d_sincos_pos_embed(
        embed_dim, int(num_patches**0.5), cls_token=False, device=device
    )

    return positional_embeddings


def get_channel_encodings(
    embed_dim: int,
    channels: tuple[int, ...],
    num_patches: int,
    device: str | torch.device,
) -> Tensor:
    """Get the channel encodings for the given channels."""
    channel_encoding = get_channel_encoding(embed_dim, channels, num_patches, device)

    return channel_encoding


# @lru_cache(128)
def get_channel_encoding(
    embed_dim: int,
    channels: tuple[int, ...],
    num_patches: int,
    device: str | torch.device,
) -> Tensor:
    """Get the channel encodings for the given channels."""
    channel_encoding = get_1d_sincos_pos_embed_from_grid(
        embed_dim,
        torch.tensor(channels, dtype=torch.float, device=device),
        device=device,
    )

    channel_encoding = channel_encoding.repeat_interleave(repeats=num_patches, dim=0)

    return channel_encoding


@lru_cache(10)
def get_mask_tokens(
    embed_dim: int,
    num_patches: int,
    channel_wise: bool = False,
    dtype: str | torch.dtype = torch.float,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Get the mask token."""
    mask_tokens = -torch.ones(num_patches, embed_dim, device=device, dtype=dtype)
    mask_tokens += get_positional_encodings(
        embed_dim, num_patches, channel_wise, device
    )
    if channel_wise:
        mask_tokens += get_channel_encodings(embed_dim, (0,), num_patches, device)
    return mask_tokens


def reduce_mask_token(
    x: Tensor,
    mask: Tensor,
    mask_tokens: Tensor,
    num_patches: int,
    keep_unreduced: bool = False,
) -> Tensor:
    """Reduce the embed token by using the values not masked in place."""
    _, PS, _ = x.shape
    mask = mask.view(-1, num_patches)  # (C, P)

    visible_pos_indices = (~mask).nonzero()[:, 1]
    sorted_visible, indices = visible_pos_indices.sort(stable=True)
    _, counts = sorted_visible.unique_consecutive(return_counts=True)  # type: ignore
    counts = counts.cumsum(dim=0)[:-1]
    counts = torch.cat([torch.zeros(1, dtype=counts.dtype, device=x.device), counts])

    mask_tokens[:, visible_pos_indices[indices[counts]]] = x[:, indices[counts]]

    if keep_unreduced:
        all_patches = torch.arange(PS, device=x.device)
        unreduced_indices = all_patches[
            (all_patches != indices[counts].view(-1, 1)).all(dim=0)
        ]
        mask_tokens = torch.cat([mask_tokens, x[:, unreduced_indices]], dim=1)

    return mask_tokens
