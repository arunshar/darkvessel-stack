"""Shared conventions for the remote-sensing task heads.

Every task head consumes backbone tokens of shape ``(B, N, D)`` where
``N == grid * grid`` patches and ``D == embed_dim`` (the contract returned by
``GeoBackbone.forward``). These helpers convert between the token sequence and a
``(B, D, grid, grid)`` feature map so spatial heads can use conv layers.

The heads built on this contract are REAL ``nn.Module`` networks, but they are
trained and evaluated on SYNTHETIC structured features (planted task signal +
noise), NOT on real xView3 / SpaceNet / fMoW / LEVIR-CD / WorldStrat / SEN12MS
data. Each task module documents its synthetic structure in ``make_synthetic``;
the repo README states the honesty boundary. The measured metrics demonstrate
that each head can learn its task on structured features; they are not benchmark
numbers.
"""
from __future__ import annotations

import math

import torch


def grid_size(n_tokens: int) -> int:
    """Side length of the square patch grid for ``n_tokens`` (must be a perfect square)."""
    g = math.isqrt(n_tokens)
    if g * g != n_tokens:
        raise ValueError(f"n_tokens={n_tokens} is not a perfect square")
    return g


def tokens_to_map(tokens: torch.Tensor, grid: int | None = None) -> torch.Tensor:
    """``(B, N, D)`` tokens -> ``(B, D, g, g)`` feature map."""
    b, n, d = tokens.shape
    g = grid or grid_size(n)
    return tokens.transpose(1, 2).reshape(b, d, g, g)


def map_to_tokens(fmap: torch.Tensor) -> torch.Tensor:
    """``(B, D, g, g)`` feature map -> ``(B, N, D)`` tokens."""
    return fmap.flatten(2).transpose(1, 2)


__all__ = ["grid_size", "tokens_to_map", "map_to_tokens"]
