"""Tests for the bi-temporal change-detection head.

Pure CPU, no downloads. The forward-shape test checks the documented output
shape; the learning test confirms the head actually learns the planted
directional change signal (F1 improves and clears the absolute bar).
"""
from __future__ import annotations

import torch

from darkvessel.heads.change import ChangeHead, make_synthetic, train_eval


def test_change_head_forward_shape() -> None:
    embed_dim, grid = 64, 8
    n_tokens = grid * grid
    head = ChangeHead(embed_dim=embed_dim, grid=grid)
    tokens_a = torch.randn(3, n_tokens, embed_dim)
    tokens_b = torch.randn(3, n_tokens, embed_dim)
    out = head(tokens_a, tokens_b)
    assert out.shape == (3, 1, grid, grid)


def test_make_synthetic_shapes() -> None:
    (tokens_a, tokens_b), target = make_synthetic(5, embed_dim=64, grid=8, seed=0)
    assert tokens_a.shape == (5, 64, 64)
    assert tokens_b.shape == (5, 64, 64)
    assert target.shape == (5, 1, 8, 8)
    # binary mask
    assert set(torch.unique(target).tolist()).issubset({0.0, 1.0})


def test_change_head_learns() -> None:
    out = train_eval(steps=250, seed=0)
    assert out["f1_end"] > out["f1_start"] + 0.1   # real learning
    assert out["f1_end"] > 0.55                      # well above the change-rate baseline
