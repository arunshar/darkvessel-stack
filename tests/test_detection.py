"""Tests for the per-cell objectness detection head.

CPU-only and fast: a shape test plus an end-to-end learning test on synthetic
planted-object tokens. No HF downloads, no real EO data.
"""
from __future__ import annotations

import torch

from darkvessel.heads.detection import DetectionHead, make_synthetic, train_eval


def test_detection_head_forward_shape() -> None:
    embed_dim, grid = 64, 8
    head = DetectionHead(embed_dim=embed_dim, grid=grid)
    tokens = torch.randn(3, grid * grid, embed_dim)
    out = head(tokens)
    assert out.shape == (3, 1, grid, grid)


def test_make_synthetic_shapes_and_labels() -> None:
    inputs, target = make_synthetic(5, embed_dim=64, grid=8, seed=0)
    assert inputs.shape == (5, 64, 64)
    assert target.shape == (5, 1, 8, 8)
    # binary heatmap with at least one object per sample (K >= 1)
    assert set(torch.unique(target).tolist()) <= {0.0, 1.0}
    assert (target.sum(dim=(1, 2, 3)) >= 1).all()


def test_detection_head_learns() -> None:
    out = train_eval(seed=0)
    assert out["f1_end"] > out["f1_start"] + 0.1   # real learning, not noise
    assert out["f1_end"] > 0.6                       # well above the sparse-object baseline
