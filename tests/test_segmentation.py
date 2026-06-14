"""Tests for the dense per-cell segmentation head.

Pure CPU, no downloads: a forward-shape check on random tokens and a learning
check that ``train_eval`` improves mIoU on a fresh synthetic set and clears the
absolute bar. Both run in a few seconds.
"""
from __future__ import annotations

import torch

from darkvessel.heads.segmentation import (
    SegmentationHead,
    make_synthetic,
    train_eval,
)


def test_segmentation_head_forward_shape() -> None:
    embed_dim, grid, n_classes = 64, 8, 4
    head = SegmentationHead(embed_dim=embed_dim, grid=grid, n_classes=n_classes)
    tokens = torch.randn(3, grid * grid, embed_dim)
    logits = head(tokens)
    assert logits.shape == (3, n_classes, grid, grid)


def test_make_synthetic_shapes_and_labels() -> None:
    embed_dim, grid, n_classes = 64, 8, 4
    tokens, target = make_synthetic(
        5, embed_dim=embed_dim, grid=grid, seed=1, n_classes=n_classes
    )
    assert tokens.shape == (5, grid * grid, embed_dim)
    assert target.shape == (5, grid, grid)
    assert target.dtype == torch.long
    assert int(target.min()) >= 0
    assert int(target.max()) < n_classes


def test_segmentation_head_learns() -> None:
    out = train_eval(steps=250, seed=0)
    assert out["miou_end"] > out["miou_start"] + 0.05   # real learning
    assert out["miou_end"] > 0.4                          # well above 4-class chance
