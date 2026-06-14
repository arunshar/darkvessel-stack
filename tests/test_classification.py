"""Tests for the whole-scene classification head.

Pure CPU, synthetic-data only (no HF downloads, no raster reads). Exercises the
forward output shape and that the head actually learns the planted class signal.
"""
from __future__ import annotations

import torch

from darkvessel.heads.classification import (
    ClassificationHead,
    make_synthetic,
    train_eval,
)


def test_classification_head_forward_shape() -> None:
    head = ClassificationHead(embed_dim=64, grid=8, n_classes=5)
    tokens = torch.randn(3, 64, 64)          # (B, N=grid*grid, D)
    logits = head(tokens)
    assert logits.shape == (3, 5)


def test_make_synthetic_shapes_and_range() -> None:
    x, y = make_synthetic(16, embed_dim=64, grid=8, seed=0, n_classes=5)
    assert x.shape == (16, 64, 64)
    assert y.shape == (16,)
    assert y.dtype == torch.long
    assert int(y.min()) >= 0 and int(y.max()) < 5


def test_classification_head_learns() -> None:
    out = train_eval(steps=250, seed=0)
    assert out["acc_end"] > out["acc_start"] + 0.1   # real learning
    assert out["acc_end"] > 0.6                        # well above 5-class chance (0.2)
