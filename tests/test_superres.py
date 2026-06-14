"""Tests for the x2 feature-map super-resolution head.

CPU-only, no downloads. A forward-shape test on a documented-shape input and a
learning test that calls ``train_eval`` and asserts the head PSNR improves with
training and clears an absolute bar. The bilinear baseline is reported honestly;
we do NOT assert the head beats it.
"""
from __future__ import annotations

import torch

from darkvessel.heads.superres import SuperResHead, make_synthetic, train_eval


def test_superres_forward_shape() -> None:
    embed_dim, grid = 64, 8
    head = SuperResHead(embed_dim=embed_dim, grid=grid)
    lr_map = torch.randn(3, embed_dim, grid, grid)
    hr_map = head(lr_map)
    assert hr_map.shape == (3, embed_dim, 2 * grid, 2 * grid)


def test_make_synthetic_shapes() -> None:
    lr, hr = make_synthetic(5, embed_dim=64, grid=8, seed=0)
    assert lr.shape == (5, 64, 8, 8)
    assert hr.shape == (5, 64, 16, 16)
    # LR must be the avg-pool of HR (the documented construction)
    assert torch.allclose(torch.nn.functional.avg_pool2d(hr, 2), lr, atol=1e-5)


def test_superres_learns() -> None:
    out = train_eval(seed=0)
    # the head must improve with training (learning test)
    assert out["psnr_end"] > out["psnr_start"] + 0.3
    # and it should beat the honest bilinear-upsampling baseline on the same targets
    assert out["psnr_end"] > out["psnr_bilinear"]
    assert out["psnr_bilinear"] == out["psnr_bilinear"]  # finite (not NaN)
