"""Tests for the next-step feature-forecasting head.

Pure CPU, no downloads. A forward-shape check plus a learning check that the
head beats its own random init AND the honest persistence baseline.
"""
from __future__ import annotations

import torch

from darkvessel.heads.forecast import ForecastHead, make_synthetic, train_eval


def test_forecast_head_forward_shape() -> None:
    embed_dim, grid, t_obs = 64, 8, 4
    n = grid * grid
    head = ForecastHead(embed_dim=embed_dim, grid=grid, n_steps=t_obs)
    seq = torch.randn(3, t_obs, n, embed_dim)
    out = head(seq)
    assert out.shape == (3, n, embed_dim)


def test_make_synthetic_shapes() -> None:
    inputs, target = make_synthetic(5, embed_dim=64, grid=8, seed=0, steps=4)
    assert inputs.shape == (5, 4, 64, 64)
    assert target.shape == (5, 64, 64)


def test_forecast_learns_and_beats_persistence() -> None:
    out = train_eval(steps=220, seed=0)
    # the head must actually improve over its random init ...
    assert out["mse_end"] < out["mse_start"]
    # ... and must beat the honest persistence baseline (the absolute bar).
    assert out["mse_end"] < out["mse_persistence"]
