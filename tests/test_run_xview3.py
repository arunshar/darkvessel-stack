"""Tests for the end-to-end xView3 detection evaluation harness.

Data-blind: runs the harness on the synthetic dataset. The CFAR detector should
recover the planted bright vessels, which validates the full
loader -> per-chip inference -> scene-coordinate mapping -> official-scorer path.
Needs scipy + tqdm + pandas (the vendored scorer); skipped when absent.
"""
from __future__ import annotations

import math

import pytest

from darkvessel.data.xview3 import XView3Dataset
from darkvessel.eval.run_xview3 import (
    cfar_predict_fn,
    constant_predict_fn,
    evaluate,
)


def _need_scorer():
    pytest.importorskip("scipy")
    pytest.importorskip("tqdm")
    pytest.importorskip("pandas")


def test_cfar_recovers_synthetic_vessels() -> None:
    _need_scorer()
    grid = 16
    ds = XView3Dataset.synthetic(n_scenes=2, scene_size=1600, chip_size=800,
                                 grid=grid, vessels_per_scene=6, seed=0)
    scores = evaluate(ds, cfar_predict_fn(grid), batch_size=4)
    assert set(scores) >= {"loc_fscore", "vessel_fscore", "fishing_fscore",
                           "length_acc", "aggregate"}
    # Detection-only: lengths are unpredicted, so length_acc and the composite
    # aggregate are NaN by the official formula. loc_fscore is the detection anchor.
    assert math.isnan(scores["aggregate"]) or 0.0 <= scores["aggregate"] <= 1.0
    # the planted vessels are bright point targets; CFAR recovers a real fraction
    assert scores["loc_fscore"] > 0.25


def test_no_detector_scores_zero() -> None:
    _need_scorer()
    grid = 16
    ds = XView3Dataset.synthetic(n_scenes=1, grid=grid, vessels_per_scene=5, seed=1)
    scores = evaluate(ds, constant_predict_fn(grid, value=-10.0))
    assert scores["loc_fscore"] == 0.0


def test_evaluate_returns_frames() -> None:
    _need_scorer()
    grid = 16
    ds = XView3Dataset.synthetic(n_scenes=1, grid=grid, vessels_per_scene=4, seed=2)
    out = evaluate(ds, cfar_predict_fn(grid), batch_size=2, return_frames=True)
    assert isinstance(out, tuple) and len(out) == 3
    scores, pred_df, gt_df = out
    assert "aggregate" in scores
    # gt has one row per planted vessel; pred columns match the scorer schema
    assert len(gt_df) == sum(len(v) for v in ds.labels_by_scene.values())
    assert {"scene_id", "detect_scene_row", "detect_scene_column"} <= set(pred_df.columns)
