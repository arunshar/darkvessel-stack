"""Tests for the xView3 / SARFish detection loader and scorer adapter.

Data-blind: everything runs on the synthetic in-memory dataset (no rasterio, no
real downloaded scenes). The coordinate-mapping tests are pure. The scorer tests
import the vendored official metric, which needs scipy + tqdm + pandas; they are
skipped (not failed) when those are absent.
"""
from __future__ import annotations

import math

import pytest
import torch

from darkvessel.data.xview3 import (
    XView3Dataset,
    cell_to_scene,
    labels_to_heatmap,
    scene_to_cell,
    xview3_collate,
)
from darkvessel.heads.detection import DetectionHead


# --------------------------------------------------------------------------- #
# Pure coordinate mapping
# --------------------------------------------------------------------------- #
def test_cell_scene_roundtrip() -> None:
    chip_size, grid = 800, 16
    row0, col0 = 1600, 800
    for cr in range(grid):
        for cc in range(grid):
            r, c = cell_to_scene(cr, cc, row0, col0, chip_size, grid)
            assert scene_to_cell(r, c, row0, col0, chip_size, grid) == (cr, cc)


def test_scene_to_cell_outside_chip_is_none() -> None:
    assert scene_to_cell(10, 10, 1000, 1000, 800, 16) is None       # before chip
    assert scene_to_cell(2000, 2000, 1000, 1000, 800, 16) is None   # after chip


def test_labels_to_heatmap_marks_object_cells() -> None:
    grid, chip_size = 8, 800
    labels = [{"detect_scene_row": 0, "detect_scene_column": 0},          # cell (0,0)
              {"detect_scene_row": 799, "detect_scene_column": 799}]      # cell (7,7)
    hm = labels_to_heatmap(labels, 0, 0, chip_size, grid)
    assert hm.shape == (1, grid, grid)
    assert hm[0, 0, 0] == 1.0 and hm[0, 7, 7] == 1.0
    assert hm.sum() == 2.0


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #
def test_synthetic_dataset_item_shapes() -> None:
    ds = XView3Dataset.synthetic(n_scenes=2, scene_size=1600, chip_size=800,
                                 grid=16, vessels_per_scene=6, seed=0)
    assert len(ds) == 2 * 4  # 1600/800 = 2 chips per side -> 4 chips per scene
    img, target = ds[0]
    assert img.shape == (2, 800, 800)
    assert target["heatmap"].shape == (1, 16, 16)
    assert {"scene_id", "row0", "col0", "chip_size", "grid", "labels"} <= set(target)


def test_synthetic_dataset_all_vessels_land_in_some_chip() -> None:
    ds = XView3Dataset.synthetic(n_scenes=1, scene_size=1600, chip_size=800,
                                 grid=16, vessels_per_scene=10, seed=1)
    planted = sum(len(v) for v in ds.labels_by_scene.values())
    on_grid = sum(int(ds[i][1]["heatmap"].sum().item()) for i in range(len(ds)))
    # every planted vessel falls inside exactly one (non-overlapping) chip
    assert on_grid == planted


def test_collate_batches_images_and_heatmaps() -> None:
    ds = XView3Dataset.synthetic(n_scenes=1, chip_size=800, grid=16, seed=2)
    batch = [ds[i] for i in range(len(ds))]
    images, heatmaps, targets = xview3_collate(batch)
    assert images.shape == (len(ds), 2, 800, 800)
    assert heatmaps.shape == (len(ds), 1, 16, 16)
    assert len(targets) == len(ds)


def test_detection_head_consumes_grid_contract() -> None:
    # the loader's grid must match what DetectionHead emits: (B,1,grid,grid)
    grid, embed_dim = 16, 64
    head = DetectionHead(embed_dim=embed_dim, grid=grid)
    tokens = torch.randn(3, grid * grid, embed_dim)
    assert head(tokens).shape == (3, 1, grid, grid)


# --------------------------------------------------------------------------- #
# Official-scorer adapter (needs scipy + tqdm + pandas)
# --------------------------------------------------------------------------- #
def _bench():
    pytest.importorskip("scipy")
    pytest.importorskip("tqdm")
    pytest.importorskip("pandas")
    import darkvessel.eval.xview3_bench as bench
    return bench


def test_scorer_perfect_prediction_scores_high() -> None:
    bench = _bench()
    ds = XView3Dataset.synthetic(n_scenes=1, vessels_per_scene=8, seed=3)
    gt_rows = [l for v in ds.labels_by_scene.values() for l in v]
    gt = bench.labels_to_dataframe(gt_rows)
    # feed ground truth back as the prediction -> detection should be ~perfect
    pred = bench.predictions_to_dataframe([
        {"scene_id": r["scene_id"],
         "detect_scene_row": r["detect_scene_row"],
         "detect_scene_column": r["detect_scene_column"],
         "is_vessel": r["is_vessel"],
         "is_fishing": r["is_fishing"],
         "vessel_length_m": r["vessel_length_m"]}
        for r in gt_rows
    ])
    scores = bench.score_predictions(pred, gt, shore_root=None)
    assert set(scores) >= {"loc_fscore", "vessel_fscore", "fishing_fscore",
                           "length_acc", "aggregate"}
    assert scores["loc_fscore"] > 0.99
    assert 0.0 <= scores["aggregate"] <= 1.0


def test_scorer_empty_prediction_scores_zero() -> None:
    bench = _bench()
    ds = XView3Dataset.synthetic(n_scenes=1, vessels_per_scene=5, seed=4)
    gt_rows = [l for v in ds.labels_by_scene.values() for l in v]
    gt = bench.labels_to_dataframe(gt_rows)
    pred = bench.predictions_to_dataframe([])
    scores = bench.score_predictions(pred, gt, shore_root=None)
    assert scores["loc_fscore"] == 0.0


def test_heatmap_to_detections_maps_to_scene_pixels() -> None:
    bench = _bench()
    grid, chip_size, row0, col0 = 16, 800, 800, 1600
    logits = torch.full((1, grid, grid), -10.0)
    logits[0, 3, 5] = 10.0  # one confident cell
    dets = bench.heatmap_to_detections(
        logits, "SYN_SCENE_000", row0, col0, chip_size, grid, threshold=0.5)
    assert len(dets) == 1
    exp_r, exp_c = cell_to_scene(3, 5, row0, col0, chip_size, grid)
    assert dets[0]["detect_scene_row"] == exp_r
    assert dets[0]["detect_scene_column"] == exp_c
    assert dets[0]["is_vessel"] is True
    assert math.isnan(dets[0]["vessel_length_m"])
