"""End-to-end xView3 / SARFish detection evaluation harness.

Ties the whole detection track together: tile a set of GRD scenes into chips,
run a per-chip objectness model, map predicted cells back to scene-pixel
detections, and score the assembled predictions against the labels with the
vendored official xView3 metric.

The model is a pluggable ``predict_fn(images) -> logits`` where ``images`` is a
``(B, 2, H, W)`` VV/VH chip batch and ``logits`` is ``(B, 1, grid, grid)``
objectness (pre-sigmoid). Two batteries-included predictors:

* :func:`cfar_predict_fn` - a classical constant-false-alarm-style detector that
  needs NO weights and NO backbone (vessels are bright point targets in SAR, so
  it max-pools amplitude to the grid and thresholds). It runs on real xView3 data
  the moment it is downloaded, and on the synthetic dataset now, which is how the
  harness self-validates.
* a learned path: compose any backbone (image -> tokens) with
  :class:`darkvessel.heads.detection.DetectionHead` (tokens -> logits) into a
  ``predict_fn`` and pass it in.

CLI::

    python -m darkvessel.eval.run_xview3 --self-test
    python -m darkvessel.eval.run_xview3 \
        --data-root /scratch.global/$USER/xview3/validation \
        --labels-csv xView3_val_label_set.csv \
        --grid 16 --limit-scenes 2

HONESTY: the CFAR detector is a crude unlearned baseline, not a competitive
result; it exists to make the pipeline turnkey and self-checking. Real numbers
come from a trained model passed as ``predict_fn`` and from real downloaded data.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from darkvessel.data.xview3 import XView3Dataset, xview3_collate
from darkvessel.eval.xview3_bench import (
    heatmap_to_detections,
    labels_to_dataframe,
    score_predictions,
)

# A predictor maps a chip batch (B,2,H,W) to objectness logits (B,1,grid,grid).
PredictFn = Callable[[torch.Tensor], torch.Tensor]


def cfar_predict_fn(grid: int, abs_threshold: float = 5.0, gain: float = 2.0) -> PredictFn:
    """Classical bright-target (CFAR-style) detector over standardized SAR chips.

    Vessels are strong point scatterers, so per-chip standardized amplitude is
    large at vessel pixels. We take the max over polarisations, max-pool to the
    ``grid x grid`` lattice, and emit ``gain * (pooled - abs_threshold)`` as a
    logit (cells brighter than ``abs_threshold`` z-units fire). No training.
    """
    def fn(images: torch.Tensor) -> torch.Tensor:
        amp = images.amax(dim=1, keepdim=True)              # (B,1,H,W) brightest pol
        pooled = F.adaptive_max_pool2d(amp, (grid, grid))   # (B,1,grid,grid)
        return gain * (pooled - abs_threshold)
    return fn


def constant_predict_fn(grid: int, value: float = -10.0) -> PredictFn:
    """Predict the same logit everywhere (e.g. a strongly negative no-detector)."""
    def fn(images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        return torch.full((b, 1, grid, grid), value, dtype=torch.float32)
    return fn


def evaluate(
    dataset: XView3Dataset,
    predict_fn: PredictFn,
    *,
    batch_size: int = 8,
    threshold: float = 0.5,
    shore_root: str | None = None,
    num_workers: int = 0,
    device: str | torch.device = "cpu",
    return_frames: bool = False,
) -> dict | tuple[dict, Any, Any]:
    """Run ``predict_fn`` over every chip and score against the dataset labels.

    Returns the official score dict (``loc_fscore`` is the detection anchor). With
    ``return_frames=True`` also returns ``(scores, pred_df, gt_df)``.

    Note: with a detection-only predictor (no predicted ``vessel_length_m``), the
    official ``length_acc`` and therefore the composite ``aggregate`` are ``NaN``
    by the metric's formula; read ``loc_fscore`` (and ``vessel_fscore`` /
    ``fishing_fscore`` once a classifier is wired in). Predict lengths to get a
    defined aggregate, as a real leaderboard submission must.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=xview3_collate)
    detections: list[dict[str, Any]] = []
    for images, _heatmaps, targets in loader:
        images = images.to(device)
        with torch.no_grad():
            logits = predict_fn(images).detach().cpu()
        for b, t in enumerate(targets):
            detections.extend(heatmap_to_detections(
                logits[b], t["scene_id"], t["row0"], t["col0"],
                t["chip_size"], t["grid"], threshold=threshold,
            ))
    gt_rows = [lab for labs in dataset.labels_by_scene.values() for lab in labs]
    gt_df = labels_to_dataframe(gt_rows)
    scores = score_predictions(detections, gt_df, shore_root=shore_root)
    if return_frames:
        from darkvessel.eval.xview3_bench import predictions_to_dataframe
        return scores, predictions_to_dataframe(detections), gt_df
    return scores


def _self_test(grid: int = 16) -> dict:
    """Run the harness on the synthetic dataset with the CFAR detector."""
    ds = XView3Dataset.synthetic(n_scenes=2, scene_size=1600, chip_size=800,
                                 grid=grid, vessels_per_scene=6, seed=0)
    scores = evaluate(ds, cfar_predict_fn(grid), batch_size=4)
    assert isinstance(scores, dict)  # return_frames defaults False
    return scores


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="xView3 detection evaluation harness")
    p.add_argument("--self-test", action="store_true",
                   help="run on the synthetic dataset (no real data / weights)")
    p.add_argument("--data-root", help="dir of GRD .SAFE.zip products")
    p.add_argument("--labels-csv", help="xView3 label CSV")
    p.add_argument("--checkpoint", help="torch-saved nn.Module mapping images->logits")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--chip-size", type=int, default=800)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--shore-root", default=None)
    p.add_argument("--limit-scenes", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    if args.self_test:
        print(json.dumps(_self_test(args.grid), indent=2))
        return 0

    if not (args.data_root and args.labels_csv):
        p.error("--data-root and --labels-csv are required unless --self-test")

    ds = XView3Dataset.from_directory(
        args.data_root, args.labels_csv,
        max_scenes=args.limit_scenes, chip_size=args.chip_size, grid=args.grid)

    if args.checkpoint:
        model = torch.load(args.checkpoint, map_location=args.device)
        model.eval()
        predict_fn: PredictFn = lambda imgs: model(imgs)
    else:
        predict_fn = cfar_predict_fn(args.grid)

    scores = evaluate(ds, predict_fn, batch_size=args.batch_size,
                      threshold=args.threshold, shore_root=args.shore_root,
                      device=args.device)
    print(json.dumps(scores, indent=2))
    return 0


__all__ = ["PredictFn", "cfar_predict_fn", "constant_predict_fn", "evaluate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
