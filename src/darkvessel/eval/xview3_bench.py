"""Adapter from DetectionHead outputs to the official xView3 scorer.

Bridges the chip-level objectness predictions produced by
:class:`darkvessel.heads.detection.DetectionHead` to the vendored official metric
``reference/xview3_official_metric.py``, which scores at the full-scene level on
DataFrames of detections.

Pipeline:
  1. :func:`heatmap_to_detections` turns a chip's ``(1, grid, grid)`` objectness
     logits into scene-pixel detections (cell centres above threshold).
  2. :func:`predictions_to_dataframe` / :func:`labels_to_dataframe` assemble the
     ``pred`` and ``gt`` DataFrames in the exact schema ``score()`` expects.
  3. :func:`score_predictions` calls the official ``score()`` and returns its dict
     (``loc_fscore``, ``loc_fscore_shore``, ``vessel_fscore``, ``fishing_fscore``,
     ``length_acc``, ``aggregate``). ``loc_fscore`` is the detection-leaderboard
     anchor metric.

This scaffold predicts DETECTION only. ``is_vessel`` / ``is_fishing`` /
``vessel_length_m`` for predictions default to ``(True, False, NaN)``; wire the
classification + length heads in via the ``vessel_fn`` / ``fishing_fn`` /
``length_fn`` hooks (or post-fill the pred DataFrame) when those heads exist.

The vendored scorer is imported lazily and robustly: it does ``from constants
import ...`` (the vendored constants file is ``xview3_official_constants.py``) and
``from tqdm import tqdm``, so it needs ``reference/`` on ``sys.path``, a
``constants`` alias, and ``tqdm`` installed (the ``xview3`` optional extra).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# Required prediction/ground-truth columns for the official scorer.
PRED_COLUMNS = [
    "scene_id", "detect_scene_row", "detect_scene_column",
    "is_vessel", "is_fishing", "vessel_length_m",
]
GT_EXTRA_COLUMNS = ["confidence", "distance_from_shore_km"]

# Returned when there are no predictions to score (the official scorer crashes on
# an empty pred DataFrame: scipy distance_matrix gets a (0,) array, not (0, 2)).
ZERO_SCORES = {
    "loc_fscore": 0.0, "loc_fscore_shore": 0.0, "vessel_fscore": 0.0,
    "fishing_fscore": 0.0, "length_acc": 0.0, "aggregate": 0.0,
}


def _reference_dir() -> Path:
    """Absolute path to the repo's ``reference/`` dir (holds the vendored scorer)."""
    # src/darkvessel/eval/xview3_bench.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "reference"


def load_official_score() -> Callable[..., dict]:
    """Import and return the official ``score`` function, fixing its import quirks.

    Keeps the vendored file pristine: we put ``reference/`` on ``sys.path`` and
    alias ``xview3_official_constants`` as ``constants`` so the scorer's
    ``from constants import ...`` resolves.
    """
    ref = _reference_dir()
    if not (ref / "xview3_official_metric.py").exists():
        raise FileNotFoundError(f"vendored scorer not found under {ref}")
    if str(ref) not in sys.path:
        sys.path.insert(0, str(ref))
    if "constants" not in sys.modules:
        import xview3_official_constants as _c  # type: ignore
        sys.modules["constants"] = _c
    import xview3_official_metric  # type: ignore
    return xview3_official_metric.score


def heatmap_to_detections(
    logits: "Any",
    scene_id: str,
    row0: int,
    col0: int,
    chip_size: int,
    grid: int,
    threshold: float = 0.5,
    vessel_fn: Callable[[int, int], bool] | None = None,
    fishing_fn: Callable[[int, int], bool] | None = None,
    length_fn: Callable[[int, int], float] | None = None,
) -> list[dict[str, Any]]:
    """Convert one chip's objectness logits to scene-pixel detections.

    ``logits`` is ``(1, grid, grid)`` or ``(grid, grid)`` (torch tensor or ndarray)
    of pre-sigmoid objectness. Cells with ``sigmoid(logit) >= threshold`` become
    detections at the cell CENTRE (mapped back to scene pixels).
    """
    from darkvessel.data.xview3 import cell_to_scene

    arr = np.asarray(logits.detach().cpu()) if hasattr(logits, "detach") \
        else np.asarray(logits)
    arr = arr.reshape(grid, grid)
    probs = 1.0 / (1.0 + np.exp(-arr))
    dets: list[dict[str, Any]] = []
    for cr in range(grid):
        for cc in range(grid):
            if probs[cr, cc] >= threshold:
                r, c = cell_to_scene(cr, cc, row0, col0, chip_size, grid)
                dets.append({
                    "scene_id": scene_id,
                    "detect_scene_row": r,
                    "detect_scene_column": c,
                    "is_vessel": vessel_fn(cr, cc) if vessel_fn else True,
                    "is_fishing": fishing_fn(cr, cc) if fishing_fn else False,
                    "vessel_length_m": length_fn(cr, cc) if length_fn else math.nan,
                    "score": float(probs[cr, cc]),
                })
    return dets


def predictions_to_dataframe(detections: Sequence[dict[str, Any]]) -> "Any":
    """Assemble a ``pred`` DataFrame with the columns the scorer requires."""
    import pandas as pd
    df = pd.DataFrame(list(detections))
    if df.empty:
        df = pd.DataFrame(columns=PRED_COLUMNS)
    for col in PRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def labels_to_dataframe(labels: "Any") -> "Any":
    """Normalise xView3 labels (DataFrame, CSV path, or row-dict list) to gt schema.

    Fills the scorer-only columns when absent: ``confidence`` -> ``"HIGH"`` and
    ``distance_from_shore_km`` -> a large value (so shore filtering is a no-op
    unless real distances are present).
    """
    import pandas as pd
    if isinstance(labels, (str, Path)):
        df = pd.read_csv(labels)
    elif isinstance(labels, pd.DataFrame):
        df = labels.copy()
    else:
        df = pd.DataFrame(list(labels))
    for col in PRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if "confidence" not in df.columns:
        df["confidence"] = "HIGH"
    if "distance_from_shore_km" not in df.columns:
        df["distance_from_shore_km"] = 1e9
    return df


def score_predictions(
    pred: "Any",
    gt: "Any",
    shore_root: str | None = None,
    distance_tolerance: int = 200,
    shore_tolerance: int = 2,
    costly_dist: bool = False,
) -> dict:
    """Score ``pred`` against ``gt`` with the official xView3 metric.

    ``pred`` / ``gt`` may be DataFrames or anything :func:`predictions_to_dataframe`
    / :func:`labels_to_dataframe` accept. With ``shore_root=None`` the shore metric
    is skipped (returns 0), per the official scorer.
    """
    import pandas as pd
    pred_df = pred if isinstance(pred, pd.DataFrame) else predictions_to_dataframe(pred)
    gt_df = gt if isinstance(gt, pd.DataFrame) else labels_to_dataframe(gt)
    # The official scorer assumes >=1 prediction; with none, its scipy
    # distance_matrix call gets a (0,) array and raises. Detection of nothing is a
    # well-defined zero score, so short-circuit.
    if len(pred_df) == 0:
        return dict(ZERO_SCORES)
    score = load_official_score()
    return score(
        pred_df, gt_df, shore_root,
        distance_tolerance=distance_tolerance,
        shore_tolerance=shore_tolerance,
        costly_dist=costly_dist,
    )


__all__ = [
    "PRED_COLUMNS",
    "GT_EXTRA_COLUMNS",
    "load_official_score",
    "heatmap_to_detections",
    "predictions_to_dataframe",
    "labels_to_dataframe",
    "score_predictions",
]
