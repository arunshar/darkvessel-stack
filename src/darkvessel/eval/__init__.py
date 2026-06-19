"""Real-data evaluation adapters for darkvessel-stack.

Currently holds the xView3 / SARFish official-scorer bridge. The heavy deps
(pandas, tqdm, the vendored scorer) are imported lazily inside the functions so
importing this package stays cheap.
"""
from darkvessel.eval.xview3_bench import (
    heatmap_to_detections,
    labels_to_dataframe,
    load_official_score,
    predictions_to_dataframe,
    score_predictions,
)

__all__ = [
    "heatmap_to_detections",
    "labels_to_dataframe",
    "load_official_score",
    "predictions_to_dataframe",
    "score_predictions",
]
