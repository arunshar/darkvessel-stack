"""Test the ORIGINAL xView3 per-scene-folder reader end to end.

Writes tiny VV_dB / VH_dB GeoTIFFs (the format DIU's aria2 download unpacks to)
into temp scene folders, builds the dataset via from_xview3_directory / from_any,
and scores the CFAR baseline. Needs rasterio + scipy + tqdm + pandas; skipped when
absent (so it runs in the container, not on the bare login node).
"""
from __future__ import annotations

import pytest

# rasterio's wheels link against system libraries (e.g. libexpat) that the lean
# CPU CI image lacks, so `import rasterio` can raise a load-time ImportError that
# is NOT a ModuleNotFoundError, which pytest.importorskip does not treat as
# skippable. Guard the whole module so it cleanly skips wherever rasterio cannot
# fully import (CI), and runs where it can (the mirror_pnemo container).
try:
    import rasterio  # noqa: F401
    from rasterio.transform import from_origin  # noqa: F401
except Exception as _exc:  # pragma: no cover - environment dependent
    pytest.skip(f"rasterio not importable: {_exc}", allow_module_level=True)


def _build_scene_dirs(tmp_path):
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.transform import from_origin

    H = W = 400
    vessels = [(100, 150), (300, 320)]
    rows = []
    for s in range(2):
        sid = f"SCENE_{s:03d}"
        d = tmp_path / sid
        d.mkdir()
        rng = np.random.RandomState(s)
        for pol in ("VV", "VH"):
            arr = rng.standard_normal((H, W)).astype("float32")
            for (r, c) in vessels:
                arr[r, c] = 40.0  # bright point target
            with rasterio.open(
                d / f"{pol}_dB.tif", "w", driver="GTiff", height=H, width=W,
                count=1, dtype="float32", crs="EPSG:32601",
                transform=from_origin(0, 0, 10, 10),
            ) as ds:
                ds.write(arr, 1)
        for (r, c) in vessels:
            rows.append(dict(scene_id=sid, detect_scene_row=r, detect_scene_column=c,
                             is_vessel=True, is_fishing=False, vessel_length_m=50.0,
                             confidence="HIGH", distance_from_shore_km=10.0))
    csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_from_xview3_directory_reads_folders(tmp_path) -> None:
    pytest.importorskip("rasterio")
    pytest.importorskip("pandas")
    from darkvessel.data.xview3 import XView3Dataset

    csv = _build_scene_dirs(tmp_path)
    ds = XView3Dataset.from_xview3_directory(tmp_path, csv, chip_size=200, grid=8)
    assert len(ds) == 2 * 4  # 400/200 = 2 chips/side -> 4 chips per scene
    img, target = ds[0]
    assert img.shape == (2, 200, 200)
    assert target["heatmap"].shape == (1, 8, 8)


def test_from_any_autodetects_folder_format_and_scores(tmp_path) -> None:
    pytest.importorskip("rasterio")
    pytest.importorskip("scipy")
    pytest.importorskip("tqdm")
    pytest.importorskip("pandas")
    from darkvessel.data.xview3 import XView3Dataset
    from darkvessel.eval.run_xview3 import cfar_predict_fn, evaluate

    csv = _build_scene_dirs(tmp_path)
    # no .SAFE.zip present -> from_any should pick the xView3 folder reader
    ds = XView3Dataset.from_any(tmp_path, csv, chip_size=200, grid=8)
    assert len(ds) == 2 * 4
    scores = evaluate(ds, cfar_predict_fn(8), batch_size=4)
    assert set(scores) >= {"loc_fscore", "vessel_fscore", "aggregate"}
    # the planted bright targets should be recoverable by the CFAR detector
    assert scores["loc_fscore"] > 0.25
