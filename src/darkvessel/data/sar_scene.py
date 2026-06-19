"""Reader for xView3 / SARFish GRD Sentinel-1 SAFE products.

A GRD ``.SAFE(.zip)`` holds full-scene VV and VH detected-amplitude GeoTIFFs
(here 16644 x 25306, ``uint16`` digital numbers, nodata 0) plus an xView3
shoreline vector. The full scene is ~840 MB per polarisation, so we never load
it whole: rasterio windowed reads pull fixed chips on demand. Amplitude DN is
mapped to a dB-like scale (``20*log10``) and standardised per chip, the standard
SAR-detection front end.

Labels are NOT bundled with the imagery; they come from the xView3 label CSV
(iuu.xview.us/download-links), keyed by ``detect_scene_row`` / ``detect_scene_column``,
and are attached to chips by the (later) chip sampler when present. The same CSV
schema scores via the vendored official metric (reference/xview3_official_metric.py).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window

POLARISATIONS = ("vv", "vh")


def _inner_tiff(zip_path: str, pol: str) -> str:
    """Find the measurement GeoTIFF for a polarisation inside the SAFE zip."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            low = name.lower()
            if low.endswith(".tiff") and f"grd-{pol}-" in low and "/measurement/" in low:
                return name
    raise FileNotFoundError(f"no GRD {pol} tiff in {zip_path}")


def _shoreline(zip_path: str) -> np.ndarray | None:
    """Load the xView3 shoreline vertex array (N, 2) if present."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.lower().endswith("_xview3_shoreline.npy"):
                with z.open(name) as f:
                    return np.load(io.BytesIO(f.read()), allow_pickle=True)
    return None


def to_db(dn: np.ndarray, floor: float = 1.0) -> np.ndarray:
    """Map uint16 amplitude DN to a dB-like scale; nodata (0) -> NaN."""
    x = dn.astype(np.float32)
    out = 20.0 * np.log10(np.clip(x, floor, None))
    out[dn == 0] = np.nan
    return out


@dataclass
class SARScene:
    """Lazy windowed reader over a GRD SAFE product's VV/VH channels."""

    zip_path: str

    def __post_init__(self) -> None:
        self._vsi = {p: f"/vsizip/{self.zip_path}/{_inner_tiff(self.zip_path, p)}"
                     for p in POLARISATIONS}
        with rasterio.open(self._vsi["vv"]) as ds:
            self.height, self.width = ds.height, ds.width
        self.shoreline = _shoreline(self.zip_path)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def read_chip(self, row0: int, col0: int, size: int = 800,
                  standardize: bool = True) -> np.ndarray:
        """Return a (2, size, size) VV/VH chip in dB; NaNs (nodata) -> 0.

        With ``standardize`` each channel is per-chip z-scored over valid pixels,
        the front end the detection head expects."""
        chans = []
        for p in POLARISATIONS:
            with rasterio.open(self._vsi[p]) as ds:
                w = ds.read(1, window=Window(col0, row0, size, size),
                            boundless=True, fill_value=0)
            db = to_db(w)
            if standardize:
                valid = np.isfinite(db)
                if valid.any():
                    mu, sd = db[valid].mean(), db[valid].std() + 1e-6
                    db = (db - mu) / sd
            chans.append(np.nan_to_num(db, nan=0.0))
        return np.stack(chans, axis=0).astype(np.float32)

    def chip_grid(self, size: int = 800, overlap: int = 0) -> list[tuple[int, int]]:
        """Top-left (row, col) of a tiling grid; the xView3 default chip is 800px."""
        step = size - overlap
        rows = list(range(0, max(self.height - size, 0) + 1, step)) or [0]
        cols = list(range(0, max(self.width - size, 0) + 1, step)) or [0]
        return [(r, c) for r in rows for c in cols]

    def valid_fraction(self, row0: int, col0: int, size: int = 800) -> float:
        """Fraction of non-nodata pixels in a chip (for skipping empty ocean-edge tiles)."""
        with rasterio.open(self._vsi["vv"]) as ds:
            w = ds.read(1, window=Window(col0, row0, size, size), boundless=True, fill_value=0)
        return float((w != 0).mean())
