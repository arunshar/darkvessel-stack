"""xView3-SAR / SARFish detection data loader.

This builds a chip-level detection dataset over Sentinel-1 GRD scenes in the
xView3 / SARFish format. A full GRD scene is ~16k x 25k pixels, so we never load
it whole: each scene is tiled into fixed chips (default 800 px, the xView3 chip
size) read on demand through :class:`darkvessel.data.sar_scene.SARScene`. Per-chip
detection targets are a binary objectness heatmap on a ``grid x grid`` lattice,
matching the contract the :class:`darkvessel.heads.detection.DetectionHead`
produces (``(B, 1, grid, grid)`` logits).

Label schema (xView3 / SARFish label CSV, iuu.xview.us/download-links): one row
per detection, columns ``scene_id``, ``detect_scene_row``, ``detect_scene_column``,
``is_vessel``, ``is_fishing``, ``vessel_length_m`` (plus ``confidence`` and
``distance_from_shore_km`` used only by the scorer). The same CSV scores via the
vendored official metric (``reference/xview3_official_metric.py``), wrapped by
:mod:`darkvessel.eval.xview3_bench`.

HONESTY BOUNDARY: the real-data path is implemented against the documented xView3 /
SARFish layout but has NOT been validated end to end on real downloaded scenes yet
(the imagery pull is pending credentials). The synthetic constructor
(:meth:`XView3Dataset.synthetic`) exercises the full loader -> heatmap -> scorer
wiring with no rasterio and no real data, which is what the unit tests run. The
chip coordinate mapping and the heatmap encoding are exact, so wiring real scenes
in is a matter of pointing :meth:`XView3Dataset.from_directory` at downloaded
``.SAFE.zip`` products and the label CSV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Chip <-> scene coordinate mapping (pure, no I/O, fully tested)
# --------------------------------------------------------------------------- #
def cell_size_px(chip_size: int, grid: int) -> float:
    """Side length in pixels of one objectness-grid cell within a chip."""
    return chip_size / grid


def scene_to_cell(
    row: float,
    col: float,
    row0: int,
    col0: int,
    chip_size: int,
    grid: int,
) -> tuple[int, int] | None:
    """Map a scene-pixel detection ``(row, col)`` to a chip grid cell.

    Returns ``(cell_row, cell_col)`` if the detection falls inside the chip whose
    top-left is ``(row0, col0)``, else ``None``.
    """
    lr, lc = row - row0, col - col0
    if not (0 <= lr < chip_size and 0 <= lc < chip_size):
        return None
    cpx = cell_size_px(chip_size, grid)
    cell_row = min(int(lr // cpx), grid - 1)
    cell_col = min(int(lc // cpx), grid - 1)
    return cell_row, cell_col


def cell_to_scene(
    cell_row: int,
    cell_col: int,
    row0: int,
    col0: int,
    chip_size: int,
    grid: int,
) -> tuple[int, int]:
    """Map a chip grid cell back to a scene-pixel coordinate (the cell CENTRE)."""
    cpx = cell_size_px(chip_size, grid)
    row = row0 + int((cell_row + 0.5) * cpx)
    col = col0 + int((cell_col + 0.5) * cpx)
    return row, col


def labels_to_heatmap(
    labels: Sequence[dict[str, Any]],
    row0: int,
    col0: int,
    chip_size: int,
    grid: int,
) -> np.ndarray:
    """Binary ``(1, grid, grid)`` objectness heatmap for the labels inside a chip."""
    hm = np.zeros((1, grid, grid), dtype=np.float32)
    for lab in labels:
        cell = scene_to_cell(
            float(lab["detect_scene_row"]),
            float(lab["detect_scene_column"]),
            row0, col0, chip_size, grid,
        )
        if cell is not None:
            hm[0, cell[0], cell[1]] = 1.0
    return hm


# --------------------------------------------------------------------------- #
# Scene readers: real (SARScene) and in-memory synthetic share one duck-typed
# interface -> read_chip(row0, col0, size, standardize) -> (2, size, size).
# --------------------------------------------------------------------------- #
@dataclass
class _ArrayScene:
    """In-memory stand-in for :class:`SARScene` used by the synthetic dataset.

    Mirrors ``SARScene.read_chip`` semantics: boundless windowed read (out-of-bounds
    filled with 0) and optional per-chip per-channel z-score over finite pixels.
    """

    array: np.ndarray  # (2, H, W) float32

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape[1], self.array.shape[2]

    def read_chip(self, row0: int, col0: int, size: int = 800,
                  standardize: bool = True) -> np.ndarray:
        h, w = self.shape
        out = np.zeros((2, size, size), dtype=np.float32)
        r1, c1 = min(row0 + size, h), min(col0 + size, w)
        r0, c0 = max(row0, 0), max(col0, 0)
        if r1 > r0 and c1 > c0:
            out[:, r0 - row0:r1 - row0, c0 - col0:c1 - col0] = self.array[:, r0:r1, c0:c1]
        if standardize:
            for ch in range(out.shape[0]):
                band = out[ch]
                mu, sd = band.mean(), band.std() + 1e-6
                out[ch] = (band - mu) / sd
        return out

    def chip_grid(self, size: int = 800, overlap: int = 0) -> list[tuple[int, int]]:
        h, w = self.shape
        step = size - overlap
        rows = list(range(0, max(h - size, 0) + 1, step)) or [0]
        cols = list(range(0, max(w - size, 0) + 1, step)) or [0]
        return [(r, c) for r in rows for c in cols]

    def valid_fraction(self, row0: int, col0: int, size: int = 800) -> float:
        return 1.0


@dataclass
class ChipRef:
    """One (scene, chip-window) sample in the dataset index."""

    scene_id: str
    scene_idx: int
    row0: int
    col0: int


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class XView3Dataset(Dataset):
    """Chip-level xView3 / SARFish detection dataset.

    Each item is ``(image, target)`` where ``image`` is a ``(2, chip_size,
    chip_size)`` VV/VH float tensor and ``target`` is a dict::

        {
          "heatmap": (1, grid, grid) float tensor   # binary objectness
          "scene_id": str,
          "row0": int, "col0": int,                 # chip offset in the scene
          "chip_size": int, "grid": int,
          "labels": [ {detect_scene_row, detect_scene_column, is_vessel,
                       is_fishing, vessel_length_m}, ... ]  # detections in this chip
        }

    Use :func:`xview3_collate` as the DataLoader ``collate_fn`` to batch the
    images + heatmaps while keeping the variable-length per-chip label lists.
    """

    def __init__(
        self,
        scenes: Sequence[Any],
        scene_ids: Sequence[str],
        labels_by_scene: dict[str, list[dict[str, Any]]],
        chip_size: int = 800,
        grid: int = 16,
        overlap: int = 0,
        standardize: bool = True,
        min_valid_fraction: float = 0.0,
        keep_empty: bool = True,
    ) -> None:
        if len(scenes) != len(scene_ids):
            raise ValueError("scenes and scene_ids must align 1:1")
        self.scenes = list(scenes)
        self.scene_ids = list(scene_ids)
        self.labels_by_scene = labels_by_scene
        self.chip_size = chip_size
        self.grid = grid
        self.overlap = overlap
        self.standardize = standardize

        self.index: list[ChipRef] = []
        for si, (scene, sid) in enumerate(zip(self.scenes, self.scene_ids)):
            for (r0, c0) in scene.chip_grid(size=chip_size, overlap=overlap):
                if min_valid_fraction > 0.0 and \
                        scene.valid_fraction(r0, c0, chip_size) < min_valid_fraction:
                    continue
                if not keep_empty:
                    has_lab = any(
                        scene_to_cell(
                            float(l["detect_scene_row"]), float(l["detect_scene_column"]),
                            r0, c0, chip_size, grid) is not None
                        for l in labels_by_scene.get(sid, [])
                    )
                    if not has_lab:
                        continue
                self.index.append(ChipRef(sid, si, r0, c0))

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        labels_csv: str | Path,
        glob: str = "**/*.SAFE.zip",
        max_scenes: int | None = None,
        **kw: Any,
    ) -> "XView3Dataset":
        """Build from a directory of GRD ``.SAFE.zip`` products + a label CSV.

        ``root`` is scanned with ``glob`` for SAFE zips; each becomes a
        :class:`SARScene`. The label CSV is grouped by ``scene_id``. The scene id
        is the SAFE product stem (filename without ``.SAFE.zip``); ensure the CSV's
        ``scene_id`` uses the same convention. ``max_scenes`` caps the number of
        scenes (sorted order) for a quick validation-subset run.
        """
        import pandas as pd  # lazy: only the real path needs pandas
        from darkvessel.data.sar_scene import SARScene  # lazy: pulls rasterio

        root = Path(root)
        zips = sorted(root.glob(glob))
        if not zips:
            raise FileNotFoundError(f"no SAFE zips matching {glob!r} under {root}")
        if max_scenes is not None:
            zips = zips[:max_scenes]
        scenes, scene_ids = [], []
        for z in zips:
            scenes.append(SARScene(str(z)))
            scene_ids.append(z.name.replace(".SAFE.zip", "").replace(".zip", ""))

        df = pd.read_csv(labels_csv)
        labels_by_scene = _group_labels(df)
        return cls(scenes, scene_ids, labels_by_scene, **kw)

    @classmethod
    def from_xview3_directory(
        cls,
        root: str | Path,
        labels_csv: str | Path,
        max_scenes: int | None = None,
        **kw: Any,
    ) -> "XView3Dataset":
        """Build from the ORIGINAL xView3 per-scene-folder layout.

        DIU's aria2 download unpacks each scene to a subdirectory named by
        ``scene_id`` holding ``VV_dB.tif`` / ``VH_dB.tif`` (plus aux layers). Use
        :meth:`from_directory` instead for Sentinel-1 ``.SAFE.zip`` products, or
        :meth:`from_any` to auto-detect. ``max_scenes`` caps the scene count.
        """
        import pandas as pd  # lazy
        from darkvessel.data.sar_scene import XView3SceneFolder  # lazy: pulls rasterio

        root = Path(root)
        scene_dirs = _find_xview3_scene_dirs(root)
        if not scene_dirs:
            raise FileNotFoundError(
                f"no xView3 scene folders (a subdir with VV/VH GeoTIFFs) under {root}")
        if max_scenes is not None:
            scene_dirs = scene_dirs[:max_scenes]
        scenes = [XView3SceneFolder(str(d)) for d in scene_dirs]
        scene_ids = [d.name for d in scene_dirs]
        df = pd.read_csv(labels_csv)
        labels_by_scene = _group_labels(df)
        return cls(scenes, scene_ids, labels_by_scene, **kw)

    @classmethod
    def from_any(
        cls,
        root: str | Path,
        labels_csv: str | Path,
        **kw: Any,
    ) -> "XView3Dataset":
        """Auto-detect the on-disk format: ``.SAFE.zip`` products vs xView3 folders."""
        root = Path(root)
        if any(root.rglob("*.SAFE.zip")):
            return cls.from_directory(root, labels_csv, **kw)
        return cls.from_xview3_directory(root, labels_csv, **kw)

    @classmethod
    def synthetic(
        cls,
        n_scenes: int = 2,
        scene_size: int = 1600,
        chip_size: int = 800,
        grid: int = 16,
        vessels_per_scene: int = 6,
        seed: int = 0,
        **kw: Any,
    ) -> "XView3Dataset":
        """Build an in-memory synthetic dataset (no rasterio, no real data).

        Plants ``vessels_per_scene`` bright targets at known pixel coordinates in
        each fake VV/VH scene and emits a matching label list, so the full
        loader -> heatmap -> scorer path is exercisable in CI.
        """
        rng = np.random.default_rng(seed)
        scenes, scene_ids, labels_by_scene = [], [], {}
        for s in range(n_scenes):
            arr = rng.standard_normal((2, scene_size, scene_size)).astype(np.float32)
            sid = f"SYN_SCENE_{s:03d}"
            labs: list[dict[str, Any]] = []
            for _ in range(vessels_per_scene):
                r = int(rng.integers(0, scene_size))
                c = int(rng.integers(0, scene_size))
                # plant a bright 3x3 blob so a future real correlation test has signal
                rr = slice(max(r - 1, 0), min(r + 2, scene_size))
                cc = slice(max(c - 1, 0), min(c + 2, scene_size))
                arr[:, rr, cc] += 8.0
                labs.append({
                    "scene_id": sid,
                    "detect_scene_row": r,
                    "detect_scene_column": c,
                    "is_vessel": True,
                    "is_fishing": bool(rng.integers(0, 2)),
                    "vessel_length_m": float(rng.uniform(20.0, 200.0)),
                    "confidence": "HIGH",
                    "distance_from_shore_km": float(rng.uniform(2.0, 50.0)),
                })
            scenes.append(_ArrayScene(arr))
            scene_ids.append(sid)
            labels_by_scene[sid] = labs
        return cls(scenes, scene_ids, labels_by_scene,
                   chip_size=chip_size, grid=grid, **kw)

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.index)

    def chip_labels(self, ref: ChipRef) -> list[dict[str, Any]]:
        """Labels whose scene-pixel centre falls inside the chip ``ref``."""
        out = []
        for lab in self.labels_by_scene.get(ref.scene_id, []):
            if scene_to_cell(
                float(lab["detect_scene_row"]), float(lab["detect_scene_column"]),
                ref.row0, ref.col0, self.chip_size, self.grid,
            ) is not None:
                out.append(lab)
        return out

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, Any]]:
        ref = self.index[i]
        scene = self.scenes[ref.scene_idx]
        chip = scene.read_chip(ref.row0, ref.col0, self.chip_size,
                               standardize=self.standardize)
        labels = self.chip_labels(ref)
        heatmap = labels_to_heatmap(labels, ref.row0, ref.col0,
                                    self.chip_size, self.grid)
        target = {
            "heatmap": torch.from_numpy(heatmap),
            "scene_id": ref.scene_id,
            "row0": ref.row0,
            "col0": ref.col0,
            "chip_size": self.chip_size,
            "grid": self.grid,
            "labels": labels,
        }
        return torch.from_numpy(chip), target


def _find_xview3_scene_dirs(root: Path) -> list[Path]:
    """Subdirectories of ``root`` that look like xView3 scenes (hold a VV GeoTIFF)."""
    import os

    def has_vv(d: Path) -> bool:
        try:
            return any(f.lower().endswith((".tif", ".tiff")) and "vv" in f.lower()
                       for f in os.listdir(d))
        except OSError:
            return False

    direct = sorted(p for p in root.iterdir() if p.is_dir() and has_vv(p))
    if direct:
        return direct
    # fall back to one level of nesting (e.g. root/validation/<scene_id>/)
    return sorted(p for p in root.rglob("*") if p.is_dir() and has_vv(p))


def _group_labels(df: Any) -> dict[str, list[dict[str, Any]]]:
    """Group a label DataFrame into ``{scene_id: [row-dict, ...]}``."""
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in df.to_dict(orient="records"):
        out.setdefault(str(rec["scene_id"]), []).append(rec)
    return out


def xview3_collate(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Collate into ``(images (B,2,H,W), heatmaps (B,1,g,g), targets list)``."""
    images = torch.stack([b[0] for b in batch], dim=0)
    heatmaps = torch.stack([b[1]["heatmap"] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, heatmaps, targets


__all__ = [
    "XView3Dataset",
    "ChipRef",
    "xview3_collate",
    "cell_size_px",
    "scene_to_cell",
    "cell_to_scene",
    "labels_to_heatmap",
]
