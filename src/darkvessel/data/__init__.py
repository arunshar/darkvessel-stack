"""Real-data loaders for darkvessel-stack (xView3 / SARFish SAR).

The chip-level :class:`XView3Dataset` and the coordinate helpers live in
``darkvessel.data.xview3`` and need only numpy + torch (rasterio is imported
lazily, real path only), so the synthetic constructor and the scorer wiring work
without rasterio. ``SARScene`` / ``to_db`` (the GRD windowed reader) DO need
rasterio, so they are exposed lazily via module ``__getattr__`` to keep
``import darkvessel.data`` rasterio-free for callers that only want ``XView3Dataset``.
"""
from darkvessel.data.xview3 import (
    XView3Dataset,
    cell_to_scene,
    labels_to_heatmap,
    scene_to_cell,
    xview3_collate,
)

_LAZY = {"SARScene": "sar_scene", "to_db": "sar_scene"}


def __getattr__(name: str):
    """Lazily import the rasterio-backed names (SARScene, to_db) on first access."""
    if name in _LAZY:
        import importlib
        module = importlib.import_module(f"darkvessel.data.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SARScene",
    "to_db",
    "XView3Dataset",
    "xview3_collate",
    "cell_to_scene",
    "scene_to_cell",
    "labels_to_heatmap",
]
