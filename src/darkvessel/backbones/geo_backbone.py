"""Unified interface over open geospatial foundation models.

Loads any of Prithvi-2 (NASA / IBM), Clay v1, SatMAE++, DOFA, SatlasNet, or
RemoteCLIP via a single ``from_pretrained`` call and exposes a common
``forward(x) -> tokens`` API so downstream heads do not need to know which
backbone is in use.

The list of supported backbones is exposed via ``GeoBackbone.list_supported()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

BackboneName = Literal[
    "prithvi-2",
    "clay-v1",
    "satmae-pp",
    "dofa",
    "satlasnet",
    "remoteclip",
]

_BACKBONE_CARDS: dict[str, dict] = {
    "prithvi-2": {
        "hf_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "patch": 16,
        "embed_dim": 1280,
        "bands": ["blue", "green", "red", "nir_narrow", "swir1", "swir2"],
        "license": "Apache-2.0",
    },
    "clay-v1": {
        "hf_id": "made-with-clay/Clay",
        "patch": 16,
        "embed_dim": 1024,
        "bands": "sensor-conditioned",
        "license": "MIT",
    },
    "satmae-pp": {
        "hf_id": "yezhen/SatMAE-PlusPlus",
        "patch": 16,
        "embed_dim": 1024,
        "bands": "fmow-sentinel-13",
        "license": "CC-BY-NC",
    },
    "dofa": {
        "hf_id": "torchgeo/DOFA-large",
        "patch": 16,
        "embed_dim": 1024,
        "bands": "wavelength-conditioned",
        "license": "MIT",
    },
    "satlasnet": {
        "hf_id": "allenai/satlas-pretrain",
        "patch": 32,
        "embed_dim": 768,
        "bands": "sentinel2-rgb-nir",
        "license": "Apache-2.0",
    },
    "remoteclip": {
        "hf_id": "chendelong/RemoteCLIP",
        "patch": 16,
        "embed_dim": 768,
        "bands": "rgb-only",
        "license": "Apache-2.0",
    },
}


@dataclass
class BackboneSpec:
    name: str
    hf_id: str
    patch: int
    embed_dim: int
    bands: object
    license: str


class GeoBackbone(nn.Module):
    """Common adapter exposing a uniform ``forward`` over six EO backbones.

    The constructor takes the name of the backbone; ``from_pretrained`` downloads
    weights from the Hugging Face hub. For unit tests and CPU smoke we expose
    a ``stub`` mode that returns a zero-initialized projection so the rest of
    the pipeline can be exercised without downloading hundreds of megabytes.
    """

    def __init__(self, name: BackboneName, in_chans: int = 6, image_size: int = 224, stub: bool = False) -> None:
        super().__init__()
        if name not in _BACKBONE_CARDS:
            raise ValueError(f"Unknown backbone {name!r}; supported: {list(_BACKBONE_CARDS)}")
        card = _BACKBONE_CARDS[name]
        self.spec = BackboneSpec(name=name, **card)
        self.in_chans = in_chans
        self.image_size = image_size
        self.stub = stub
        if stub:
            self._stub_proj = nn.Conv2d(in_chans, self.spec.embed_dim, kernel_size=self.spec.patch, stride=self.spec.patch)
        else:
            self._module = self._load_hf_backbone()

    def _load_hf_backbone(self) -> nn.Module:
        from transformers import AutoModel  # local import keeps test path light
        return AutoModel.from_pretrained(self.spec.hf_id, trust_remote_code=True)

    @classmethod
    def from_pretrained(cls, name: BackboneName, **kwargs) -> "GeoBackbone":
        return cls(name=name, stub=False, **kwargs)

    @classmethod
    def list_supported(cls) -> list[str]:
        return list(_BACKBONE_CARDS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-patch tokens of shape ``(B, N, embed_dim)``.

        ``x`` is expected to be ``(B, C, H, W)`` with ``C == self.in_chans``.
        """
        if x.dim() != 4 or x.shape[1] != self.in_chans:
            raise ValueError(
                f"expected input of shape (B, {self.in_chans}, H, W), got {tuple(x.shape)}"
            )
        if self.stub:
            tokens = self._stub_proj(x)
            return tokens.flatten(2).transpose(1, 2)
        return self._module(pixel_values=x).last_hidden_state
