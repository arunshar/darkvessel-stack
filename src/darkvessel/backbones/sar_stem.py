"""Learned SAR detector: a from-scratch conv stem feeding the DetectionHead.

The repo's :class:`GeoBackbone` wraps optical-EO foundation models (Prithvi,
Clay, SatMAE++, ...), which are 6-band RGB / multispectral and the wrong domain
for 2-channel Sentinel-1 SAR amplitude. For xView3 vessel detection we instead
learn a small SAR-native conv stem that maps a ``(B, 2, H, W)`` VV/VH chip to a
``(B, embed_dim, grid, grid)`` feature map, then reuse the existing
:class:`darkvessel.heads.detection.DetectionHead` (tokens -> per-cell objectness
logits) unchanged on top.

Design note: vessels are bright POINT scatterers in SAR, so the stem ends in an
adaptive MAX pool to the objectness grid (it keeps the peak response per cell,
like the CFAR baseline but over LEARNED features that can suppress sea/land
clutter), rather than an average pool that would dilute a point target. Trained
from scratch, this is a real learned model to compare against the unlearned CFAR
baseline on the official xView3 scorer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from darkvessel.heads.common import map_to_tokens
from darkvessel.heads.detection import DetectionHead


def _block(cin: int, cout: int) -> nn.Sequential:
    """3x3 stride-2 conv + GroupNorm + GELU (halves spatial resolution)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
        nn.GELU(),
    )


class SARStem(nn.Module):
    """``(B, 2, H, W)`` VV/VH SAR chip -> ``(B, embed_dim, grid, grid)`` feature map.

    Four stride-2 conv blocks downsample by 16x, then an adaptive max pool lands
    the feature map on the exact ``grid x grid`` objectness lattice (robust to the
    chip_size / grid choice, and never upsamples for the xView3 defaults where
    chip_size / 16 >= grid).
    """

    def __init__(self, in_chans: int = 2, embed_dim: int = 64, grid: int = 16) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid = grid
        self.encoder = nn.Sequential(
            _block(in_chans, 24),    # H/2
            _block(24, 48),          # H/4
            _block(48, 96),          # H/8
            _block(96, embed_dim),   # H/16
        )
        self.pool = nn.AdaptiveMaxPool2d((grid, grid))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_chans:
            raise ValueError(
                f"expected input of shape (B, {self.in_chans}, H, W), got {tuple(x.shape)}"
            )
        return self.pool(self.encoder(x))    # (B, embed_dim, grid, grid)


class SARDetector(nn.Module):
    """``SARStem`` + ``DetectionHead`` -> ``(B, 1, grid, grid)`` objectness logits.

    Matches the ``run_xview3`` predict_fn contract: images ``(B, 2, H, W)`` ->
    logits ``(B, 1, grid, grid)``. The DetectionHead is reused unchanged (tokens
    -> logits); the stem supplies the learned image->token features it needs.
    """

    def __init__(self, in_chans: int = 2, embed_dim: int = 64, grid: int = 16,
                 hidden: int = 64) -> None:
        super().__init__()
        self.grid = grid
        self.stem = SARStem(in_chans=in_chans, embed_dim=embed_dim, grid=grid)
        self.head = DetectionHead(embed_dim=embed_dim, grid=grid, hidden=hidden)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        fmap = self.stem(images)             # (B, embed_dim, grid, grid)
        tokens = map_to_tokens(fmap)         # (B, grid*grid, embed_dim)
        return self.head(tokens)             # (B, 1, grid, grid)


__all__ = ["SARStem", "SARDetector"]
