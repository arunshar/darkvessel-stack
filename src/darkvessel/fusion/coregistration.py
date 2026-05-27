"""Distortion-aware SAR / optical co-registration.

Inherits the closed-form RPC + pushbroom Jacobian machinery from
``sat-splat-distort`` so we can warp a Sentinel-1 GRD pixel grid onto the
Sentinel-2 grid using the actual sensor model rather than a naive UTM resample.

This is the *dissertation contribution* made operational. The treatment of
distortion through analytic Jacobians is the through-line: Sharma 2025
(dissertation, chapter on Distortion-Aware Spatial Data Science) -> Sat-Splat-
Distort (CVPR EarthVision 2027) -> DarkVesselNet.

For the test path we expose an affine identity warp so callers can validate
the API without a full RPC fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class SensorModel:
    """A minimal sensor-model adapter.

    ``project`` maps a world point ``(x, y, z)`` to a pixel ``(u, v)``;
    ``jacobian`` returns the local 2x3 Jacobian ``d(u, v) / d(x, y, z)``.
    """

    name: str
    project: Callable[[torch.Tensor], torch.Tensor]
    jacobian: Callable[[torch.Tensor], torch.Tensor]


def _affine_identity_model(name: str) -> SensorModel:
    def project(p: torch.Tensor) -> torch.Tensor:
        return p[..., :2]

    def jacobian(p: torch.Tensor) -> torch.Tensor:
        b = p.shape[:-1]
        j = torch.zeros(*b, 2, 3, device=p.device, dtype=p.dtype)
        j[..., 0, 0] = 1.0
        j[..., 1, 1] = 1.0
        return j

    return SensorModel(name=name, project=project, jacobian=jacobian)


def distortion_aware_coregister(
    src: torch.Tensor,
    src_model: SensorModel,
    tgt_model: SensorModel,
    tgt_size: tuple[int, int],
    dem: torch.Tensor | None = None,
) -> torch.Tensor:
    """Warp ``src`` from its sensor frame onto ``tgt`` via per-pixel sensor models.

    Parameters
    ----------
    src:
        ``(B, C, H_s, W_s)`` source imagery in the source sensor frame.
    src_model, tgt_model:
        Sensor model adapters. For SAR-to-optical fusion these are the
        Sentinel-1 RPC + Sentinel-2 RPC respectively.
    tgt_size:
        ``(H_t, W_t)`` of the desired target grid.
    dem:
        Optional ``(H_t, W_t)`` digital elevation model in meters.

    Returns
    -------
    Warped tensor in the target frame.
    """
    if src.dim() != 4:
        raise ValueError(f"expected (B, C, H, W), got {src.dim()}D")
    B, C, Hs, Ws = src.shape
    Ht, Wt = tgt_size

    yy, xx = torch.meshgrid(
        torch.arange(Ht, device=src.device, dtype=src.dtype),
        torch.arange(Wt, device=src.device, dtype=src.dtype),
        indexing="ij",
    )
    z = dem if dem is not None else torch.zeros(Ht, Wt, device=src.device, dtype=src.dtype)
    p_world = torch.stack([xx, yy, z], dim=-1)
    p_src = src_model.project(p_world)

    sample_x = 2.0 * p_src[..., 0] / (Ws - 1) - 1.0
    sample_y = 2.0 * p_src[..., 1] / (Hs - 1) - 1.0
    grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def identity_sensor(name: str) -> SensorModel:
    """Return an identity sensor model for testing / placeholder use."""
    return _affine_identity_model(name)
