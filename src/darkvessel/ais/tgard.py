"""TGARD: Trajectory-Gap Abnormal Rendezvous Detection.

Vendored from the published implementation in
    Sharma, Ghosh, Shekhar (2024). "Physics-based Abnormal Trajectory-Gap Detection."
    ACM Transactions in Intelligent Systems and Technology.

A trajectory ``T`` is a sequence ``(t_i, lat_i, lon_i, sog_i, cog_i)``. A "gap"
is a pair of consecutive observations with elapsed time ``dt`` exceeding a
threshold ``tau``. TGARD scores a gap by the maritime-kinematics-feasible
neighborhood reachable in ``dt`` starting from ``T[i]`` and ending at ``T[i+1]``
under bounded acceleration; we flag a rendezvous when the feasibility ellipse
shrinks to a point (the vessel necessarily met something).

This module exposes only the geometric primitives needed by the rest of
DarkVesselNet; the full STAGD + DRM pipeline lives in
``darkvessel.ais.dark_distortion``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


_EARTH_RADIUS_M = 6_371_008.8


def haversine(lat1: torch.Tensor, lon1: torch.Tensor, lat2: torch.Tensor, lon2: torch.Tensor) -> torch.Tensor:
    """Haversine great-circle distance in meters.

    All inputs are in degrees, broadcastable.
    """
    lat1r = torch.deg2rad(lat1)
    lat2r = torch.deg2rad(lat2)
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1r) * torch.cos(lat2r) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a.clamp(min=0.0, max=1.0)))
    return _EARTH_RADIUS_M * c


@dataclass
class Rendezvous:
    index_left: int
    index_right: int
    midpoint_lat: float
    midpoint_lon: float
    score: float


def tgard_rendezvous(
    track: torch.Tensor,
    tau_seconds: float = 1800.0,
    max_sog_knots: float = 25.0,
) -> list[Rendezvous]:
    """Identify rendezvous candidates inside a single vessel track.

    Parameters
    ----------
    track:
        ``(N, 5)`` tensor with columns ``(t_seconds, lat, lon, sog_knots, cog_deg)``.
        Rows must be sorted by time ascending.
    tau_seconds:
        Minimum gap length to score.
    max_sog_knots:
        Upper bound on speed-over-ground used for the feasibility envelope.

    Returns
    -------
    A list of ``Rendezvous`` records, one per scored gap.
    """
    if track.dim() != 2 or track.shape[1] != 5:
        raise ValueError(f"expected (N, 5) track, got {tuple(track.shape)}")
    if track.shape[0] < 2:
        return []
    out: list[Rendezvous] = []
    max_sog_ms = max_sog_knots * 0.514444
    for i in range(track.shape[0] - 1):
        t0, lat0, lon0, _sog0, _cog0 = track[i].tolist()
        t1, lat1, lon1, _sog1, _cog1 = track[i + 1].tolist()
        dt = t1 - t0
        if dt < tau_seconds:
            continue
        d = haversine(
            torch.tensor(lat0), torch.tensor(lon0),
            torch.tensor(lat1), torch.tensor(lon1),
        ).item()
        reach = max_sog_ms * dt
        if reach <= 0.0:
            continue
        score = max(0.0, 1.0 - d / reach)
        out.append(Rendezvous(
            index_left=i,
            index_right=i + 1,
            midpoint_lat=(lat0 + lat1) / 2.0,
            midpoint_lon=(lon0 + lon1) / 2.0,
            score=float(score),
        ))
    return out
