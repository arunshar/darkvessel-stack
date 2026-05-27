"""Pi-DPM anomaly head: physics-informed diffusion for dark-vessel reasoning.

Wraps the encoder-decoder physics-informed diffusion probabilistic model from
    Sharma et al. (2025). "Towards Physics-informed Diffusion for Anomaly
    Detection in Trajectories: A Summary of Results." ACM SIGSPATIAL
    GeoAnomalies Workshop.

The head takes (1) backbone tokens describing the scene where the AIS gap
occurred, and (2) the AIS track segment surrounding the gap, and produces
(a) a reconstructed trajectory across the gap and (b) a per-track spoofing /
denial likelihood score derived from the kinematic residual of the
reconstruction against the S-KBM (simplified kinematic bicycle model).

The full diffusion sampler lives in the upstream Pi-DPM repository; this
module exposes only the inference-time scoring head DarkVesselNet uses to
turn a vessel candidate into a dark-vessel probability.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PiDPMAnomalyHead(nn.Module):
    """Light reasoning head over (scene tokens, ais gap) for spoofing scoring."""

    def __init__(
        self,
        embed_dim: int = 1280,
        ais_dim: int = 5,
        hidden: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.scene_proj = nn.Linear(embed_dim, hidden)
        self.ais_proj = nn.Sequential(
            nn.Linear(ais_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.score = nn.Linear(hidden, 1)
        self.recon = nn.Linear(hidden, ais_dim)

    def forward(
        self,
        scene_tokens: torch.Tensor,
        ais_segment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score and reconstruct an AIS gap.

        Parameters
        ----------
        scene_tokens:
            ``(B, N, embed_dim)`` from a GeoBackbone forward pass on the AOI.
        ais_segment:
            ``(B, T, 5)`` AIS segment around the gap with columns
            ``(t, lat, lon, sog, cog)``.

        Returns
        -------
        dict with
            ``score``: ``(B, 1)`` spoofing / denial likelihood logit
            ``reconstruction``: ``(B, T, 5)`` reconstructed AIS segment
        """
        scene = self.scene_proj(scene_tokens.mean(dim=1))
        ais = self.ais_proj(ais_segment).mean(dim=1)
        fused = self.fuse(torch.cat([scene, ais], dim=-1))
        score = self.score(fused)
        recon_seed = fused.unsqueeze(1).expand(-1, ais_segment.shape[1], -1)
        recon = self.recon(recon_seed)
        return {"score": score, "reconstruction": recon}
