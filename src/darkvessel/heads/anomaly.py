"""Pi-DPM anomaly head: physics-informed diffusion for dark-vessel reasoning.

Implements the inference-time scoring head of

    Sharma et al. (2025). "Towards Physics-informed Diffusion for Anomaly
    Detection in Trajectories: A Summary of Results." ACM SIGSPATIAL
    GeoAnomalies Workshop.

The head takes (1) backbone tokens describing the scene where the AIS gap
occurred and (2) the AIS track segment surrounding the gap, and produces
(a) a reconstructed trajectory across the gap from a conditional diffusion
denoiser and (b) a per-track spoofing / denial logit derived from the
reconstruction residual and a kinematic-smoothness residual.

This is a genuine conditional diffusion head (not the previous Linear MLP): the
denoiser, the Gaussian diffusion, and the S-KBM physics envelope are the
vendored Pi-DPM package under ``darkvessel.pidpm`` (the canonical model, with its
training / eval harness and metric S-KBM envelope, lives in the pi-grpo repo).
The AIS columns are on heterogeneous scales (deg, deg, kn, deg), so the score's
physics term here is the scale-free kinematic-smoothness residual of the
reconstructed lat/lon; the full metric S-KBM envelope is applied upstream where
tracks are projected to local metres.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from darkvessel.pidpm.config import PiDPMConfig
from darkvessel.pidpm.diffusion import GaussianDiffusion
from darkvessel.pidpm.model import TrajectoryDenoiser


class PiDPMAnomalyHead(nn.Module):
    """Conditional Pi-DPM reconstruction + scoring head over (scene tokens, AIS gap)."""

    def __init__(
        self,
        embed_dim: int = 1280,
        ais_dim: int = 5,
        hidden: int = 512,
        dropout: float = 0.1,
        max_len: int = 96,
        recon_noise_t: int = 40,
    ) -> None:
        super().__init__()
        self.ais_dim = ais_dim
        self.max_len = max_len
        self.cfg = PiDPMConfig(
            seq_len=max_len, in_dim=ais_dim, cond_dim=hidden,
            d_model=hidden // 2, n_heads=4, n_layers=3, dropout=dropout,
            eval_noise_t=recon_noise_t,
            # the metric S-KBM envelope assumes projected metres; AIS columns here
            # are heterogeneous (deg/kn), so train the denoiser on eps-MSE only and
            # score with the scale-free smoothness residual below
            physics_weight=0.0,
        )
        self.scene_proj = nn.Linear(embed_dim, hidden)
        self.denoiser = TrajectoryDenoiser(self.cfg)
        self.diffusion = GaussianDiffusion(self.denoiser, self.cfg)
        # learned mapping from (reconstruction residual, smoothness residual) to a logit
        self.score_head = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.recon_noise_t = recon_noise_t

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _normalise(seg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = seg.mean(dim=1, keepdim=True)
        sd = seg.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (seg - mu) / sd, mu, sd

    def _pad_to_work(self, seg: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Crop/pad the time axis to max_len (pad by repeating the last frame)."""
        t = seg.shape[1]
        if t == self.max_len:
            return seg, t
        if t > self.max_len:
            return seg[:, : self.max_len], self.max_len
        pad = seg[:, -1:].expand(-1, self.max_len - t, -1)
        return torch.cat([seg, pad], dim=1), t

    @staticmethod
    def _smoothness(latlon: torch.Tensor) -> torch.Tensor:
        """Mean squared jerk of (lat, lon), scale-free -> (B,)."""
        if latlon.shape[1] < 4:
            return latlon.new_zeros(latlon.shape[0])
        vel = torch.diff(latlon, dim=1)
        acc = torch.diff(vel, dim=1)
        jerk = torch.diff(acc, dim=1)
        return jerk.pow(2).mean(dim=(1, 2))

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        scene_tokens: torch.Tensor,
        ais_segment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score and reconstruct an AIS gap.

        Parameters
        ----------
        scene_tokens: ``(B, N, embed_dim)`` from a GeoBackbone forward pass.
        ais_segment: ``(B, T, 5)`` AIS segment around the gap, columns
            ``(t, lat, lon, sog, cog)``.

        Returns dict with ``score`` ``(B, 1)`` spoofing / denial logit and
        ``reconstruction`` ``(B, T, 5)`` reconstructed AIS segment.
        """
        b, t_in, _ = ais_segment.shape
        ais_n, mu, sd = self._normalise(ais_segment)
        cond = self.scene_proj(scene_tokens.mean(dim=1))           # (B, hidden)

        x_pad, _ = self._pad_to_work(ais_n)                        # (B, L, 5)
        # one differentiable diffusion denoise step at a fixed noise level
        t = torch.full((b,), self.recon_noise_t, device=ais_segment.device, dtype=torch.long)
        noise = torch.randn_like(x_pad)
        x_noised = self.diffusion.q_sample(x_pad, t, noise)
        eps = self.denoiser(x_noised, t, cond)
        x0_hat = self.diffusion.predict_x0(x_noised, t, eps)        # (B, L, 5)
        recon_n = x0_hat[:, :t_in]                                  # back to input length

        recon_resid = (ais_n - recon_n).pow(2).mean(dim=(1, 2))     # (B,)
        smooth_resid = self._smoothness(recon_n[..., [2, 1]])       # (lon, lat) jerk
        score = self.score_head(torch.stack([recon_resid, smooth_resid], dim=-1))  # (B, 1)

        reconstruction = recon_n * sd + mu                          # de-normalise to AIS units
        return {"score": score, "reconstruction": reconstruction}

    def diffusion_loss(self, scene_tokens: torch.Tensor, ais_segment: torch.Tensor) -> torch.Tensor:
        """Training objective: conditional diffusion eps-MSE + physics regulariser."""
        ais_n, _, _ = self._normalise(ais_segment)
        x_pad, _ = self._pad_to_work(ais_n)
        cond = self.scene_proj(scene_tokens.mean(dim=1))
        return self.diffusion.loss(x_pad, cond)["loss"]
