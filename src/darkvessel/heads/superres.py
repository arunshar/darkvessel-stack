"""Feature-map super-resolution head (x2) for backbone feature maps.

A small residual CNN that upsamples a ``(B, D, g, g)`` backbone feature map to
``(B, D, 2g, 2g)``. It is trained to recover the high-frequency structure that
average-pooling removes when the high-resolution map is downsampled.

The data here is SYNTHETIC and is NOT a real EO super-resolution benchmark
(no WorldStrat / SEN2VENUS / PROBA-V pairs). It is a controlled probe: we build
HR feature maps whose high-frequency detail is a DETERMINISTIC, nonlinear
function of the low-frequency content, so the detail is genuinely recoverable
from the LR input rather than random aliasing noise. The reported PSNR shows the
head can learn that mapping; it is not a benchmark number. We also report the
honest bilinear-upsampling baseline PSNR so any gain is measured against it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    """3x3 conv residual block (pre-activation), channel-preserving."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(x))
        h = self.conv2(F.relu(h))
        return x + h


class SuperResHead(nn.Module):
    """x2 feature-map super-resolution head.

    Operates on feature MAPS, not tokens. A bilinear upsample provides a base
    estimate; a small residual CNN (with a learned PixelShuffle upsampler)
    predicts the high-frequency residual that bilinear cannot recover, so the
    network only has to learn the detail signal on top of the smooth baseline.
    """

    def __init__(self, embed_dim: int = 64, grid: int = 8, hidden: int = 48, n_blocks: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        self.head_in = nn.Conv2d(embed_dim, hidden, 3, padding=1)
        self.body = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_blocks)])
        # learned x2 upsampler: conv to 4*hidden then PixelShuffle(2)
        self.up = nn.Sequential(
            nn.Conv2d(hidden, 4 * hidden, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.head_out = nn.Conv2d(hidden, embed_dim, 3, padding=1)

    def forward(self, lr_map: torch.Tensor) -> torch.Tensor:
        """Upsample ``(B, D, g, g)`` -> ``(B, D, 2g, 2g)``.

        The output is the bilinear base plus a learned high-frequency residual.
        """
        base = F.interpolate(lr_map, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.head_in(lr_map)
        h = self.body(h)
        h = self.up(h)              # (B, hidden, 2g, 2g)
        residual = self.head_out(h)  # (B, D, 2g, 2g)
        return base + residual


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio (dB) over a batch, peak fixed to the data range.

    The synthetic HR maps are constructed in roughly the unit range, so we use a
    fixed peak of 1.0; this makes PSNR comparable across the head and the
    bilinear baseline on the same targets.
    """
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def make_synthetic(
    n: int,
    embed_dim: int = 64,
    grid: int = 8,
    seed: int = 0,
    detail_strength: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (LR, HR) feature-map super-resolution pairs.

    Structure
    ---------
    1. Draw a smooth low-frequency HR base ``low`` by upsampling a random
       ``(g, g)`` field to ``(2g, 2g)`` with bilinear interpolation. This is the
       content that survives 2x average pooling.
    2. Add a DETERMINISTIC high-frequency detail field that is a fixed nonlinear
       function of ``low`` modulated by a fixed 2x2 checkerboard phase:
       ``detail = detail_strength * checker * sin(pi * low) * cos(2*pi * low_shift)``.
       Because ``detail`` is a function of ``low`` (not of fresh noise), it is
       recoverable from the LR input; but it lives at the sub-pixel scale that
       2x average pooling destroys, so bilinear upsampling cannot reproduce it.
    3. ``HR = low + detail`` (shape ``(n, D, 2g, 2g)``), and the input
       ``LR = F.avg_pool2d(HR, 2)`` (shape ``(n, D, g, g)``).

    Returns ``(LR, HR)`` = ``(inputs, target)``.
    """
    gen = torch.Generator().manual_seed(seed)
    g2 = 2 * grid

    # 1. smooth low-frequency base, shared structure across channels with per-channel scale
    seed_field = torch.randn(n, embed_dim, grid, grid, generator=gen)
    low = F.interpolate(seed_field, scale_factor=2, mode="bilinear", align_corners=False)

    # 2. deterministic high-frequency detail as a fixed function of `low`
    #    a fixed checkerboard phase puts the detail at the sub-pixel (HR-only) scale
    ys = torch.arange(g2).view(1, 1, g2, 1)
    xs = torch.arange(g2).view(1, 1, 1, g2)
    checker = ((ys + xs) % 2).float() * 2.0 - 1.0          # +/-1 checkerboard, (1,1,2g,2g)
    low_shift = torch.roll(low, shifts=(1, 1), dims=(2, 3))
    detail = detail_strength * checker * torch.sin(math.pi * low) * torch.cos(2.0 * math.pi * low_shift)

    hr = low + detail                                      # (n, D, 2g, 2g)
    lr = F.avg_pool2d(hr, 2)                               # (n, D, g, g)
    return lr, hr


def train_eval(steps: int = 400, seed: int = 0, n_train: int = 64, n_eval: int = 64,
               lr_rate: float = 3e-3, **kw) -> dict:
    """Train the SuperResHead and evaluate PSNR on a FRESH synthetic set.

    Trains with MSE(forward(LR), HR) using Adam, then reports the head PSNR and
    the honest bilinear-upsampling baseline PSNR on held-out data. Returns
    ``{psnr, psnr_bilinear, psnr_start, psnr_end}`` (all dB floats).
    """
    torch.manual_seed(seed)
    embed_dim = kw.pop("embed_dim", 64)
    grid = kw.pop("grid", 8)

    lr_tr, hr_tr = make_synthetic(n_train, embed_dim=embed_dim, grid=grid, seed=seed)
    lr_ev, hr_ev = make_synthetic(n_eval, embed_dim=embed_dim, grid=grid, seed=seed + 1)

    head = SuperResHead(embed_dim=embed_dim, grid=grid)
    opt = torch.optim.Adam(head.parameters(), lr=lr_rate)

    # PSNR before any training (eval set)
    head.eval()
    with torch.no_grad():
        psnr_start = _psnr(head(lr_ev), hr_ev)

    head.train()
    for _ in range(steps):
        opt.zero_grad()
        pred = head(lr_tr)
        loss = F.mse_loss(pred, hr_tr)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred_ev = head(lr_ev)
        psnr_end = _psnr(pred_ev, hr_ev)
        # honest baseline: plain bilinear upsample of the LR input
        bilinear = F.interpolate(lr_ev, scale_factor=2, mode="bilinear", align_corners=False)
        psnr_bilinear = _psnr(bilinear, hr_ev)

    return {
        "psnr": psnr_end,
        "psnr_bilinear": psnr_bilinear,
        "psnr_start": psnr_start,
        "psnr_end": psnr_end,
    }


__all__ = ["SuperResHead", "make_synthetic", "train_eval"]
