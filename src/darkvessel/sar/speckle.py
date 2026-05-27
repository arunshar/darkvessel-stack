"""SAR speckle filtering.

Lee filter (Lee, 1980) is the canonical first-pass speckle suppressor for
Sentinel-1 GRD imagery and the one we apply before backbone embedding. We
implement it in pure PyTorch so it stays differentiable and GPU-friendly.

The filter weight at pixel i is
    w_i = max(0, 1 - C_u^2 / C_i^2)
where C_u is the noise coefficient of variation (sqrt(1/L) for L looks) and
C_i is the local coefficient of variation in the window. The output is
    y_i = mean_i + w_i * (x_i - mean_i)
which reduces to the local mean in homogeneous regions and preserves edges
in heterogeneous regions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def lee_filter(x: torch.Tensor, window: int = 5, looks: float = 4.4) -> torch.Tensor:
    """Apply the Lee speckle filter to a SAR amplitude / sigma0 tensor.

    Parameters
    ----------
    x:
        ``(B, C, H, W)`` SAR tensor (amplitude or sigma0). Polarizations stack
        along ``C``.
    window:
        Side length of the box window; must be odd.
    looks:
        Effective number of looks (ENL). Sentinel-1 IW GRD: ENL ~ 4.4.
    """
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    if x.dim() != 4:
        raise ValueError(f"expected 4D tensor (B, C, H, W), got {x.dim()}D")
    # F.pad(mode="reflect") requires pad < min(H, W); shrink the window for tiny tiles.
    min_hw = min(x.shape[-2], x.shape[-1])
    pad = min(window // 2, max(min_hw - 1, 0))
    window = 2 * pad + 1
    kernel = torch.ones(1, 1, window, window, device=x.device, dtype=x.dtype) / (window * window)
    c = x.shape[1]
    kernel = kernel.expand(c, 1, window, window)

    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    mean = F.conv2d(x_pad, kernel, padding=0, groups=c)
    mean_sq = F.conv2d(x_pad * x_pad, kernel, padding=0, groups=c)
    var = (mean_sq - mean * mean).clamp(min=0.0)
    cv2 = var / (mean * mean + 1e-8)
    cu2 = 1.0 / looks
    w = ((cv2 - cu2) / (cv2 + 1e-8)).clamp(min=0.0, max=1.0)
    return mean + w * (x - mean)
