"""Per-cell objectness detection head over backbone tokens.

Treats object detection as dense per-grid-cell objectness classification: every
patch token is scored for whether it contains an object (a small vessel /
target). The head consumes ``(B, N, D)`` backbone tokens, reshapes them to a
``(B, D, grid, grid)`` feature map via the shared contract, and runs a small
convolutional stack that produces one objectness logit per cell.

The data here is SYNTHETIC, not a real EO detection benchmark (xView3,
SpaceNet, fMoW, ...). ``make_synthetic`` plants a learnable "object signature"
direction in D-space at a handful of cells per sample, buried in Gaussian
noise; the head must learn to project tokens onto that latent direction to
separate object cells from background. The reported F1 demonstrates the head
can learn the task on structured features; it is not a benchmark number.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from darkvessel.heads.common import grid_size, tokens_to_map


class DetectionHead(nn.Module):
    """Dense objectness head: ``(B, N, D)`` tokens -> ``(B, 1, grid, grid)`` logits."""

    def __init__(self, embed_dim: int = 64, grid: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        # 3x3 conv lets a cell borrow local context; 1x1 conv reduces to a logit.
        # padding=1 keeps the (grid, grid) resolution so output is one logit/cell.
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Score each grid cell for objectness.

        Parameters
        ----------
        tokens: ``(B, N, D)`` backbone tokens with ``N == grid * grid``.

        Returns
        -------
        ``(B, 1, grid, grid)`` objectness LOGITS (pre-sigmoid).
        """
        grid = grid_size(tokens.shape[1])
        fmap = tokens_to_map(tokens, grid)          # (B, D, grid, grid)
        return self.net(fmap)                        # (B, 1, grid, grid)


def make_synthetic(
    n: int,
    embed_dim: int = 64,
    grid: int = 8,
    seed: int = 0,
    alpha: float = 4.5,
    noise: float = 1.0,
    max_objects: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build synthetic tokens with planted objects.

    A single seeded unit "object signature" vector ``s`` in R^D is fixed across
    the whole dataset. For each of the ``n`` samples we choose ``K`` object cells
    uniformly at random (``K`` drawn uniformly in ``1..max_objects``). Object
    cells get feature ``alpha * s + noise * eps``; background cells get
    independent noise ``noise * eps``. The signature direction is shared, so the
    head must LEARN to project onto ``s``; it is never given the cell labels.

    Returns
    -------
    inputs: ``(n, grid*grid, embed_dim)`` tokens.
    target: ``(n, 1, grid, grid)`` binary objectness heatmap (1 at object cells).
    """
    g = grid
    n_cells = g * g
    gen = torch.Generator().manual_seed(seed)
    # The object signature defines the TASK and must be identical across the
    # train and eval splits (which use different sample seeds); draw it from a
    # fixed task generator, not the per-split `seed`.
    sig_gen = torch.Generator().manual_seed(20240601)

    # fixed shared object signature (unit vector in D-space)
    s = torch.randn(embed_dim, generator=sig_gen)
    s = s / s.norm().clamp_min(1e-8)

    inputs = noise * torch.randn(n, n_cells, embed_dim, generator=gen)
    target = torch.zeros(n, 1, g, g)

    for i in range(n):
        k = int(torch.randint(1, max_objects + 1, (1,), generator=gen).item())
        # choose k distinct cells without replacement
        perm = torch.randperm(n_cells, generator=gen)[:k]
        inputs[i, perm] += alpha * s
        target[i, 0].view(-1)[perm] = 1.0

    return inputs, target


def train_eval(
    steps: int = 400,
    seed: int = 0,
    n_train: int = 256,
    n_eval: int = 256,
    embed_dim: int = 64,
    grid: int = 8,
    lr: float = 5e-3,
    **kw,
) -> dict:
    """Train the detection head, then evaluate F1 on a FRESH synthetic set.

    Trains with ``binary_cross_entropy_with_logits`` (Adam) on a planted-object
    dataset and reports cell-wise F1 at threshold 0.5 before and after training.
    Everything is seeded for reproducibility.

    Returns a dict with ``f1``, ``f1_start``, ``f1_end``.
    """
    torch.manual_seed(seed)

    x_train, y_train = make_synthetic(n_train, embed_dim, grid, seed=seed, **kw)
    # fresh, independently seeded evaluation set
    x_eval, y_eval = make_synthetic(n_eval, embed_dim, grid, seed=seed + 9999, **kw)

    head = DetectionHead(embed_dim=embed_dim, grid=grid)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    # objects are sparse (a few cells in grid*grid), so positives are rare;
    # weight the positive class in BCE to counter that imbalance.
    pos = y_train.sum()
    pos_weight = (y_train.numel() - pos) / pos.clamp_min(1.0)

    def f1_at_half(logits: torch.Tensor, target: torch.Tensor) -> float:
        pred = (torch.sigmoid(logits) >= 0.5).float()
        tp = (pred * target).sum()
        fp = (pred * (1.0 - target)).sum()
        fn = ((1.0 - pred) * target).sum()
        denom = 2.0 * tp + fp + fn
        if denom.item() == 0.0:
            return 1.0  # no positives anywhere and none predicted
        return (2.0 * tp / denom).item()

    head.eval()
    with torch.no_grad():
        f1_start = f1_at_half(head(x_eval), y_eval)

    head.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = head(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        f1_end = f1_at_half(head(x_eval), y_eval)

    return {"f1": f1_end, "f1_start": f1_start, "f1_end": f1_end}


__all__ = ["DetectionHead", "make_synthetic", "train_eval"]
