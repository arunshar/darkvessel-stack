"""Bi-temporal change-detection head over backbone tokens.

Given two co-registered observations of the same scene (time ``a`` and time
``b``), each encoded by a GeoBackbone into ``(B, N, D)`` tokens, this head emits
a per-cell change logit ``(B, 1, grid, grid)`` flagging which patches changed
between the two acquisitions (the LEVIR-CD / OSCD style task).

The data here is SYNTHETIC and is NOT a real EO change-detection benchmark
(LEVIR-CD, OSCD, S2Looking, ...). ``make_synthetic`` plants a genuine,
non-trivial task signal: every cell draws a shared random base feature vector
present in BOTH ``a`` and ``b``; "changed" cells additionally have a single
fixed, seeded "change signature" direction added into ``tokens_b`` (plus noise),
while unchanged cells differ between ``a`` and ``b`` only by small i.i.d. noise.

Because unchanged cells also carry a non-zero ``a->b`` difference (noise), the
raw difference magnitude ``|a-b|`` is an unreliable detector: a noisy unchanged
cell can have a larger ``|a-b|`` than a faint change. The signal the head must
LEARN is the *direction* of the difference in D-space, i.e. its alignment with
the (unknown to the head) change signature, separated from the isotropic noise
floor. The head is given only ``(tokens_a, tokens_b)``; the signature and mask
are never fed in, so training is not trivial.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from darkvessel.heads.common import grid_size, map_to_tokens, tokens_to_map


class ChangeHead(nn.Module):
    """Siamese bi-temporal change detector: ``(tokens_a, tokens_b) -> (B,1,g,g)`` logits.

    The two timestamps are embedded by a shared per-token projection (the
    Siamese trunk). We then form a small set of interaction features per cell,
    absolute difference, signed difference, and the projected pair, and fuse
    them with a 3x3 conv stem so a cell decision can borrow a little spatial
    context. The learnable projection is what lets the head pick out the change
    *direction* in D-space rather than relying on raw difference magnitude.
    """

    def __init__(self, embed_dim: int = 64, grid: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        # shared (Siamese) per-token embedding applied to each timestamp
        self.embed = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # spatial fusion over [|d|, d, e_a, e_b] stacked on the channel axis
        fuse_in = 4 * hidden
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> torch.Tensor:
        """Return per-cell change logits ``(B, 1, grid, grid)``.

        Parameters
        ----------
        tokens_a, tokens_b: ``(B, N, D)`` backbone tokens for the two timestamps,
            with ``N == grid * grid`` and ``D == embed_dim``.
        """
        if tokens_a.shape != tokens_b.shape:
            raise ValueError("tokens_a and tokens_b must share shape")
        g = grid_size(tokens_a.shape[1])
        e_a = self.embed(tokens_a)                 # (B, N, hidden)
        e_b = self.embed(tokens_b)                 # (B, N, hidden)
        diff = e_b - e_a                           # signed difference (direction-aware)
        feats = torch.cat([diff.abs(), diff, e_a, e_b], dim=-1)  # (B, N, 4*hidden)
        fmap = tokens_to_map(feats, g)             # (B, 4*hidden, g, g)
        fused = self.fuse(fmap)                    # (B, hidden, g, g)
        return self.classifier(fused)              # (B, 1, g, g)


def make_synthetic(
    n: int,
    embed_dim: int = 64,
    grid: int = 8,
    seed: int = 0,
    change_prob: float = 0.3,
    signal: float = 2.0,
    noise: float = 0.7,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Synthetic bi-temporal tokens with a planted directional change signal.

    Structure (all draws from a single seeded ``torch.Generator``):

    * A fixed unit ``signature`` direction in R^D is drawn ONCE (shared across the
      whole dataset, unknown to the head).
    * For each sample and each of the ``grid*grid`` cells, a shared ``base`` vector
      is drawn. ``tokens_a = base + noise * eps_a`` for every cell.
    * A Bernoulli(``change_prob``) ``mask`` marks changed cells. For changed cells
      ``tokens_b = base + signal * signature + noise * eps_b``; for unchanged cells
      ``tokens_b = base + noise * eps_b``.

    The ``a->b`` difference is ``signal * signature * mask + noise * (eps_b - eps_a)``.
    Detecting change therefore requires reading the *direction* (alignment with
    ``signature``), since the isotropic noise term gives unchanged cells a non-zero,
    sometimes large, ``|a-b|``.

    Returns ``((tokens_a, tokens_b), target)`` where ``tokens_*`` are ``(n, N, D)``
    and ``target`` is the binary changed-cell mask ``(n, 1, grid, grid)`` (float).
    """
    g = grid
    n_tokens = g * g
    gen = torch.Generator().manual_seed(seed)
    # The change signature defines the TASK and must match across train/eval
    # splits (which use different sample seeds); draw it from a fixed task generator.
    sig_gen = torch.Generator().manual_seed(20240601)

    # one shared change direction in D-space, unit-normalised
    signature = torch.randn(embed_dim, generator=sig_gen)
    signature = signature / signature.norm().clamp_min(1e-8)

    base = torch.randn(n, n_tokens, embed_dim, generator=gen)
    eps_a = torch.randn(n, n_tokens, embed_dim, generator=gen)
    eps_b = torch.randn(n, n_tokens, embed_dim, generator=gen)

    mask = (torch.rand(n, n_tokens, 1, generator=gen) < change_prob).float()

    tokens_a = base + noise * eps_a
    tokens_b = base + signal * mask * signature + noise * eps_b

    target = mask.reshape(n, 1, g, g)
    return (tokens_a, tokens_b), target


def _f1_from_logits(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    """Binary F1 of ``sigmoid(logits) > thresh`` against ``target`` (both (B,1,g,g))."""
    pred = (torch.sigmoid(logits) > thresh).float()
    tgt = (target > 0.5).float()
    tp = (pred * tgt).sum()
    fp = (pred * (1.0 - tgt)).sum()
    fn = ((1.0 - pred) * tgt).sum()
    denom = 2.0 * tp + fp + fn
    if denom.item() == 0.0:
        return 1.0  # no positives anywhere and none predicted -> perfect
    return (2.0 * tp / denom).item()


def train_eval(
    steps: int = 250,
    seed: int = 0,
    n_train: int = 64,
    n_eval: int = 64,
    embed_dim: int = 64,
    grid: int = 8,
    hidden: int = 64,
    lr: float = 3e-3,
    **kw: float,
) -> dict:
    """Train ``ChangeHead`` on planted bi-temporal data; report F1 start vs end.

    Trains with ``F.binary_cross_entropy_with_logits`` (Adam) on a fixed seeded
    training set and evaluates F1 on a FRESH synthetic set with a different seed.
    ``f1_start`` is measured on the eval set before any optimisation, ``f1_end``
    after training. Everything is seeded for reproducibility.
    """
    torch.manual_seed(seed)

    (xa, xb), ytr = make_synthetic(n_train, embed_dim, grid, seed=seed, **kw)
    (ea, eb), yev = make_synthetic(n_eval, embed_dim, grid, seed=seed + 1000, **kw)

    head = ChangeHead(embed_dim=embed_dim, grid=grid, hidden=hidden)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    head.eval()
    with torch.no_grad():
        f1_start = _f1_from_logits(head(ea, eb), yev)

    head.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = head(xa, xb)
        loss = F.binary_cross_entropy_with_logits(logits, ytr)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        f1_end = _f1_from_logits(head(ea, eb), yev)

    return {
        "f1": f1_end,
        "f1_start": f1_start,
        "f1_end": f1_end,
        "loss_end": float(loss.detach()),
    }


__all__ = ["ChangeHead", "make_synthetic", "train_eval"]
