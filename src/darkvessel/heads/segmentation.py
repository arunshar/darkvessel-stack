"""Dense per-cell semantic segmentation head over backbone tokens.

Consumes backbone tokens ``(B, N, D)`` (``N == grid * grid``), reshapes them to
a ``(B, D, grid, grid)`` feature map via the shared contract, and predicts a
per-cell class map with a small conv stack. This is the land-cover / scene-class
style head a GeoBackbone would feed in the dark-vessel stack (e.g. water / land /
wake / vessel cells), kept deliberately light so it trains on CPU in seconds.

HONESTY: the data here is SYNTHETIC, NOT a real EO segmentation benchmark
(no LandCover.ai / DynamicEarthNet / SEN12MS labels). ``make_synthetic`` plants
a genuine but non-trivial signal: ``n_classes`` random prototype directions in
R^D, with each cell's class drawn from a few random spatial Gaussian "blobs"
(so labels are spatially contiguous, NOT iid), and the cell feature set to the
class prototype plus per-cell Gaussian noise. The head never sees the label; it
must learn to separate the prototype directions through the noise AND exploit
the spatial smoothness of the blobs. The reported mIoU shows the head can learn
this structured task; it is not a benchmark number.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from darkvessel.heads.common import grid_size, map_to_tokens, tokens_to_map


class SegmentationHead(nn.Module):
    """Small conv segmentation head: tokens ``(B, N, D)`` -> logits ``(B, C, g, g)``.

    The two 3x3 convolutions (padding 1) keep the spatial resolution at
    ``grid x grid`` and give the head a receptive field over neighbouring cells,
    which is what lets it exploit the spatial contiguity of the blob labels.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        grid: int = 8,
        n_classes: int = 4,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        self.n_classes = n_classes
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, n_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, N, D)`` tokens -> per-cell class logits ``(B, n_classes, g, g)``."""
        g = grid_size(tokens.shape[1])
        fmap = tokens_to_map(tokens, g)          # (B, D, g, g)
        feats = self.body(fmap)                  # (B, hidden, g, g)
        return self.classifier(feats)            # (B, n_classes, g, g)


def make_synthetic(
    n: int,
    embed_dim: int = 64,
    grid: int = 8,
    seed: int = 0,
    n_classes: int = 4,
    n_blobs: int = 3,
    noise: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic dense-segmentation tokens with spatially contiguous labels.

    Structure:
      * ``n_classes`` fixed prototype vectors are drawn once in R^D (the class
        "directions"), shared across every sample.
      * For each sample a label map ``(grid, grid)`` is built by scattering
        ``n_blobs`` random Gaussian blobs per class over the grid and taking, at
        each cell, the argmax over the accumulated class affinity fields. This
        yields spatially contiguous regions rather than iid per-cell labels.
      * Each cell feature is ``prototype[label] + noise * N(0, I)`` in R^D, then
        packed as tokens ``(B, N, D)`` with ``N == grid * grid``.

    Returns
    -------
    tokens : ``(n, grid * grid, embed_dim)`` float32 cell features.
    target : ``(n, grid, grid)`` int64 per-cell class ids in ``[0, n_classes)``.
    """
    # grid_size validates that N == grid*grid is a perfect square (the contract).
    if grid_size(grid * grid) != grid:
        raise ValueError("grid must describe a square token grid")
    gen = torch.Generator().manual_seed(seed)
    # Class prototypes define the TASK and must match across train/eval splits
    # (which use different sample seeds); draw them from a fixed task generator.
    sig_gen = torch.Generator().manual_seed(20240601)

    prototypes = torch.randn(n_classes, embed_dim, generator=sig_gen)
    # normalise prototype directions so the planted signal magnitude is comparable
    # to the per-cell noise (keeps the task non-trivial across embed_dim).
    prototypes = prototypes / prototypes.norm(dim=1, keepdim=True).clamp_min(1e-6)
    prototypes = prototypes * math.sqrt(embed_dim)

    # coordinate grid for the affinity fields
    ys, xs = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32),
        torch.arange(grid, dtype=torch.float32),
        indexing="ij",
    )
    sigma = max(grid / 4.0, 1.0)

    labels = torch.empty(n, grid, grid, dtype=torch.long)
    for b in range(n):
        affinity = torch.zeros(n_classes, grid, grid)
        for c in range(n_classes):
            for _ in range(n_blobs):
                cy = torch.rand(1, generator=gen).item() * (grid - 1)
                cx = torch.rand(1, generator=gen).item() * (grid - 1)
                amp = 0.5 + torch.rand(1, generator=gen).item()
                d2 = (ys - cy) ** 2 + (xs - cx) ** 2
                affinity[c] += amp * torch.exp(-d2 / (2.0 * sigma * sigma))
        # tiny per-cell jitter breaks ties / empty regions so every class can appear
        affinity += 0.05 * torch.randn(n_classes, grid, grid, generator=gen)
        labels[b] = affinity.argmax(dim=0)

    # build features from prototypes + noise
    feat_map = prototypes[labels]                       # (n, grid, grid, D)
    feat_map = feat_map + noise * torch.randn(
        n, grid, grid, embed_dim, generator=gen
    )
    fmap = feat_map.permute(0, 3, 1, 2).contiguous()    # (n, D, grid, grid)
    tokens = map_to_tokens(fmap)                        # (n, N, D)
    return tokens, labels


def _miou(logits: torch.Tensor, target: torch.Tensor, n_classes: int) -> float:
    """Mean IoU over classes PRESENT in ``target`` (pixel/cell-wise)."""
    pred = logits.argmax(dim=1)                         # (B, g, g)
    ious: list[float] = []
    for c in range(n_classes):
        gt_c = target == c
        if not gt_c.any():
            continue
        pr_c = pred == c
        inter = (pr_c & gt_c).sum().item()
        union = (pr_c | gt_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def train_eval(
    steps: int = 250,
    seed: int = 0,
    embed_dim: int = 64,
    grid: int = 8,
    n_classes: int = 4,
    n_train: int = 192,
    n_eval: int = 96,
    lr: float = 3e-3,
    noise: float = 1.0,
    **kw,
) -> dict:
    """Train the segmentation head on synthetic blobs; eval on a FRESH set.

    Returns a dict with mIoU before vs after training (cell-wise, over classes
    present in the eval targets) so the learning test can assert improvement.
    """
    torch.manual_seed(seed)

    head = SegmentationHead(embed_dim=embed_dim, grid=grid, n_classes=n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    x_tr, y_tr = make_synthetic(
        n_train, embed_dim=embed_dim, grid=grid, seed=seed, n_classes=n_classes, noise=noise
    )
    x_ev, y_ev = make_synthetic(
        n_eval, embed_dim=embed_dim, grid=grid, seed=seed + 9991, n_classes=n_classes, noise=noise
    )

    head.eval()
    with torch.no_grad():
        miou_start = _miou(head(x_ev), y_ev, n_classes)

    head.train()
    batch = 64
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        idx = torch.randint(0, n_train, (batch,), generator=gen)
        logits = head(x_tr[idx])                        # (batch, C, g, g)
        loss = F.cross_entropy(logits, y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        miou_end = _miou(head(x_ev), y_ev, n_classes)

    return {
        "miou": miou_end,
        "miou_start": miou_start,
        "miou_end": miou_end,
        "final_loss": float(loss.detach()),
        "steps": steps,
    }


__all__ = ["SegmentationHead", "make_synthetic", "train_eval"]
