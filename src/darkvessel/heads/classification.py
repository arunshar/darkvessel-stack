"""Whole-scene classification head over backbone tokens.

A lightweight scene-level classifier for the dark-vessel stack: given the
``(B, N, D)`` patch tokens returned by a ``GeoBackbone`` forward pass, it
mean-pools over the ``N`` spatial patches and maps the pooled descriptor through
an MLP to ``n_classes`` logits (e.g. a coarse scene category such as open-water /
coastal / port / fishing-ground / shipping-lane).

The data used here is SYNTHETIC and not a real Earth-observation benchmark (no
fMoW / EuroSAT / BigEarthNet read). ``make_synthetic`` plants a per-class
prototype direction in ``R^D`` and adds independent per-token Gaussian noise, so
the class signal survives mean-pooling in expectation but is buried under
per-token noise in any single token. The head still has to LEARN to separate the
pooled prototypes; the planted prototypes are never handed to the head as labels.
The reported accuracy demonstrates the head can learn this structured task; it is
not a benchmark number.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Mean-pool the patch tokens then classify the scene with a small MLP."""

    def __init__(
        self,
        embed_dim: int = 64,
        grid: int = 8,
        n_classes: int = 5,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        self.n_classes = n_classes
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, N, D)`` tokens -> ``(B, n_classes)`` class logits.

        Mean-pools over the ``N`` patch tokens to a single scene descriptor,
        normalises it, and projects to per-class logits.
        """
        pooled = tokens.mean(dim=1)          # (B, D)
        pooled = self.norm(pooled)
        return self.mlp(pooled)              # (B, n_classes)


def make_synthetic(
    n: int,
    embed_dim: int = 64,
    grid: int = 8,
    seed: int = 0,
    n_classes: int = 5,
    noise: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic scene-classification tokens with a planted per-class direction.

    Structure: ``n_classes`` unit-norm class prototypes are drawn once in
    ``R^embed_dim`` from a seeded generator. For each of the ``n`` scenes a class
    label is sampled uniformly; the scene's tokens are the class prototype
    (broadcast over all ``N = grid*grid`` patches) plus independent per-token
    Gaussian noise of scale ``noise``. Mean-pooling over ``N`` averages the noise
    down toward the prototype, so the class direction is recoverable in
    expectation, but every individual token is dominated by noise and the head
    must learn the linear separation between pooled prototypes.

    Returns
    -------
    inputs: ``(n, N, embed_dim)`` token tensor, ``N == grid * grid``.
    target: ``(n,)`` long tensor of class indices in ``[0, n_classes)``.
    """
    g = torch.Generator().manual_seed(seed)
    n_tok = grid * grid

    # Prototypes define the TASK and must be identical across train/eval splits
    # (which use different sample seeds); draw them from a fixed task generator.
    sig_gen = torch.Generator().manual_seed(20240601)
    prototypes = torch.randn(n_classes, embed_dim, generator=sig_gen)
    # scale to ~sqrt(D) so the class signal survives the per-token noise after
    # mean-pooling (the task stays noisy at the token level but separable once pooled)
    prototypes = F.normalize(prototypes, dim=1) * (embed_dim ** 0.5)

    target = torch.randint(0, n_classes, (n,), generator=g)
    proto_per_scene = prototypes[target]                              # (n, embed_dim)
    signal = proto_per_scene.unsqueeze(1).expand(n, n_tok, embed_dim)  # broadcast over N
    perturbation = noise * torch.randn(n, n_tok, embed_dim, generator=g)
    inputs = signal + perturbation
    return inputs, target


def train_eval(
    steps: int = 250,
    seed: int = 0,
    n_classes: int = 5,
    embed_dim: int = 64,
    grid: int = 8,
    n_train: int = 512,
    n_eval: int = 512,
    lr: float = 3e-3,
    noise: float = 1.0,
    **kw: object,
) -> dict:
    """Train the classification head on synthetic scenes and evaluate accuracy.

    Builds the head, fits it with Adam under ``F.cross_entropy`` on a seeded
    synthetic training set, and measures classification accuracy on a FRESH
    synthetic set (different seed) before and after training.

    Returns a dict with ``acc`` (final eval accuracy), ``acc_start`` (eval
    accuracy at initialisation), and ``acc_end`` (== ``acc``).
    """
    torch.manual_seed(seed)

    head = ClassificationHead(embed_dim=embed_dim, grid=grid, n_classes=n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    x_tr, y_tr = make_synthetic(
        n_train, embed_dim=embed_dim, grid=grid, seed=seed, n_classes=n_classes, noise=noise
    )
    x_ev, y_ev = make_synthetic(
        n_eval, embed_dim=embed_dim, grid=grid, seed=seed + 1, n_classes=n_classes, noise=noise
    )

    @torch.no_grad()
    def accuracy() -> float:
        head.eval()
        logits = head(x_ev)
        pred = logits.argmax(dim=1)
        return (pred == y_ev).float().mean().item()

    acc_start = accuracy()

    head.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = head(x_tr)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()

    acc_end = accuracy()
    return {"acc": acc_end, "acc_start": acc_start, "acc_end": acc_end}
