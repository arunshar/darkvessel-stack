#!/usr/bin/env python
"""Train a SAR vessel detector on xView3 and compare it to the CFAR baseline.

Trains :class:`darkvessel.backbones.sar_stem.SARDetector` (a from-scratch SAR
conv stem feeding the reused DetectionHead) on a SCENE-DISJOINT split of the
labeled xView3 scenes, then scores the held-out scenes with the vendored official
xView3 metric. It also scores the unlearned CFAR baseline on the SAME held-out
scenes, so the learned-vs-CFAR loc_fscore comparison is apples-to-apples (same
grid, same heatmap_to_detections extraction, same scenes).

HONESTY: the labeled scenes are the xView3 *validation* partition (the test
partition's labels are private). This is therefore a held-out-validation-split
result with the official scorer, NOT a public-leaderboard rank. The milestone is
a real, scorer-validated beat of the unlearned baseline, not a competitive
submission.

Real data (GPU)::

    python scripts/train_xview3_detector.py \
        --data-root /scratch.global/$USER/xview3/validation \
        --labels-csv /scratch.global/$USER/xview3/validation.csv \
        --grid 16 --epochs 12 --batch-size 16 --device cuda \
        --out results/xview3/detector_vs_cfar.json \
        --ckpt results/xview3/sar_detector.pt

Smoke (synthetic, no rasterio / no GPU)::

    python scripts/train_xview3_detector.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from darkvessel.backbones.sar_stem import SARDetector
from darkvessel.data.xview3 import XView3Dataset, labels_to_heatmap, xview3_collate
from darkvessel.eval.run_xview3 import cfar_predict_fn, evaluate


def _f(x: Any) -> float:
    """Coerce a score to a finite float (None / NaN -> 0.0) for safe arithmetic."""
    if x is None:
        return 0.0
    x = float(x)
    return 0.0 if math.isnan(x) else x


def _scene_map(ds: XView3Dataset) -> dict[str, Any]:
    return dict(zip(ds.scene_ids, ds.scenes))


def _subset(full: XView3Dataset, ids: list[str], *, grid: int, chip_size: int,
            keep_empty: bool) -> XView3Dataset:
    """A scene-disjoint XView3Dataset over ``ids`` with labels scoped to those scenes.

    Scoping the labels to ``ids`` is essential: ``evaluate`` builds the ground
    truth from ``dataset.labels_by_scene``, so leaving in labels for scenes that
    are not in this split would inject unmatchable false negatives.
    """
    smap = _scene_map(full)
    scenes = [smap[s] for s in ids]
    labels = {s: full.labels_by_scene.get(s, []) for s in ids}
    return XView3Dataset(scenes, ids, labels, chip_size=chip_size, grid=grid,
                         keep_empty=keep_empty)


def _split_pos_neg(ds: XView3Dataset) -> tuple[list, list]:
    """Split a dataset's chip index into (positive, negative) refs by in-chip labels."""
    pos, neg = [], []
    for ref in ds.index:
        (pos if ds.chip_labels(ref) else neg).append(ref)
    return pos, neg


def _pos_weight(ds: XView3Dataset, refs: list, grid: int, cap: float = 20.0) -> float:
    """BCE pos_weight = #negative cells / #positive cells over the training chips.

    Capped (``cap``) to bound the over-prediction pressure: too high a pos_weight
    drives the detector to fire everywhere, which kills precision on the mostly
    empty held-out scenes.
    """
    pos_cells = 0
    for ref in refs:
        hm = labels_to_heatmap(ds.chip_labels(ref), ref.row0, ref.col0,
                               ds.chip_size, grid)
        pos_cells += int(hm.sum())
    total = len(refs) * grid * grid
    neg_cells = max(total - pos_cells, 1)
    return min(neg_cells / max(pos_cells, 1), cap)


def build_training_set(full: XView3Dataset, train_ids: list[str], *, grid: int,
                       chip_size: int, neg_per_pos: float, max_chips: int | None,
                       seed: int) -> tuple[XView3Dataset, int, int]:
    """All positive chips + sampled negatives from the train scenes.

    Negatives (empty water / land / clutter) are essential to teach the detector
    to NOT fire, which is exactly where a learned model can beat the fixed CFAR
    threshold. Keep every positive chip; sample negatives at ``neg_per_pos`` per
    positive, capped by ``max_chips``.
    """
    ds = _subset(full, train_ids, grid=grid, chip_size=chip_size, keep_empty=True)
    pos, neg = _split_pos_neg(ds)
    rng = random.Random(seed)
    rng.shuffle(neg)
    n_neg = min(len(neg), int(neg_per_pos * max(len(pos), 1)))
    if max_chips is not None:
        n_neg = min(n_neg, max(max_chips - len(pos), 0))
    if n_neg < len(pos):
        print(f"[warn] only {n_neg} negative chips vs {len(pos)} positive; the detector "
              f"sees little clutter and will over-fire. Raise --max-train-chips.", flush=True)
    chosen = list(pos) + neg[:n_neg]
    rng.shuffle(chosen)
    ds.index = chosen
    return ds, len(pos), n_neg


def train(model: torch.nn.Module, ds: XView3Dataset, *, epochs: int, batch_size: int,
          lr: float, pos_weight: float, device: str, num_workers: int) -> list[float]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, collate_fn=xview3_collate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], device=device)
    model.train()
    history: list[float] = []
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for images, heatmaps, _targets in loader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(images), heatmaps, pos_weight=pw)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        avg = tot / max(nb, 1)
        history.append(avg)
        print(f"[train] epoch {ep + 1}/{epochs}  loss {avg:.4f}", flush=True)
    return history


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu", flush=True)
        device = "cpu"

    if args.smoke:
        full = XView3Dataset.synthetic(n_scenes=6, scene_size=512, chip_size=256,
                                       grid=args.grid, vessels_per_scene=5, seed=0)
        chip_size = 256
    else:
        full = XView3Dataset.from_any(args.data_root, args.labels_csv,
                                      max_scenes=args.limit_scenes,
                                      chip_size=args.chip_size, grid=args.grid)
        chip_size = args.chip_size

    ids = sorted(set(full.scene_ids))
    random.Random(args.seed).shuffle(ids)
    n_test = min(args.n_test_scenes, max(len(ids) - 1, 1))
    test_ids = sorted(ids[:n_test])
    train_ids = sorted(ids[n_test:])
    print(f"[split] {len(train_ids)} train scenes, {len(test_ids)} held-out test scenes",
          flush=True)

    train_ds, n_pos, n_neg = build_training_set(
        full, train_ids, grid=args.grid, chip_size=chip_size,
        neg_per_pos=args.neg_per_pos, max_chips=args.max_train_chips, seed=args.seed)
    print(f"[train set] {n_pos} positive + {n_neg} negative chips", flush=True)

    pw = _pos_weight(train_ds, train_ds.index, args.grid, cap=args.pos_weight_cap)
    print(f"[pos_weight] {pw:.2f}", flush=True)

    model = SARDetector(in_chans=2, embed_dim=args.embed_dim, grid=args.grid).to(device)
    history = train(model, train_ds, epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, pos_weight=pw, device=device, num_workers=args.num_workers)

    test_ds = _subset(full, test_ids, grid=args.grid, chip_size=chip_size, keep_empty=True)
    print(f"[eval] scoring {len(test_ds)} held-out chips (learned vs CFAR) ...", flush=True)
    model.eval()
    learned = evaluate(test_ds, lambda imgs: model(imgs), batch_size=args.batch_size,
                       device=device, num_workers=args.num_workers)
    cfar = evaluate(test_ds, cfar_predict_fn(args.grid), batch_size=args.batch_size,
                    device="cpu", num_workers=args.num_workers)
    assert isinstance(learned, dict) and isinstance(cfar, dict)  # return_frames=False

    learned_f = _f(learned.get("loc_fscore"))
    cfar_f = _f(cfar.get("loc_fscore"))
    result = {
        "train_scenes": train_ids,
        "test_scenes": test_ids,
        "n_pos_chips": n_pos,
        "n_neg_chips": n_neg,
        "pos_weight": pw,
        "epochs": args.epochs,
        "final_train_loss": history[-1] if history else None,
        "grid": args.grid,
        "chip_size": chip_size,
        "embed_dim": args.embed_dim,
        "learned_loc_fscore": learned_f,
        "cfar_loc_fscore": cfar_f,
        "delta_loc_fscore": learned_f - cfar_f,
        "learned_scores": learned,
        "cfar_scores": cfar,
        "smoke": args.smoke,
    }
    print(json.dumps({k: result[k] for k in
                      ("learned_loc_fscore", "cfar_loc_fscore", "delta_loc_fscore")},
                     indent=2), flush=True)
    print(f"[milestone] learned detector "
          f"{'BEATS' if result['delta_loc_fscore'] > 0 else 'does NOT beat'} "
          f"CFAR on held-out loc_fscore", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[saved] {args.out}", flush=True)
    if args.ckpt:
        Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.ckpt)
        print(f"[saved] {args.ckpt}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", help="xView3 scene-folder root (or .SAFE.zip dir)")
    p.add_argument("--labels-csv", help="xView3 validation label CSV")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--chip-size", type=int, default=800)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--neg-per-pos", type=float, default=3.0)
    p.add_argument("--max-train-chips", type=int, default=20000,
                   help="keep all positive chips + up to neg-per-pos negatives; set high "
                        "enough that negatives are not starved (clutter teaches precision)")
    p.add_argument("--pos-weight-cap", type=float, default=20.0)
    p.add_argument("--n-test-scenes", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-scenes", type=int, default=None,
                   help="cap total scenes (sorted) for a quick subset run")
    p.add_argument("--out", default=None, help="write the full result JSON here")
    p.add_argument("--ckpt", default=None, help="save the trained model state_dict here")
    p.add_argument("--smoke", action="store_true",
                   help="run on the synthetic dataset (no real data / GPU)")
    args = p.parse_args(argv)
    if not args.smoke and not (args.data_root and args.labels_csv):
        p.error("--data-root and --labels-csv are required unless --smoke")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
