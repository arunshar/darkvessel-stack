"""Unit tests for the learned SAR detector (SARStem + DetectionHead)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from darkvessel.backbones.sar_stem import SARDetector, SARStem
from darkvessel.data.xview3 import XView3Dataset, xview3_collate


def test_sar_detector_output_contract() -> None:
    # images (B, 2, H, W) -> objectness logits (B, 1, grid, grid)
    det = SARDetector(in_chans=2, embed_dim=32, grid=16)
    y = det(torch.randn(3, 2, 256, 256))
    assert y.shape == (3, 1, 16, 16)


def test_sar_stem_feature_map_shape() -> None:
    stem = SARStem(in_chans=2, embed_dim=48, grid=8)
    f = stem(torch.randn(2, 2, 512, 512))
    assert f.shape == (2, 48, 8, 8)


def test_sar_stem_rejects_wrong_channel_count() -> None:
    stem = SARStem(in_chans=2, embed_dim=16, grid=8)
    try:
        stem(torch.randn(1, 3, 256, 256))
    except ValueError:
        return
    raise AssertionError("SARStem should reject an input whose channel count != in_chans")


def test_sar_detector_learns_on_synthetic_xview3() -> None:
    # Exercise the real loader -> model -> BCE path on a tiny synthetic scene set
    # and confirm training reduces the loss (the head can learn the objectness task).
    torch.manual_seed(0)
    ds = XView3Dataset.synthetic(n_scenes=2, scene_size=512, chip_size=256,
                                 grid=16, vessels_per_scene=6, seed=1)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=xview3_collate)
    model = SARDetector(in_chans=2, embed_dim=32, grid=16)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    pos_weight = torch.tensor([40.0])  # objects are sparse cells in grid*grid

    losses = []
    for _ in range(40):
        for images, heatmaps, _targets in loader:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(images), heatmaps, pos_weight=pos_weight)
            loss.backward()
            opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0]  # training reduces the loss
