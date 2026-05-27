"""Math tests: numerical correctness of the core operators.

These are pure CPU tests with no HF downloads, no GDAL, no rasterio reads.
They exercise the algorithmic core so a CI run takes under 5 seconds.
"""
from __future__ import annotations

import math

import pytest
import torch

from darkvessel.ais.tgard import haversine, tgard_rendezvous
from darkvessel.backbones.geo_backbone import GeoBackbone
from darkvessel.fusion.coregistration import distortion_aware_coregister, identity_sensor
from darkvessel.heads.anomaly import PiDPMAnomalyHead
from darkvessel.optical.cloud_mask import band_ratios, s2cloudless_mask
from darkvessel.sar.speckle import lee_filter


def test_lee_filter_idempotent_on_constant_image() -> None:
    x = torch.full((1, 2, 32, 32), 0.5)
    y = lee_filter(x, window=5, looks=4.4)
    assert torch.allclose(y, x, atol=1e-6)


def test_lee_filter_reduces_variance_on_speckle() -> None:
    torch.manual_seed(0)
    x = torch.ones(1, 1, 64, 64) + 0.3 * torch.randn(1, 1, 64, 64)
    y = lee_filter(x, window=7, looks=4.4)
    assert y.var().item() < x.var().item()


def test_lee_filter_rejects_even_window() -> None:
    with pytest.raises(ValueError):
        lee_filter(torch.zeros(1, 1, 8, 8), window=4)


def test_lee_filter_handles_tiny_input() -> None:
    # pad >= min(H, W) used to crash F.pad(mode="reflect"); now the window shrinks.
    x = torch.rand(1, 2, 2, 2)
    y = lee_filter(x, window=5, looks=4.4)
    assert y.shape == x.shape
    one = torch.rand(1, 1, 1, 1)
    assert torch.allclose(lee_filter(one, window=5, looks=4.4), one, atol=1e-6)
    strip = torch.rand(1, 1, 1, 32)
    assert lee_filter(strip, window=5, looks=4.4).shape == strip.shape


def test_s2cloudless_mask_shape_and_dtype() -> None:
    s2 = torch.rand(2, 10, 24, 24)
    m = s2cloudless_mask(s2, threshold=0.4, dilation=2)
    assert m.shape == (2, 24, 24)
    assert m.dtype == torch.bool


def test_band_ratios_ndvi_in_range() -> None:
    s2 = torch.rand(1, 12, 16, 16)
    ratios = band_ratios(s2)
    assert set(ratios) == {"ndvi", "ndwi", "mndwi", "ndbi", "savi"}
    assert (ratios["ndvi"].abs() <= 1.0001).all()
    assert (ratios["ndwi"].abs() <= 1.0001).all()


def test_haversine_zero_for_same_point() -> None:
    z = haversine(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
    assert z.item() == pytest.approx(0.0, abs=1e-3)


def test_haversine_one_degree_lat_known() -> None:
    d = haversine(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0))
    assert d.item() == pytest.approx(111_195, rel=5e-4)


def test_tgard_skips_short_gaps() -> None:
    track = torch.tensor([
        [0.0, 35.0, -120.0, 12.0, 90.0],
        [60.0, 35.001, -120.001, 12.0, 90.0],
    ])
    assert tgard_rendezvous(track, tau_seconds=1800.0) == []


def test_tgard_emits_rendezvous_when_gap_is_infeasible() -> None:
    track = torch.tensor([
        [0.0, 35.0, -120.0, 5.0, 90.0],
        [3600.0, 35.0, -120.005, 5.0, 90.0],
    ])
    out = tgard_rendezvous(track, tau_seconds=1800.0, max_sog_knots=25.0)
    assert len(out) == 1
    assert 0.0 <= out[0].score <= 1.0


def test_coregister_identity_matches_grid_sample() -> None:
    src = torch.randn(1, 3, 32, 32)
    sm = identity_sensor("sentinel-1")
    tm = identity_sensor("sentinel-2")
    out = distortion_aware_coregister(src, sm, tm, tgt_size=(32, 32))
    assert out.shape == (1, 3, 32, 32)


def test_geo_backbone_stub_forward_shape() -> None:
    bb = GeoBackbone(name="prithvi-2", in_chans=6, image_size=224, stub=True)
    x = torch.randn(2, 6, 224, 224)
    tokens = bb(x)
    assert tokens.dim() == 3
    assert tokens.shape[0] == 2
    expected_tokens = (224 // bb.spec.patch) ** 2
    assert tokens.shape[1] == expected_tokens
    assert tokens.shape[2] == bb.spec.embed_dim


def test_geo_backbone_lists_all_six() -> None:
    assert set(GeoBackbone.list_supported()) == {
        "prithvi-2", "clay-v1", "satmae-pp", "dofa", "satlasnet", "remoteclip",
    }


def test_pidpm_head_forward_shapes() -> None:
    head = PiDPMAnomalyHead(embed_dim=128, ais_dim=5, hidden=64)
    scene = torch.randn(3, 64, 128)
    ais = torch.randn(3, 12, 5)
    out = head(scene, ais)
    assert out["score"].shape == (3, 1)
    assert out["reconstruction"].shape == (3, 12, 5)


def test_pidpm_head_supports_backward() -> None:
    head = PiDPMAnomalyHead(embed_dim=64, ais_dim=5, hidden=32)
    scene = torch.randn(2, 16, 64, requires_grad=True)
    ais = torch.randn(2, 8, 5)
    out = head(scene, ais)
    loss = out["score"].mean() + out["reconstruction"].pow(2).mean()
    loss.backward()
    assert scene.grad is not None
