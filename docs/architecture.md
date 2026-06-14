# Architecture

> Status: this document is a DESIGN sketch, not a description of shipped code. As of today the only implemented pieces are the Lee speckle filter, the cloud-mask stub + band ratios, the TGARD haversine / rendezvous geometry, an identity-affine `grid_sample` coregistration, a stub `GeoBackbone` adapter, and the real Pi-DPM conditional-diffusion anomaly head. The six non-anomaly heads, the foundation-model loading, the STAC ingest, the data loaders, the eval/benchmark code, and the training loop are PLANNED and do not exist. See the README "Status" section for the authoritative breakdown.

## Design goals

1. **One repo, every canonical EO task.** Detection, segmentation, classification, change detection, super-resolution, time-series forecasting, and anomaly reasoning all hang off one shared backbone via swappable heads.
2. **One backbone interface, six open foundation models.** Prithvi-2 (default), Clay v1, SatMAE++, DOFA, SatlasNet, and RemoteCLIP all load through `GeoBackbone.from_pretrained(name)`.
3. **Sensor-faithful fusion.** SAR-to-optical co-registration uses the actual RPC / pushbroom Jacobians, not naive UTM reprojection.
4. **AIS-grounded anomaly reasoning.** TGARD detects gaps; Pi-DPM reconstructs the missing segment; the head outputs a calibrated dark-vessel probability and a reasoning trace.

## Top-level data flow (design target; only the anomaly head and a few primitives are coded)

Steps marked PLANNED have no implementation; the only coded path is the Lee filter, band ratios, the coregistration warp (identity-affine), the stub backbone forward, and the Pi-DPM anomaly head.

```
STAC catalog query (Microsoft Planetary Computer)                 PLANNED (no ingest code)
   |
   v
stackstac lazy stack (Sentinel-1 GRD VV/VH, Sentinel-2 L2A)        PLANNED
   |
   v
Preprocessing
   - SAR: calibration LUT -> Lee filter -> terrain correction      only Lee filter is coded
   - Optical: s2cloudless mask -> sun-glint suppress -> band ratios only cloud-mask stub + ratios are coded
   |
   v
Coregistration (identity-affine grid_sample today; RPC/pushbroom Jacobians PLANNED)
   |
   v
GeoBackbone (stub forward only; no real foundation-model weights loaded or tested)
   |
   +--- detection head (DETR)                 PLANNED
   +--- segmentation head (SAM 2 + Satlas)    PLANNED
   +--- classification head (linear probe)    PLANNED
   +--- change head (siamese Clay)            PLANNED
   +--- SR head (kriging-informed diffusion)  PLANNED
   +--- forecast head (TimeSformer)           PLANNED
   +--- anomaly head (Pi-DPM)                 IMPLEMENTED (conditional diffusion)
```

## Module map (EXISTS vs PLANNED)

```
src/darkvessel/
├── backbones/        EXISTS  geo_backbone.py only (stub adapter; no real weights)
├── heads/            EXISTS  anomaly.py only (real Pi-DPM); other 6 heads PLANNED
├── pidpm/            EXISTS  vendored Pi-DPM diffusion package
├── sar/              EXISTS  speckle.py only (Lee filter); calibration/terrain PLANNED
├── optical/          EXISTS  cloud_mask.py only (stub + band ratios); atm/sun-glint PLANNED
├── fusion/           EXISTS  coregistration.py only (identity-affine warp; RPC PLANNED)
├── ais/              EXISTS  tgard.py only (haversine + rendezvous); pidpm.py/dark_distortion.py do NOT exist
├── data/             PLANNED  (directory absent; no STAC/AIS dataloaders)
├── eval/             PLANNED  (directory absent; no xView3/OSCD/fMoW benchmarks)
├── training/         PLANNED  (no train.py, no Hydra configs)
└── viz/              PLANNED  (directory absent)
```

## Why this design

Most EO repos lock you into one task and one backbone. DarkVesselNet was built to be the *answer* to "tell me about modern remote sensing" in an interview: every modality, every canonical task, every leading open backbone, and a downstream decision (dark-vessel probability) that ties to a published anomaly-detection line. The architecture is intentionally over-built so you can talk about any cross-section for as long as the interviewer wants to dig.
