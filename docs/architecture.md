# Architecture

## Design goals

1. **One repo, every canonical EO task.** Detection, segmentation, classification, change detection, super-resolution, time-series forecasting, and anomaly reasoning all hang off one shared backbone via swappable heads.
2. **One backbone interface, six open foundation models.** Prithvi-2 (default), Clay v1, SatMAE++, DOFA, SatlasNet, and RemoteCLIP all load through `GeoBackbone.from_pretrained(name)`.
3. **Sensor-faithful fusion.** SAR-to-optical co-registration uses the actual RPC / pushbroom Jacobians, not naive UTM reprojection.
4. **AIS-grounded anomaly reasoning.** TGARD detects gaps; Pi-DPM reconstructs the missing segment; the head outputs a calibrated dark-vessel probability and a reasoning trace.

## Top-level data flow

```
STAC catalog query (Microsoft Planetary Computer)
   |
   v
stackstac lazy stack (Sentinel-1 GRD VV/VH, Sentinel-2 L2A 12 bands)
   |
   v
Preprocessing
   - SAR: calibration LUT -> Lee filter -> terrain correction
   - Optical: s2cloudless mask -> sun-glint suppress -> band ratios
   |
   v
Distortion-aware coregistration (analytic RPC / pushbroom Jacobians)
   |
   v
GeoBackbone (Prithvi-2 | Clay | SatMAE++ | DOFA | SatlasNet | RemoteCLIP)
   |
   +--- detection head (DETR)
   +--- segmentation head (SAM 2 + SatlasNet adapter)
   +--- classification head (linear probe)
   +--- change head (siamese Clay)
   +--- SR head (PhysFlow-Earth pipeline)
   +--- forecast head (TimeSformer)
   +--- anomaly head (Pi-DPM) <----- joined with AIS via traj-CLIP + TGARD
```

## Module map

```
src/darkvessel/
├── backbones/        # GeoBackbone unifies six foundation models
├── heads/            # seven task heads, all consume backbone tokens
├── sar/              # Sentinel-1 preprocessing
├── optical/          # Sentinel-2 preprocessing
├── fusion/           # distortion-aware co-registration
├── ais/              # TGARD, Pi-DPM, traj-CLIP integration
├── data/             # STAC + AIS dataloaders
├── eval/             # xView3, OSCD, fMoW, WorldStrat benchmarks
├── training/         # Hydra-configured train loop
└── viz/              # Folium map + attention rollout + reasoning trace
```

## Why this design

Most EO repos lock you into one task and one backbone. DarkVesselNet was built to be the *answer* to "tell me about modern remote sensing" in an interview: every modality, every canonical task, every leading open backbone, and a downstream decision (dark-vessel probability) that ties to a published anomaly-detection line. The architecture is intentionally over-built so you can talk about any cross-section for as long as the interviewer wants to dig.
