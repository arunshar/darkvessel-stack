# DarkVesselNet

> A work-in-progress reference scaffold for multi-modal dark-vessel reasoning: Sentinel-1 SAR + Sentinel-2 optical + AIS, with a real physics-informed conditional-diffusion anomaly head from the TGARD / Pi-DPM thesis line and a design sketch for the surrounding Earth-observation stack. Most of the EO stack is a documented roadmap, not shipped code. Read the "Status" section before judging what is implemented.

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Maritime domain awareness fails when ships go "dark," that is, when an AIS transponder is silenced, spoofed, or jammed. Detecting these vessels requires *fusing* what the sea is broadcasting (AIS) with what the sky is observing (Sentinel-1 SAR, Sentinel-2 optical, thermal). DarkVesselNet is meant to grow into a single repository that ingests these heterogeneous sensor streams, embeds them through a shared geospatial foundation model backbone, and runs the canonical remote sensing tasks end-to-end. As of today the implemented core is: the math primitives for SAR / optical preprocessing and fusion geometry, a backbone adapter stub, a genuine conditional-diffusion Pi-DPM anomaly head, and the seven canonical task heads as real networks evaluated on synthetic structured features. The real-data EO pipeline (foundation-model weights, STAC ingest, real benchmarks) is still a design document.

This README is written to be honest for anyone who clones the repo. Where a feature is described as a goal, it is labeled as such.

## Status: what is real vs. planned

This is the most important section. Everything below is either REAL (in `src/`, exercised by passing tests) or PLANNED (described here as a roadmap but not implemented).

### REAL and tested (lives in `src/`, covered by `tests/`, 33 tests pass: 15 in `test_math.py` plus 18 across the six task-head suites)

- **Lee SAR speckle filter** (`src/darkvessel/sar/speckle.py`). Pure-PyTorch Lee (1980) filter; tests check idempotence on constant images, variance reduction on speckle, and edge cases.
- **s2cloudless-surrogate cloud mask + band ratios** (`src/darkvessel/optical/cloud_mask.py`). A stub linear cloud classifier (NOT the real s2cloudless LightGBM model) plus exact NDVI / NDWI / MNDWI / NDBI / SAVI band ratios.
- **Haversine + TGARD rendezvous geometry** (`src/darkvessel/ais/tgard.py`). Great-circle distance and the trajectory-gap feasibility / rendezvous score from the TGARD paper. This is the geometric core only, not the full STAGD + DRM pipeline.
- **Grid-sample coregistration** (`src/darkvessel/fusion/coregistration.py`). A `grid_sample`-based warp driven by pluggable sensor models. The shipped sensor model is an **identity-affine warp** for the test path. The real closed-form RPC / pushbroom Jacobians are NOT implemented here; the docstring's reference to `sat-splat-distort` is aspirational.
- **GeoBackbone adapter stub** (`src/darkvessel/backbones/geo_backbone.py`). Holds a registry of backbone "cards" (HF ids, patch sizes, embed dims) and a `stub` mode that returns a shape-consistent Conv2d projection. `from_pretrained` would call `transformers.AutoModel`, but no foundation model is actually loaded or tested; only the stub path is exercised.
- **Pi-DPM conditional-diffusion anomaly head** (`src/darkvessel/heads/anomaly.py`), backed by the vendored package `src/darkvessel/pidpm/`. This is now a genuine conditional diffusion model: a `TrajectoryDenoiser`, a `GaussianDiffusion` process, and a learned score head over the reconstruction residual and a scale-free kinematic-smoothness residual. The test exercises a forward pass and the diffusion loss on CPU.
- **The six remote-sensing task heads** (`src/darkvessel/heads/{detection,segmentation,classification,change,superres,forecast}.py`, on the shared backbone-token contract in `heads/common.py`). Each is a real `nn.Module` with a self-contained synthetic eval that demonstrates the head genuinely learns its task on structured features: detection cell-F1 0.07 -> 0.86, segmentation mIoU 0.13 -> 1.0, classification accuracy 0.23 -> 1.0, change-detection F1 0.47 -> 0.71, super-resolution PSNR 8.6 -> 9.3 dB (beats the bilinear baseline 9.0), next-step forecasting MSE 0.12 -> 0.0014 (beats persistence 0.11). These numbers are measured on SYNTHETIC structured features (a planted, learnable signal plus noise), NOT on xView3 / SpaceNet / fMoW / LEVIR-CD / WorldStrat / SEN12MS. They show each head works end to end; they are not benchmark results.

### NOT implemented (roadmap only, no code yet)

- **Real-data training of the task heads.** All seven heads exist as real modules, but the six EO heads run on synthetic structured features only; none is trained on real xView3 / SpaceNet / fMoW / LEVIR-CD / WorldStrat / SEN12MS data, and no real-data checkpoints are shipped.
- **Foundation-model loading and fusion.** No Prithvi-2 / Clay / SatMAE++ / DOFA / SatlasNet / RemoteCLIP weights are actually loaded or fine-tuned. Only the `stub` backbone path runs.
- **STAC / Microsoft Planetary Computer ingest.** No `stackstac`, no STAC queries, no tile fetch. The end-to-end diagram below is a design, not a running pipeline.
- **All data loaders.** No xView3, SEN12MS, SpaceNet, fMoW, MarineCadastre, or Planetary-Computer loaders exist (`darkvessel.data.*` is not present).
- **Real EO benchmark code.** `darkvessel.eval.xview3_bench` and the other real-dataset benchmark modules do NOT exist. The only measured numbers in this repo are the task heads' SYNTHETIC self-checks (above); there is no real xView3 / ScanNet / LEVIR-CD benchmark result.
- **Training.** `darkvessel.training.train` and the Hydra `configs/` tree do NOT exist. There is no shipped checkpoint or released dataset.
- **The xView3 SOTA run.** Not run, not runnable from this repo.

## Why this repo exists

This is the bridge between the author's dissertation on distortion-aware spatial data science and the modern geospatial ML stack. The implemented core is the published TGARD / Pi-DPM anomaly line; the surrounding EO scaffold documents how that anomaly head would slot into a full multi-sensor pipeline. Treat the EO portions as a literacy-and-design reference, and the `src/` modules listed under "REAL" as the working code.

## The seven canonical tasks (all 7 implemented as real heads; the 6 EO heads on synthetic evals)

All seven heads are real modules. Heads 1-6 are simple conv/MLP networks on the backbone-token contract, evaluated on synthetic structured features (the Status column gives the measured synthetic metric, NOT a real-dataset benchmark). The "Intended backbone/dataset" columns are the real-data targets that are not yet wired in. Task 7 is the conditional-diffusion anomaly head.

| # | Task | Status (synthetic eval) | Intended backbone | Intended dataset |
| --- | --- | --- | --- | --- |
| 1 | Object detection | real head, cell-F1 0.07->0.86 | DETR head on backbone features | xView3-SAR |
| 2 | Instance segmentation | real head, mIoU 0.13->1.0 | SAM 2 + SatlasNet adapter | SpaceNet 6, xView3 |
| 3 | Classification | real head, acc 0.23->1.0 | linear probe on backbone | fMoW-Sentinel, BigEarthNet, EuroSAT |
| 4 | Change detection | real head, F1 0.47->0.71 | siamese Clay encoder | LEVIR-CD, OSCD, xBD |
| 5 | Super-resolution | real head, PSNR 9.3 dB > bilinear | kriging-informed diffusion pipeline | WorldStrat, SEN2VENuS |
| 6 | Time-series forecasting | real head, MSE 0.001 < persistence | TimeSformer / Prithvi-WxC | SEN12MS-CR-TS |
| 7 | **Anomaly reasoning (Pi-DPM)** | **conditional diffusion, real** | conditional diffusion over AIS + scene tokens | (trains on AIS segments; no released split) |

## End-to-end pipeline (design sketch, NOT a running pipeline)

The diagram below is the intended architecture. The six upstream task heads and the bottom Pi-DPM anomaly block are implemented as real modules (evaluated on synthetic features); STAC ingest and the backbone forward on real foundation-model weights are not.

```
[Microsoft Planetary Computer STAC]   <-- PLANNED, no ingest code
   |
   v
[stackstac] --Sentinel-1 VV/VH--+      +--[ AIS feed: MarineCadastre / DMA / xView3 AIS ]  <-- PLANNED
                                 |      |
                                 v      v
                          [GeoFoundation backbone]      <-- only a stub adapter exists
                                 |
              +------------------+------------------+----------+----------+
              v                  v                  v          v          v
         [Detection]      [Segmentation]      [Classify]   [SR]    [Change det]   <-- REAL heads (synthetic eval)
              |                  |                  |          |          |
              +--------+---------+---------+--------+----------+----------+
                       v
                [Pi-DPM conditional-diffusion anomaly head]   <-- IMPLEMENTED (heads/anomaly.py + pidpm/)
                       |
                       v
                [Dark vessel score + reconstructed AIS segment]
```

## Quickstart

```bash
git clone https://github.com/arunshar/darkvessel-stack
cd darkvessel-stack
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                                          # the math tests pass (15 tests)
```

The following commands are PLANNED and NOT yet runnable. The referenced modules, configs, and scripts do not exist in this repo:

```bash
# NOT RUNNABLE YET (no data loader, no download script):
bash scripts/download_xview3_sample.sh

# NOT RUNNABLE YET (no darkvessel.training.train module, no configs/ tree):
python -m darkvessel.training.train +experiment=xview3_prithvi
```

## Tests

```bash
pytest                                          # math tests only; these pass
```

The math tests cover what is actually implemented: SAR Lee-filter consistency and variance reduction, band-ratio correctness, GeoBackbone stub patch-embed shape, TGARD haversine distance, fusion grid-sample shape, and a Pi-DPM conditional-diffusion forward pass plus diffusion loss. There is no benchmark, no checkpoint, and no measured leaderboard number in this repo.

## Demo Space (CPU stub, returns a seeded pseudo-score, NOT a model output)

`space/app.py` is a Gradio CPU stub. It does NOT load any foundation model and does NOT run the real anomaly head. On click it seeds the RNG from the AOI name and returns a deterministic pseudo-random "dark vessel probability" plus a hard-coded reasoning string. It exists to show the intended UI shape only. Do not read its output as a detection.

## Sensor primer (background, not implemented capability)

The table below is reference material on the four sensor modalities the design targets. It documents what the pipeline *would* fuse; the repo does not currently ingest any of these sensors.

| Modality | Sentinel mission | Spatial | Temporal | Spectral | Strength | Weakness |
| --- | --- | --- | --- | --- | --- | --- |
| SAR | Sentinel-1 (C-band) | 5 m x 20 m (IW GRD) | 6 d (one sat) / 12 d (constellation) | VV / VH polarizations | All-weather, day-night, sees through cloud | Speckle, layover, foreshortening |
| Optical multispectral | Sentinel-2 | 10 m (RGB+NIR) / 20 m (red edge, SWIR) / 60 m (aerosol, cirrus) | 5 d at equator | 13 bands, 443-2190 nm | Vegetation indices, true color, easy interp | Clouds, shadows, sun angle |
| Thermal | Landsat-9 TIRS | 100 m | 16 d | 10.8 + 12.0 um | Heat signatures (engines, fires) | Low spatial res |
| High-res optical | PlanetScope / SkySat | 3 m / 0.5 m | Daily | RGB-NIR | Visible vessel structure | Tasking cost, cloud-limited |

The intent: SAR as the workhorse for vessel detection in the open ocean (the only modality that sees at night and through cloud), optical to confirm, thermal to disambiguate wakes and engine heat, and high-res optical (where licensed) to classify vessel type.

## SAR / optical preprocessing notes (background; only the Lee filter and band ratios are coded)

- SAR calibration to sigma0 via the official calibration LUT, terrain correction via SNAP / pyroSAR against Copernicus DEM-30, and a close-to-shore layover head are described in the xView3 protocol but are NOT implemented here. The one implemented SAR primitive is the Lee speckle filter.
- Optical atmospheric correction (Sen2Cor L2A) and sun-glint suppression are background notes, not code. The implemented optical primitives are the cloud-mask *stub* and the band ratios.
- SAR-to-optical co-registration: the shipped warp is an identity-affine `grid_sample`, NOT the RPC + pushbroom Jacobians the design calls for.

## Foundation model menu (design notes; none are loaded or tested)

This menu records candidate open backbones the `GeoBackbone` adapter is *designed* to wrap. The adapter only carries their metadata cards and a stub forward today; no weights are downloaded, loaded, or fine-tuned in any tested path.

| Backbone | Source | Params | Pretraining data | Intended use |
| --- | --- | --- | --- | --- |
| Prithvi-2 | NASA + IBM, Apache 2.0 | 600M | 4.2 TB HLS L30/S30 | default detection + classification head |
| Clay v1 | Made with Clay, MIT | 300M | 70 TB multi-sensor | change detection (siamese) |
| SatMAE++ | Cong et al., NeurIPS 2023 | 305M | fMoW-Sentinel | scarce-label fine-tunes |
| DOFA | Xiong et al., CVPR 2024 | 350M | multi-sensor unified | cross-sensor transfer |
| SatlasNet | Allen AI | 90M Swin-B | Satlas | instance segmentation head |
| RemoteCLIP | Liu et al., CVPR 2024 | 304M | RS-image text pairs | open-vocabulary retrieval |

The adapter signature is `darkvessel.backbones.GeoBackbone.from_pretrained(...)`, but only the `stub=True` path is exercised by tests.

## Repository layout (annotated: which files exist vs. are planned)

Files marked EXISTS are present in the repo. Files marked PLANNED are referenced in the design but are NOT in the repo.

```
darkvessel-stack/
├── src/darkvessel/
│   ├── backbones/
│   │   └── geo_backbone.py            # EXISTS (stub adapter; no real weights loaded)
│   │   # prithvi.py, clay.py, satmae.py, dofa.py, satlas.py, remoteclip.py  -> PLANNED
│   ├── heads/
│   │   └── anomaly.py                 # EXISTS (real conditional-diffusion Pi-DPM head)
│   │   # detection.py, segmentation.py, classification.py, change.py, sr.py, forecast.py -> PLANNED
│   ├── pidpm/                         # EXISTS (vendored Pi-DPM: config, data, diffusion,
│   │   │                             #         model, physics, scoring, train, eval)
│   ├── sar/
│   │   └── speckle.py                 # EXISTS (Lee filter)
│   │   # calibration.py, terrain_correction.py, polarimetry.py -> PLANNED
│   ├── optical/
│   │   └── cloud_mask.py              # EXISTS (cloud-mask stub + band ratios)
│   │   # atm_correction.py, sun_glint.py, band_ratios.py(standalone) -> PLANNED
│   ├── fusion/
│   │   └── coregistration.py          # EXISTS (identity-affine grid_sample warp)
│   │   # token_fusion.py, late_fusion.py -> PLANNED
│   ├── ais/
│   │   └── tgard.py                   # EXISTS (haversine + rendezvous geometry only)
│   │   # trajclip.py, pidpm.py, dark_distortion.py -> PLANNED (do NOT exist)
│   ├── data/                          # PLANNED (entire directory absent)
│   ├── eval/                          # PLANNED (entire directory absent; no xview3_bench)
│   ├── training/                      # PLANNED (no train.py, no Hydra configs)
│   └── viz/                           # PLANNED (entire directory absent)
├── space/app.py                       # EXISTS (Gradio CPU stub, seeded pseudo-score)
├── configs/                           # PLANNED (no Hydra config tree)
├── tests/test_math.py                 # EXISTS (15 math tests, pass)
├── paper/{main.tex, main.pdf}         # EXISTS (draft)
├── docs/                              # EXISTS (architecture, sensor_primer,
│                                      #         foundation_models, anomaly_pipeline)
└── scripts/                           # download/submit scripts -> PLANNED
```

## Planned evaluation (NOT measured, no eval code yet)

There is no eval code in this repo (`darkvessel.eval.xview3_bench` does not exist), no trained checkpoint, and no released split, so DarkVesselNet has produced no measured number. The table below lists only published xView3-SAR baselines as a reference for what a future evaluation would compare against. DarkVesselNet does NOT appear in it because it has not been run.

Reference baselines (from Paolo et al., NeurIPS Datasets and Benchmarks 2022, and the public xView3 leaderboard at challenge close):

| Method | Aggregate score | Detection F1 | Length RMSE (m) | Close-to-shore F1 |
| --- | --- | --- | --- | --- |
| xView3 baseline (FRCNN) | 27.7 | 0.39 | 32.7 | 0.27 |
| 1st place (BloodAxe et al.) | 50.6 | 0.62 | 19.5 | 0.42 |
| SatlasNet-XL (Allen AI) | 47.8 | 0.59 | 21.1 | 0.40 |

A training and evaluation harness against this benchmark is on the roadmap; none of it is shipped here.

## Connection to my published work

| Thesis contribution | Where it lives in this repo | Status | Original venue |
| --- | --- | --- | --- |
| Pi-DPM (physics-informed diffusion for trajectory anomalies) | `src/darkvessel/heads/anomaly.py` + `src/darkvessel/pidpm/` | real conditional diffusion now | ACM SIGSPATIAL GeoAnomalies 2025 |
| TGARD (trajectory rendezvous + anomaly gap geometry) | `src/darkvessel/ais/tgard.py` | geometric core only (no full STAGD/DRM) | ACM TIST 2024 |
| STAGD + DRM (denial-based distortion) | not in this repo (`ais/dark_distortion.py` does NOT exist) | planned | ACM TIST 2024 |
| Distortion-aware projection (RPC / pushbroom Jacobians) | `src/darkvessel/fusion/coregistration.py` | only an identity-affine warp; the RPC Jacobians are NOT implemented | reused from sat-splat-distort (draft) |
| Kriging-informed conditional diffusion for downscaling | not in this repo (no `heads/sr.py`) | planned | SIGSPATIAL 2024 |

Note: the modules `src/darkvessel/ais/pidpm.py` and `src/darkvessel/ais/dark_distortion.py` referenced in earlier drafts do NOT exist. The real Pi-DPM implementation is under `heads/anomaly.py` and the vendored `pidpm/` package.

## Citation

```bibtex
@misc{sharma_darkvesselnet,
  title  = {DarkVesselNet: A Multi-Modal Remote Sensing Scaffold for Dark Vessel Detection (work in progress)},
  author = {Sharma, Arun},
  note   = {Reference scaffold; only the Pi-DPM anomaly head and core math primitives are implemented},
  year   = {2026}
}
```

## License

Apache 2.0. The foundation models named in the design notes retain their own licenses (Prithvi-2 Apache 2.0; Clay v1 MIT; SatMAE++ CC-BY-NC); none are redistributed here. xView3 data is under the Allen AI Impact License.
