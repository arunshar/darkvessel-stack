# Geospatial foundation models

> Status: this is BACKGROUND / design reference on candidate backbones, not a description of shipped capability. The `GeoBackbone` adapter today only stores these metadata cards and runs a shape-consistent stub forward. No Prithvi-2 / Clay / SatMAE++ / DOFA / SatlasNet / RemoteCLIP weights are actually loaded, fine-tuned, or tested in any path. There are no Hydra configs.

Six open backbones the `GeoBackbone` adapter is *designed* to unify. The intended default is Prithvi-2; today only the stub path runs.

## Prithvi-2 (NASA / IBM, 2024)

- 600M parameters, ViT-G, 16x16 patches.
- Pretrained on 4.2 TB of Harmonized Landsat-Sentinel (HLS) L30 + S30 (six bands: blue, green, red, narrow NIR, SWIR1, SWIR2).
- Masked autoencoder pretraining objective.
- License: Apache 2.0.
- Best for: detection and classification heads on HLS / Sentinel-2 inputs.

## Clay v1 (Made with Clay, 2024)

- 300M parameters, ViT-L.
- Pretrained on 70 TB across Sentinel-2, Sentinel-1, Landsat, MODIS, NAIP, LINZ.
- Sensor-conditioned: a metadata token (sensor id, bands, GSD, latitude) is concatenated to the patch tokens at every layer.
- License: MIT.
- Best for: cross-sensor transfer (e.g., training on Sentinel-2 and applying to PlanetScope) and change detection in siamese mode.

## SatMAE++ (Cong et al., NeurIPS 2023)

- 305M parameters, ViT-L.
- Multispectral MAE pretraining on fMoW-Sentinel.
- License: CC-BY-NC.
- Best for: scarce-label fine-tunes on fMoW-style classification.

## DOFA (Xiong et al., CVPR 2024)

- 350M parameters, dynamic one-for-all multispectral transformer.
- Conditions on wavelength of each input band so it generalizes across sensors with different bands.
- License: MIT.
- Best for: cross-sensor settings where each tile may come from a different mission (Sentinel-2 + Landsat + PlanetScope).

## SatlasNet (Allen AI, 2023)

- 90M-parameter Swin-B.
- Pretrained on Satlas, a 36-task supervised collection over global Sentinel-2.
- License: Apache 2.0.
- Best for: instance segmentation when paired with SAM 2 mask decoder.

## RemoteCLIP (Liu et al., CVPR 2024)

- 304M parameters, ViT-L.
- CLIP-style contrastive pretraining on remote sensing image / text pairs.
- License: Apache 2.0.
- Best for: open-vocabulary retrieval ("show me ports with unusual activity"), zero-shot classification.

## Why six

Interviewers ask "have you used Prithvi?" / "what about Clay?" / "do you know SatMAE?" The adapter is the wrapper that would unify them behind one interface. As shipped it carries their metadata and a stub forward; loading and fine-tuning the real weights is on the roadmap.
