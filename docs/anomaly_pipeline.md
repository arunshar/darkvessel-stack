# Anomaly pipeline: from candidate vessel to dark-vessel probability

The seven RS heads tell us *where* and *what* the vessel is. The anomaly pipeline tells us *whether it should be there*. This is where the dissertation contributions plug in.

## Inputs

- Vessel candidates from the detection head: bounding boxes with confidence in SAR pixel coords.
- Vessel-type predictions from the classification head: fishing / cargo / tanker / passenger / other.
- AIS feed within the AOI and time window: ``(t, mmsi, lat, lon, sog, cog)`` tuples from MarineCadastre or DMA Danish AIS.

## Step 1: traj-CLIP alignment (from trajprompt)

For each AIS track in the AOI we compute the traj-CLIP embedding (Sharma 2026, NeurIPS Datasets and Benchmarks, in submission). Each vessel candidate from the detection head is matched to the nearest traj-CLIP embedding by cosine similarity weighted by spatial proximity. Unmatched candidates are flagged as **AIS-absent** (the strict definition of dark).

## Step 2: TGARD gap detection (Sharma, Ghosh, Shekhar, ACM TIST 2024)

For every AIS track that *is* in the area, we walk the time series and flag every gap longer than tau (default 30 min). Each gap is scored by the feasibility envelope of the maritime kinematic model:
- ``reach = max_sog * dt``
- ``score = max(0, 1 - haversine(start, end) / reach)``

A score near 1 means the vessel barely could have made it; a score near 0 means it definitely did (and likely met someone in between).

## Step 3: STAGD + DRM (Sharma, Ghosh, Shekhar, ACM TIST 2024)

For each flagged gap we evaluate the spatiotemporal-anomaly graph dilation (STAGD) and apply the denial-based reasoning module (DRM). DRM distinguishes:
- benign denial (signal lost due to cloud, atmosphere, GPS occlusion)
- malicious denial (transponder deliberately spoofed or silenced)

Inputs are the gap envelope, the historical port-of-call signature for the MMSI, and the local AIS-density prior.

## Step 4: Pi-DPM physics-informed reconstruction (Sharma et al., ACM SIGSPATIAL GeoAnomalies 2025)

For each high-DRM gap we reconstruct the missing trajectory segment with the physics-informed diffusion probabilistic model. Pi-DPM samples plausible reconstructions consistent with the S-KBM (simplified kinematic bicycle model) and conditions on the surrounding AIS, the scene backbone tokens, and the SAR / optical evidence (detected wake, port-state features, neighboring traffic).

The reconstruction loss vs. the actual endpoint provides a calibrated spoofing likelihood.

## Step 5: dark-vessel probability and reasoning trace

The anomaly head produces a final probability per detected vessel via a calibrated logistic combination of:

- ``p_match``: 1 - cosine to nearest traj-CLIP track
- ``p_gap``: TGARD gap score, if any
- ``p_drm``: DRM denial-type confidence
- ``p_pidpm``: Pi-DPM reconstruction loss

with weights fit on the xView3 validation split. The viz module renders each detection with a reasoning trace listing which signal fired.

## End-to-end example

> "Vessel 41 detected at (24.612, 58.103) on Sentinel-1 GRD acquired 2026-02-14T03:17Z, classified as **tanker** (Prithvi-2 + linear probe, p=0.91). No traj-CLIP match within 800 m of any active AIS track (p_match = 0.87). MMSI 477394650 has a TGARD gap (score 0.93) starting 2026-02-14T01:45Z. DRM classifies the gap as **malicious denial** (p_drm = 0.71): port-of-call history is Strait of Hormuz only, no documented sanctions exemption. Pi-DPM reconstruction places the vessel at (24.604, 58.099) at 03:17Z with kinematic residual 0.21 m/s^2 (p_pidpm = 0.78). **Dark-vessel probability: 0.86.**"

This is the reasoning trace the Gradio Space prints next to the Folium map.
