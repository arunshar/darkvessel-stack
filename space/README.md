---
title: DarkVesselNet
emoji: "\U0001F6F0"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multi-modal RS stack for dark vessel detection (S1+S2+AIS).
tags:
  - remote-sensing
  - sentinel-1
  - sentinel-2
  - ais
  - dark-vessel
  - geospatial
  - prithvi-2
  - clay-v1
  - satmae
  - dofa
  - satlasnet
  - remoteclip
  - xview3
  - tgard
  - pi-dpm
  - earth-observation
  - sar
---

# DarkVesselNet

Multi-modal remote sensing stack for dark vessel detection. Sentinel-1 SAR plus Sentinel-2 optical plus AIS, fused through a geospatial foundation model backbone (Prithvi-2 / Clay v1 / SatMAE++ / DOFA / SatlasNet / RemoteCLIP), with physics-informed anomaly reasoning via TGARD and Pi-DPM.

Click an area of interest on the Mapbox globe (Strait of Hormuz, South China Sea, Galapagos EEZ, Sea of Japan, Gulf of Oman) and the Space pulls the latest cloud-free Sentinel-2 chip with the nearest Sentinel-1 GRD scene from Microsoft Planetary Computer, runs the seven canonical EO heads, joins to AIS, and renders a Folium overlay of candidate dark vessels with a per-detection reasoning trace.

See the [GitHub repository](https://github.com/arunshar/darkvessel-stack) for the full pipeline, training recipes, and xView3-SAR leaderboard reproduction.
