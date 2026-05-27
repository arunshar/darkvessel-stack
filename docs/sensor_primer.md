# Remote sensing sensor primer

Quick reference for the four sensor modalities DarkVesselNet fuses.

## Optical, multispectral (Sentinel-2)

- 13 bands spanning 443 to 2190 nm.
- Three native ground sampling distances: 10 m (B02, B03, B04, B08), 20 m (B05, B06, B07, B8A, B11, B12), 60 m (B01, B09, B10).
- 5-day equator revisit with Sentinel-2A + 2B; 2-3 days at mid-latitudes.
- L1C is top-of-atmosphere radiance; L2A is surface reflectance after Sen2Cor.
- Useful indices for maritime: NDWI (water), MNDWI (turbid water), NDVI (kelp, mangroves), Sun-glint index (specular returns).

## SAR (Sentinel-1, C-band)

- 5.405 GHz, dual-pol VV / VH (over ocean we use VV primarily).
- Interferometric Wide-swath (IW) mode: 250 km swath, 5 m x 20 m pixel.
- 6-day repeat (Sentinel-1A) or 12-day with one satellite, day or night, all-weather.
- Two product levels we use:
  - **GRD** (Ground-Range Detected): amplitude only, multi-looked, ground-projected.
  - **SLC** (Single Look Complex): preserves phase for interferometry.
- Speckle is fully developed for L >= 4 looks; we apply a 5x5 Lee filter pre-backbone.
- Vessel signal: bright point targets against the near-specular (dark) ocean. The signal-to-clutter ratio degrades near coast (layover, foreshortening) and in high sea state (Bragg scattering brightens the sea).

## Thermal (Landsat-9 TIRS)

- Two bands: 10.8 um and 12.0 um.
- 100 m native, resampled to 30 m in delivered products.
- 16-day repeat (Landsat-9 alone); 8-day with Landsat-8 paired.
- Use case: engine and exhaust signatures, sometimes wakes against thermal contrast; supplementary only.

## High-resolution optical (PlanetScope, SkySat, WorldView)

- PlanetScope: ~3 m, RGB-NIR, daily near-global revisit, free for research via NICFI / Planet Labs.
- SkySat: 0.5 m panchromatic / 1 m multispectral, tasking only.
- WorldView-3 / -4: 0.31 m, tasking only.
- Use case: classify vessel type once SAR has flagged a candidate.

## Common preprocessing concerns and where they live in this repo

| Concern | Module |
| --- | --- |
| Speckle | `darkvessel.sar.speckle.lee_filter` |
| SAR sigma0 calibration | `darkvessel.sar.calibration` |
| Terrain correction | `darkvessel.sar.terrain_correction` |
| Cloud masking | `darkvessel.optical.cloud_mask.s2cloudless_mask` |
| Atmospheric correction (L1C only) | `darkvessel.optical.atm_correction` |
| Sun-glint suppression | `darkvessel.optical.sun_glint` |
| Co-registration across sensors | `darkvessel.fusion.coregistration.distortion_aware_coregister` |

## Coordinate systems

DarkVesselNet works in EPSG:4326 (WGS84 lat / lon) externally and in the native sensor frame internally. Co-registration is *not* done by reprojecting to a common UTM grid (that introduces resampling artifacts at the swath edge); we sample the source through the source sensor model, exactly as in `sat-splat-distort`. This is the dissertation contribution made operational.
