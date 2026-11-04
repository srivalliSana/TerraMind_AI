# Scientific methodology

Everything here describes a **screening** method. It highlights places that are unusual for their season compared with earlier years.
It does not identify causes.

## 1. Input data (verified against NASA HLS v2.0 documentation)
- Reflectance: int16, scale factor 0.0001, fill value -9999. Fmask: uint8, fill 255 (tested before bits are decoded).
- Fmask bits: 1 cloud, 2 adjacent to cloud/shadow, 3 cloud shadow, 4 snow/ice, 5 water; 6-7 aerosol (not used for masking).
- Bands: use the narrow NIR on both sensors (HLSL30 B05, HLSS30 B8A) and SWIR1 (B06 / B11). See `configs/sensors.yaml`.
- Granule-level cloud coverage from CMR describes the whole ~110 km tile, not the AOI; AOI validity is computed from the mask.

## 2. Indices
| Index | Formula | Note |
|---|---|---|
| NDVI | (NIR - Red) / (NIR + Red) | greenness; saturates at dense canopy |
| NDMI | (NIR - SWIR1) / (NIR + SWIR1) | canopy water content proxy |
| SAVI | 1.5 (NIR - Red) / (NIR + Red + 0.5) | reduces soil background (L = 0.5) |
| EVI | needs blue band | implemented, not used (blue not downloaded) |

Division uses an epsilon of 1e-6. Reflectance outside -0.01..1.0 is rejected as implausible.

## 3. AOI-level baseline and anomaly
- Same-date granules are merged (pixel-count weighted).
- For each date, the baseline uses **only other years and only earlier dates**, within a +/-15-day circular window.
  Median and MAD (robust) with a sigma floor of 0.03; at least 5 observations and 2 baseline years.
- The seasonal slope in the baseline window is removed with a Theil-Sen estimator (`detrend_baseline`), otherwise the
  steep green-up/senescence slopes hide real changes.
- z = (observed - baseline) / sigma; only **declines** are graded: watch -1.5, elevated -2.0, high -3.0.
- Two consecutive anomalous observations count as "sustained".

## 4. Per-pixel affected area
- Each 30 m pixel is compared with the same pixel in earlier years (same grid required, +/-15 days, area-wide slope removed,
  per-pixel MAD floored at 0.05, at least 5 observations).
- A pixel is "below normal" at z <= -2.0 and counts as affected only inside an 8-connected patch of >= 9 pixels (~0.81 ha).
- The share expected by chance (about 2.3% under a normal distribution) is always shown beside the result.
- **Confounding:** a field with a different crop, planting date or fallow status in earlier years will look anomalous.
  With two baseline years this cannot be separated from stress. Treat patches as places to look.

## 5. Risk score
`score = 100 * (magnitude 50 + persistence 30 + extent 20) / available points`, where
magnitude = min(max(-z, 0)/3, 1) weighted NDVI 1.0 / NDMI 0.7, persistence is full at 4 consecutive observations, extent is full at 30% affected.
Categories: informational <= 25, watch <= 50, elevated <= 75, high above. If valid data < 30% or the baseline is missing,
**no score** is produced.
The weights and cut-offs are engineering choices, not fitted to outcomes.

**Confidence** (heuristic, not a probability): baseline depth 0.4, data quality 0.3, index agreement 0.15, freshness 0.15;
x0.9 if extent unknown; capped at 0.6 with fewer than 3 baseline years. Labels: high >= 0.70, medium >= 0.45.

## 6. What has and has not been checked
- Checked: arithmetic and logic on synthetic series and rasters (264 automated tests); a real Iowa run completed end to end.
- Not checked: agreement with field observations, yields, weather or drought indices; sensitivity of thresholds; other regions or crops;
  performance with long (5+ year) baselines. The thresholds file carries `validated: false` for this reason.
- Real Iowa result (2024-07-27, 2 baseline years): z = -0.61, informational, score 11.8, confidence medium (0.6 cap).
  3.2% of pixels were below z <= -2 against ~2.3% by chance; the larger patches were field-shaped, which suggests crop/planting differences.

## 7. Recommended validation before operational use
1. Collect labelled stress events (agency reports, field scouting, yield records) for several seasons and regions.
2. Extend history to at least 3-5 years per AOI.
3. Evaluate detection against those labels (precision, recall, lead time) and tune thresholds on a separate period.
4. Only then consider a learned model, with spatial/temporal hold-out splits.
