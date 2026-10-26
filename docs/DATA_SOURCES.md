# Data Sources

## NASA HLS v2.0 (primary)

Source of truth: **HLS User Guide V2.0** (LP DAAC), <https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf>.
Verified facts are recorded in `configs/sensors.yaml`.

| Product | Sensor | Resolution | Layers |
|---|---|---|---|
| HLSL30 | Landsat 8/9 OLI | 30 m, MGRS tiles | B01–B07, B09 (TOA), B10–B11 (TOA thermal BT), Fmask, angles |
| HLSS30 | Sentinel-2 MSI | 30 m, MGRS tiles | B01–B08, B8A, B09–B10 (TOA), B11–B12, Fmask, angles |

Each layer is a separate Cloud-Optimized GeoTIFF, so only the needed layers are downloaded.
Granule names look like `HLS.L30.T17SLU.2020209T155956.v2.0.B05.tif`; the timestamp is a
unique identifier and **not** an exact sensing time over the tile.

### Band mapping used by TerraMind

| Quantity | HLSL30 | HLSS30 |
|---|---|---|
| Red | B04 | B04 |
| NIR (narrow) | B05 | **B8A** (not B08) |
| SWIR1 | B06 | B11 |
| SWIR2 | B07 | B12 |

Using narrow NIR on both sensors keeps NDVI/NDMI comparable. HLS applies BRDF normalisation
and a bandpass adjustment so S30 resembles L30, but small residual sensor differences remain
and are a documented limitation of combining the two.

### Scaling and fill

- Reflectance bands: `int16`, scale factor **0.0001**, fill value **−9999** (must become NoData, never reflectance).
- The guide lists `ADD_OFFSET` as a per-granule metadata attribute without a value. TerraMind reads
  scale/offset from granule metadata and warns if absent; it never assumes silently.
- Fmask: `uint8`, fill value **255**.

### Fmask bits (Table 9, bit 0 = least significant)

| Bit | Meaning |
|---|---|
| 0 | Cirrus — reserved, unused in v2.0 |
| 1 | Cloud |
| 2 | Adjacent to cloud/shadow (5-pixel dilation) |
| 3 | Cloud shadow |
| 4 | Snow/ice |
| 5 | Water |
| 6–7 | Aerosol level (not used for masking by default) |

Caveats from the guide: bits 1–5 may not be mutually exclusive after resampling to 30 m, and a
value of 255 (fill) has every bit set, so fill must be tested before decoding bits. The aerosol
bit-pair ordering is written ambiguously in Table 9 versus the Appendix A example; since it is
not used for masking, it is deliberately not interpreted until clarified.

### Access

- **Catalog search**: public, no credentials (NASA CMR via `earthaccess`).
- **Downloads**: need NASA Earthdata Login (token preferred). See README for setup.

## Planned adapters (not implemented)

Sentinel-2 via STAC, Sentinel-1 SAR, MODIS/VIIRS, weather/precipitation. Mixing sensors or
resolutions will be documented per adapter before use.
