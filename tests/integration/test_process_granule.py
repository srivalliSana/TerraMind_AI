"""End-to-end processing on SYNTHETIC granules (not real satellite data); runs offline.

Known inputs: red=0.1, NIR=0.5, SWIR1=0.25  ->  NDVI = 0.4/0.6 = 2/3, NDMI = 0.25/0.75 = 1/3.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from src.data.manifest import FileRecord, GranuleManifest
from src.geospatial import aoi as A
from src.pipeline.jobs import ProcessingError, process_granule
from src.storage.cache import sha256_file

GID = "HLS.L30.T15TVG.2024186T165841.v2.0"
SIZE, X0, Y0 = 200, 399960.0, 4700040.0
DN = {"B04": 1000, "B05": 5000, "B06": 2500}   # reflectance 0.1 / 0.5 / 0.25


def _write(path, arr, dtype, nodata, scale=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1, dtype=dtype, crs="EPSG:32615",
                       transform=from_origin(X0, Y0, 30, 30), nodata=nodata) as d:
        d.write(arr, 1)
        if scale:
            d.scales = (scale,)
            d.offsets = (0.0,)


def build(tmp_path, cloud_cols=100, fill_all=False, scale=0.0001, drop=None, catalog_scale=0.0001):
    raw = tmp_path / "raw"
    fm = np.zeros((SIZE, SIZE), dtype="uint8")
    fm[:, :cloud_cols] = 2                      # cloud bit in the left part
    if fill_all:
        fm[:] = 255
    recs = {}
    layers = {"B04": None, "B05": None, "B06": None, "Fmask": fm}
    for layer, arr in layers.items():
        if layer == drop:
            continue
        p = raw / "nasa_hls" / "HLSL30" / "T15TVG" / GID / f"{GID}.{layer}.tif"
        if layer == "Fmask":
            _write(p, fm, "uint8", 255)
        else:
            full = np.full((SIZE, SIZE), DN[layer], dtype="int16")
            if fill_all:
                full[:] = -9999
            _write(p, full, "int16", -9999, scale=scale)
        recs[layer] = FileRecord(layer, "https://example.test", p.relative_to(raw).as_posix(),
                                 p.stat().st_size, sha256_file(p), "2024-01-01T00:00:00+00:00")
    m = GranuleManifest(GID, "nasa_hls", "HLSL30", "T15TVG", datetime(2024, 7, 4, tzinfo=timezone.utc).isoformat(),
                        {"scale_factor": catalog_scale, "cloud_cover": 28, "spatial_coverage": 99}, recs)
    return m, raw


@pytest.fixture
def aoi():
    lon, lat = Transformer.from_crs("EPSG:32615", "EPSG:4326", always_xy=True).transform(X0 + 100 * 30, Y0 - 100 * 30)
    return A.from_bbox(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01, name="synthetic")


def test_end_to_end_values_masking_and_outputs(tmp_path, aoi):
    m, raw = build(tmp_path)
    res = process_granule(m, aoi, raw_dir=raw, out_root=tmp_path / "out")

    with rasterio.open(res.output_dir / "ndvi.tif") as s:
        ndvi = s.read(1)
        assert s.crs.to_string() == "EPSG:32615" and s.tags()["index"] == "NDVI"
    with rasterio.open(res.output_dir / "ndmi.tif") as s:
        ndmi = s.read(1)
    vm = rasterio.open(res.output_dir / "valid_mask.tif").read(1).astype(bool)

    clear = vm
    assert clear.any() and (~vm & np.isfinite(ndvi)).sum() == 0             # NaN wherever mask says unusable
    np.testing.assert_allclose(ndvi[clear], 2 / 3, rtol=1e-4)
    np.testing.assert_allclose(ndmi[clear], 1 / 3, rtol=1e-4)
    assert res.stats["ndvi"]["mean"] == pytest.approx(2 / 3, rel=1e-4)
    assert res.stats["ndvi"]["std"] < 1e-5
    # cloud covers roughly the left half of the AOI
    assert 0.35 < res.quality.cloud_fraction < 0.65
    assert res.quality.valid_fraction == pytest.approx(1 - res.quality.cloud_fraction, abs=1e-9)  # only cloud bit is set
    assert res.quality.observed_fraction == 1.0 and res.quality.fill_fraction == 0.0
    assert res.usable_for_analysis
    summary = json.loads((res.output_dir / "summary.json").read_text())
    assert summary["layers"]["nir"] == "B05" and "caveat" in summary and len(summary["input_sha256"]) == 4


def test_cloudy_aoi_flagged_not_usable(tmp_path, aoi):
    m, raw = build(tmp_path, cloud_cols=150)         # AOI almost entirely cloud
    res = process_granule(m, aoi, raw_dir=raw, out_root=tmp_path / "out")
    assert res.quality.valid_fraction < 0.30 and not res.usable_for_analysis
    assert any("excluded" in w for w in res.warnings)


def test_aoi_with_no_observations_fails_clearly(tmp_path, aoi):
    m, raw = build(tmp_path, fill_all=True)
    with pytest.raises(ProcessingError, match="no observed pixels"):
        process_granule(m, aoi, raw_dir=raw, out_root=tmp_path / "out")


def test_missing_layer_requires_download(tmp_path, aoi):
    m, raw = build(tmp_path, drop="B06")
    with pytest.raises(ProcessingError, match="not downloaded"):
        process_granule(m, aoi, raw_dir=raw, out_root=tmp_path / "out")


def test_tampered_file_is_rejected(tmp_path, aoi):
    m, raw = build(tmp_path)
    (raw / m.files["B05"].path).write_bytes(b"tampered")
    with pytest.raises(ProcessingError, match="does not match its manifest"):
        process_granule(m, aoi, raw_dir=raw, out_root=tmp_path / "out")


def test_scale_disagreement_between_file_and_catalog_is_an_error(tmp_path, aoi):
    m, raw = build(tmp_path, scale=0.0001, catalog_scale=0.001)
    with pytest.raises(ProcessingError, match="Scale mismatch"):
        process_granule(m, aoi, raw_dir=raw, out_root=tmp_path / "out")


def test_aoi_outside_raster_fails(tmp_path):
    m, raw = build(tmp_path)
    from src.geospatial.raster_io import RasterError
    with pytest.raises(RasterError, match="overlap"):
        process_granule(m, A.from_bbox(10, 10, 10.01, 10.01), raw_dir=raw, out_root=tmp_path / "out")
