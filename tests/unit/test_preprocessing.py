"""Scaling, Fmask decoding, quality metrics, grid checks and AOI clipping.
All arrays are small SYNTHETIC examples with hand-computed expected values."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.config import ProcessingSettings
from src.geospatial import aoi as A
from src.geospatial.alignment import GridMismatchError, assert_same_grid
from src.geospatial.raster_io import RasterError, read_band_clipped, read_meta, write_geotiff
from src.preprocessing.cloud_mask import build_valid_mask, decode_fmask
from src.preprocessing.quality_control import compute_quality
from src.preprocessing.scaling import scale_reflectance


# ---- scaling -------------------------------------------------------------
def test_scale_known_values():
    out = scale_reflectance(np.array([1000, 2500, 0], dtype="int16"))
    np.testing.assert_allclose(out, [0.1, 0.25, 0.0], rtol=1e-6)
    assert out.dtype == np.float32


def test_fill_becomes_nan_not_reflectance():
    out = scale_reflectance(np.array([-9999, 1000], dtype="int16"), valid_range=None)
    assert np.isnan(out[0]) and out[1] == pytest.approx(0.1)
    # unscaled, -9999 would be -0.9999; it must never survive as a number
    assert not np.any(out == pytest.approx(-0.9999))


def test_out_of_range_masked():
    out = scale_reflectance(np.array([-500, 5000, 10000, 12000], dtype="int16"), valid_range=(-0.01, 1.0))
    assert np.isnan(out[0])           # -0.05 below range
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)   # boundary kept
    assert np.isnan(out[3])           # 1.2 above range


def test_nonzero_offset_rejected_and_bad_scale():
    with pytest.raises(ValueError, match="ADD_OFFSET"):
        scale_reflectance(np.array([1], dtype="int16"), offset=0.1)
    with pytest.raises(ValueError):
        scale_reflectance(np.array([1], dtype="int16"), scale=0)


def test_scaling_rejects_non_numeric():
    with pytest.raises(TypeError):
        scale_reflectance(np.array(["a"]))


# ---- Fmask ---------------------------------------------------------------
# bit values: cloud=2, adjacent=4, shadow=8, snow=16, water=32, aerosol bits 64/128
FM = np.array([[0, 2, 4], [8, 16, 32], [64, 255, 66]], dtype=np.uint8)


def test_decode_each_bit():
    f = decode_fmask(FM)
    assert f.cloud.tolist() == [[False, True, False], [False, False, False], [False, False, True]]
    assert f.adjacent[0, 2] and f.shadow[1, 0] and f.snow_ice[1, 1] and f.water[1, 2]
    assert f.fill[2, 1] and f.fill.sum() == 1


def test_fill_255_does_not_trigger_other_flags():
    f = decode_fmask(np.array([[255]], dtype=np.uint8))
    assert f.fill[0, 0]
    assert not (f.cloud | f.adjacent | f.shadow | f.snow_ice | f.water)[0, 0]  # 255 has all bits set


def test_aerosol_bits_alone_do_not_mask():
    valid = build_valid_mask(np.array([[64, 128, 192]], dtype=np.uint8))
    assert valid.all()


def test_valid_mask_default_excludes_all_flags_and_fill():
    valid = build_valid_mask(FM)
    assert valid.tolist() == [[True, False, False], [False, False, False], [True, False, False]]


def test_valid_mask_options():
    v = build_valid_mask(FM, mask_water=False, mask_adjacent=False)
    assert v[1, 2] and v[0, 2]          # water and adjacent now allowed
    assert not v[0, 1] and not v[2, 1]  # cloud and fill still masked


def test_decode_requires_uint8():
    with pytest.raises(TypeError):
        decode_fmask(np.zeros((2, 2), dtype="int16"))


# ---- quality -------------------------------------------------------------
def test_quality_fractions_inside_aoi_only():
    fm = np.array([[0, 2], [255, 0]], dtype=np.uint8)
    inside = np.array([[True, True], [True, False]])      # bottom-right excluded
    flags = decode_fmask(fm)
    q = compute_quality(flags, build_valid_mask(fm), inside)
    assert q.aoi_pixels == 3
    assert q.valid_fraction == pytest.approx(1 / 3)       # only (0,0) usable
    assert q.cloud_fraction == pytest.approx(1 / 3)
    assert q.fill_fraction == pytest.approx(1 / 3) and q.observed_fraction == pytest.approx(2 / 3)


def test_quality_errors():
    flags = decode_fmask(np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="no pixels"):
        compute_quality(flags, np.ones((2, 2), bool), np.zeros((2, 2), bool))
    with pytest.raises(ValueError, match="same shape"):
        compute_quality(flags, np.ones((3, 3), bool), np.ones((2, 2), bool))


# ---- rasters: clipping and grid checks -----------------------------------
def make_raster(path, *, crs="EPSG:32615", x0=399960.0, y0=4700040.0, size=100, dtype="int16", fill=5):
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype=dtype, crs=crs,
                       transform=from_origin(x0, y0, 30, 30), nodata=-9999 if dtype == "int16" else 255) as d:
        d.write(np.full((1, size, size), fill, dtype=dtype))
    return path


def test_clip_reads_only_aoi_window(tmp_path):
    p = make_raster(tmp_path / "a.tif")
    # raster covers x 399960..402960 (E), y 4697040..4700040 (N) in UTM 15N (~ -93.9..-93.86 lon, 42.45..42.48 lat)
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:32615", "EPSG:4326", always_xy=True).transform(401400, 4698500)
    aoi = A.from_bbox(lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002)
    data, meta, inside = read_band_clipped(p, aoi)
    assert 0 < data.shape[0] < 100 and 0 < data.shape[1] < 100      # a window, not the full tile
    assert inside.shape == data.shape and inside.any()
    assert (data == 5).all()
    assert meta.shape == data.shape and meta.crs == "EPSG:32615"


def test_clip_outside_raster_raises(tmp_path):
    p = make_raster(tmp_path / "a.tif")
    with pytest.raises(RasterError, match="overlap"):
        read_band_clipped(p, A.from_bbox(10, 10, 10.01, 10.01))


def test_grid_check_accepts_identical_and_rejects_mismatch(tmp_path):
    a, b = make_raster(tmp_path / "a.tif"), make_raster(tmp_path / "b.tif")
    assert_same_grid(read_meta(a), read_meta(b))
    with pytest.raises(GridMismatchError, match="CRS"):
        assert_same_grid(read_meta(a), read_meta(make_raster(tmp_path / "c.tif", crs="EPSG:32616")))
    with pytest.raises(GridMismatchError, match="Shape"):
        assert_same_grid(read_meta(a), read_meta(make_raster(tmp_path / "d.tif", size=50)))
    with pytest.raises(GridMismatchError, match="grid"):
        assert_same_grid(read_meta(a), read_meta(make_raster(tmp_path / "e.tif", x0=399990.0)))


def test_write_geotiff_roundtrip_with_tags(tmp_path):
    arr = np.array([[0.1, np.nan], [0.5, 0.9]], dtype="float32")
    p = write_geotiff(tmp_path / "o" / "ndvi.tif", arr, crs="EPSG:32615", transform=from_origin(0, 60, 30, 30),
                      tags={"source": "synthetic", "index": "NDVI"})
    with rasterio.open(p) as s:
        back = s.read(1)
        assert s.crs.to_string() == "EPSG:32615" and s.tags()["source"] == "synthetic"
    np.testing.assert_array_equal(np.isnan(back), np.isnan(arr))


def test_processing_settings_validate_range():
    with pytest.raises(ValueError):
        ProcessingSettings(reflectance_valid_range=(1.0, 0.0))
