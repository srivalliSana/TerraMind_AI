"""Index tests with hand-computed expected values (SYNTHETIC reflectances)."""
import numpy as np
import pytest

from src.indices.moisture import ndmi
from src.indices.vegetation import evi, ndvi, safe_divide, savi


def test_ndvi_known_values():
    nir = np.array([0.8, 0.6, 0.3])
    red = np.array([0.2, 0.3, 0.2])
    # (0.8-0.2)/(1.0)=0.6 ; (0.6-0.3)/0.9=1/3 ; (0.3-0.2)/0.5=0.2
    np.testing.assert_allclose(ndvi(nir, red), [0.6, 1 / 3, 0.2], rtol=1e-5)


def test_ndmi_known_values():
    # (0.5-0.1)/(0.6)=2/3 ; (0.4-0.4)/0.8=0
    np.testing.assert_allclose(ndmi(np.array([0.5, 0.4]), np.array([0.1, 0.4])), [2 / 3, 0.0], atol=1e-6)


def test_swir1_not_swir2_documented_difference():
    nir, swir1, swir2 = np.array([0.4]), np.array([0.2]), np.array([0.1])
    assert ndmi(nir, swir1)[0] != pytest.approx(ndmi(nir, swir2)[0])


def test_savi_known_value():
    # (0.8-0.2)*1.5/(0.8+0.2+0.5)=0.9/1.5=0.6
    assert savi(np.array([0.8]), np.array([0.2]))[0] == pytest.approx(0.6, rel=1e-5)
    with pytest.raises(ValueError):
        savi(np.array([0.8]), np.array([0.2]), soil_factor=2)


def test_evi_known_value():
    # 2.5*(0.5-0.1)/(0.5+0.6-7.5*0.05+1)=1.0/(2.1-0.375)=1/1.725
    assert evi(np.array([0.5]), np.array([0.1]), np.array([0.05]))[0] == pytest.approx(1 / 1.725, rel=1e-5)


def test_evi_out_of_range_becomes_nan():
    # denominator tiny positive -> huge EVI -> flagged NaN
    assert np.isnan(evi(np.array([0.9]), np.array([0.0]), np.array([0.3333]))[0])


def test_division_by_zero_and_near_zero_gives_nan():
    out = ndvi(np.array([0.0, 1e-9, 0.5]), np.array([0.0, 1e-9, 0.1]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])


def test_nan_inputs_propagate_and_inf_rejected():
    out = ndvi(np.array([np.nan, 0.5, np.inf]), np.array([0.1, np.nan, 0.1]))
    assert np.isnan(out).all()


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shapes differ"):
        ndvi(np.zeros((2, 2)), np.zeros((3, 3)))


def test_2d_shape_and_dtype_preserved():
    out = ndvi(np.full((4, 5), 0.6), np.full((4, 5), 0.2))
    assert out.shape == (4, 5) and out.dtype == np.float32
    assert np.allclose(out, 0.5)


def test_bounds_for_valid_reflectance():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000)
    out = ndvi(a, b)
    assert np.nanmin(out) >= -1 and np.nanmax(out) <= 1


def test_safe_divide_basic():
    np.testing.assert_allclose(safe_divide(np.array([1.0, 2.0]), np.array([2.0, 0.0])), [0.5, np.nan])
