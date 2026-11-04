"""Per-pixel baseline / z-score / patch tests on tiny SYNTHETIC arrays with hand-computed answers."""
import numpy as np
import pytest

from src.features.spatial import (
    affected_area,
    expected_fraction_by_chance,
    pixel_baseline,
    pixel_zscores,
    signed_day_offsets,
)

DX = np.array([-10, -8, -3, 2, 5, 9], dtype=float)


def stack_with_trend(offset, slope=0.01, base=0.8):
    return np.stack([base + slope * d + offset for d in DX]).astype("float32")


def test_signed_offsets_wrap_year_end():
    np.testing.assert_array_equal(signed_day_offsets(np.array([360, 5, 10]), 3), [-8, 2, 7])
    assert signed_day_offsets(np.array([200]), 200)[0] == 0


def test_detrended_level_recovers_trend_and_per_pixel_offsets():
    offset = np.linspace(-0.1, 0.1, 12, dtype="float32").reshape(3, 4)          # each pixel has its own normal level
    b = pixel_baseline(stack_with_trend(offset), DX, min_obs=5, sigma_floor=0.05, detrend=True)
    assert b.area_slope == pytest.approx(0.01, abs=1e-6)
    np.testing.assert_allclose(b.level, 0.8 + offset, atol=1e-5)                 # level on the target day, per pixel
    assert np.allclose(b.sigma, 0.05)                                            # perfect trend -> spread at the floor


def test_without_detrend_slope_leaks_into_level():
    plain = pixel_baseline(stack_with_trend(np.zeros((2, 2), "float32")), DX, detrend=False)
    assert plain.area_slope == 0.0
    assert plain.level[0, 0] == pytest.approx(0.8 + 0.01 * np.median(DX), abs=1e-5)   # 0.795, not 0.8


def test_pixels_with_too_few_valid_observations_are_nan():
    s = stack_with_trend(np.zeros((2, 2), "float32"))
    s[:2, 0, 0] = np.nan                                                          # pixel (0,0): 4 valid < 5
    b = pixel_baseline(s, DX, min_obs=5)
    assert np.isnan(b.level[0, 0]) and np.isnan(b.sigma[0, 0]) and b.count[0, 0] == 4
    assert np.isfinite(b.level[1, 1])


def test_sigma_uses_mad_when_above_floor():
    vals = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.7], dtype="float32").reshape(6, 1, 1)
    b = pixel_baseline(vals, np.zeros(6), min_obs=5, sigma_floor=0.01, detrend=False)
    # median(0.5,0.6,0.7,0.7,0.8,0.9)=0.7 ; |dev| = .2,.1,0,0,.1,.2 -> median .1 -> sigma .14826
    assert b.level[0, 0] == pytest.approx(0.7) and b.sigma[0, 0] == pytest.approx(0.14826, rel=1e-4)


def test_zscores_and_nan_propagation():
    s = stack_with_trend(np.zeros((1, 3), "float32"))
    b = pixel_baseline(s, DX, sigma_floor=0.05)
    target = np.array([[0.8, 0.7, np.nan]], dtype="float32")
    z = pixel_zscores(target, b)
    assert z[0, 0] == pytest.approx(0.0, abs=1e-4) and z[0, 1] == pytest.approx(-2.0, abs=1e-3) and np.isnan(z[0, 2])


def test_shape_validation():
    with pytest.raises(ValueError, match="observations"):
        pixel_baseline(np.zeros((4, 4)), np.zeros(4))
    with pytest.raises(ValueError, match="one entry"):
        pixel_baseline(np.zeros((3, 2, 2)), np.zeros(4))


def test_patch_filter_keeps_blocks_and_drops_isolated_pixels():
    z = np.zeros((10, 10), dtype="float32")
    z[1:4, 1:4] = -3                     # 3x3 block = 9 px
    z[7, 7] = -5                         # isolated pixel
    z[8, 1:3] = -3                       # 2-pixel sliver
    a = affected_area(z, -2.0, min_patch_pixels=9)
    assert (a.raw_pixels, a.pixels, a.patches, a.largest_patch) == (12, 9, 1, 9)
    assert a.mask[2, 2] and not a.mask[7, 7] and not a.mask[8, 1]
    assert a.fraction == pytest.approx(9 / 100)


def test_diagonal_chain_counts_as_one_patch_8_connectivity():
    z = np.zeros((9, 9), dtype="float32")
    for i in range(9):
        z[i, i] = -4
    a = affected_area(z, -2.0, min_patch_pixels=9)
    assert a.patches == 1 and a.pixels == 9


def test_nan_pixels_are_not_evaluated():
    z = np.full((4, 4), np.nan, dtype="float32")
    z[0, :2] = [-3, 0]
    a = affected_area(z, -2.0, min_patch_pixels=1)
    assert a.evaluated == 2 and a.pixels == 1 and a.fraction == pytest.approx(0.5)


def test_no_affected_pixels():
    a = affected_area(np.zeros((5, 5), "float32"), -2.0, 4)
    assert a.pixels == 0 and a.patches == 0 and a.fraction == 0.0


def test_expected_by_chance_is_gaussian_tail():
    assert expected_fraction_by_chance(-2.0) == pytest.approx(0.02275, abs=1e-4)
    assert expected_fraction_by_chance(-1.5) == pytest.approx(0.0668, abs=1e-3)
