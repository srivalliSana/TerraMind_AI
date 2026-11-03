"""Time-series, baseline and anomaly tests on SYNTHETIC series with hand-computed answers."""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_thresholds
from src.detection.anomaly_detection import NO_BASELINE, classify_z, flag_anomalies
from src.detection.baseline import add_baseline
from src.features.temporal import add_gap_info, aggregate_by_date, load_observations, rolling_median

T = load_thresholds()   # watch -1.5, elevated -2.0, high -3.0, persistence 2


def hist(dates, vals):
    return pd.DataFrame({"date": pd.to_datetime(dates), "ndvi": vals})


H5 = hist(["2022-07-01", "2022-07-05", "2022-07-09", "2022-07-13", "2022-07-17"], [0.76, 0.78, 0.80, 0.82, 0.84])


def cur(date, v):
    return pd.DataFrame({"date": pd.to_datetime([date]), "ndvi": [v]})


# ---- baseline -------------------------------------------------------------
def test_baseline_median_sigma_floor_and_z():
    r = add_baseline(cur("2024-07-10", 0.74), H5, "ndvi", min_obs=5, sigma_floor=0.03).iloc[0]
    assert r["ndvi_base_n"] == 5 and r["ndvi_base_median"] == pytest.approx(0.80)
    # raw 1.4826*MAD = 1.4826*0.02 = 0.02965 < floor -> 0.03 ; z = (0.74-0.80)/0.03 = -2
    assert r["ndvi_base_sigma"] == pytest.approx(0.03)
    assert r["ndvi_z"] == pytest.approx(-2.0)


def test_baseline_counts_distinct_years():
    two_years = pd.concat([H5.iloc[:3], hist(["2023-07-05", "2023-07-09"], [0.8, 0.8])])
    r = add_baseline(cur("2024-07-10", 0.8), two_years, "ndvi", min_obs=5).iloc[0]
    assert r["ndvi_base_n"] == 5 and r["ndvi_base_years"] == 2


def test_baseline_uses_mad_when_larger_than_floor():
    h = hist(["2022-07-01", "2022-07-05", "2022-07-09", "2022-07-13", "2022-07-17"], [0.5, 0.6, 0.7, 0.8, 0.9])
    r = add_baseline(cur("2024-07-10", 0.4), h, "ndvi", min_obs=5, sigma_floor=0.03).iloc[0]
    # median 0.7 ; |dev| = .2,.1,0,.1,.2 -> MAD .1 -> sigma .14826 ; z = -0.3/.14826
    assert r["ndvi_base_sigma"] == pytest.approx(0.14826)
    assert r["ndvi_z"] == pytest.approx(-0.3 / 0.14826, rel=1e-4)


def test_insufficient_history_gives_nan_not_a_number():
    r = add_baseline(cur("2024-07-10", 0.5), H5.iloc[:4], "ndvi", min_obs=5).iloc[0]
    assert r["ndvi_base_n"] == 4 and np.isnan(r["ndvi_z"]) and np.isnan(r["ndvi_base_median"])


def test_no_future_or_same_year_leakage():
    leaky = pd.concat([H5.iloc[:4], hist(["2025-07-10"], [0.80]),      # future year
                       hist(["2024-07-08"], [0.80])])                  # same year as target
    r = add_baseline(cur("2024-07-10", 0.5), leaky, "ndvi", min_obs=5).iloc[0]
    assert r["ndvi_base_n"] == 4          # neither the future nor same-year point counted


def test_seasonal_window_excludes_far_dates():
    far = hist(["2022-06-01", "2022-07-05", "2022-07-09", "2022-07-13", "2022-07-17"], [0.1, 0.78, 0.80, 0.82, 0.84])
    r = add_baseline(cur("2024-07-10", 0.8), far, "ndvi", min_obs=4, window_days=15).iloc[0]
    assert r["ndvi_base_n"] == 4 and r["ndvi_base_median"] == pytest.approx(0.81)


def test_window_wraps_around_new_year():
    h = hist(["2022-12-26", "2022-12-28", "2022-12-30", "2023-01-03", "2023-01-05"], [0.3] * 5)
    r = add_baseline(cur("2024-01-02", 0.3), h, "ndvi", min_obs=5, window_days=15).iloc[0]
    assert r["ndvi_base_n"] == 5


def test_nan_current_value_gives_nan_z():
    r = add_baseline(cur("2024-07-10", np.nan), H5, "ndvi", min_obs=5).iloc[0]
    assert np.isnan(r["ndvi_z"]) and r["ndvi_base_n"] == 5


# ---- anomaly classification ----------------------------------------------
@pytest.mark.parametrize("z,level", [(0.5, "normal"), (-1.49, "normal"), (-1.5, "watch"), (-1.99, "watch"),
                                     (-2.0, "elevated"), (-2.99, "elevated"), (-3.0, "high"), (-8, "high"),
                                     (np.nan, NO_BASELINE)])
def test_classify_z_boundaries(z, level):
    assert classify_z(z, T) == level


def test_positive_anomalies_never_flagged():
    assert classify_z(5.0, T) == "normal"


def test_persistence_distinguishes_noise_from_sustained():
    df = pd.DataFrame({"z": [-0.1, -1.6, -2.1, 0.0, -1.6, np.nan, -3.5]})
    out = flag_anomalies(df, "z", T)
    assert out["z_level"].tolist() == ["normal", "watch", "elevated", "normal", "watch", NO_BASELINE, "high"]
    assert out["z_run"].tolist() == [0, 1, 2, 0, 1, 0, 1]
    assert out["z_sustained"].tolist() == [False, False, True, False, False, False, False]


# ---- observation assembly -------------------------------------------------
def write_summary(root, aoi, gid, sensor, when, ndvi_med, n, valid, usable=True):
    d = root / aoi / gid
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "granule_id": gid, "sensor": sensor, "acquired": when, "usable_for_analysis": usable,
        "quality": {"valid_fraction": valid},
        "index_stats": {"ndvi": {"median": ndvi_med, "std": 0.05, "count": n},
                        "ndmi": {"median": 0.3, "std": 0.05, "count": n}}}))


def test_load_and_merge_same_day_granules(tmp_path):
    write_summary(tmp_path, "aoi_x", "g1", "HLSL30", "2024-07-04T17:00:00+00:00", 0.80, 1000, 0.8)
    write_summary(tmp_path, "aoi_x", "g2", "HLSS30", "2024-07-04T10:00:00+00:00", 0.60, 3000, 0.4)   # same date
    write_summary(tmp_path, "aoi_x", "g3", "HLSS30", "2024-07-09T10:00:00+00:00", 0.70, 500, 0.9)
    write_summary(tmp_path, "aoi_x", "g4", "HLSS30", "2024-07-14T10:00:00+00:00", 0.1, 500, 0.1, usable=False)
    obs = load_observations(tmp_path, "aoi_x")
    assert len(obs) == 4
    daily = aggregate_by_date(obs)
    assert len(daily) == 2                                   # merged same day; unusable dropped
    first = daily.iloc[0]
    # weighted by pixel count: (0.8*1000 + 0.6*3000)/4000 = 0.65 ; NOT a simple mean (0.70)
    assert first["ndvi"] == pytest.approx(0.65) and first["n_granules"] == 2
    assert first["sensors"] == "HLSL30+HLSS30"
    assert first["valid_fraction"] == pytest.approx((0.8 * 1000 + 0.4 * 3000) / 4000)


def test_load_skips_corrupt_summary_and_handles_empty(tmp_path):
    (tmp_path / "aoi_y" / "bad").mkdir(parents=True)
    (tmp_path / "aoi_y" / "bad" / "summary.json").write_text("{not json")
    assert load_observations(tmp_path, "aoi_y").empty
    assert load_observations(tmp_path, "missing_aoi").empty


def test_gap_flags_and_rolling_median():
    d = pd.DataFrame({"date": pd.to_datetime(["2024-07-01", "2024-07-05", "2024-08-10"]), "ndvi": [0.8, 0.2, 0.8]})
    g = add_gap_info(d, gap_days=20)
    assert g["days_since_prev"].tolist()[1:] == [4, 36] and g["gap"].tolist() == [False, False, True]
    assert rolling_median(d, "ndvi", 3).tolist() == [0.8, 0.5, 0.8]      # median of (0.8,0.2,0.8)=0.8 at the end
