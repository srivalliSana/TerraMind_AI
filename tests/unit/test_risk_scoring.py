"""Risk scoring tests on SYNTHETIC analysed series. Expected scores are computed by hand (see comments)."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.alerts.risk_scoring import INSUFFICIENT_BASELINE, INSUFFICIENT_DATA, assess_risk, category_for
from src.config import load_thresholds

T = load_thresholds()


def level(z):
    if not np.isfinite(z):
        return "no_baseline"
    return "high" if z <= -3 else "elevated" if z <= -2 else "watch" if z <= -1.5 else "normal"


def series(z_ndvi, z_ndmi=0.0, run=1, run_ndmi=None, valid=0.9, base_n=10, n=1, dates=None):
    """One-row (or n-row) analysed series; the last row carries the given z-scores/runs."""
    dates = dates or pd.date_range("2024-07-01", periods=n, freq="5D")
    rows = []
    for i, d in enumerate(dates):
        last = i == len(dates) - 1
        zn, zm = (z_ndvi, z_ndmi) if last else (0.0, 0.0)
        rows.append({"date": d, "n_granules": 1, "valid_fraction": valid, "ndvi": 0.6, "ndmi": 0.3,
                     "ndvi_base_median": 0.8, "ndmi_base_median": 0.4, "ndvi_base_n": base_n, "ndmi_base_n": base_n,
                     "ndvi_z": zn, "ndmi_z": zm, "ndvi_z_level": level(zn), "ndmi_z_level": level(zm),
                     "ndvi_z_run": run if last else 0, "ndmi_z_run": (run if run_ndmi is None else run_ndmi) if last else 0})
    return pd.DataFrame(rows)


def test_hand_computed_elevated_without_extent():
    # magnitude: ndvi z=-3.5 -> sev 1 -> 50 ; ndmi z=-1 -> .333*.7=.233 (ignored, max used) ; persistence run1 -> 30*.25=7.5
    # score = 100*57.5/80 = 71.875 -> 71.9 -> elevated (<=75)
    a = assess_risk(series(-3.5, -1.0, run=1), T)
    assert a.score == pytest.approx(71.9, abs=0.05) and a.category == "elevated"
    assert a.components["magnitude"] == pytest.approx(50) and a.components["persistence"] == pytest.approx(7.5)
    assert a.components["extent"] is None and a.affected_fraction is None


def test_full_score_with_extent_is_high():
    a = assess_risk(series(-3.0, -3.0, run=4), T, affected_fraction=0.30)
    assert a.score == 100.0 and a.category == "high" and a.components["extent"] == pytest.approx(20)


def test_extent_scales_between_zero_and_full():
    a = assess_risk(series(-3.0, 0, run=4), T, affected_fraction=0.15)
    # 50 + 30 + 20*(0.15/0.30)=10 -> 90
    assert a.score == pytest.approx(90.0)


def test_watch_and_informational_cases():
    assert assess_risk(series(-1.5, 0, run=1), T).category == "watch"            # (25+7.5)/80 = 40.6
    low = assess_risk(series(-0.5, 0, run=0), T)                                 # 8.33/80 = 10.4
    assert low.category == "informational" and low.score == pytest.approx(10.4, abs=0.05)
    greener = assess_risk(series(+2.0, 1.0, run=0), T)                           # above normal is never risk
    assert greener.score == 0.0 and greener.category == "informational"


def test_moisture_only_anomaly_is_weighted_down():
    a = assess_risk(series(0.0, -3.0, run=1), T)       # 0.7*50=35 + 7.5 = 42.5 -> 53.1 -> elevated
    assert a.score == pytest.approx(53.1, abs=0.05) and a.category == "elevated"


def test_persistence_raises_score():
    one = assess_risk(series(-2.0, 0, run=1), T).score
    three = assess_risk(series(-2.0, 0, run=3), T).score
    assert three > one


def test_no_baseline_gives_no_score_not_zero():
    a = assess_risk(series(np.nan, np.nan, run=0, base_n=2), T)
    assert a.category == INSUFFICIENT_BASELINE and a.score is None and a.confidence is None
    assert "baseline" in a.evidence[0].lower()


def test_low_valid_coverage_gives_no_score():
    a = assess_risk(series(-3.0, -3.0, valid=0.2), T)
    assert a.category == INSUFFICIENT_DATA and a.score is None


def test_empty_series():
    a = assess_risk(pd.DataFrame(), T)
    assert a.category == INSUFFICIENT_DATA and a.score is None and a.observation_dates == []


def test_confidence_high_vs_low_and_label():
    strong = assess_risk(series(-3.0, -3.0, run=3, valid=0.9, base_n=10), T, affected_fraction=0.3)
    assert strong.confidence == pytest.approx(1.0) and strong.confidence_label == "high"
    weak = assess_risk(series(-3.0, 0.0, run=1, valid=0.31, base_n=5), T, as_of=date(2024, 9, 15))  # stale, thin, mixed
    assert weak.confidence_label == "low" and weak.confidence < strong.confidence
    assert any("Low confidence" in r for r in weak.recommendations)
    assert any("days old" in r for r in weak.recommendations)


def test_thin_baseline_years_cap_confidence():
    s = series(-3.0, -3.0, run=3, valid=0.9, base_n=12)
    s["ndvi_base_years"], s["ndmi_base_years"] = 2, 2
    a = assess_risk(s, T, affected_fraction=0.3)
    assert a.confidence == pytest.approx(0.60) and a.confidence_label == "medium"       # would be 1.0 with 3+ years
    assert any("only 2 year" in r for r in a.recommendations) and "in 2 year(s)" in a.evidence[0]
    s["ndvi_base_years"], s["ndmi_base_years"] = 5, 5
    assert assess_risk(s, T, affected_fraction=0.3).confidence == pytest.approx(1.0)


def test_unknown_extent_lowers_confidence():
    s = series(-3.0, -3.0, run=3)
    assert assess_risk(s, T).confidence < assess_risk(s, T, affected_fraction=0.3).confidence


def test_recommendations_single_vs_persistent():
    assert any("wait for the next clear" in r for r in assess_risk(series(-3.0, 0, run=1), T).recommendations)
    assert any("persistent" in r for r in assess_risk(series(-3.0, 0, run=3), T).recommendations)


def test_observation_dates_cover_the_anomalous_run():
    a = assess_risk(series(-3.0, 0, run=3, n=5), T)
    assert len(a.observation_dates) == 3 and a.observation_dates == sorted(a.observation_dates)


def test_output_is_json_serialisable_and_carries_caveat_and_unvalidated_flag():
    a = assess_risk(series(-3.0, -2.0, run=2), T)
    d = json.loads(json.dumps(a.to_dict()))
    assert d["validated"] is False and "cannot identify the cause" in d["caveats"][0]
    assert d["evidence"] and d["quality"]["valid_fraction"] == pytest.approx(0.9)


@pytest.mark.parametrize("score,cat", [(0, "informational"), (25, "informational"), (25.1, "watch"), (50, "watch"),
                                       (75, "elevated"), (75.1, "high"), (100, "high")])
def test_category_boundaries(score, cat):
    assert category_for(score, T) == cat
