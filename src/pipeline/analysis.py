"""AOI time-series analysis: processed outputs -> baseline z-scores and anomaly flags."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.detection.anomaly_detection import flag_anomalies
from src.detection.baseline import add_baseline
from src.features.temporal import add_gap_info, aggregate_by_date, load_observations

INDICES = ("ndvi", "ndmi")


def analyze_daily(daily: pd.DataFrame, indices: tuple[str, ...] = INDICES) -> pd.DataFrame:
    """Add gap info, seasonal baselines and anomaly flags to a one-row-per-date table.

    Shared by real data (``analyze_aoi``) and the clearly-labelled synthetic demo, so both go through
    identical statistical code. Baselines for each date use only EARLIER observations from OTHER years.
    """
    if daily.empty:
        return daily
    ts = config.get_settings().timeseries
    thresholds = config.load_thresholds()
    daily = add_gap_info(daily, ts.gap_days)
    for idx in indices:
        daily = add_baseline(daily, daily, idx, window_days=ts.baseline_window_days,
                             min_obs=ts.min_observations_for_baseline, sigma_floor=ts.sigma_floor,
                             min_years=ts.min_baseline_years, detrend=ts.detrend_baseline)
        daily = flag_anomalies(daily, f"{idx}_z", thresholds)
    return daily


def analyze_aoi(processed_root: Path, aoi_id: str, *, indices: tuple[str, ...] = INDICES) -> pd.DataFrame:
    """Build the per-date series for an AOI with seasonal baselines and anomaly levels."""
    daily = aggregate_by_date(load_observations(processed_root, aoi_id, indices), indices)
    return analyze_daily(daily, indices)
