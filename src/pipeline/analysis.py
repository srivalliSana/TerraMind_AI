"""AOI time-series analysis: processed outputs -> baseline z-scores and anomaly flags."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.detection.anomaly_detection import flag_anomalies
from src.detection.baseline import add_baseline
from src.features.temporal import add_gap_info, aggregate_by_date, load_observations

INDICES = ("ndvi", "ndmi")


def analyze_aoi(processed_root: Path, aoi_id: str, *, indices: tuple[str, ...] = INDICES) -> pd.DataFrame:
    """Build the per-date series for an AOI with seasonal baselines and anomaly levels.

    Baselines for each date use only EARLIER observations from OTHER years, so the first year of data has
    no baseline (NaN z-scores, level ``no_baseline``) by construction.
    """
    ts = config.get_settings().timeseries
    thresholds = config.load_thresholds()
    daily = aggregate_by_date(load_observations(processed_root, aoi_id, indices), indices)
    if daily.empty:
        return daily
    daily = add_gap_info(daily, ts.gap_days)
    for idx in indices:
        daily = add_baseline(daily, daily, idx, window_days=ts.baseline_window_days,
                             min_obs=ts.min_observations_for_baseline, sigma_floor=ts.sigma_floor)
        daily = flag_anomalies(daily, f"{idx}_z", thresholds)
    return daily
