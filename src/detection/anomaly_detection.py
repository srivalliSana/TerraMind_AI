"""Turn z-scores into graded, persistence-aware anomaly flags.

Only DECLINES are flagged (z below zero): above-normal greenness is not stress.
A single low observation is labelled transient; it becomes ``sustained`` only after
``min_persistence_obs`` consecutive anomalous observations, which separates
short-term noise (haze, missed cloud edges) from persistent change.
Thresholds come from ``configs/thresholds.yaml`` and are NOT validated universal rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Thresholds

LEVELS = ["normal", "watch", "elevated", "high"]   # ordered by severity
NO_BASELINE = "no_baseline"


def classify_z(z: float, t: Thresholds) -> str:
    """Map a z-score to an anomaly level using configured (negative) thresholds."""
    if not np.isfinite(z):
        return NO_BASELINE
    a = t.anomaly
    if z <= a["zscore_high"]:
        return "high"
    if z <= a["zscore_elevated"]:
        return "elevated"
    if z <= a["zscore_watch"]:
        return "watch"
    return "normal"


def flag_anomalies(df: pd.DataFrame, z_col: str, t: Thresholds) -> pd.DataFrame:
    """Add ``<z_col>_level``, ``<z_col>_run`` (consecutive anomalous observations) and ``<z_col>_sustained``."""
    out = df.copy()
    levels = [classify_z(z, t) for z in out[z_col]]
    anomalous = np.array([lv in ("watch", "elevated", "high") for lv in levels])
    run, count = [], 0
    for is_anom, lv in zip(anomalous, levels):
        if lv == NO_BASELINE:
            count = 0            # unknown breaks the chain: we cannot claim persistence across missing baselines
        else:
            count = count + 1 if is_anom else 0
        run.append(count)
    min_obs = int(t.anomaly.get("min_persistence_obs", 2))
    out[f"{z_col}_level"] = levels
    out[f"{z_col}_run"] = run
    out[f"{z_col}_sustained"] = np.array(run) >= min_obs
    return out
