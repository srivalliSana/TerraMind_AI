"""Seasonal baselines and z-scores.

For an observation on day-of-year D in year Y, the baseline is built from observations in OTHER years
(strictly earlier dates only: no future leakage) whose day-of-year lies within +/- ``window_days`` of D,
with wrap-around at year end. Centre = median, spread = 1.4826 * MAD (a robust sigma), floored at
``sigma_floor`` so unusually stable history cannot inflate z-scores.

With too little history the z-score is left NaN and flagged, never invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def _circular_doy_distance(doy: np.ndarray, target: int) -> np.ndarray:
    d = np.abs(doy - target)
    return np.minimum(d, 365 - d)


def add_baseline(
    current: pd.DataFrame,
    history: pd.DataFrame,
    col: str,
    *,
    window_days: int = 15,
    min_obs: int = 5,
    sigma_floor: float = 0.03,
) -> pd.DataFrame:
    """Return ``current`` plus ``<col>_base_median``, ``<col>_base_sigma``, ``<col>_base_n``, ``<col>_z``.

    ``current`` and ``history`` need ``date`` and ``col``. ``history`` may include the current series; only
    observations from different calendar years and earlier than each target date are used.
    """
    hist = history[["date", col]].dropna()
    h_dates = pd.to_datetime(hist["date"])
    h_year = h_dates.dt.year.to_numpy()
    h_doy = h_dates.dt.dayofyear.to_numpy()
    h_val = hist[col].to_numpy(dtype=float)
    h_time = h_dates.to_numpy()

    meds, sigs, ns, zs, yrs = [], [], [], [], []
    for date, x in zip(current["date"], current[col].to_numpy(dtype=float)):
        d = pd.Timestamp(date)
        mask = (h_year != d.year) & (h_time < np.datetime64(d)) & (
            _circular_doy_distance(h_doy, d.dayofyear) <= window_days)
        vals = h_val[mask]
        n = int(vals.size)
        yrs.append(int(np.unique(h_year[mask]).size))
        if n >= min_obs:
            med = float(np.median(vals))
            sig = max(MAD_TO_SIGMA * float(np.median(np.abs(vals - med))), sigma_floor)
            z = (x - med) / sig if np.isfinite(x) else np.nan
        else:
            med = sig = z = np.nan
        meds.append(med), sigs.append(sig), ns.append(n), zs.append(z)

    out = current.copy()
    out[f"{col}_base_median"], out[f"{col}_base_sigma"] = meds, sigs
    out[f"{col}_base_n"], out[f"{col}_z"] = ns, zs
    out[f"{col}_base_years"] = yrs     # distinct calendar years behind the baseline (n alone can hide a thin record)
    return out
