"""Time-series assembly: processed observations -> one row per acquisition date.

Key rule: two granules on the same date (e.g. Landsat + Sentinel-2, or overlapping
tiles) describe the same moment, so they are merged into ONE observation instead of
being counted as independent evidence.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_observations(processed_root: Path, aoi_id: str, indices: tuple[str, ...] = ("ndvi", "ndmi")) -> pd.DataFrame:
    """Read every ``summary.json`` for an AOI into a table (one row per granule)."""
    rows = []
    for path in sorted((Path(processed_root) / aoi_id).glob("*/summary.json")):
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
            row = {
                "date": pd.Timestamp(s["acquired"]).tz_convert("UTC").tz_localize(None).normalize(),
                "granule_id": s["granule_id"], "sensor": s["sensor"],
                "valid_fraction": s["quality"]["valid_fraction"], "usable": bool(s["usable_for_analysis"]),
                "output_dir": str(path.parent),
            }
            for idx in indices:
                st = (s.get("index_stats") or {}).get(idx)
                row[f"{idx}_median"] = st["median"] if st else np.nan
                row[f"{idx}_std"] = st["std"] if st else np.nan
                row[f"n_{idx}"] = st["count"] if st else 0
            rows.append(row)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Skipping unreadable summary %s: %s", path, exc)
    cols = ["date", "granule_id", "sensor", "valid_fraction", "usable", "output_dir"] + [
        c for idx in indices for c in (f"{idx}_median", f"{idx}_std", f"n_{idx}")]
    return pd.DataFrame(rows, columns=cols).sort_values("date").reset_index(drop=True)


def aggregate_by_date(obs: pd.DataFrame, indices: tuple[str, ...] = ("ndvi", "ndmi"), usable_only: bool = True) -> pd.DataFrame:
    """Merge granules per date. Values are pixel-count-weighted means of per-granule medians.

    ``valid_fraction`` is the weighted mean across granules (conservative: overlapping coverage is not assumed
    to add up). Dates with no usable granule are dropped when ``usable_only``.
    """
    df = obs[obs["usable"]] if usable_only else obs
    out = []
    for date, g in df.groupby("date"):
        row = {"date": date, "n_granules": len(g), "sensors": "+".join(sorted(set(g["sensor"])))}
        for idx in indices:
            w = g[f"n_{idx}"].to_numpy(dtype=float)
            v = g[f"{idx}_median"].to_numpy(dtype=float)
            ok = np.isfinite(v) & (w > 0)
            row[idx] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
        w_all = np.maximum(g[f"n_{indices[0]}"].to_numpy(dtype=float), 1.0)
        row["valid_fraction"] = float(np.average(g["valid_fraction"], weights=w_all))
        out.append(row)
    daily = pd.DataFrame(out, columns=["date", "n_granules", "sensors", *indices, "valid_fraction"])
    return daily.sort_values("date").reset_index(drop=True)


def add_gap_info(daily: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """Add ``days_since_prev`` and a ``gap`` flag (irregular revisits are normal; long gaps weaken inference)."""
    out = daily.copy()
    out["days_since_prev"] = out["date"].diff().dt.days
    out["gap"] = out["days_since_prev"] > gap_days
    return out


def rolling_median(daily: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Smooth over the last ``window`` observations (not calendar days), robust to single outliers."""
    return daily[col].rolling(window=window, min_periods=1).median()


def change_from_previous(daily: pd.DataFrame, col: str) -> pd.Series:
    """Difference from the previous observation (NaN for the first)."""
    return daily[col].diff()
