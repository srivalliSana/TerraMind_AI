"""SYNTHETIC demonstration data. These are generated numbers, NOT satellite observations.

Used only when the dashboard is in Demo mode so the interface can be explored without credentials or
downloads. Each AOI runs through the same baseline/anomaly/risk/alert code as real data, but its results
never touch the real database and are labelled as synthetic everywhere they are shown.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.alerts.alert_manager import AlertManager
from src.alerts.risk_scoring import RiskAssessment, assess_risk
from src.geospatial import aoi as aoi_mod
from src.geospatial.aoi import AOI
from src.pipeline.analysis import analyze_daily
from src.storage import repositories as repo
from src.storage.database import open_db

DEMO_BANNER = ("SYNTHETIC DEMO DATA: generated numbers for exploring the interface. "
               "Not satellite observations and not a real crop-stress assessment.")
SEED = 42


@dataclass
class DemoAOI:
    aoi: AOI
    scenario: str
    daily: pd.DataFrame
    assessment: RiskAssessment


def _season_curve(doy: np.ndarray, peak: float) -> np.ndarray:
    return 0.22 + (peak - 0.22) * np.exp(-(((doy - 205) / 42.0) ** 2))


def make_daily(years: list[int], *, seed: int, decline_from: str | None = None, decline_depth: float = 0.0) -> pd.DataFrame:
    """One row per ~5-day observation between Jun 1 and Sep 10 of each year, with optional late-season decline."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        peak = 0.84 + rng.normal(0, 0.015)
        for date in pd.date_range(f"{year}-06-01", f"{year}-09-10", freq="5D"):
            ndvi = float(_season_curve(np.array([date.dayofyear]), peak)[0] + rng.normal(0, 0.012))
            if decline_from and date >= pd.Timestamp(decline_from):
                ndvi -= decline_depth * min((date - pd.Timestamp(decline_from)).days / 30.0, 1.0)
            ndmi = 0.12 + 0.45 * (ndvi - 0.22) / 0.62 + float(rng.normal(0, 0.012))
            rows.append({"date": date, "n_granules": 1, "sensors": "DEMO", "ndvi": ndvi, "ndmi": ndmi,
                         "valid_fraction": float(rng.uniform(0.55, 0.98))})
    return pd.DataFrame(rows)


def build_demo(config_dir=None) -> list[DemoAOI]:
    """Three synthetic fields: normal season, simulated sustained decline, and too little history."""
    t = config.load_thresholds()
    min_obs = config.get_settings().timeseries.min_observations_for_baseline
    specs = [
        ("DEMO field A (normal season)", (-93.75, 41.90, -93.70, 41.95), [2021, 2022, 2023, 2024], None, 0.0, 1),
        ("DEMO field B (simulated decline)", (-93.65, 41.90, -93.60, 41.95), [2021, 2022, 2023, 2024], "2024-07-25", 0.28, 2),
        ("DEMO field C (thin history)", (-93.55, 41.90, -93.50, 41.95), [2023, 2024], None, 0.0, 3),
    ]
    out = []
    for name, bbox, years, decline_from, depth, seed in specs:
        daily = analyze_daily(make_daily(years, seed=SEED + seed, decline_from=decline_from, decline_depth=depth))
        # cut the last year at a fixed point so "latest" is mid-decline for field B
        daily = daily[daily["date"] <= "2024-08-25"].reset_index(drop=True)
        aoi = aoi_mod.from_bbox(*bbox, name=name)
        out.append(DemoAOI(aoi, name, daily, assess_risk(daily, t, aoi_id=aoi.id, min_baseline_obs=min_obs)))
    return out


def build_demo_db(demo: list[DemoAOI] | None = None) -> tuple[sqlite3.Connection, list[DemoAOI]]:
    """In-memory database with demo AOIs and an alert history built by replaying the series date by date."""
    demo = demo or build_demo()
    t = config.load_thresholds()
    min_obs = config.get_settings().timeseries.min_observations_for_baseline
    conn = open_db(":memory:")
    state = {"day": "2024-06-01"}
    mgr = AlertManager(conn, clock=lambda: f"{state['day']}T12:00:00+00:00")
    for d in demo:
        repo.upsert_aoi(conn, d.aoi)
        for i in range(len(d.daily)):
            if d.daily.iloc[i]["date"].year != 2024:
                continue
            state["day"] = d.daily.iloc[i]["date"].date().isoformat()
            mgr.process(assess_risk(d.daily.iloc[: i + 1], t, aoi_id=d.aoi.id, min_baseline_obs=min_obs))
    return conn, demo
