"""Data layer for the dashboard: pure functions, no Streamlit imports (so they are unit-testable).

A ``DataContext`` bundles everything the pages show, for either REAL data (processed outputs on disk +
SQLite) or the clearly-labelled SYNTHETIC demo. Pages never mix the two.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

from src import config
from src.alerts.risk_scoring import RiskAssessment, assess_risk
from src.geospatial import aoi as aoi_mod
from src.geospatial.aoi import AOI
from src.pipeline.analysis import analyze_aoi
from src.storage import repositories as repo
from src.storage.database import open_db
from src.utils import demo_data

CATEGORY_COLORS = {
    "informational": "#2e9e6b", "watch": "#e0b100", "elevated": "#e8761e", "high": "#d6334a",
    "insufficient_data": "#7c8794", "insufficient_baseline": "#7c8794",
}
CATEGORY_LABELS = {
    "informational": "Informational", "watch": "Watch", "elevated": "Elevated", "high": "High risk",
    "insufficient_data": "Insufficient data", "insufficient_baseline": "Insufficient baseline",
}


@dataclass
class DataContext:
    mode: str                                     # 'real' | 'demo'
    conn: sqlite3.Connection
    aois: dict[str, AOI]
    daily: dict[str, pd.DataFrame]
    assessments: dict[str, RiskAssessment]
    processed_root: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"


def processed_aoi_ids(processed_root: Path) -> list[str]:
    """AOI ids that have processed outputs and a saved AOI shape."""
    root = Path(processed_root)
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/aoi.geojson"))


def has_real_data(settings: config.Settings | None = None) -> bool:
    s = settings or config.get_settings()
    return bool(processed_aoi_ids(s.data_dir / "processed"))


def load_real_context(settings: config.Settings | None = None) -> DataContext:
    """Open the database, register on-disk AOIs/observations (idempotent) and analyse each AOI."""
    s = settings or config.get_settings()
    root = s.data_dir / "processed"
    conn = open_db(s.db_path, check_same_thread=False)
    t = config.load_thresholds()
    aois, daily, assessments, notes = {}, {}, {}, []
    for aid in processed_aoi_ids(root):
        try:
            aoi = aoi_mod.from_geojson(root / aid / "aoi.geojson")
            repo.upsert_aoi(conn, aoi)
            repo.sync_processed_dir(conn, root, aoi.id)
            df = analyze_aoi(root, aoi.id)
            aois[aoi.id], daily[aoi.id] = aoi, df
            assessments[aoi.id] = assess_risk(df, t, aoi_id=aoi.id,
                                              min_baseline_obs=s.timeseries.min_observations_for_baseline)
        except Exception as exc:  # noqa: BLE001 - one broken AOI must not take the dashboard down
            notes.append(f"Could not load AOI {aid}: {exc}")
    return DataContext("real", conn, aois, daily, assessments, root, notes)


def load_demo_context() -> DataContext:
    conn, demo = demo_data.build_demo_db()
    return DataContext("demo", conn, {d.aoi.id: d.aoi for d in demo}, {d.aoi.id: d.daily for d in demo},
                       {d.aoi.id: d.assessment for d in demo}, None, [demo_data.DEMO_BANNER])


def default_mode(settings: config.Settings | None = None) -> str:
    """Real if processed data exist; otherwise demo (never silently the other way round)."""
    return "real" if has_real_data(settings) else "demo"


def aoi_status_table(ctx: DataContext) -> pd.DataFrame:
    rows = []
    for aid, aoi in ctx.aois.items():
        df, a = ctx.daily.get(aid, pd.DataFrame()), ctx.assessments.get(aid)
        active = ctx.conn.execute("SELECT COUNT(*) FROM alerts WHERE aoi_id=? AND status='active'", (aid,)).fetchone()[0]
        rows.append({
            "aoi_id": aid, "name": aoi.name, "area_km2": round(aoi.area_km2, 2), "observations": len(df),
            "latest_observation": df["date"].max().date().isoformat() if len(df) else None,
            "category": a.category if a else "insufficient_data", "score": a.score if a else None,
            "confidence": a.confidence_label if a else None, "active_alerts": active})
    return pd.DataFrame(rows, columns=["aoi_id", "name", "area_km2", "observations", "latest_observation", "category",
                                       "score", "confidence", "active_alerts"])


def overview_metrics(ctx: DataContext) -> dict[str, object]:
    t = aoi_status_table(ctx)
    n_obs = int(ctx.conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]) if not ctx.is_demo else int(t["observations"].sum())
    latest = t["latest_observation"].dropna().max() if len(t) and t["latest_observation"].notna().any() else None
    return {"aois": len(t), "observations": n_obs, "latest_observation": latest,
            "active_alerts": int(t["active_alerts"].sum()) if len(t) else 0}


def recent_activity(ctx: DataContext, limit: int = 10) -> pd.DataFrame:
    if ctx.is_demo:
        rows = [{"recorded_at": "(demo)", "aoi": ctx.aois[a].name, "granule": "synthetic", "acquired": str(df['date'].max().date()),
                 "valid_fraction": float(df['valid_fraction'].iloc[-1])} for a, df in ctx.daily.items() if len(df)]
        return pd.DataFrame(rows)
    q = ("SELECT o.recorded_at, a.name AS aoi, o.granule_id AS granule, substr(o.acquired,1,10) AS acquired, o.valid_fraction "
         "FROM observations o JOIN aois a ON a.id = o.aoi_id ORDER BY o.recorded_at DESC, o.acquired DESC LIMIT ?")
    return pd.DataFrame([dict(r) for r in ctx.conn.execute(q, (limit,))],
                        columns=["recorded_at", "aoi", "granule", "acquired", "valid_fraction"])


def list_observations(processed_root: Path, aoi_id: str) -> list[dict]:
    """Processed observations for an AOI (newest first), each with its summary.json contents."""
    out = []
    for p in sorted((Path(processed_root) / aoi_id).glob("*/summary.json"), reverse=True):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            out.append({"granule_id": s["granule_id"], "acquired": s["acquired"], "dir": p.parent, "summary": s})
        except (KeyError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda o: o["acquired"], reverse=True)


def raster_overlay(path: Path, cmap: str, vmin: float, vmax: float, max_px: int = 700):
    """Colourised RGBA preview and its lat/lon bounds ``[[south, west], [north, east]]`` for a map overlay.

    NaN pixels become transparent. The image stays in the raster's own grid, so a very slight rotation
    relative to lat/lon is possible (negligible at AOI scale); it is for visual orientation, not measurement.
    """
    import matplotlib

    with rasterio.open(path) as src:
        scale = min(1.0, max_px / max(src.width, src.height))
        shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
        data = src.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.nearest).astype("float32")
        w, s_, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    rgba = (matplotlib.colormaps[cmap](norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(data), 255, 0)
    return rgba, [[s_, w], [n, e]]


def evaluation_runs(conn: sqlite3.Connection) -> list[dict]:
    """Stored model evaluation runs with parsed metrics. Empty until a real model has been trained and evaluated."""
    rows = conn.execute(
        "SELECT e.id, e.run_at, e.dataset, e.metrics_json, e.notes, m.name AS model_name, m.version AS model_version "
        "FROM evaluation_runs e LEFT JOIN model_versions m ON m.id = e.model_version_id ORDER BY e.run_at DESC").fetchall()
    out = []
    for r in rows:
        try:
            metrics = json.loads(r["metrics_json"])
        except json.JSONDecodeError:
            metrics = {}
        out.append({**dict(r), "metrics": metrics})
    return out


def evaluate_alerts(ctx: DataContext, aoi_id: str):
    """Apply the AOI's current assessment to the alert state (same code path as the CLI)."""
    from src.alerts.alert_manager import AlertManager

    return AlertManager(ctx.conn).process(ctx.assessments[aoi_id])
