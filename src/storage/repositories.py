"""Data-access helpers: AOIs, granules, observations, index summaries."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from src.geospatial import aoi as aoi_mod
from src.geospatial.aoi import AOI


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_aoi(conn: sqlite3.Connection, aoi: AOI, now: str | None = None) -> str:
    """Insert or update an AOI by its stable, geometry-derived id."""
    now = now or utc_now()
    minx, miny, maxx, maxy = aoi.bounds
    conn.execute(
        """INSERT INTO aois (id, name, source, geometry_geojson, min_lon, min_lat, max_lon, max_lat, area_km2, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET name=excluded.name, source=excluded.source, updated_at=excluded.updated_at""",
        (aoi.id, aoi.name, aoi.source, json.dumps(aoi.to_geojson()["geometry"]), minx, miny, maxx, maxy,
         aoi.area_km2, now, now))
    conn.commit()
    return aoi.id


def get_aoi(conn: sqlite3.Connection, aoi_id: str) -> AOI | None:
    row = conn.execute("SELECT * FROM aois WHERE id = ?", (aoi_id,)).fetchone()
    if row is None:
        return None
    return aoi_mod.from_geometry(shape(json.loads(row["geometry_geojson"])), name=row["name"], source=row["source"])


def list_aois(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM aois ORDER BY name").fetchall()


def record_observation(conn: sqlite3.Connection, aoi_id: str, summary: dict[str, Any], output_dir: str | Path | None = None,
                       now: str | None = None) -> int:
    """Store (or refresh) a processed observation and its index summaries from a ``summary.json`` dict."""
    now = now or utc_now()
    gid = summary["granule_id"]
    conn.execute(
        """INSERT INTO granules (granule_id, provider, sensor, tile_id, acquired, cloud_cover, spatial_coverage, recorded_at)
           VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(granule_id) DO NOTHING""",
        (gid, "nasa_hls", summary["sensor"], gid.split(".")[2] if gid.count(".") >= 3 else None, summary["acquired"],
         (summary.get("catalog") or {}).get("cloud_cover"), (summary.get("catalog") or {}).get("spatial_coverage"), now))
    conn.execute(
        """INSERT INTO observations (aoi_id, granule_id, acquired, valid_fraction, usable, output_dir, recorded_at)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(aoi_id, granule_id) DO UPDATE SET valid_fraction=excluded.valid_fraction,
               usable=excluded.usable, output_dir=excluded.output_dir, recorded_at=excluded.recorded_at""",
        (aoi_id, gid, summary["acquired"], summary["quality"]["valid_fraction"], int(summary["usable_for_analysis"]),
         str(output_dir) if output_dir else None, now))
    obs_id = conn.execute("SELECT id FROM observations WHERE aoi_id=? AND granule_id=?", (aoi_id, gid)).fetchone()["id"]
    for name, st in (summary.get("index_stats") or {}).items():
        if st is None:
            continue
        conn.execute(
            """INSERT INTO index_summaries (observation_id, index_name, pixel_count, mean, median, std, p10, p90)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(observation_id, index_name) DO UPDATE SET pixel_count=excluded.pixel_count, mean=excluded.mean,
                   median=excluded.median, std=excluded.std, p10=excluded.p10, p90=excluded.p90""",
            (obs_id, name, st["count"], st["mean"], st["median"], st["std"], st["p10"], st["p90"]))
    conn.commit()
    return int(obs_id)


def sync_processed_dir(conn: sqlite3.Connection, processed_root: Path, aoi_id: str) -> int:
    """Record every ``summary.json`` under ``processed_root/aoi_id`` (idempotent). Returns the number recorded."""
    n = 0
    for path in sorted((Path(processed_root) / aoi_id).glob("*/summary.json")):
        try:
            record_observation(conn, aoi_id, json.loads(path.read_text(encoding="utf-8")), path.parent)
            n += 1
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return n
