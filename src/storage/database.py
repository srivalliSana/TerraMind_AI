"""SQLite metadata store with numbered migrations.

Only metadata lives here. Rasters stay on disk as GeoTIFFs (never BLOBs). The schema maps cleanly to
PostgreSQL/PostGIS (see docs/DEPLOYMENT.md): TEXT timestamps become timestamptz and the GeoJSON
column becomes a geometry column.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA_V1 = """
CREATE TABLE aois (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    geometry_geojson TEXT NOT NULL,
    min_lon REAL NOT NULL, min_lat REAL NOT NULL, max_lon REAL NOT NULL, max_lat REAL NOT NULL,
    area_km2 REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE granules (
    granule_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    sensor TEXT NOT NULL,
    tile_id TEXT,
    acquired TEXT NOT NULL,
    cloud_cover REAL,
    spatial_coverage REAL,
    manifest_path TEXT,
    recorded_at TEXT NOT NULL
);

CREATE TABLE observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    aoi_id TEXT NOT NULL REFERENCES aois(id) ON DELETE CASCADE,
    granule_id TEXT NOT NULL REFERENCES granules(granule_id) ON DELETE CASCADE,
    acquired TEXT NOT NULL,
    valid_fraction REAL NOT NULL,
    usable INTEGER NOT NULL CHECK (usable IN (0, 1)),
    output_dir TEXT,
    recorded_at TEXT NOT NULL,
    UNIQUE (aoi_id, granule_id)
);

CREATE TABLE index_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    index_name TEXT NOT NULL,
    pixel_count INTEGER NOT NULL,
    mean REAL, median REAL, std REAL, p10 REAL, p90 REAL,
    UNIQUE (observation_id, index_name)
);

CREATE TABLE processing_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    aoi_id TEXT REFERENCES aois(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    params_json TEXT
);

CREATE TABLE alerts (
    id TEXT PRIMARY KEY,
    aoi_id TEXT NOT NULL REFERENCES aois(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'resolved', 'dismissed')),
    category TEXT NOT NULL,
    score REAL,
    confidence REAL,
    confidence_label TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    last_as_of TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    dismissed_at TEXT,
    dismissed_category TEXT,
    clear_streak INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL,
    assessment_json TEXT NOT NULL
);
-- At most one ACTIVE alert per AOI and kind: duplicate prevention enforced by the database itself.
CREATE UNIQUE INDEX ux_alert_active ON alerts (aoi_id, kind) WHERE status = 'active';
CREATE INDEX ix_alerts_aoi ON alerts (aoi_id, status);

CREATE TABLE alert_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    at TEXT NOT NULL,
    from_category TEXT,
    to_category TEXT,
    note TEXT
);
CREATE INDEX ix_alert_events_alert ON alert_events (alert_id, at);

CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    stage TEXT NOT NULL,
    trained_at TEXT,
    metadata_json TEXT,
    UNIQUE (name, version)
);

CREATE TABLE evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version_id INTEGER REFERENCES model_versions(id) ON DELETE SET NULL,
    run_at TEXT NOT NULL,
    dataset TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    notes TEXT
);
"""

# (version, script). Append new migrations; never edit an applied one.
MIGRATIONS: list[tuple[int, str]] = [(1, SCHEMA_V1)]


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with foreign keys on and rows accessible by name."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"] or 0)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order; returns the resulting schema version. Idempotent."""
    version = current_version(conn)
    for ver, script in MIGRATIONS:
        if ver <= version:
            continue
        log.info("applying schema migration %d", ver)
        try:
            conn.executescript("BEGIN;\n" + script + f"\nINSERT INTO schema_version VALUES ({ver}, datetime('now'));\nCOMMIT;")
        except sqlite3.Error:
            if conn.in_transaction:
                conn.rollback()
            raise
        version = ver
    return version


def open_db(path: str | Path) -> sqlite3.Connection:
    """Connect and migrate to the latest schema."""
    conn = connect(path)
    migrate(conn)
    return conn
