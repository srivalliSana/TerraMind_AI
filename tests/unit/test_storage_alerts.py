"""Database + alert lifecycle tests (SYNTHETIC assessments, in-memory SQLite)."""
import csv
import io
import json
import sqlite3

import pytest

from src.alerts.alert_manager import AlertManager
from src.alerts.risk_scoring import RiskAssessment
from src.geospatial import aoi as A
from src.storage import database as db
from src.storage import repositories as repo

SCORE = {"informational": 5.0, "watch": 40.0, "elevated": 60.0, "high": 90.0}


def assessment(category, as_of, aoi_id):
    scored = category in SCORE
    return RiskAssessment(aoi_id, as_of, category, SCORE.get(category), {"magnitude": 1.0, "persistence": 1.0, "extent": None},
                          [f"evidence for {category}"], [as_of], {"valid_fraction": 0.9}, 0.7 if scored else None,
                          "high" if scored else None, None, ["follow up"])


@pytest.fixture
def conn():
    c = db.open_db(":memory:")
    yield c
    c.close()


@pytest.fixture
def aoi(conn):
    a = A.from_bbox(-93.7, 41.9, -93.6, 42.0, name="iowa")
    repo.upsert_aoi(conn, a)
    return a


class Clock:
    def __init__(self, day="2024-07-01"):
        self.day = day

    def __call__(self):
        return f"{self.day}T12:00:00+00:00"


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def mgr(conn, clock, aoi):
    return AlertManager(conn, clock=clock)


# ---- database -------------------------------------------------------------------------------------
def test_migration_is_idempotent_and_creates_all_tables(conn):
    assert db.migrate(conn) == 1 and db.migrate(conn) == 1
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"aois", "granules", "observations", "index_summaries", "processing_jobs", "alerts", "alert_events",
            "model_versions", "evaluation_runs", "schema_version"} <= names
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1


def test_foreign_keys_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO observations (aoi_id, granule_id, acquired, valid_fraction, usable, recorded_at) "
                     "VALUES ('nope','nope','2024-07-01',1,1,'x')")


def test_aoi_roundtrip_and_stable_id(conn, aoi):
    assert repo.upsert_aoi(conn, aoi) == aoi.id
    assert conn.execute("SELECT COUNT(*) FROM aois").fetchone()[0] == 1          # upsert, not duplicate
    back = repo.get_aoi(conn, aoi.id)
    assert back.id == aoi.id and back.name == "iowa"
    assert repo.get_aoi(conn, "aoi_missing") is None


def test_record_observation_is_idempotent(conn, aoi):
    summary = {"granule_id": "HLS.S30.T15TVG.2024199T170849.v2.0", "sensor": "HLSS30", "acquired": "2024-07-17T17:08:49+00:00",
               "usable_for_analysis": True, "quality": {"valid_fraction": 0.46},
               "catalog": {"cloud_cover": 18, "spatial_coverage": 57},
               "index_stats": {"ndvi": {"count": 10, "mean": 0.8, "median": 0.85, "std": 0.1, "p10": 0.6, "p90": 0.9}, "ndmi": None}}
    repo.record_observation(conn, aoi.id, summary)
    repo.record_observation(conn, aoi.id, summary)
    assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM index_summaries").fetchone()[0] == 1       # null stats skipped
    assert conn.execute("SELECT tile_id FROM granules").fetchone()[0] == "T15TVG"


def test_sync_processed_dir_skips_corrupt(conn, aoi, tmp_path):
    good = tmp_path / aoi.id / "g1"
    good.mkdir(parents=True)
    (good / "summary.json").write_text(json.dumps({
        "granule_id": "HLS.S30.T15TVG.2024199T170849.v2.0", "sensor": "HLSS30", "acquired": "2024-07-17T17:00:00+00:00",
        "usable_for_analysis": True, "quality": {"valid_fraction": 0.9}, "index_stats": {}}))
    bad = tmp_path / aoi.id / "g2"
    bad.mkdir()
    (bad / "summary.json").write_text("{oops")
    assert repo.sync_processed_dir(conn, tmp_path, aoi.id) == 1


def test_database_blocks_two_active_alerts_even_if_code_is_bypassed(conn, mgr, aoi):
    mgr.process(assessment("watch", "2024-07-01", aoi.id))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO alerts (id, aoi_id, kind, status, category, first_seen, last_seen, last_as_of, created_at, "
                     "updated_at, evidence_json, assessment_json) VALUES ('dup', ?, 'crop_stress', 'active', 'high', "
                     "'x','x','x','x','x','[]','{}')", (aoi.id,))


# ---- lifecycle ------------------------------------------------------------------------------------
def test_create_then_duplicate_is_idempotent(mgr, aoi):
    r1 = mgr.process(assessment("watch", "2024-07-01", aoi.id))
    r2 = mgr.process(assessment("watch", "2024-07-01", aoi.id))          # same observation reprocessed
    assert r1.action == "created" and r2.action == "unchanged" and r1.alert_id == r2.alert_id
    assert len(mgr.list()) == 1 and len(mgr.events(r1.alert_id)) == 1


def test_update_escalate_deescalate(mgr, aoi):
    a = mgr.process(assessment("watch", "2024-07-01", aoi.id)).alert_id
    assert mgr.process(assessment("watch", "2024-07-06", aoi.id)).action == "updated"
    assert mgr.process(assessment("high", "2024-07-11", aoi.id)).action == "escalated"
    assert mgr.process(assessment("elevated", "2024-07-16", aoi.id)).action == "de-escalated"
    row = mgr.get(a)
    assert row["category"] == "elevated" and row["last_seen"] == "2024-07-16" and row["first_seen"] == "2024-07-01"
    assert [e["event_type"] for e in mgr.events(a)] == ["created", "updated", "escalated", "de-escalated"]
    assert len(mgr.list()) == 1                                           # still a single alert


def test_resolution_needs_consecutive_clear_observations(mgr, aoi):
    a = mgr.process(assessment("elevated", "2024-07-01", aoi.id)).alert_id
    assert mgr.process(assessment("informational", "2024-07-06", aoi.id)).action == "clear_pending"
    assert mgr.get(a)["status"] == "active"
    assert mgr.process(assessment("informational", "2024-07-11", aoi.id)).action == "resolved"
    assert mgr.get(a)["status"] == "resolved" and mgr.get(a)["resolved_at"]
    new = mgr.process(assessment("watch", "2024-07-20", aoi.id))
    assert new.action == "created" and new.alert_id != a                  # a fresh alert, history preserved


def test_clear_streak_resets_when_anomaly_returns(mgr, aoi):
    a = mgr.process(assessment("watch", "2024-07-01", aoi.id)).alert_id
    mgr.process(assessment("informational", "2024-07-06", aoi.id))
    mgr.process(assessment("watch", "2024-07-11", aoi.id))
    assert mgr.get(a)["clear_streak"] == 0
    assert mgr.process(assessment("informational", "2024-07-16", aoi.id)).action == "clear_pending"   # needs 2 again


def test_clear_duplicate_observation_does_not_double_count(mgr, aoi):
    mgr.process(assessment("watch", "2024-07-01", aoi.id))
    mgr.process(assessment("informational", "2024-07-06", aoi.id))
    assert mgr.process(assessment("informational", "2024-07-06", aoi.id)).action == "unchanged"


def test_unknown_assessments_never_create_or_resolve(mgr, aoi):
    assert mgr.process(assessment("insufficient_data", "2024-07-01", aoi.id)).action == "no_change"
    assert mgr.list() == []
    mgr.process(assessment("high", "2024-07-02", aoi.id))
    assert mgr.process(assessment("insufficient_baseline", "2024-07-07", aoi.id)).action == "no_change"
    assert mgr.list(status="active")[0]["category"] == "high"


def test_informational_without_alert_does_nothing(mgr, aoi):
    assert mgr.process(assessment("informational", "2024-07-01", aoi.id)).action == "no_change" and mgr.list() == []


def test_dismissal_suppresses_same_or_lower_but_not_escalation(mgr, aoi, clock):
    a = mgr.process(assessment("elevated", "2024-07-01", aoi.id)).alert_id
    clock.day = "2024-07-02"
    mgr.dismiss(a, "known harvest")
    assert mgr.get(a)["status"] == "dismissed" and mgr.get(a)["dismissed_category"] == "elevated"
    assert mgr.process(assessment("elevated", "2024-07-10", aoi.id)).action == "suppressed"
    assert mgr.process(assessment("watch", "2024-07-12", aoi.id)).action == "suppressed"
    assert mgr.list(status="active") == []
    clock.day = "2024-07-14"
    esc = mgr.process(assessment("high", "2024-07-14", aoi.id))
    assert esc.action == "created" and esc.alert_id != a
    assert "after escalation" in mgr.events(esc.alert_id)[0]["note"]


def test_dismissal_expires_after_suppress_window(mgr, aoi, clock):
    a = mgr.process(assessment("watch", "2024-07-01", aoi.id)).alert_id
    clock.day = "2024-07-02"
    mgr.dismiss(a)
    assert mgr.process(assessment("watch", "2024-09-15", aoi.id)).action == "created"      # > 30 days later


def test_user_actions_validate_state(mgr, aoi):
    a = mgr.process(assessment("watch", "2024-07-01", aoi.id)).alert_id
    mgr.resolve(a, "checked in field")
    with pytest.raises(ValueError, match="already resolved"):
        mgr.dismiss(a)
    with pytest.raises(KeyError):
        mgr.dismiss("alert_nope")


def test_list_filters(mgr, conn, aoi):
    other = A.from_bbox(10, 50, 10.1, 50.1, name="other")
    repo.upsert_aoi(conn, other)
    mgr.process(assessment("watch", "2024-07-01", aoi.id))
    mgr.process(assessment("high", "2024-07-03", other.id))
    assert len(mgr.list()) == 2
    assert [r["aoi_name"] for r in mgr.list(category="high")] == ["other"]
    assert len(mgr.list(aoi_id=aoi.id)) == 1 and len(mgr.list(since="2024-07-02")) == 1 and mgr.list(status="resolved") == []


def test_exports(mgr, aoi):
    mgr.process(assessment("high", "2024-07-01", aoi.id))
    rows = list(csv.DictReader(io.StringIO(mgr.export_csv())))
    assert len(rows) == 1 and rows[0]["category"] == "high" and rows[0]["preliminary_indicator"] == "yes"
    gj = mgr.export_geojson()
    f = gj["features"][0]
    assert f["geometry"]["type"] == "Polygon" and f["properties"]["preliminary_indicator"] is True
    assert f["properties"]["evidence"] == ["evidence for high"]


def test_assessment_without_aoi_rejected(mgr):
    with pytest.raises(ValueError):
        mgr.process(assessment("watch", "2024-07-01", None))
