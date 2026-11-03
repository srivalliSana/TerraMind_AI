"""Dashboard data layer + safe settings persistence (no Streamlit, no network)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src import config
from src.utils import dashboard_data as dd


@pytest.fixture(scope="module")
def demo_ctx():
    return dd.load_demo_context()


def test_demo_context_is_flagged_and_complete(demo_ctx):
    assert demo_ctx.is_demo and "SYNTHETIC" in demo_ctx.notes[0]
    assert len(demo_ctx.aois) == 3 and set(demo_ctx.daily) == set(demo_ctx.assessments) == set(demo_ctx.aois)


def test_status_table_and_metrics(demo_ctx):
    t = dd.aoi_status_table(demo_ctx)
    assert len(t) == 3 and set(t["category"]) >= {"informational", "insufficient_baseline"}
    assert t["active_alerts"].sum() == 1                                   # only the simulated decline
    m = dd.overview_metrics(demo_ctx)
    assert m["aois"] == 3 and m["active_alerts"] == 1 and m["latest_observation"]


def test_recent_activity_demo_labelled(demo_ctx):
    a = dd.recent_activity(demo_ctx)
    assert (a["granule"] == "synthetic").all()


def test_every_category_has_color_and_label():
    for c in ("informational", "watch", "elevated", "high", "insufficient_data", "insufficient_baseline"):
        assert dd.CATEGORY_COLORS[c].startswith("#") and dd.CATEGORY_LABELS[c]


def test_default_mode_is_demo_when_no_real_data(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRAMIND_DATA_DIR", str(tmp_path))
    assert dd.default_mode(config.load_settings()) == "demo"
    (tmp_path / "processed" / "aoi_x").mkdir(parents=True)
    (tmp_path / "processed" / "aoi_x" / "aoi.geojson").write_text("{}")
    assert dd.default_mode(config.load_settings()) == "real"


def test_real_context_empty_dir_is_empty_not_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRAMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TERRAMIND_DB_PATH", str(tmp_path / "t.sqlite"))
    ctx = dd.load_real_context(config.load_settings())
    assert not ctx.is_demo and ctx.aois == {} and len(dd.aoi_status_table(ctx)) == 0
    assert dd.overview_metrics(ctx)["aois"] == 0


def test_broken_aoi_is_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRAMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TERRAMIND_DB_PATH", str(tmp_path / "t.sqlite"))
    (tmp_path / "processed" / "aoi_bad").mkdir(parents=True)
    (tmp_path / "processed" / "aoi_bad" / "aoi.geojson").write_text("{not json")
    ctx = dd.load_real_context(config.load_settings())
    assert ctx.aois == {} and "aoi_bad" in ctx.notes[0]


def test_raster_overlay_transparent_nan_and_bounds(tmp_path):
    arr = np.array([[0.0, 0.5], [1.0, np.nan]], dtype="float32")
    p = tmp_path / "x.tif"
    with rasterio.open(p, "w", driver="GTiff", height=2, width=2, count=1, dtype="float32", crs="EPSG:32615",
                       transform=from_origin(399960, 4700040, 30, 30), nodata=np.nan) as d:
        d.write(arr, 1)
    rgba, bounds = dd.raster_overlay(p, "RdYlGn", 0, 1)
    assert rgba.shape == (2, 2, 4) and rgba[1, 1, 3] == 0 and rgba[0, 0, 3] == 255      # NaN transparent
    (s, w), (n, e) = bounds
    assert s < n and w < e and 41 < s < 44 and -96 < w < -92


def test_list_observations_skips_corrupt(tmp_path):
    good = tmp_path / "aoi_a" / "g1"
    good.mkdir(parents=True)
    (good / "summary.json").write_text(json.dumps({"granule_id": "g1", "acquired": "2024-07-01T00:00:00+00:00"}))
    bad = tmp_path / "aoi_a" / "g2"
    bad.mkdir()
    (bad / "summary.json").write_text("{x")
    assert [o["granule_id"] for o in dd.list_observations(tmp_path, "aoi_a")] == ["g1"]


def test_evaluation_runs_empty_until_a_model_exists(demo_ctx):
    assert dd.evaluation_runs(demo_ctx.conn) == []


def test_evaluation_runs_parse(demo_ctx):
    c = demo_ctx.conn
    c.execute("INSERT INTO model_versions (name, version, stage) VALUES ('rf', '0.1', 'experimental')")
    c.execute("INSERT INTO evaluation_runs (model_version_id, run_at, dataset, metrics_json) VALUES (1,'2024-01-01','synthetic-test', ?)",
              (json.dumps({"f1": 0.5}),))
    r = dd.evaluation_runs(c)
    assert r[0]["model_name"] == "rf" and r[0]["metrics"]["f1"] == 0.5


def test_apply_alert_evaluation_is_idempotent():
    ctx = dd.load_demo_context()
    ids = [a for a, x in ctx.assessments.items() if x.category == "elevated" or x.category == "high"]
    first = dd.evaluate_alerts(ctx, ids[0])
    assert first.action in ("unchanged", "updated", "escalated", "de-escalated")
    assert dd.evaluate_alerts(ctx, ids[0]).action == "unchanged"


# ---- settings persistence --------------------------------------------------------------------------
@pytest.fixture
def cfg(tmp_path):
    shutil.copytree(config.CONFIG_DIR, tmp_path / "c", ignore=shutil.ignore_patterns("local.yaml"))
    return tmp_path / "c"


def test_save_override_validates_then_applies(cfg, monkeypatch):
    monkeypatch.delenv("TERRAMIND_IGNORE_LOCAL_CONFIG")
    config.save_local_overrides({"aoi": {"max_area_km2": 500}}, {"anomaly": {"zscore_watch": -1.2}}, config_dir=cfg)
    assert config.load_settings(cfg).aoi.max_area_km2 == 500
    assert config.load_thresholds(cfg).anomaly["zscore_watch"] == -1.2
    assert config.load_settings(cfg).search.max_results == 50                  # untouched defaults preserved


def test_invalid_override_rejected_and_nothing_written(cfg):
    with pytest.raises(config.ConfigError, match="nothing was saved"):
        config.save_local_overrides({"aoi": {"max_area_km2": -5}}, config_dir=cfg)
    with pytest.raises(config.ConfigError):
        config.save_local_overrides(None, {"anomaly": {"zscore_watch": -3.5}}, config_dir=cfg)   # breaks ordering
    assert not (cfg / "local.yaml").exists()


def test_overrides_accumulate_and_reset(cfg, monkeypatch):
    monkeypatch.delenv("TERRAMIND_IGNORE_LOCAL_CONFIG")
    config.save_local_overrides({"aoi": {"max_area_km2": 500}}, config_dir=cfg)
    config.save_local_overrides({"search": {"max_results": 10}}, config_dir=cfg)
    s = config.load_settings(cfg)
    assert s.aoi.max_area_km2 == 500 and s.search.max_results == 10
    assert config.reset_local_overrides(cfg) is True and config.reset_local_overrides(cfg) is False
    assert config.load_settings(cfg).aoi.max_area_km2 == 2500


def test_local_override_ignored_when_flag_set(cfg):
    config.save_local_overrides({"aoi": {"max_area_km2": 500}}, config_dir=cfg)
    assert config.load_settings(cfg).aoi.max_area_km2 == 2500      # conftest sets TERRAMIND_IGNORE_LOCAL_CONFIG=1
