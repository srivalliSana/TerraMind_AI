"""Headless execution of every dashboard page with Streamlit's AppTest (no browser, no network).

Three situations are covered for each page: demo mode, real mode with an empty data directory, and real mode
with SYNTHETIC processed outputs. The goal is "renders without exceptions and shows the right honest state".
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import config
from src.geospatial import aoi as A
from src.pipeline.jobs import process_granule
from tests.integration.test_process_granule import GID, aoi as _aoi_fixture, build  # noqa: F401

APP = Path(__file__).resolve().parents[2] / "app"
PAGES = ["dashboard.py"] + sorted(p.name and f"pages/{p.name}" for p in (APP / "pages").glob("*.py"))
DEMO = "Demo data (synthetic)"


def run(page: str, mode: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP / page), default_timeout=90)
    if mode:
        at.session_state["mode_choice"] = mode
    return at.run()


def all_text(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.info] + [e.value for e in at.warning] + [e.value for e in at.success]
    parts += [e.value for e in at.title] + [e.value for e in at.subheader] + [e.value for e in at.caption] + [str(m.value) for m in at.metric]
    return "\n".join(map(str, parts))


@pytest.fixture
def empty_real(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRAMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TERRAMIND_DB_PATH", str(tmp_path / "t.sqlite"))
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


@pytest.fixture
def real_data(tmp_path, monkeypatch, _aoi_fixture):
    """Real-mode data directory containing one SYNTHETIC processed observation."""
    monkeypatch.setenv("TERRAMIND_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TERRAMIND_DB_PATH", str(tmp_path / "data" / "t.sqlite"))
    config.get_settings.cache_clear()
    manifest, raw = build(tmp_path)
    process_granule(manifest, _aoi_fixture, raw_dir=raw, out_root=tmp_path / "data" / "processed")
    yield tmp_path / "data"
    config.get_settings.cache_clear()


@pytest.fixture
def real_with_extent(tmp_path, monkeypatch):
    """Real-mode data with enough SYNTHETIC history for the per-pixel affected-area analysis (a 10x10 px decline)."""
    import numpy as np

    from tests.integration.test_extent import healthy, write_obs

    monkeypatch.setenv("TERRAMIND_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TERRAMIND_DB_PATH", str(tmp_path / "data" / "t.sqlite"))
    config.get_settings.cache_clear()
    aoi = A.from_bbox(-93.7, 41.9, -93.6, 42.0, name="synthetic-extent")
    root = tmp_path / "data" / "processed"
    (root / aoi.id).mkdir(parents=True)
    (root / aoi.id / "aoi.geojson").write_text(json.dumps(aoi.to_geojson()))
    rng = np.random.default_rng(11)
    for year in (2022, 2023):
        for day in ("07-10", "07-15", "07-20"):
            write_obs(root, f"{year}-{day}", healthy(rng), aoi=aoi.id)
    arr = healthy(rng)
    arr[:10, :10] -= 0.3
    write_obs(root, "2024-07-15", arr, aoi=aoi.id)
    yield aoi
    config.get_settings.cache_clear()


def test_stress_page_shows_affected_area_when_history_allows(real_with_extent):
    at = run("pages/04_Crop_Stress_Analysis.py", "Real data")
    assert not at.exception, [e.value for e in at.exception]
    labels = {m.label: m.value for m in at.metric}
    assert labels["Affected area"] == "6.2%" and labels["Patches"] == "1"          # 100 of 1600 pixels
    assert any("different crop" in w.value for w in at.warning)                     # rotation caveat is shown


def test_stress_page_explains_when_extent_unavailable(real_data):
    at = run("pages/04_Crop_Stress_Analysis.py", "Real data")
    assert not at.exception
    assert any("No per-pixel anomaly map" in i.value for i in at.info)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_in_demo_mode(page, empty_real):
    at = run(page, DEMO)
    assert not at.exception, [e.value for e in at.exception]
    assert "SYNTHETIC DEMO DATA" in "".join(e.value for e in at.markdown)       # banner on every page


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_real_mode_with_no_data(page, empty_real):
    at = run(page, "Real data")
    assert not at.exception, [e.value for e in at.exception]
    assert "SYNTHETIC DEMO DATA" not in "".join(e.value for e in at.markdown)   # never demo content in real mode


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_real_mode_with_data(page, real_data):
    at = run(page, "Real data")
    assert not at.exception, [e.value for e in at.exception]
    assert "SYNTHETIC DEMO DATA" not in "".join(e.value for e in at.markdown)


def test_overview_empty_state_is_helpful(empty_real):
    at = run("pages/01_Global_Overview.py", "Real data")
    assert any("No processed observations yet" in i.value for i in at.info)


def test_overview_demo_shows_three_aois_and_one_alert(empty_real):
    at = run("pages/01_Global_Overview.py", DEMO)
    vals = {m.label: m.value for m in at.metric}
    assert vals["Monitored AOIs"] == "3" and vals["Active alerts"] == "1"


def test_model_page_states_no_model_exists(empty_real):
    at = run("pages/06_Model_Performance.py", "Real data")
    assert any("No trained model and no evaluation runs exist" in i.value for i in at.info)


def test_stress_page_demo_declining_field_is_flagged(empty_real):
    at = run("pages/04_Crop_Stress_Analysis.py", DEMO)
    picker = at.selectbox(key="stress_aoi")
    picker.select_index(next(i for i, o in enumerate(picker.options) if "simulated decline" in o))
    at = at.run()
    assert not at.exception
    assert any("High risk" in m.value for m in at.markdown)
    assert any("100" in m.value for m in at.metric)                          # score shown


def test_stress_page_thin_history_gives_no_score(empty_real):
    at = run("pages/04_Crop_Stress_Analysis.py", DEMO)
    picker = at.selectbox(key="stress_aoi")
    picker.select_index(next(i for i, o in enumerate(picker.options) if "thin history" in o))
    at = at.run()
    assert not at.exception and any("No risk score" in w.value for w in at.warning)


def test_alerts_page_demo_lists_active_alert_and_dismiss_works(empty_real):
    at = run("pages/05_Alerts.py", DEMO)
    assert not at.exception and at.dataframe                       # table of alerts rendered
    dismiss = [b for b in at.button if b.key and b.key.startswith("dismiss_")]
    assert dismiss
    dismiss[0].click()
    at = at.run()
    assert not at.exception
    assert any("No alerts match" in i.value for i in at.info)       # active filter now empty after dismissal


def test_settings_page_never_prints_secrets(empty_real, monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "super-secret-token-value")
    at = run("pages/07_Settings.py", DEMO)
    assert not at.exception
    blob = " ".join(str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.text) + list(at.json) + list(at.info) + list(at.warning))
    assert "super-secret-token-value" not in blob
