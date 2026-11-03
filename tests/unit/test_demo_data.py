"""The synthetic demo must exercise the real pipeline and stay clearly separate from real data."""
import pytest

from src.alerts.alert_manager import AlertManager
from src.alerts.risk_scoring import INSUFFICIENT_BASELINE
from src.utils.demo_data import DEMO_BANNER, build_demo, build_demo_db


@pytest.fixture(scope="module")
def demo():
    return build_demo()


def test_banner_says_synthetic():
    assert "SYNTHETIC" in DEMO_BANNER and "Not satellite observations" in DEMO_BANNER


def test_demo_is_deterministic(demo):
    again = build_demo()
    assert [d.assessment.score for d in demo] == [d.assessment.score for d in again]


def test_scenarios_behave_as_labelled(demo):
    a, b, c = demo
    assert a.assessment.category == "informational"
    assert b.assessment.category in ("elevated", "high") and b.assessment.score > 50      # simulated decline detected
    assert c.assessment.category == INSUFFICIENT_BASELINE and c.assessment.score is None   # honest: not enough history


def test_demo_names_are_labelled_demo(demo):
    assert all(d.aoi.name.startswith("DEMO") for d in demo)


def test_demo_alert_history_has_lifecycle_events():
    conn, demo = build_demo_db()
    mgr = AlertManager(conn)
    alerts = mgr.list()
    assert len(alerts) == 1                                           # only the declining field alerts; no duplicates
    assert alerts[0]["aoi_name"].startswith("DEMO field B")
    events = [e["event_type"] for e in mgr.events(alerts[0]["id"])]
    assert events[0] == "created" and len(events) >= 2                # created, then updated/escalated as it worsened
    assert alerts[0]["category"] in ("elevated", "high")
