"""Chart builders must not crash on empty/partial data and must encode what they claim."""
import pandas as pd

from src.alerts.risk_scoring import assess_risk
from src.config import load_thresholds
from src.utils import charts
from src.utils.demo_data import build_demo

T = load_thresholds()
A, B, C = build_demo()


def test_timeseries_has_baseline_band_and_levels():
    fig = charts.timeseries_figure(B.daily, "ndvi")
    names = [t.name for t in fig.data]
    assert "seasonal median" in names and "NDVI" in names
    colors = list(fig.data[-1].marker.color)
    assert charts.LEVEL_COLORS["high"] in colors or charts.LEVEL_COLORS["elevated"] in colors      # the simulated decline is visible


def test_timeseries_without_baseline_still_draws_points():
    fig = charts.timeseries_figure(C.daily, "ndvi")
    assert [t.name for t in fig.data] == ["NDVI"] or "NDVI" in [t.name for t in fig.data]


def test_empty_inputs_give_message_figures():
    for fig in (charts.timeseries_figure(pd.DataFrame(), "ndvi"), charts.seasonal_figure(pd.DataFrame(), "ndvi"),
                charts.zscore_figure(pd.DataFrame(), "ndvi", T)):
        assert fig.layout.annotations and not fig.data


def test_seasonal_one_trace_per_year():
    assert len(charts.seasonal_figure(A.daily, "ndvi").data) == 4


def test_zscore_figure_marks_thresholds():
    fig = charts.zscore_figure(B.daily, "ndvi", T)
    assert len(fig.layout.shapes) == 3                  # watch / elevated / high lines


def test_components_figure_scored_and_unscored():
    assert len(charts.components_figure(B.assessment, T).data) == 2
    assert not charts.components_figure(C.assessment, T).data          # no score -> message only


def test_confusion_matrix():
    fig = charts.confusion_matrix_figure([[5, 1], [2, 7]], ["ok", "stress"])
    assert fig.data[0].z[1][1] == 7
