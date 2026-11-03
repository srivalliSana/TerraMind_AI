import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import _ui  # noqa: E402
from src import config  # noqa: E402
from src.utils import charts  # noqa: E402
from src.utils import dashboard_data as dd  # noqa: E402

_ui.setup("Crop Stress Analysis", ":material/monitoring:")
ctx = _ui.get_context()
_ui.header("Crop Stress Analysis", "Observed vegetation indices vs. a seasonal baseline, with an evidence-backed preliminary risk assessment")

if not ctx.aois:
    _ui.empty_state(ctx)
    st.stop()

aid = _ui.aoi_picker(ctx, "stress_aoi")
daily: pd.DataFrame = ctx.daily[aid]
a = ctx.assessments[aid]
t = config.load_thresholds()

st.markdown(f"### {ctx.aois[aid].name} &nbsp; {_ui.badge(a.category)}", unsafe_allow_html=True)
if a.is_scored:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{a.score:.0f} / 100", help="0 = no indication, 100 = strongest indication under the configured weights")
    c2.metric("Confidence", f"{a.confidence_label} ({a.confidence:.2f})", help="Heuristic from baseline depth/years, data quality, index agreement and recency. Not a probability.")
    c3.metric("As of", a.as_of)
    c4.metric("Affected area", "not assessed" if a.affected_fraction is None else f"{a.affected_fraction:.0%}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Evidence**")
        for e in a.evidence:
            st.markdown(f"- {e}")
        st.markdown(f"**Observation dates in the current run:** {', '.join(a.observation_dates)}")
    with right:
        st.markdown("**Score components**")
        st.plotly_chart(charts.components_figure(a, t), use_container_width=True)
else:
    st.warning(f"No risk score: **{dd.CATEGORY_LABELS.get(a.category, a.category)}**.")
    for e in a.evidence:
        st.markdown(f"- {e}")

st.markdown("**Recommended follow-up**")
for r in a.recommendations:
    st.markdown(f"- {r}")

warnings = []
if a.quality.get("valid_fraction", 1) < 0.5:
    warnings.append(f"Only {a.quality['valid_fraction']:.0%} of the AOI is valid on the latest date.")
if a.quality.get("days_since_latest_observation", 0) > t.scoring["stale_after_days"]:
    warnings.append(f"The latest observation is {a.quality['days_since_latest_observation']} days old.")
if len(daily) and daily["gap"].tail(5).any():
    warnings.append("There are long gaps between recent observations; trends are less certain.")
if a.is_scored and a.confidence_label == "low":
    warnings.append("Confidence is low: treat this as a prompt for review only.")
for w in warnings:
    st.warning(w)

st.subheader("Time series")
if daily.empty:
    st.info("No usable observations for this AOI.")
else:
    tab_ndvi, tab_ndmi, tab_season, tab_z = st.tabs(["NDVI timeline", "NDMI timeline", "Seasonal view", "z-scores"])
    with tab_ndvi:
        st.plotly_chart(charts.timeseries_figure(daily, "ndvi"), use_container_width=True)
        st.caption("Points are coloured by anomaly level. The band is the seasonal baseline (median ± 2σ) built only from earlier years; "
                   "it is absent where fewer than the required years of history exist.")
    with tab_ndmi:
        st.plotly_chart(charts.timeseries_figure(daily, "ndmi"), use_container_width=True)
    with tab_season:
        st.plotly_chart(charts.seasonal_figure(daily, "ndvi"), use_container_width=True)
        st.caption("Each line is a year against day-of-year; the latest year is highlighted. Seasonal growth, planting, harvest and "
                   "irrigation can all move these curves without any crop stress.")
    with tab_z:
        st.plotly_chart(charts.zscore_figure(daily, "ndvi", t), use_container_width=True)
        st.caption("Only declines below the baseline are graded. Thresholds are configurable engineering defaults, not validated rules.")

    with st.expander("Series data"):
        st.dataframe(daily, hide_index=True, use_container_width=True)
        st.download_button("Download series (CSV)", daily.to_csv(index=False).encode("utf-8"), f"{aid}_timeseries.csv", "text/csv")

st.subheader("Anomaly map")
st.info("A per-pixel anomaly map needs pixel-level seasonal baselines, which are not implemented yet. Until then, assessments use "
        "AOI-level statistics (hence \"affected area: not assessed\"). Index maps for single dates are on the Satellite Observations page.")
_ui.prelim_note()
