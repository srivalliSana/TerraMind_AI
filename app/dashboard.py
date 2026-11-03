"""TerraMind AI: home page. Launch with ``streamlit run app/dashboard.py``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st  # noqa: E402

import _ui  # noqa: E402
from src.utils import dashboard_data as dd  # noqa: E402

_ui.setup("Home", ":material/satellite_alt:")
ctx = _ui.get_context()
_ui.header("TerraMind AI", "Satellite-based crop stress detection and early-warning intelligence")

m = dd.overview_metrics(ctx)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Monitored AOIs", m["aois"])
c2.metric("Observations", m["observations"])
c3.metric("Latest observation", m["latest_observation"] or "none")
c4.metric("Active alerts", m["active_alerts"])

if not ctx.aois:
    _ui.empty_state(ctx)

st.subheader("Where to go")
left, right = st.columns(2)
with left:
    st.page_link("pages/01_Global_Overview.py", label="Global Overview: map, status, activity", icon=":material/public:")
    st.page_link("pages/02_AOI_Explorer.py", label="AOI Explorer: define an area, discover and process observations", icon=":material/draw:")
    st.page_link("pages/03_Satellite_Observations.py", label="Satellite Observations: index maps and quality", icon=":material/layers:")
    st.page_link("pages/04_Crop_Stress_Analysis.py", label="Crop Stress Analysis: time series, baseline, risk", icon=":material/monitoring:")
with right:
    st.page_link("pages/05_Alerts.py", label="Alerts: preliminary warnings with evidence", icon=":material/notifications:")
    st.page_link("pages/06_Model_Performance.py", label="Model Performance: what has (and has not) been evaluated", icon=":material/fact_check:")
    st.page_link("pages/07_Settings.py", label="Settings: limits, thresholds, credentials status", icon=":material/settings:")

st.subheader("How to read the results")
st.markdown(
    "- **Observed** values are measurements from satellite imagery (NASA HLS), masked for cloud, shadow, snow and water.\n"
    "- **Modelled** values (baselines, z-scores, risk scores) are statistics computed from those observations.\n"
    "- Outputs are **potential stress indicators**. Satellite indices alone cannot diagnose disease, nutrient deficiency or the "
    "exact cause of a change, and thresholds here are engineering defaults that have **not** been validated against field data.\n"
    "- Coverage is limited to areas you have actually processed; this is not a global monitoring service."
)
_ui.prelim_note()
