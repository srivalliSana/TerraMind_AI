import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import _ui  # noqa: E402
from src import config  # noqa: E402
from src.utils import charts  # noqa: E402
from src.utils import dashboard_data as dd  # noqa: E402

_ui.setup("Model Performance", ":material/fact_check:")
ctx = _ui.get_context()
_ui.header("Model Performance", "Evaluation results appear here only when an actual evaluation has been run")

t = config.load_thresholds()
runs = dd.evaluation_runs(ctx.conn) if not ctx.is_demo else []

st.subheader("Current method")
st.markdown(
    "TerraMind currently uses **rule-based statistical indicators**, not a trained machine-learning classifier:\n\n"
    "1. NDVI / NDMI per acquisition date (masked for cloud, shadow, snow, water).\n"
    "2. A seasonal baseline from earlier years (robust median with detrending).\n"
    "3. z-scores of declines, graded by configurable thresholds, plus persistence over consecutive observations.\n"
    "4. A transparent risk score (magnitude, persistence, extent) with confidence.\n")
st.warning(f"These thresholds are engineering defaults. Validated against field data: **{t.validated}**. "
           "No accuracy, precision or recall can be claimed for them.")

st.subheader("Evaluation runs")
if not runs:
    st.info("**No trained model and no evaluation runs exist.** There are no validated stress labels, so no accuracy metrics are "
            "shown and none are implied.")
    with st.expander("What is needed to train and evaluate a supervised model"):
        st.markdown(
            "- **Labels:** field-verified stress / no-stress observations with locations and dates (agronomist scouting, yield loss records, "
            "insurance claims, trusted crop-condition reports), covering several seasons and regions.\n"
            "- **Splits that prevent leakage:** hold out whole fields, regions or years, never random pixels or dates of the same field.\n"
            "- **Class imbalance:** stress events are rare; report precision, recall, F1 and PR/ROC-AUC, not accuracy alone.\n"
            "- **Baseline comparison:** any model must beat the rule-based indicators above on held-out data.\n"
            "- **Calibration and out-of-distribution checks** before probabilities are shown to users.")
else:
    for run in runs:
        m = run["metrics"]
        st.markdown(f"**{run['model_name'] or 'unnamed model'} {run['model_version'] or ''}** · dataset `{run['dataset']}` · {run['run_at']}")
        scalar = {k: v for k, v in m.items() if isinstance(v, (int, float))}
        if scalar:
            cols = st.columns(min(len(scalar), 6))
            for col, (k, v) in zip(cols, scalar.items()):
                col.metric(k, f"{v:.3f}")
        if "confusion_matrix" in m:
            st.plotly_chart(charts.confusion_matrix_figure(m["confusion_matrix"], m.get("labels", ["no stress", "stress"])),
                            use_container_width=True)
        if run["notes"]:
            st.caption(run["notes"])

st.subheader("Known limitations")
st.markdown(
    "- Indices respond to season, planting, harvest, irrigation, clouds and sensor differences, not only stress.\n"
    "- Baselines depend on how many earlier years exist; short records give low-confidence results.\n"
    "- Assessments are AOI-level; sub-field damage can be diluted in an AOI-wide median.\n"
    "- HLS combines Landsat and Sentinel-2; small residual sensor differences remain.")
_ui.prelim_note()
