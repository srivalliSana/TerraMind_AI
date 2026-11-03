import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import _ui  # noqa: E402
from src.alerts.alert_manager import AlertManager  # noqa: E402
from src.utils import dashboard_data as dd  # noqa: E402

_ui.setup("Alerts", ":material/notifications:")
ctx = _ui.get_context()
_ui.header("Alerts", "Preliminary warnings with the evidence behind them. Not diagnoses; not yet validated against field observations")
mgr = AlertManager(ctx.conn)

if ctx.aois:
    with st.expander("Evaluate current assessments", expanded=False):
        st.caption("Applies each AOI's latest assessment to its alert: creates, updates, escalates, or (after 2 consecutive clear "
                   "observations) resolves. Re-running on the same observation changes nothing.")
        if st.button("Evaluate all AOIs"):
            for aoi_id in ctx.aois:
                res = dd.evaluate_alerts(ctx, aoi_id)
                st.write(f"{ctx.aois[aoi_id].name}: **{res.action}**" + (f" ({res.category})" if res.category else ""))

f1, f2, f3, f4 = st.columns(4)
status = f1.selectbox("Status", ["any", "active", "resolved", "dismissed"], index=1)
category = f2.selectbox("Category", ["any", "watch", "elevated", "high"])
aoi_choice = f3.selectbox("AOI", ["any"] + list(ctx.aois), format_func=lambda i: "any" if i == "any" else ctx.aois[i].name)
since = f4.date_input("Updated since", date.today() - timedelta(days=3650))
filters = {"status": None if status == "any" else status, "category": None if category == "any" else category,
           "aoi_id": None if aoi_choice == "any" else aoi_choice, "since": since.isoformat()}
rows = mgr.list(**filters)

if not rows:
    st.info("No alerts match these filters. That is the expected state when conditions look normal.")
else:
    df = pd.DataFrame([{"alert": r["id"], "AOI": r["aoi_name"], "status": r["status"], "category": r["category"], "score": r["score"],
                        "confidence": r["confidence_label"], "first seen": r["first_seen"], "last seen": r["last_seen"]} for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.0f")})

    pick = st.selectbox("Inspect alert", [r["id"] for r in rows],
                        format_func=lambda i: next(f"{r['aoi_name']} · {r['category']} · {r['last_seen']} · {i}" for r in rows if r["id"] == i))
    r = mgr.get(pick)
    st.markdown(f"#### {_ui.badge(r['category'])} &nbsp; {r['status']}", unsafe_allow_html=True)
    ec1, ec2 = st.columns([3, 2])
    with ec1:
        st.markdown("**Supporting evidence**")
        for e in json.loads(r["evidence_json"]):
            st.markdown(f"- {e}")
        assessment = json.loads(r["assessment_json"])
        st.markdown("**Recommended follow-up**")
        for rec in assessment.get("recommendations", []):
            st.markdown(f"- {rec}")
    with ec2:
        st.markdown("**History**")
        st.dataframe(pd.DataFrame([dict(e) for e in mgr.events(pick)])[["at", "event_type", "from_category", "to_category", "note"]],
                     hide_index=True, use_container_width=True)
    note = st.text_input("Note (optional)", key=f"note_{pick}")
    b1, b2, _ = st.columns([1, 1, 4])
    if r["status"] == "active":
        if b1.button("Dismiss", key=f"dismiss_{pick}"):
            mgr.dismiss(pick, note or None)
            st.rerun()
        if b2.button("Mark resolved", key=f"resolve_{pick}"):
            mgr.resolve(pick, note or None)
            st.rerun()
    else:
        st.caption("Only active alerts can be dismissed or resolved.")
    with st.expander("Full assessment (JSON)"):
        st.json(assessment)

st.subheader("Export")
e1, e2 = st.columns(2)
e1.download_button("Download alerts (CSV)", mgr.export_csv(**filters), "terramind_alerts.csv", "text/csv")
e2.download_button("Download alerts (GeoJSON)", json.dumps(mgr.export_geojson(**filters), indent=2), "terramind_alerts.geojson", "application/geo+json")
st.caption("Every exported row is marked as a preliminary indicator.")
_ui.prelim_note()
