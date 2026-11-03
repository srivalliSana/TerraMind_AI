import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import folium  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

import _ui  # noqa: E402
from src.utils import dashboard_data as dd  # noqa: E402

_ui.setup("Global Overview", ":material/public:")
ctx = _ui.get_context()
_ui.header("Global Overview", "All monitored areas, their latest status and recent processing activity")

m = dd.overview_metrics(ctx)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Monitored AOIs", m["aois"])
c2.metric("Observations", m["observations"])
c3.metric("Latest observation", m["latest_observation"] or "none")
c4.metric("Active alerts", m["active_alerts"])

table = dd.aoi_status_table(ctx)
if table.empty:
    _ui.empty_state(ctx)
    st.stop()

fmap = _ui.base_map()
all_bounds = []
for _, row in table.iterrows():
    aoi = ctx.aois[row["aoi_id"]]
    color = dd.CATEGORY_COLORS.get(row["category"], "#7c8794")
    score = "n/a" if row["score"] != row["score"] or row["score"] is None else f"{row['score']:.0f}/100"
    folium.GeoJson(
        aoi.to_geojson(), name=row["name"],
        style_function=lambda _f, c=color: {"color": c, "weight": 3, "fillColor": c, "fillOpacity": 0.35},
        tooltip=f"{row['name']}: {dd.CATEGORY_LABELS.get(row['category'], row['category'])} (score {score})",
    ).add_to(fmap)
    minx, miny, maxx, maxy = aoi.bounds
    all_bounds += [[miny, minx], [maxy, maxx]]
_ui.fit(fmap, all_bounds)
st_folium(fmap, height=460, use_container_width=True, returned_objects=[])
st.caption("Polygon colour = latest preliminary risk category. Grey = not enough data or history to assess.")

st.subheader("Areas")
view = table.assign(category=table["category"].map(lambda c: dd.CATEGORY_LABELS.get(c, c)))
st.dataframe(
    view.drop(columns=["aoi_id"]), hide_index=True, use_container_width=True,
    column_config={"score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f"),
                   "area_km2": st.column_config.NumberColumn("Area km²"), "active_alerts": "Active alerts"})
picked = _ui.aoi_picker(ctx, "overview_aoi", "Focus on an area")
if picked:
    a = ctx.assessments[picked]
    st.markdown(f"**{ctx.aois[picked].name}** &nbsp; {_ui.badge(a.category)}", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        _ui.link("pages/04_Crop_Stress_Analysis.py", "Open crop stress analysis", ":material/monitoring:")
    with cols[1]:
        _ui.link("pages/03_Satellite_Observations.py", "Open satellite observations", ":material/layers:")

left, right = st.columns([3, 2])
with left:
    st.subheader("Recent processing activity")
    activity = dd.recent_activity(ctx)
    if activity.empty:
        st.caption("No processing recorded yet.")
    else:
        st.dataframe(activity, hide_index=True, use_container_width=True,
                     column_config={"valid_fraction": st.column_config.ProgressColumn("Valid", min_value=0, max_value=1, format="%.0f%%")})
with right:
    st.subheader("Data availability and quality")
    q = table[["name", "observations", "latest_observation"]]
    st.dataframe(q, hide_index=True, use_container_width=True)
    low = [ctx.aois[a].name for a, df in ctx.daily.items() if len(df) and df["valid_fraction"].iloc[-1] < 0.5]
    if low:
        st.warning("Latest observation has under 50% valid pixels for: " + ", ".join(low))
    st.caption("Coverage reflects only the areas processed here. Querying a global catalog does not mean global coverage.")
_ui.prelim_note()
