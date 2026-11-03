import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import folium  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

import _ui  # noqa: E402
from src.utils import dashboard_data as dd  # noqa: E402

_ui.setup("Satellite Observations", ":material/layers:")
ctx = _ui.get_context()
_ui.header("Satellite Observations", "Index layers for one acquisition, masked for cloud, shadow, snow and water")

if ctx.is_demo:
    st.info("Demo mode has no imagery: the demo AOIs are synthetic time series. Switch to **Real data** to view index maps.")
    st.stop()
if not ctx.aois:
    _ui.empty_state(ctx)
    st.stop()

aid = _ui.aoi_picker(ctx, "obs_aoi")
obs = dd.list_observations(ctx.processed_root, aid)
if not obs:
    _ui.empty_state(ctx, "processed observations for this AOI")
    st.stop()

label = lambda o: f"{o['acquired'][:10]} · {o['summary']['sensor']} · valid {o['summary']['quality']['valid_fraction']:.0%}"  # noqa: E731
choice = st.selectbox("Observation", range(len(obs)), format_func=lambda i: label(obs[i]))
o = obs[choice]
s = o["summary"]
q = s["quality"]

if not s["usable_for_analysis"]:
    st.warning(f"Only {q['valid_fraction']:.0%} of the AOI is valid on this date; it is excluded from time-series analysis by default.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Acquired", o["acquired"][:10])
c2.metric("Sensor", s["sensor"])
c3.metric("Resolution", f"{s['resolution_m'][0]:.0f} m")
c4.metric("Valid pixels in AOI", f"{q['valid_fraction']:.0%}")

layers = {"NDVI": ("ndvi.tif", "RdYlGn", 0.0, 1.0), "NDMI": ("ndmi.tif", "Blues", -0.2, 0.7), "Usable pixels": ("valid_mask.tif", "gray", 0, 1)}
fmap = _ui.base_map()
bounds_all = []
for i, (name, (fname, cmap, vmin, vmax)) in enumerate(layers.items()):
    path = o["dir"] / fname
    if not path.is_file():
        continue
    rgba, bounds = dd.raster_overlay(path, cmap, vmin, vmax)
    folium.raster_layers.ImageOverlay(image=rgba, bounds=bounds, name=name, opacity=0.9, show=(i == 0)).add_to(fmap)
    bounds_all += bounds
if bounds_all:
    _ui.fit(fmap, bounds_all)
folium.LayerControl(collapsed=False).add_to(fmap)
st_folium(fmap, height=520, use_container_width=True, returned_objects=[], key=f"obs_map_{o['granule_id']}")

lc1, lc2 = st.columns(2)
with lc1:
    st.markdown("**Legend**")
    st.markdown("<div style='height:14px;border-radius:7px;background:linear-gradient(90deg,#a50026,#f46d43,#fee08b,#a6d96a,#006837)'></div>"
                "<div style='display:flex;justify-content:space-between;font-size:0.8rem' class='tm-muted'>"
                "<span>NDVI 0</span><span>0.5</span><span>1</span></div>", unsafe_allow_html=True)
    st.markdown("NDVI: 0 = bare/sparse, ~0.3 = sparse or stressed, 0.6–0.9 = dense healthy vegetation.  \n"
                "NDMI: higher = more canopy water content (also affected by structure and soil).  \n"
                "Transparent = masked (cloud, shadow, adjacent, snow/ice, water, or no data).")
with lc2:
    st.markdown("**Index statistics over valid AOI pixels**")
    rows = [{"index": k.upper(), **{kk: round(vv, 3) for kk, vv in v.items() if kk != "count"}, "pixels": v["count"]}
            for k, v in s["index_stats"].items() if v]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("Quality breakdown (inside the AOI)")
qdf = pd.DataFrame({"component": ["valid", "cloud", "adjacent to cloud/shadow", "cloud shadow", "snow/ice", "water", "no data"],
                    "fraction": [q["valid_fraction"], q["cloud_fraction"], q["adjacent_fraction"], q["shadow_fraction"],
                                 q["snow_ice_fraction"], q["water_fraction"], q["fill_fraction"]]})
st.dataframe(qdf, hide_index=True, use_container_width=True,
             column_config={"fraction": st.column_config.ProgressColumn("share of AOI", min_value=0, max_value=1, format="%.1f%%")})
st.caption("Cloud, adjacent, shadow, snow and water can overlap, so shares may add up to more than 100%.")

st.subheader("RGB composite")
st.info("No RGB composite for this observation: only the Red, NIR and SWIR1 bands and the Fmask were downloaded. "
        "Download Blue (B02) and Green (B03) as well to enable true-colour display.")

with st.expander("Processing details and provenance"):
    st.json({k: s[k] for k in ("granule_id", "sensor", "acquired", "crs", "resolution_m", "window_shape", "layers", "masking",
                                "catalog", "pipeline_version", "input_sha256")})
st.caption(s["caveat"])
