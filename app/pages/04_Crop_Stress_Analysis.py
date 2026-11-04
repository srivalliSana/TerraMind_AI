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

st.subheader("Anomaly map and affected area")
ext = ctx.extents.get(aid)
if ctx.is_demo:
    st.info("Demo mode has no imagery, so there is no per-pixel map. Switch to **Real data** to see one.")
elif ext is None or not ext.ok:
    reason = ext.reason if ext is not None else "no usable observations"
    st.info(f"No per-pixel anomaly map: {reason}. A pixel baseline needs observations from at least two earlier years in the "
            "same season, on the same pixel grid.")
else:
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Affected area", f"{ext.affected_fraction:.1%}", help=f"Valid pixels in connected patches of at least {ext.min_patch_pixels} pixels with z ≤ {ext.pixel_zscore}")
    e2.metric("Affected land", f"{ext.area_ha:,.0f} ha")
    e3.metric("Patches", ext.patches, help=f"Largest patch: {ext.largest_patch_pixels} pixels")
    e4.metric("Expected by chance", f"{ext.expected_by_chance:.1%}", help="Share of pixels below the threshold in a normal year if the baseline were exact")
    raw = ext.raw_affected_pixels / ext.evaluated_pixels
    if raw <= ext.expected_by_chance * 1.5:
        st.success(f"{raw:.1%} of pixels are below the threshold, close to the {ext.expected_by_chance:.1%} expected by chance: "
                   "no widespread decline stands out.")
    out = Path(ext.output_dir)
    z_path = out / "anomaly_z.tif"
    if z_path.is_file():
        import folium
        from streamlit_folium import st_folium

        rgba, bounds = dd.raster_overlay(z_path, "RdBu", -4.0, 4.0)
        amap = _ui.base_map()
        folium.raster_layers.ImageOverlay(image=rgba, bounds=bounds, name="pixel z-score (red = below normal)", opacity=0.85).add_to(amap)
        mask_path = out / "affected_mask.tif"
        if mask_path.is_file():
            import numpy as np
            import rasterio
            from rasterio.warp import transform_bounds

            with rasterio.open(mask_path) as ms:
                mk = ms.read(1)
                w, s_, e, n = transform_bounds(ms.crs, "EPSG:4326", *ms.bounds)
            ov = np.zeros(mk.shape + (4,), dtype=np.uint8)
            ov[mk == 1] = (255, 200, 0, 230)
            folium.raster_layers.ImageOverlay(image=ov, bounds=[[s_, w], [n, e]], name="affected patches", opacity=1.0).add_to(amap)
        _ui.fit(amap, bounds)
        folium.LayerControl(collapsed=False).add_to(amap)
        st_folium(amap, height=480, use_container_width=True, returned_objects=[], key=f"anomaly_map_{aid}")
        st.caption(f"Colour = z-score of each pixel vs the same place in {ext.history_years} earlier year(s), {len(ext.history_granules)} observations "
                   f"within ±{ext.window_days} days of {ext.target_date}. Blue = above normal, red = below. Yellow = affected patches.")
    st.warning(ext.caveat)
    for w in ext.warnings:
        st.caption(f"warning: {w}")
_ui.prelim_note()
