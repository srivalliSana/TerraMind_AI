import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import folium  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from folium.plugins import Draw  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

import _ui  # noqa: E402
from src import config  # noqa: E402
from src.data.providers.base import AuthError, ProviderError  # noqa: E402
from src.data.search import get_provider, search_granules  # noqa: E402
from src.geospatial import aoi as aoi_mod  # noqa: E402
from src.pipeline.orchestrator import run_series, select_per_slot  # noqa: E402
from src.storage import repositories as repo  # noqa: E402

_ui.setup("AOI Explorer", ":material/draw:")
ctx = _ui.get_context()
_ui.header("AOI Explorer", "Define an area of interest, discover satellite observations over it, then process them")
limits = config.get_settings().aoi


def use(builder, *args, **kw):
    """Build an AOI, store it as the working AOI, or show the validation error."""
    try:
        st.session_state["work_aoi"] = builder(*args, limits=limits, **kw)
        st.session_state.pop("search_result", None)
    except (aoi_mod.AOIError, ValueError) as exc:
        st.error(f"Invalid AOI: {exc}")


st.subheader("1. Define the area")
t_box, t_geojson, t_shp, t_draw, t_saved = st.tabs(["Bounding box", "GeoJSON upload", "Shapefile upload", "Draw on map", "Saved AOIs"])

with t_box:
    c = st.columns(4)
    min_lon = c[0].number_input("Min longitude", value=-93.7, format="%.5f")
    min_lat = c[1].number_input("Min latitude", value=41.9, format="%.5f")
    max_lon = c[2].number_input("Max longitude", value=-93.6, format="%.5f")
    max_lat = c[3].number_input("Max latitude", value=42.0, format="%.5f")
    name = st.text_input("Name", "my-area")
    if st.button("Use this bounding box"):
        use(aoi_mod.from_bbox, min_lon, min_lat, max_lon, max_lat, name=name)

with t_geojson:
    up = st.file_uploader("GeoJSON file (Polygon/MultiPolygon, WGS84)", type=["geojson", "json"], key="up_geojson")
    if up is not None and st.button("Use uploaded GeoJSON"):
        use(aoi_mod.from_geojson, up.getvalue().decode("utf-8", errors="replace"), name=Path(up.name).stem)

with t_shp:
    st.caption("Upload the .shp together with its .shx and .dbf (and .prj, needed to know the coordinate system).")
    ups = st.file_uploader("Shapefile parts", type=["shp", "shx", "dbf", "prj", "cpg"], accept_multiple_files=True, key="up_shp")
    if ups and st.button("Use uploaded shapefile"):
        shp_files = [u for u in ups if u.name.lower().endswith(".shp")]
        if not shp_files:
            st.error("Include the .shp file.")
        else:
            tmp = Path(tempfile.mkdtemp(prefix="tm_shp_"))
            for u in ups:
                (tmp / Path(u.name).name).write_bytes(u.getvalue())
            use(aoi_mod.from_shapefile, tmp / Path(shp_files[0].name).name)

with t_draw:
    st.caption("Draw a polygon or rectangle, then press the button below.")
    dm = _ui.base_map((42.0, -93.6), 8)
    Draw(export=False, draw_options={"polygon": True, "rectangle": True, "polyline": False, "circle": False,
                                     "marker": False, "circlemarker": False}).add_to(dm)
    out = st_folium(dm, height=420, use_container_width=True, returned_objects=["last_active_drawing"], key="draw_map")
    drawn = (out or {}).get("last_active_drawing")
    if drawn and st.button("Use drawn shape"):
        use(aoi_mod.from_geojson, {"type": "Feature", "properties": {}, "geometry": drawn["geometry"]}, name="drawn-area")

with t_saved:
    saved = repo.list_aois(ctx.conn) if not ctx.is_demo else []
    if not saved:
        st.caption("No saved AOIs in this mode." + (" (Demo AOIs are synthetic and not offered here.)" if ctx.is_demo else ""))
    else:
        pick = st.selectbox("Saved AOI", [r["id"] for r in saved], format_func=lambda i: next(f"{r['name']}  ·  {r['id']}" for r in saved if r["id"] == i))
        if st.button("Load saved AOI"):
            st.session_state["work_aoi"] = repo.get_aoi(ctx.conn, pick)
            st.session_state.pop("search_result", None)

work = st.session_state.get("work_aoi")
if work is None:
    st.info("Define an area above to continue. Limits come from configuration: "
            f"{limits.min_area_km2}–{limits.max_area_km2:,.0f} km².")
    st.stop()

st.success(f"Working AOI: **{work.name}** · id `{work.id}` · {work.area_km2:,.2f} km² · bounds {tuple(round(b, 4) for b in work.bounds)}")
pm = _ui.base_map()
folium.GeoJson(work.to_geojson(), style_function=lambda _f: {"color": "#2dd4bf", "weight": 3, "fillOpacity": 0.25}).add_to(pm)
minx, miny, maxx, maxy = work.bounds
_ui.fit(pm, [[miny, minx], [maxy, maxx]])
st_folium(pm, height=300, use_container_width=True, returned_objects=[], key="preview_map")
if st.button("Save AOI to database", disabled=ctx.is_demo, help="Disabled in demo mode"):
    repo.upsert_aoi(ctx.conn, work)
    st.toast("AOI saved")

st.subheader("2. Discover observations")
provider = get_provider()
c1, c2, c3 = st.columns(3)
dates = c1.date_input("Date range", (date.today() - timedelta(days=30), date.today()), key="search_dates")
provider_name = c2.selectbox("Provider", [provider.name])
sensors = c3.multiselect("Sensors", provider.available_sensors, default=provider.available_sensors)
c4, c5 = st.columns(2)
max_cloud = c4.slider("Max tile cloud cover (%)", 0, 100, int(config.get_settings().search.max_cloud_cover),
                      help="Catalog value for the whole ~110 km tile. Real cloud cover over your AOI is measured after processing.")
limit = c5.number_input("Max results per sensor", 1, config.get_settings().search.max_results, min(20, config.get_settings().search.max_results))

if st.button("Search catalog", type="primary"):
    if not (isinstance(dates, tuple) and len(dates) == 2):
        st.error("Pick a start and end date.")
    else:
        with st.spinner("Querying NASA CMR (no login needed)..."):
            try:
                st.session_state["search_result"] = (search_granules(
                    work, dates[0], dates[1], sensors=tuple(sensors) or None, max_cloud_cover=float(max_cloud), max_results=int(limit)),
                    dates)
            except (ProviderError, ValueError) as exc:
                st.error(str(exc))

found = st.session_state.get("search_result")
if found:
    result, used_dates = found
    for w in result.warnings:
        st.warning(w)
    if not result.granules:
        st.info("No observations match. Try a wider date range or a higher cloud limit.")
    else:
        df = pd.DataFrame([{"date": g.acquired_date, "sensor": g.sensor, "tile": g.tile_id, "tile cloud %": g.cloud_cover,
                            "tile coverage %": g.spatial_coverage, "granule": g.granule_id} for g in result.granules])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(f"{len(df)} observation(s) found. This is catalog metadata only; nothing has been downloaded.")
        cov = _ui.base_map()
        folium.GeoJson(work.to_geojson(), style_function=lambda _f: {"color": "#2dd4bf", "weight": 3, "fillOpacity": 0.2}).add_to(cov)
        for tile in {g.tile_id: g for g in result.granules}.values():
            if tile.footprint is not None:
                folium.GeoJson(tile.footprint.__geo_interface__, tooltip=f"tile {tile.tile_id}",
                               style_function=lambda _f: {"color": "#8fb3e8", "weight": 1, "fillOpacity": 0.05}).add_to(cov)
        _ui.fit(cov, [[miny - 0.3, minx - 0.3], [maxy + 0.3, maxx + 0.3]])
        st_folium(cov, height=300, use_container_width=True, returned_objects=[], key="coverage_map")

        st.subheader("3. Process")
        slot = st.number_input("One best observation per N days", 1, 30, 7)
        plan = select_per_slot(result.granules, int(slot))
        cap = st.number_input("Maximum granules to download", 1, 20, min(len(plan), 6) or 1)
        plan = sorted(plan, key=lambda g: g.cloud_cover or 100)[: int(cap)]
        est = len(plan) * 46
        st.write(f"Plan: **{len(plan)}** granule(s), roughly **{est} MB** (3 bands + Fmask for Sentinel-2; Landsat is similar).")
        sensor_args = " ".join(sensors)
        cmd = (f"python -m src.cli run-series --bbox {min_lon} {min_lat} {max_lon} {max_lat} --ranges {used_dates[0]}:{used_dates[1]} "
               f"--sensors {sensor_args} --slot-days {int(slot)} --max-granules {int(cap)}")
        st.code(cmd, language="powershell")
        status = provider.auth_status(verify=False)
        can = (not ctx.is_demo) and status.state in ("unchecked", "ok")
        if status.state == "not_configured":
            st.warning(status.message)
        if ctx.is_demo:
            st.info("Processing is disabled in demo mode.")
        if st.button("Download and process here", disabled=not can, help="Runs in this session; large jobs are better run from the CLI."):
            s = config.get_settings()
            with st.status("Downloading and processing...", expanded=True) as box:
                try:
                    rep = run_series(work, [(used_dates[0], used_dates[1])], raw_dir=s.data_dir / "raw",
                                     manifest_dir=s.data_dir / "manifests", out_root=s.data_dir / "processed",
                                     sensors=tuple(sensors) or None, slot_days=int(slot), max_granules=int(cap))
                    box.update(label=f"Done: {len(rep.processed)} processed, {len(rep.failed)} skipped", state="complete")
                    for gid, why in rep.failed:
                        st.write(f"skipped `{gid}`: {why}")
                    repo.upsert_aoi(ctx.conn, work)
                    st.session_state.pop("ctx_real", None)
                    st.success("Finished. Open **Crop Stress Analysis** or use *Refresh from disk*.")
                except AuthError as exc:
                    box.update(label="Authentication problem (search works, download does not)", state="error")
                    st.error(str(exc))
                except ProviderError as exc:
                    box.update(label="Failed", state="error")
                    st.error(str(exc))
_ui.prelim_note()
