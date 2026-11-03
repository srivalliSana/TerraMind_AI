import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st  # noqa: E402

import _ui  # noqa: E402
from src import config  # noqa: E402
from src.data.search import get_provider  # noqa: E402
from src.utils import dashboard_data as dd  # noqa: E402
from src.storage.database import current_version  # noqa: E402

_ui.setup("Settings", ":material/settings:")
ctx = _ui.get_context()
_ui.header("Settings", "Limits, thresholds and data-source status. Secrets are never shown")

s = config.get_settings()
t = config.load_thresholds()
local = (config.CONFIG_DIR / "local.yaml").is_file()

tab_status, tab_limits, tab_thresholds, tab_system = st.tabs(["Authentication and data", "Limits and defaults", "Thresholds", "System"])

with tab_status:
    st.markdown("**NASA Earthdata credentials** (presence only; values are never displayed or logged)")
    for k, present in config.credential_status().items():
        st.write(f"{'✅' if present else '⚪'} `{k}`: {'set' if present else 'not set'}")
    provider = get_provider()
    status = provider.auth_status(verify=False)
    st.write(f"Catalog search: **{'available' if status.can_search else 'unavailable'}** (public, no login needed)")
    st.write(f"Downloads: **{status.state}**. {status.message}")
    if st.button("Verify credentials with Earthdata", help="Contacts urs.earthdata.nasa.gov; shows only ok/failed."):
        with st.spinner("Verifying..."):
            v = provider.auth_status(verify=True)
        (st.success if v.state == "ok" else st.error)(f"{v.state}: {v.message}")
    st.caption("Set `EARTHDATA_TOKEN` in a local `.env` file (never in the repository). See the README for step-by-step setup.")

    st.markdown("**Data on disk**")
    for name in ("raw", "processed", "manifests"):
        p = s.data_dir / name
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0
        st.write(f"`{p}`: {size:,.0f} MB")

with tab_limits:
    st.caption("Saved to `configs/local.yaml` (gitignored) after validation. Defaults in `configs/default.yaml` are never modified.")
    with st.form("limits"):
        c1, c2 = st.columns(2)
        max_area = c1.number_input("Max AOI area (km²)", 1.0, 100000.0, float(s.aoi.max_area_km2))
        min_area = c2.number_input("Min AOI area (km²)", 0.001, 1000.0, float(s.aoi.min_area_km2), format="%.3f")
        max_results = c1.number_input("Max search results per sensor", 1, 2000, int(s.search.max_results))
        max_cloud = c2.number_input("Default max tile cloud cover (%)", 0, 100, int(s.search.max_cloud_cover))
        min_valid = c1.slider("Min valid fraction for an observation to be analysed", 0.0, 1.0, float(s.processing.min_valid_fraction))
        workers = c2.number_input("Max workers", 1, 64, int(s.processing.max_workers))
        st.markdown("Masking")
        m1, m2, m3, m4, m5 = st.columns(5)
        mask_cloud = m1.checkbox("Cloud", s.processing.mask_cloud)
        mask_adj = m2.checkbox("Adjacent", s.processing.mask_adjacent)
        mask_sh = m3.checkbox("Shadow", s.processing.mask_shadow)
        mask_snow = m4.checkbox("Snow/ice", s.processing.mask_snow_ice)
        mask_water = m5.checkbox("Water", s.processing.mask_water)
        if st.form_submit_button("Save limits"):
            try:
                config.save_local_overrides({
                    "aoi": {"max_area_km2": max_area, "min_area_km2": min_area},
                    "search": {"max_results": int(max_results), "max_cloud_cover": float(max_cloud)},
                    "processing": {"min_valid_fraction": min_valid, "max_workers": int(workers), "mask_cloud": mask_cloud,
                                   "mask_adjacent": mask_adj, "mask_shadow": mask_sh, "mask_snow_ice": mask_snow, "mask_water": mask_water}})
                st.success("Saved. Reload the page to apply.")
            except config.ConfigError as exc:
                st.error(str(exc))

with tab_thresholds:
    st.warning(f"Thresholds are engineering defaults (validated against field data: **{t.validated}**). "
               "Calibrate per crop and region before relying on them.")
    with st.form("thresholds"):
        a = t.anomaly
        z1, z2, z3 = st.columns(3)
        zw = z1.number_input("z watch", -10.0, 0.0, float(a["zscore_watch"]), step=0.1)
        ze = z2.number_input("z elevated", -10.0, 0.0, float(a["zscore_elevated"]), step=0.1)
        zh = z3.number_input("z high", -10.0, 0.0, float(a["zscore_high"]), step=0.1)
        persist = st.number_input("Consecutive observations for 'sustained'", 1, 10, int(a["min_persistence_obs"]))
        if st.form_submit_button("Save thresholds"):
            try:
                config.save_local_overrides(thresholds={"anomaly": {"zscore_watch": zw, "zscore_elevated": ze, "zscore_high": zh,
                                                                    "min_persistence_obs": float(persist)}})
                st.success("Saved. Reload the page to apply.")
            except config.ConfigError as exc:
                st.error(str(exc))
    st.json({"anomaly": t.anomaly, "affected_area": t.affected_area, "quality": t.quality, "scoring": t.scoring,
             "risk_categories": t.risk_categories})
    if local and st.button("Reset all local overrides to defaults"):
        config.reset_local_overrides()
        st.success("Local overrides removed. Reload the page.")

with tab_system:
    st.write(f"Database: `{s.db_path}` · schema version **{current_version(ctx.conn)}**")
    st.write(f"Local overrides active: **{'yes' if local else 'no'}**")
    st.write(f"Data mode on open: **{dd.default_mode(s)}** (demo only when no processed data exist)")
    st.json({"search": s.search.model_dump(), "aoi": s.aoi.model_dump(mode="json"), "processing": s.processing.model_dump(),
             "timeseries": s.timeseries.model_dump()})
_ui.prelim_note()
