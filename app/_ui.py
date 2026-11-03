"""Shared UI helpers for all dashboard pages. Logic lives in ``src/``; this file only presents it."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.utils import dashboard_data as dd  # noqa: E402
from src.utils.demo_data import DEMO_BANNER  # noqa: E402

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.tm-badge {display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; font-size:0.85rem; color:#0b1220;}
.tm-card {background:#111a2e; border:1px solid #1f2b45; border-radius:12px; padding:14px 18px; margin-bottom:10px;}
.tm-muted {color:#8a98b3; font-size:0.9rem;}
.tm-demo {background:#3a2a05; border:1px solid #b88a14; color:#ffd166; padding:10px 16px; border-radius:10px; margin-bottom:14px; font-weight:600;}
.tm-prelim {color:#8a98b3; font-size:0.82rem; border-left:3px solid #2dd4bf; padding-left:10px; margin:8px 0;}
[data-testid="stMetricValue"] {font-size:1.7rem;}
</style>
"""

PRELIM = ("Preliminary indicator derived from satellite-observed vegetation indices vs. a statistical seasonal baseline. "
          "It cannot identify the cause of a change and has not been validated against field observations.")


def setup(title: str, icon: str) -> None:
    st.set_page_config(page_title=f"{title} · TerraMind AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<div class='tm-muted'>{subtitle}</div>", unsafe_allow_html=True)
    st.write("")


def badge(category: str) -> str:
    color = dd.CATEGORY_COLORS.get(category, "#7c8794")
    return f"<span class='tm-badge' style='background:{color}'>{dd.CATEGORY_LABELS.get(category, category)}</span>"


def prelim_note() -> None:
    st.markdown(f"<div class='tm-prelim'>{PRELIM}</div>", unsafe_allow_html=True)


def get_context() -> dd.DataContext:
    """Sidebar data-mode switch + cached context. Demo data are always flagged; never blended with real data."""
    default = dd.default_mode()
    with st.sidebar:
        st.markdown("### TerraMind AI")
        choice = st.radio("Data mode", ["Real data", "Demo data (synthetic)"], index=0 if default == "real" else 1,
                          key="mode_choice", help="Demo mode uses generated numbers, not satellite observations.")
        mode = "real" if choice == "Real data" else "demo"
        if st.button("Refresh from disk", use_container_width=True):
            st.session_state.pop(f"ctx_{mode}", None)
    key = f"ctx_{mode}"
    if key not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state[key] = dd.load_real_context() if mode == "real" else dd.load_demo_context()
    ctx: dd.DataContext = st.session_state[key]
    if ctx.is_demo:
        st.markdown(f"<div class='tm-demo'>{DEMO_BANNER}</div>", unsafe_allow_html=True)
    else:
        with st.sidebar:
            st.caption("Showing REAL processed observations.")
    for note in ctx.notes if not ctx.is_demo else []:
        st.warning(note)
    return ctx


def empty_state(ctx: dd.DataContext, what: str = "processed observations") -> None:
    st.info(f"No {what} yet. Process an area on the **AOI Explorer** page "
            f"(or run `python -m src.cli run-series ...`), or switch to **Demo data** in the sidebar to explore the interface.")


def aoi_picker(ctx: dd.DataContext, key: str, label: str = "Area of interest") -> str | None:
    """Select an AOI; the choice is shared across pages through session state."""
    if not ctx.aois:
        return None
    ids = list(ctx.aois)
    current = st.session_state.get("aoi_id")
    idx = ids.index(current) if current in ids else 0
    picked = st.selectbox(label, ids, index=idx, key=key, format_func=lambda i: f"{ctx.aois[i].name}  ·  {i}")
    st.session_state["aoi_id"] = picked
    return picked


def base_map(center=(20.0, 0.0), zoom=2) -> folium.Map:
    """Map with free basemaps that need no API key: Esri satellite imagery (default) and OpenStreetMap."""
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri, Maxar, Earthstar Geographics", name="Satellite imagery (Esri)", control=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True, show=False).add_to(m)
    return m


def link(page: str, label: str, icon: str | None = None) -> None:
    """Page link that degrades to plain text when the page is run outside the multipage entrypoint."""
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:  # noqa: BLE001 - StreamlitPageNotFoundError when run standalone (e.g. in tests)
        st.caption(f"→ {label}")


def fit(m: folium.Map, bounds: list[list[float]]) -> None:
    m.fit_bounds(bounds, padding=(20, 20))
