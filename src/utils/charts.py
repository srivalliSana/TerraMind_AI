"""Plotly figure builders for the dashboard (no Streamlit dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.alerts.risk_scoring import RiskAssessment
from src.config import Thresholds

LEVEL_COLORS = {"normal": "#2e9e6b", "watch": "#e0b100", "elevated": "#e8761e", "high": "#d6334a", "no_baseline": "#7c8794"}
TEMPLATE = "plotly_dark"


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font={"size": 15})
    fig.update_layout(template=TEMPLATE, xaxis={"visible": False}, yaxis={"visible": False}, height=320)
    return fig


def timeseries_figure(daily: pd.DataFrame, index: str) -> go.Figure:
    """Observed index per date, coloured by anomaly level, with the seasonal-baseline band (median +/- 2 sigma)."""
    if daily.empty or index not in daily:
        return _empty("No observations to plot")
    fig = go.Figure()
    base = daily.dropna(subset=[f"{index}_base_median"])
    if not base.empty:
        med, sig = base[f"{index}_base_median"], base[f"{index}_base_sigma"]
        fig.add_trace(go.Scatter(x=base["date"], y=med + 2 * sig, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=base["date"], y=med - 2 * sig, mode="lines", line={"width": 0}, fill="tonexty",
                                 fillcolor="rgba(120,160,220,0.18)", name="seasonal baseline ±2σ", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=base["date"], y=med, mode="lines", line={"dash": "dot", "color": "#8fb3e8"}, name="seasonal median"))
    levels = daily.get(f"{index}_z_level", pd.Series(["no_baseline"] * len(daily)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily[index], mode="lines+markers", line={"color": "#9aa7b8", "width": 1},
                             marker={"size": 9, "color": [LEVEL_COLORS.get(l, "#7c8794") for l in levels], "line": {"width": 1, "color": "#0b1220"}},
                             name=index.upper(), customdata=np.stack([levels, daily.get(f"{index}_z", pd.Series([np.nan] * len(daily))).fillna(np.nan)], axis=-1),
                             hovertemplate="%{x|%Y-%m-%d}<br>" + index.upper() + " %{y:.3f}<br>level: %{customdata[0]}<br>z: %{customdata[1]:.2f}<extra></extra>"))
    fig.update_layout(template=TEMPLATE, height=380, yaxis_title=index.upper(), margin={"t": 30, "b": 30},
                      legend={"orientation": "h", "y": 1.12})
    return fig


def seasonal_figure(daily: pd.DataFrame, index: str) -> go.Figure:
    """One line per year against day-of-year; shows the latest year against earlier ones."""
    if daily.empty or index not in daily:
        return _empty("No observations to plot")
    d = daily.assign(year=daily["date"].dt.year, doy=daily["date"].dt.dayofyear)
    years = sorted(d["year"].unique())
    fig = go.Figure()
    for y in years:
        s = d[d["year"] == y]
        latest = y == years[-1]
        fig.add_trace(go.Scatter(x=s["doy"], y=s[index], mode="lines+markers", name=str(y),
                                 line={"width": 3.5 if latest else 1.5, "color": "#2dd4bf" if latest else None},
                                 opacity=1.0 if latest else 0.55))
    fig.update_layout(template=TEMPLATE, height=380, xaxis_title="day of year", yaxis_title=index.upper(), margin={"t": 30, "b": 30})
    return fig


def zscore_figure(daily: pd.DataFrame, index: str, t: Thresholds) -> go.Figure:
    z = f"{index}_z"
    if daily.empty or z not in daily or daily[z].dropna().empty:
        return _empty("No baseline z-scores yet (needs more years of history)")
    fig = go.Figure(go.Bar(x=daily["date"], y=daily[z], marker_color=[LEVEL_COLORS.get(l, "#7c8794") for l in daily[f"{z}_level"]],
                           hovertemplate="%{x|%Y-%m-%d}<br>z %{y:.2f}<extra></extra>", name="z-score"))
    for name, col in (("watch", "#e0b100"), ("elevated", "#e8761e"), ("high", "#d6334a")):
        fig.add_hline(y=t.anomaly[f"zscore_{name}"], line={"dash": "dash", "color": col, "width": 1}, annotation_text=name,
                      annotation_position="bottom right")
    fig.update_layout(template=TEMPLATE, height=300, yaxis_title="z-score vs seasonal baseline", margin={"t": 20, "b": 30})
    return fig


def components_figure(a: RiskAssessment, t: Thresholds) -> go.Figure:
    if not a.is_scored:
        return _empty("No score: " + a.category.replace("_", " "))
    s = t.scoring
    maxes = {"magnitude": s["max_magnitude"], "persistence": s["max_persistence"], "extent": s["max_extent"]}
    names = [k for k, v in a.components.items() if v is not None]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[maxes[k] for k in names], orientation="h", marker_color="rgba(120,160,220,0.18)", name="max", hoverinfo="skip"))
    fig.add_trace(go.Bar(y=names, x=[a.components[k] for k in names], orientation="h", marker_color="#2dd4bf", name="points"))
    fig.update_layout(template=TEMPLATE, barmode="overlay", height=190, margin={"t": 10, "b": 10, "l": 10}, showlegend=False,
                      xaxis_title="points (extent not assessed when absent)")
    return fig


def confusion_matrix_figure(matrix: list[list[float]], labels: list[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Blues", text=matrix, texttemplate="%{text}"))
    fig.update_layout(template=TEMPLATE, height=360, xaxis_title="predicted", yaxis_title="actual", yaxis={"autorange": "reversed"})
    return fig
