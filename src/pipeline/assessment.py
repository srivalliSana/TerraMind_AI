"""One call that turns an AOI's processed outputs into series, extent and a risk assessment."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src import config
from src.alerts.risk_scoring import RiskAssessment, assess_risk
from src.pipeline.analysis import analyze_aoi
from src.pipeline.extent import ExtentResult, compute_extent


@dataclass
class AOIAssessment:
    daily: pd.DataFrame
    assessment: RiskAssessment
    extent: ExtentResult | None


def assess_aoi(processed_root: Path, aoi_id: str, *, with_extent: bool = True, as_of: date | None = None) -> AOIAssessment:
    """Analyse an AOI. The affected-area component is included only when the per-pixel analysis succeeded."""
    s = config.get_settings()
    daily = analyze_aoi(processed_root, aoi_id)
    extent = None
    if with_extent and not daily.empty:
        extent = compute_extent(Path(processed_root), aoi_id)
    use = extent if extent is not None and extent.ok else None
    a = assess_risk(daily, config.load_thresholds(), aoi_id=aoi_id, as_of=as_of,
                    affected_fraction=use.affected_fraction if use else None,
                    extent_note=use.note() if use else None,
                    min_baseline_obs=s.timeseries.min_observations_for_baseline)
    return AOIAssessment(daily, a, extent)
