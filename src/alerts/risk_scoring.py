"""Interpretable crop-stress risk scoring (PRELIMINARY indicator, not a diagnosis).

score (0-100) = 100 * (magnitude + persistence + extent) / (maximum of the components that are available)

- magnitude   : depth of the decline below the seasonal baseline (z-score, declines only)
- persistence : consecutive anomalous observations (separates noise from sustained change)
- extent      : fraction of the AOI affected; when unknown it is excluded and confidence is reduced

Weights, thresholds and categories come from ``configs/thresholds.yaml`` and are NOT validated
universal rules; calibrate per crop/region against field observations.
The confidence value is a heuristic combining baseline depth, data quality, index agreement and
recency. It is NOT a calibrated probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.config import Thresholds

CATEGORIES = ["informational", "watch", "elevated", "high"]
INSUFFICIENT_DATA = "insufficient_data"
INSUFFICIENT_BASELINE = "insufficient_baseline"
CAVEAT = ("Satellite-derived indices cannot identify the cause of a decline (disease, nutrients, water, "
          "harvest, planting stage, residual cloud...). This is a preliminary indicator until validated "
          "against field observations.")


@dataclass
class RiskAssessment:
    aoi_id: str | None
    as_of: str
    category: str
    score: float | None
    components: dict[str, float | None]
    evidence: list[str]
    observation_dates: list[str]
    quality: dict[str, float | int | None]
    confidence: float | None
    confidence_label: str | None
    affected_fraction: float | None
    recommendations: list[str]
    caveats: list[str] = field(default_factory=lambda: [CAVEAT])
    validated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_scored(self) -> bool:
        return self.score is not None


def _severity(z: float, z_high: float) -> float:
    """0..1 depth of decline; positive z (greener than normal) counts as 0."""
    if not np.isfinite(z):
        return float("nan")
    return float(min(max(-z, 0.0) / abs(z_high), 1.0))


def category_for(score: float, t: Thresholds) -> str:
    bounds = t.risk_categories
    for name in CATEGORIES:
        if score <= bounds[name]:
            return name
    return CATEGORIES[-1]


def assess_risk(
    daily: pd.DataFrame,
    t: Thresholds,
    *,
    aoi_id: str | None = None,
    affected_fraction: float | None = None,
    as_of: date | None = None,
    min_baseline_obs: int = 5,
    extent_note: str | None = None,
) -> RiskAssessment:
    """Assess the latest observation of an analysed series (output of ``analyze_aoi``)."""
    s = t.scoring
    if daily.empty:
        return _no_answer(aoi_id, as_of or date.today(), INSUFFICIENT_DATA, [],
                          ["No usable observations are available for this AOI."],
                          ["Run observations over this AOI (clear-sky acquisitions) before relying on any assessment."], {})
    last = daily.iloc[-1]
    last_date = pd.Timestamp(last["date"]).date()
    as_of = as_of or last_date
    stale_days = (as_of - last_date).days
    quality = {"valid_fraction": float(last["valid_fraction"]), "baseline_observations": int(last.get("ndvi_base_n", 0)),
               "days_since_latest_observation": stale_days, "n_granules": int(last.get("n_granules", 1))}

    if quality["valid_fraction"] < t.quality.get("min_valid_fraction", 0.3):
        return _no_answer(aoi_id, as_of, INSUFFICIENT_DATA, [str(last_date)],
                          [f"Only {quality['valid_fraction']:.0%} of the AOI is valid on {last_date}."],
                          ["Wait for the next clear observation."], quality)

    z_hi = t.anomaly["zscore_high"]
    sev = {i: _severity(float(last.get(f"{i}_z", np.nan)), z_hi) for i in ("ndvi", "ndmi")}
    if all(np.isnan(v) for v in sev.values()):
        return _no_answer(aoi_id, as_of, INSUFFICIENT_BASELINE, [str(last_date)],
                          [f"Fewer than {min_baseline_obs} comparable observations from earlier years exist for this time "
                           f"of year (have {quality['baseline_observations']}); a seasonal baseline cannot be built."],
                          ["Accumulate more historical seasons (ideally 5+ years) or add more observations in the window."],
                          quality)

    weights = {"ndvi": s["weight_ndvi"], "ndmi": s["weight_ndmi"]}
    mag_frac = max(weights[i] * sev[i] for i in sev if not np.isnan(sev[i]))
    run = int(max(last.get("ndvi_z_run", 0), last.get("ndmi_z_run", 0)))
    pers_frac = min(run / s["persistence_full_obs"], 1.0)
    comps = {"magnitude": s["max_magnitude"] * min(mag_frac, 1.0), "persistence": s["max_persistence"] * pers_frac,
             "extent": None}
    available_max = s["max_magnitude"] + s["max_persistence"]
    total = comps["magnitude"] + comps["persistence"]
    if affected_fraction is not None:
        ext = min(affected_fraction / t.affected_area["min_fraction_high"], 1.0)
        comps["extent"] = s["max_extent"] * ext
        available_max += s["max_extent"]
        total += comps["extent"]
    score = round(100.0 * total / available_max, 1)
    category = category_for(score, t)

    # --- evidence ---------------------------------------------------------------------------------
    evidence = []
    for idx in ("ndvi", "ndmi"):
        z = last.get(f"{idx}_z", np.nan)
        if np.isfinite(z):
            evidence.append(f"{idx.upper()} {last[idx]:.3f} vs seasonal median {last[f'{idx}_base_median']:.3f} "
                            f"(z = {z:+.2f}, {last[f'{idx}_z_level']}, baseline from {int(last[f'{idx}_base_n'])} earlier "
                            f"observations" + (f" in {int(last[f'{idx}_base_years'])} year(s))" if f"{idx}_base_years" in last else ")"))
    evidence.append(f"{run} consecutive anomalous observation(s)" if run else "latest observation is not anomalous")
    evidence.append(f"{quality['valid_fraction']:.0%} of the AOI valid (cloud/shadow/snow/water excluded)")
    evidence.append("affected area: " + (f"{affected_fraction:.0%} of valid pixels" + (f" ({extent_note})" if extent_note else "")
                                         if affected_fraction is not None else "not assessed (AOI-level statistics only)"))
    dates = [pd.Timestamp(d).date().isoformat() for d in daily["date"].iloc[-max(run, 1):]]

    # --- confidence (heuristic) ---------------------------------------------------------------------
    base_conf = min(quality["baseline_observations"] / (2 * min_baseline_obs), 1.0)
    qual_conf = min(quality["valid_fraction"] / 0.7, 1.0)
    both = [i for i in ("ndvi", "ndmi") if np.isfinite(last.get(f"{i}_z", np.nan))]
    anomalous = [i for i in both if last[f"{i}_z_level"] in ("watch", "elevated", "high")]
    agree = 1.0 if len(both) == 2 and len(anomalous) in (0, 2) else 0.5
    fresh = 1.0 if stale_days <= s["stale_after_days"] else 0.5
    confidence = round(0.4 * base_conf + 0.3 * qual_conf + 0.15 * agree + 0.15 * fresh, 2)
    if affected_fraction is None:
        confidence = round(confidence * 0.9, 2)         # extent unknown -> less certain
    base_years = int(last.get("ndvi_base_years", last.get("ndmi_base_years", 99)))
    thin_baseline = base_years < s["min_baseline_years"]
    if thin_baseline:                                   # observations from only 1-2 years are not a climatology
        confidence = min(confidence, s["confidence_cap_thin_baseline"])
    label = "high" if confidence >= s["confidence_high"] else "medium" if confidence >= s["confidence_medium"] else "low"

    # --- recommendations ----------------------------------------------------------------------------
    rec = []
    if category in ("watch", "elevated", "high"):
        if run < 2:
            rec.append("Single low observation: wait for the next clear acquisition (about 3-5 days) to see if the decline persists.")
        else:
            rec.append("Decline is persistent: inspect imagery and consider a field check or local crop-calendar review.")
        if affected_fraction is None:
            rec.append("Assess spatial extent (per-pixel anomaly map) to see whether specific parts of the AOI drive the signal.")
    if stale_days > s["stale_after_days"]:
        rec.append(f"Latest observation is {stale_days} days old; acquire newer data.")
    if thin_baseline:
        rec.append(f"Baseline covers only {base_years} year(s) (< {int(s['min_baseline_years'])}); add more seasons to firm up what 'normal' is.")
    if label == "low":
        rec.append("Low confidence: thin baseline, limited clear coverage or extent unknown. Treat as a prompt for review only.")
    if category == "informational":
        rec.append("No action indicated; continue routine monitoring.")

    return RiskAssessment(aoi_id, as_of.isoformat(), category, score, comps, evidence, dates, quality,
                          confidence, label, affected_fraction, rec)


def _no_answer(aoi_id, as_of, category, dates, evidence, rec, quality) -> RiskAssessment:
    return RiskAssessment(aoi_id, as_of.isoformat(), category, None, {"magnitude": None, "persistence": None, "extent": None},
                          evidence, dates, quality, None, None, None, rec)
