"""Affected-area analysis: per-pixel anomaly map and patch statistics for an AOI's latest observation.

Uses the NDVI rasters already produced by the processing job. History is restricted to EARLIER observations from
OTHER years within the seasonal window, on the same pixel grid as the target; anything else is skipped with a
warning. If there is not enough history the result says so and no map is invented.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src import config
from src.features.spatial import (
    affected_area,
    expected_fraction_by_chance,
    pixel_baseline,
    pixel_zscores,
    signed_day_offsets,
)
from src.features.temporal import load_observations
from src.geospatial.raster_io import write_geotiff

log = logging.getLogger(__name__)

CAVEAT = ("Each pixel is compared with the same place in earlier years. A field that grew a different crop, was planted on a "
          "different date, or was fallow in those years will look 'below normal' without any stress, and with only a few "
          "earlier years this cannot be separated from real decline. Treat patches as places to look, not as diagnoses.")


@dataclass
class ExtentResult:
    status: str                                   # 'ok' | 'unavailable'
    reason: str | None = None
    target_granule: str | None = None
    target_date: str | None = None
    history_granules: list[str] = field(default_factory=list)
    history_years: int = 0
    window_days: int = 0
    pixel_zscore: float = 0.0
    min_patch_pixels: int = 0
    evaluated_pixels: int = 0
    raw_affected_pixels: int = 0
    affected_pixels: int = 0
    patches: int = 0
    largest_patch_pixels: int = 0
    affected_fraction: float | None = None
    expected_by_chance: float = 0.0
    area_ha: float | None = None
    area_slope_per_day: float | None = None
    output_dir: str | None = None
    warnings: list[str] = field(default_factory=list)
    caveat: str = CAVEAT

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return asdict(self)

    def note(self) -> str:
        """Short evidence phrase for the risk assessment."""
        return (f"{self.patches} connected patch(es) of >= {self.min_patch_pixels} pixels; about "
                f"{self.expected_by_chance:.1%} of pixels would fall below the threshold by chance in a normal year")


def _unavailable(reason: str, **kw) -> ExtentResult:
    return ExtentResult(status="unavailable", reason=reason, **kw)


def _read(path: Path) -> tuple[np.ndarray, tuple, str]:
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), tuple(src.transform)[:6], src.crs.to_string()


def compute_extent(processed_root: Path, aoi_id: str, *, target_granule: str | None = None, write: bool = True) -> ExtentResult:
    """Compute the per-pixel anomaly map and affected area for the latest usable observation (or ``target_granule``)."""
    ts = config.get_settings().timeseries
    z_thr = config.load_thresholds().affected_area["pixel_zscore"]
    base_kw = dict(window_days=ts.baseline_window_days, pixel_zscore=z_thr, min_patch_pixels=ts.min_patch_pixels,
                   expected_by_chance=expected_fraction_by_chance(z_thr))

    obs = load_observations(Path(processed_root), aoi_id, ("ndvi",))
    obs = obs[obs["usable"]].reset_index(drop=True)
    if obs.empty:
        return _unavailable("no usable observations", **base_kw)
    if target_granule:
        sel = obs[obs["granule_id"] == target_granule]
        if sel.empty:
            return _unavailable(f"granule {target_granule} is not a usable observation of this AOI", **base_kw)
        target = sel.iloc[0]
    else:
        latest = obs[obs["date"] == obs["date"].max()]
        target = latest.sort_values("valid_fraction", ascending=False).iloc[0]

    t_date = pd.Timestamp(target["date"])
    doy_dist = lambda d: np.minimum(np.abs(d - t_date.dayofyear), 365 - np.abs(d - t_date.dayofyear))  # noqa: E731
    hist = obs[(obs["date"].dt.year != t_date.year) & (obs["date"] < t_date)
               & (doy_dist(obs["date"].dt.dayofyear.to_numpy()) <= ts.baseline_window_days)]
    years = int(hist["date"].dt.year.nunique())
    kw = dict(target_granule=target["granule_id"], target_date=t_date.date().isoformat(), **base_kw)
    if years < ts.min_baseline_years:
        return _unavailable(f"needs observations from at least {ts.min_baseline_years} earlier years in the seasonal window "
                            f"(have {years})", history_years=years, **kw)

    t_arr, t_tf, t_crs = _read(Path(target["output_dir"]) / "ndvi.tif")
    layers, offsets, used, warns = [], [], [], []
    for _, h in hist.iterrows():
        try:
            arr, tf, crs = _read(Path(h["output_dir"]) / "ndvi.tif")
        except (rasterio.errors.RasterioError, OSError) as exc:
            warns.append(f"skipped {h['granule_id']}: cannot read NDVI raster ({exc})")
            continue
        if arr.shape != t_arr.shape or crs != t_crs or any(abs(a - b) > 1e-6 for a, b in zip(tf, t_tf)):
            warns.append(f"skipped {h['granule_id']}: pixel grid differs from the target's")
            continue
        layers.append(arr)
        offsets.append(float(signed_day_offsets(np.array([h["date"].dayofyear]), t_date.dayofyear)[0]))
        used.append(h["granule_id"])
    kw["warnings"] = warns
    if len(layers) < ts.min_observations_for_baseline:
        return _unavailable(f"only {len(layers)} comparable earlier observations (need {ts.min_observations_for_baseline})",
                            history_years=years, history_granules=used, **kw)

    base = pixel_baseline(np.stack(layers), np.array(offsets), min_obs=ts.min_observations_for_baseline,
                          sigma_floor=ts.pixel_sigma_floor, detrend=ts.detrend_baseline)
    z = pixel_zscores(t_arr, base)
    area = affected_area(z, z_thr, ts.min_patch_pixels)
    if area.evaluated == 0:
        return _unavailable("no pixel has both a valid value today and enough history", history_years=years,
                            history_granules=used, **kw)

    pixel_m2 = abs(t_tf[0] * t_tf[4])
    result = ExtentResult(
        status="ok", history_granules=used, history_years=years, evaluated_pixels=area.evaluated,
        raw_affected_pixels=area.raw_pixels, affected_pixels=area.pixels, patches=area.patches,
        largest_patch_pixels=area.largest_patch, affected_fraction=area.fraction, area_ha=area.pixels * pixel_m2 / 10_000.0,
        area_slope_per_day=base.area_slope, **kw)
    if write:
        out = Path(target["output_dir"])
        write_geotiff(out / "anomaly_z.tif", z, crs=t_crs, transform=rasterio.Affine(*t_tf),
                      nodata=np.nan, tags={"meaning": "pixel z-score vs seasonal baseline (declines negative)", "indices": "NDVI"})
        write_geotiff(out / "affected_mask.tif", np.where(np.isfinite(z), area.mask, False).astype(np.uint8), crs=t_crs,
                      transform=rasterio.Affine(*t_tf), nodata=None,
                      tags={"meaning": f"1 = in a patch of >= {ts.min_patch_pixels} pixels with z <= {z_thr}"})
        result.output_dir = str(out)
        (out / "extent.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
