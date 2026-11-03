"""Observation quality metrics, always computed *inside the AOI*.

Tile-level cloud cover from the catalog says little about a small AOI, so valid
and masked percentages are measured on the clipped AOI pixels only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.preprocessing.cloud_mask import FmaskFlags


@dataclass(frozen=True)
class QualityReport:
    aoi_pixels: int
    observed_fraction: float     # AOI pixels that have any data (not fill)
    valid_fraction: float        # AOI pixels usable after all masks
    cloud_fraction: float
    adjacent_fraction: float
    shadow_fraction: float
    snow_ice_fraction: float
    water_fraction: float
    fill_fraction: float         # AOI pixels with no observation (outside swath/tile coverage)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_quality(flags: FmaskFlags, usable: np.ndarray, inside: np.ndarray) -> QualityReport:
    """Summarise quality over AOI pixels. ``usable`` is the final per-pixel validity mask."""
    if not (flags.fill.shape == usable.shape == inside.shape):
        raise ValueError("flags, usable mask and AOI mask must have the same shape")
    n = int(inside.sum())
    if n == 0:
        raise ValueError("AOI contains no pixels")

    def frac(mask: np.ndarray) -> float:
        return float((mask & inside).sum()) / n

    return QualityReport(
        aoi_pixels=n,
        observed_fraction=1.0 - frac(flags.fill),
        valid_fraction=frac(usable),
        cloud_fraction=frac(flags.cloud),
        adjacent_fraction=frac(flags.adjacent),
        shadow_fraction=frac(flags.shadow),
        snow_ice_fraction=frac(flags.snow_ice),
        water_fraction=frac(flags.water),
        fill_fraction=frac(flags.fill),
    )
