"""Per-pixel seasonal baselines, z-scores and affected-area patches.

For a target observation, each pixel's expected value comes from the SAME pixel in earlier years' observations
taken at the same time of year. The area-wide seasonal slope (robust Theil-Sen on the area medians) is removed
first, so a steep green-up or senescence does not distort every pixel. Pixel spread is a robust MAD, floored,
because single pixels are noisier than the area median.

Counting "affected" pixels is deliberately conservative: only connected patches of at least ``min_patch_pixels``
count, since isolated low pixels occur by chance (about 2.3% of pixels fall below z = -2 in a perfectly normal
year if the baseline were exact).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from src.detection.baseline import MAD_TO_SIGMA, theil_sen


def signed_day_offsets(history_doy: np.ndarray, target_doy: int) -> np.ndarray:
    """Signed day-of-year offset of each history observation from the target, wrapping at the year end."""
    return ((np.asarray(history_doy) - target_doy + 182) % 365 - 182).astype(float)


@dataclass(frozen=True)
class PixelBaseline:
    level: np.ndarray        # expected value per pixel on the target day
    sigma: np.ndarray        # robust spread per pixel (floored)
    count: np.ndarray        # valid history observations per pixel
    area_slope: float        # area-wide seasonal slope removed (index units per day)


def pixel_baseline(
    stack: np.ndarray,
    day_offsets: np.ndarray,
    *,
    min_obs: int = 5,
    sigma_floor: float = 0.05,
    detrend: bool = True,
) -> PixelBaseline:
    """Baseline from a (K, H, W) stack of earlier observations (NaN = invalid) at known day offsets."""
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError("stack must be (observations, rows, cols)")
    if stack.shape[0] != len(day_offsets):
        raise ValueError("day_offsets must have one entry per observation")

    slope = 0.0
    if detrend:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(stack.reshape(stack.shape[0], -1), axis=1)
        ok = np.isfinite(medians)
        if ok.sum() >= 3:
            slope, _, _ = theil_sen(np.asarray(day_offsets, dtype=float)[ok], medians[ok].astype(float))
    adjusted = stack - np.float32(slope) * np.asarray(day_offsets, dtype=np.float32)[:, None, None]

    count = np.isfinite(adjusted).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmedian(adjusted, axis=0)
        mad = np.nanmedian(np.abs(adjusted - level), axis=0)
    sigma = np.maximum(MAD_TO_SIGMA * mad, np.float32(sigma_floor))
    enough = count >= min_obs
    return PixelBaseline(np.where(enough, level, np.nan).astype(np.float32),
                         np.where(enough, sigma, np.nan).astype(np.float32), count, float(slope))


def pixel_zscores(target: np.ndarray, base: PixelBaseline) -> np.ndarray:
    """z = (today - expected) / spread, NaN wherever either side is invalid."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((np.asarray(target, dtype=np.float32) - base.level) / base.sigma).astype(np.float32)


@dataclass(frozen=True)
class AffectedArea:
    mask: np.ndarray          # bool, pixels in patches of >= min_patch_pixels
    raw_pixels: int           # below threshold before the patch filter
    pixels: int               # after the patch filter
    patches: int
    largest_patch: int
    evaluated: int            # pixels with a valid z-score

    @property
    def fraction(self) -> float:
        return self.pixels / self.evaluated if self.evaluated else float("nan")


def affected_area(z: np.ndarray, z_threshold: float, min_patch_pixels: int) -> AffectedArea:
    """Pixels at or below ``z_threshold``, kept only where they form 8-connected patches of enough pixels."""
    finite = np.isfinite(z)
    below = finite & (z <= z_threshold)
    labels, n = ndimage.label(below, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return AffectedArea(np.zeros_like(below), 0, 0, 0, 0, int(finite.sum()))
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(sizes >= min_patch_pixels) + 1
    mask = np.isin(labels, keep)
    return AffectedArea(mask, int(below.sum()), int(mask.sum()), int(len(keep)),
                        int(sizes[keep - 1].max()) if len(keep) else 0, int(finite.sum()))


def expected_fraction_by_chance(z_threshold: float) -> float:
    """Fraction of pixels expected at or below ``z_threshold`` if the baseline were exact and noise Gaussian."""
    return 0.5 * math.erfc(-z_threshold / math.sqrt(2.0))
