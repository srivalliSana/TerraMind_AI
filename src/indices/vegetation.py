"""Vegetation indices from surface reflectance (0-1 floats, NaN = invalid).

Limitations (also written into output metadata):
- An index decline is NOT proof of crop stress. Seasonal growth, planting, harvest,
  irrigation, residual cloud and sensor differences all change these values.
- NDVI saturates over dense canopy; SAVI/EVI reduce but do not remove this.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6   # |denominator| below this is treated as undefined


def _check(*arrays: np.ndarray) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=np.float32) for a in arrays]
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError(f"Input shapes differ: {[a.shape for a in arrs]}")
    return arrs


def safe_divide(num: np.ndarray, den: np.ndarray, eps: float = EPS) -> np.ndarray:
    """num/den with NaN where the denominator is ~0 or any input is non-finite."""
    num, den = _check(num, den)
    out = np.full(num.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    out[ok] = num[ok] / den[ok]
    return out


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where undefined. Mathematically bounded to [-1, 1] for non-negative inputs."""
    a, b = _check(a, b)
    return safe_divide(a - b, a + b)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)."""
    return normalized_difference(nir, red)


def savi(nir: np.ndarray, red: np.ndarray, soil_factor: float = 0.5) -> np.ndarray:
    """SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L); L=0.5 is the standard default (Huete, 1988)."""
    if not 0 <= soil_factor <= 1:
        raise ValueError("soil_factor L must be within [0, 1]")
    nir, red = _check(nir, red)
    return safe_divide((nir - red) * (1 + soil_factor), nir + red + soil_factor)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)  (standard MODIS coefficients).

    Needs the blue band. Output is clipped to nothing; extreme values are flagged NaN
    outside the physically meaningful [-1, 1.5] range.
    """
    nir, red, blue = _check(nir, red, blue)
    out = safe_divide(2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0)
    with np.errstate(invalid="ignore"):
        out[(out < -1.0) | (out > 1.5)] = np.nan
    return out
