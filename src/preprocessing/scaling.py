"""Convert HLS integer digital numbers to surface reflectance.

HLS v2.0 stores reflectance as int16 with scale 0.0001 and fill value -9999
(HLS User Guide V2.0, Tables 6-7). The guide lists ``ADD_OFFSET`` as per-granule
metadata but does not spell out how a non-zero offset is applied, and every
verified granule has offset 0, so a non-zero offset is rejected instead of guessed.
"""
from __future__ import annotations

import numpy as np


def scale_reflectance(
    dn: np.ndarray,
    *,
    scale: float = 0.0001,
    offset: float = 0.0,
    fill_value: float = -9999,
    valid_range: tuple[float, float] | None = (-0.01, 1.0),
) -> np.ndarray:
    """Return float32 reflectance with NaN for fill and out-of-range pixels.

    Fill pixels are identified on the *raw* integers before scaling so that
    ``-9999`` can never be mistaken for a reflectance value.
    """
    dn = np.asarray(dn)
    if not np.issubdtype(dn.dtype, np.number):
        raise TypeError(f"Expected a numeric array, got {dn.dtype}")
    if offset != 0:
        raise ValueError(
            f"Non-zero ADD_OFFSET ({offset}) is not supported: its application is not specified in the HLS guide."
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid scale factor: {scale}")

    out = dn.astype(np.float32) * np.float32(scale)
    out[dn == fill_value] = np.nan
    if np.issubdtype(dn.dtype, np.floating):
        out[~np.isfinite(dn)] = np.nan
    if valid_range is not None:
        lo, hi = valid_range
        with np.errstate(invalid="ignore"):
            out[(out < lo) | (out > hi)] = np.nan
    return out
