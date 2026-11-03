"""Decode the HLS v2.0 Fmask quality layer (HLS User Guide V2.0, Table 9).

Bit 0 is the least significant bit:
    1 cloud | 2 adjacent to cloud/shadow | 3 cloud shadow | 4 snow/ice | 5 water | 6-7 aerosol level
Fill value is 255 (every bit set), so fill is tested *before* decoding bits.
Bits 1-5 are not necessarily mutually exclusive after resampling to 30 m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FMASK_FILL = 255
BIT_CLOUD, BIT_ADJACENT, BIT_SHADOW, BIT_SNOW_ICE, BIT_WATER = 1, 2, 3, 4, 5


@dataclass(frozen=True)
class FmaskFlags:
    """Boolean per-pixel flags. ``fill`` pixels have all other flags False."""

    fill: np.ndarray
    cloud: np.ndarray
    adjacent: np.ndarray
    shadow: np.ndarray
    snow_ice: np.ndarray
    water: np.ndarray


def decode_fmask(fmask: np.ndarray, fill_value: int = FMASK_FILL) -> FmaskFlags:
    """Decode Fmask bits into boolean arrays."""
    fmask = np.asarray(fmask)
    if fmask.dtype != np.uint8:
        raise TypeError(f"Fmask must be uint8, got {fmask.dtype}")
    fill = fmask == fill_value

    def bit(n: int) -> np.ndarray:
        return (((fmask >> n) & 1) == 1) & ~fill

    return FmaskFlags(
        fill=fill, cloud=bit(BIT_CLOUD), adjacent=bit(BIT_ADJACENT), shadow=bit(BIT_SHADOW),
        snow_ice=bit(BIT_SNOW_ICE), water=bit(BIT_WATER),
    )


def build_valid_mask(
    fmask: np.ndarray,
    *,
    mask_cloud: bool = True,
    mask_adjacent: bool = True,
    mask_shadow: bool = True,
    mask_snow_ice: bool = True,
    mask_water: bool = True,
) -> np.ndarray:
    """True where a pixel is usable: observed (not fill) and none of the selected flags are set."""
    f = decode_fmask(fmask)
    bad = f.fill.copy()
    for enabled, flag in (
        (mask_cloud, f.cloud), (mask_adjacent, f.adjacent), (mask_shadow, f.shadow),
        (mask_snow_ice, f.snow_ice), (mask_water, f.water),
    ):
        if enabled:
            bad |= flag
    return ~bad
