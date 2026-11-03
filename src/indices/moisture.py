"""Moisture-sensitive indices.

NDMI = (NIR - SWIR1) / (NIR + SWIR1). SWIR1 is HLSL30 B06 / HLSS30 B11 (NOT SWIR2:
B07 / B12). Higher values suggest more canopy water content, but soil moisture,
residue and canopy structure also affect it, so it is a stress *indicator* only.
"""
from __future__ import annotations

import numpy as np

from src.indices.vegetation import normalized_difference


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """NDMI from narrow NIR and SWIR1 reflectance."""
    return normalized_difference(nir, swir1)
