"""Grid compatibility checks. We verify; we do not silently resample."""
from __future__ import annotations

from src.geospatial.raster_io import RasterMeta


class GridMismatchError(ValueError):
    """Rasters do not share the same grid and cannot be combined pixel-by-pixel."""


def assert_same_grid(*metas: RasterMeta, tol: float = 1e-6) -> None:
    """Raise :class:`GridMismatchError` unless CRS, shape and affine transform all match."""
    if len(metas) < 2:
        return
    ref = metas[0]
    for m in metas[1:]:
        if m.crs != ref.crs:
            raise GridMismatchError(f"CRS differs: {ref.crs} vs {m.crs} ({m.path})")
        if m.shape != ref.shape:
            raise GridMismatchError(f"Shape differs: {ref.shape} vs {m.shape} ({m.path})")
        if any(abs(a - b) > tol for a, b in zip(tuple(ref.transform)[:6], tuple(m.transform)[:6])):
            raise GridMismatchError(f"Pixel grid/origin differs ({m.path}); resampling is required and not done implicitly")
