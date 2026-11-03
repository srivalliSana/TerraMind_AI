"""Raster reading, AOI clipping and GeoTIFF writing with provenance tags."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_window, geometry_mask
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import mapping

from src.geospatial.aoi import AOI


class RasterError(RuntimeError):
    """Raster missing, unreadable or not usable."""


@dataclass(frozen=True)
class RasterMeta:
    """What we need to know about a raster to combine it safely with others."""

    path: str
    crs: str
    transform: Affine
    width: int
    height: int
    dtype: str
    nodata: float | None
    scale: float
    offset: float
    tags: dict[str, str]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def resolution(self) -> tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))


def read_meta(path: str | Path) -> RasterMeta:
    """Read georeferencing and scaling metadata (no pixel data)."""
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise RasterError(f"{Path(path).name}: raster has no CRS")
            return RasterMeta(
                path=str(path), crs=src.crs.to_string(), transform=src.transform, width=src.width,
                height=src.height, dtype=src.dtypes[0], nodata=src.nodata,
                scale=float(src.scales[0]), offset=float(src.offsets[0]), tags=dict(src.tags()),
            )
    except rasterio.errors.RasterioError as exc:
        raise RasterError(f"Cannot read {path}: {exc}") from exc


def read_band_clipped(path: str | Path, aoi: AOI) -> tuple[np.ndarray, RasterMeta, np.ndarray]:
    """Read only the AOI window of band 1.

    Returns ``(data, meta, inside)``: ``data`` is the raw (unscaled) array for the
    window, ``meta`` carries the *window's* transform and shape, and ``inside`` is a
    boolean mask that is True for pixels whose centre lies inside the AOI polygon.
    Raises :class:`RasterError` if the AOI does not overlap the raster.
    """
    full = read_meta(path)
    geom = transform_geom("EPSG:4326", full.crs, mapping(aoi.geometry))
    try:
        with rasterio.open(path) as src:
            window = geometry_window(src, [geom]).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if window.width < 1 or window.height < 1:
                raise RasterError("AOI does not overlap this raster")
            data = src.read(1, window=window)
            transform = src.window_transform(window)
    except rasterio.errors.WindowError as exc:  # empty intersection of AOI window and raster extent
        raise RasterError(f"AOI does not overlap {Path(path).name}") from exc
    except rasterio.errors.RasterioError as exc:
        raise RasterError(f"AOI window read failed for {Path(path).name}: {exc}") from exc
    except ValueError as exc:  # geometry_window raises when completely outside
        raise RasterError(f"AOI does not overlap this raster ({exc})") from exc

    inside = geometry_mask([geom], out_shape=data.shape, transform=transform, invert=True)
    meta = RasterMeta(
        path=full.path, crs=full.crs, transform=transform, width=data.shape[1], height=data.shape[0],
        dtype=full.dtype, nodata=full.nodata, scale=full.scale, offset=full.offset, tags=full.tags,
    )
    return data, meta, inside


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    *,
    crs: str,
    transform: Affine,
    nodata: float | None = np.nan,
    tags: dict[str, Any] | None = None,
) -> Path:
    """Write a single-band float32/uint8 GeoTIFF (tiled, deflate) with provenance tags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim != 2:
        raise ValueError("write_geotiff expects a 2-D array")
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype,
        crs=crs, transform=transform, nodata=nodata, compress="deflate", tiled=True,
        blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(array, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path
