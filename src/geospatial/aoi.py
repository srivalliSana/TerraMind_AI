"""Area-of-interest (AOI) model and loaders.

An :class:`AOI` is always stored in EPSG:4326 (lon/lat). Loaders accept a
bounding box, GeoJSON (dict, string or file), a shapefile, a GeoDataFrame or a
shapely geometry, and validate CRS, geometry validity, emptiness, coordinate
bounds and area limits before returning.

Limitation: AOIs crossing the antimeridian (180° longitude) are rejected, not
silently mangled. Split them into two AOIs.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import CRS, Geod
from shapely import make_valid
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from src.config import AOISettings

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
_GEOD = Geod(ellps="WGS84")
_SHAPEFILE_REQUIRED = (".shp", ".shx", ".dbf")


class AOIError(ValueError):
    """Raised when an AOI cannot be created or fails validation."""


def geodesic_area_km2(geometry: BaseGeometry) -> float:
    """Geodesic area of a lon/lat geometry in square kilometres."""
    area_m2, _ = _GEOD.geometry_area_perimeter(geometry)
    return abs(area_m2) / 1e6


@dataclass(frozen=True)
class AOI:
    """A validated polygonal area of interest in EPSG:4326."""

    geometry: BaseGeometry
    name: str = "unnamed-aoi"
    source: str = "unknown"

    @property
    def id(self) -> str:
        """Stable identifier derived from the (rounded) geometry, not the name."""
        digest = hashlib.sha1(_round_wkt(self.geometry, 6).encode()).hexdigest()
        return f"aoi_{digest[:12]}"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)."""
        minx, miny, maxx, maxy = self.geometry.bounds
        return (minx, miny, maxx, maxy)

    @property
    def area_km2(self) -> float:
        return geodesic_area_km2(self.geometry)

    @property
    def centroid(self) -> tuple[float, float]:
        """(lon, lat) of the centroid."""
        c = self.geometry.centroid
        return (c.x, c.y)

    def to_geojson(self) -> dict[str, Any]:
        """GeoJSON Feature with identifying properties."""
        return {
            "type": "Feature",
            "id": self.id,
            "properties": {"name": self.name, "source": self.source, "area_km2": round(self.area_km2, 4)},
            "geometry": mapping(self.geometry),
        }

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"aoi_id": [self.id], "name": [self.name], "source": [self.source]},
            geometry=[self.geometry],
            crs=WGS84,
        )


def _round_wkt(geometry: BaseGeometry, digits: int) -> str:
    from shapely import set_precision

    return set_precision(geometry.normalize(), 10 ** -digits).wkt


def _clean_geometry(geometry: BaseGeometry) -> BaseGeometry:
    """Return a valid (Multi)Polygon or raise :class:`AOIError`."""
    if geometry is None or geometry.is_empty:
        raise AOIError("Geometry is empty.")
    if not geometry.is_valid:
        log.warning("Invalid geometry detected; repairing with make_valid.")
        geometry = make_valid(geometry)
    # make_valid can return a collection; keep only the polygonal parts.
    polys = [g for g in getattr(geometry, "geoms", [geometry]) if g.geom_type in ("Polygon", "MultiPolygon")]
    if not polys:
        raise AOIError(f"AOI must be polygonal (got {geometry.geom_type}).")
    geometry = unary_union(polys)
    if geometry.is_empty or geometry.area == 0:
        raise AOIError("Geometry has zero area.")
    return geometry


def _check_bounds(geometry: BaseGeometry) -> None:
    minx, miny, maxx, maxy = geometry.bounds
    if not (-180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90):
        raise AOIError(
            f"Coordinates out of range for lon/lat: bounds={geometry.bounds}. "
            "Check that the CRS is correct and that longitude/latitude are not swapped."
        )


def validate_aoi(aoi: AOI, limits: AOISettings | None = None) -> AOI:
    """Enforce area limits from configuration. Returns the AOI unchanged."""
    if limits is None:
        return aoi
    area = aoi.area_km2
    if area < limits.min_area_km2:
        raise AOIError(f"AOI area {area:.4f} km² is below the minimum of {limits.min_area_km2} km².")
    if area > limits.max_area_km2:
        raise AOIError(
            f"AOI area {area:,.1f} km² exceeds the processing limit of {limits.max_area_km2:,.0f} km². "
            "Use a smaller area or raise aoi.max_area_km2 in configs/default.yaml."
        )
    return aoi


def _build(geometry: BaseGeometry, name: str, source: str, limits: AOISettings | None) -> AOI:
    geometry = _clean_geometry(geometry)
    _check_bounds(geometry)
    return validate_aoi(AOI(geometry=geometry, name=name, source=source), limits)


def from_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    *,
    name: str = "bbox-aoi",
    limits: AOISettings | None = None,
) -> AOI:
    """Create an AOI from lon/lat bounding coordinates."""
    vals = (min_lon, min_lat, max_lon, max_lat)
    if not all(isinstance(v, (int, float)) and v == v and abs(v) != float("inf") for v in vals):
        raise AOIError("Bounding box values must be finite numbers.")
    if min_lon > max_lon:
        raise AOIError("min_lon > max_lon: antimeridian-crossing boxes are not supported; split into two AOIs.")
    if min_lat >= max_lat or min_lon == max_lon:
        raise AOIError("Bounding box must have positive width and height (min < max).")
    return _build(box(min_lon, min_lat, max_lon, max_lat), name, "bbox", limits)


def from_geodataframe(
    gdf: gpd.GeoDataFrame,
    *,
    name: str = "aoi",
    source: str = "geodataframe",
    default_crs: str | None = None,
    limits: AOISettings | None = None,
) -> AOI:
    """Create an AOI from a GeoDataFrame, reprojecting to EPSG:4326 and dissolving features."""
    if gdf is None or len(gdf) == 0:
        raise AOIError("No features found.")
    if gdf.crs is None:
        if default_crs is None:
            raise AOIError("Input has no CRS; refusing to guess. Provide a CRS (e.g. add a .prj file).")
        gdf = gdf.set_crs(default_crs)
    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    if len(gdf) == 0:
        raise AOIError("All features have empty geometry.")
    if CRS(gdf.crs) != CRS(WGS84):
        gdf = gdf.to_crs(WGS84)
    return _build(unary_union(list(gdf.geometry)), name, source, limits)


def from_geometry(
    geometry: BaseGeometry,
    *,
    crs: str = WGS84,
    name: str = "aoi",
    source: str = "geometry",
    limits: AOISettings | None = None,
) -> AOI:
    """Create an AOI from a shapely geometry in the given CRS."""
    return from_geodataframe(
        gpd.GeoDataFrame(geometry=[geometry], crs=crs), name=name, source=source, limits=limits
    )


def from_geojson(
    data: dict[str, Any] | str | Path,
    *,
    name: str | None = None,
    limits: AOISettings | None = None,
) -> AOI:
    """Create an AOI from GeoJSON (dict, JSON string or file path).

    Per RFC 7946, GeoJSON coordinates are WGS84 lon/lat. A legacy ``crs``
    member naming another CRS is honoured.
    """
    source = "geojson"
    if isinstance(data, Path) or (isinstance(data, str) and not data.lstrip().startswith("{")):
        path = Path(data)
        if not path.is_file():
            raise AOIError(f"GeoJSON file not found: {path}")
        name = name or path.stem
        source = f"geojson:{path.name}"
        data = path.read_text(encoding="utf-8")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise AOIError(f"Invalid GeoJSON: {exc}") from exc
    if not isinstance(data, dict) or "type" not in data:
        raise AOIError("GeoJSON must be an object with a 'type'.")

    crs = WGS84
    legacy = (data.get("crs") or {}).get("properties", {}).get("name")
    if legacy:
        try:
            crs = CRS.from_user_input(legacy).to_string()
        except Exception as exc:  # noqa: BLE001
            raise AOIError(f"Unrecognised GeoJSON crs '{legacy}'.") from exc

    kind = data["type"]
    try:
        if kind == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
        elif kind == "Feature":
            geoms = [shape(data["geometry"])] if data.get("geometry") else []
        else:
            geoms = [shape(data)]
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise AOIError(f"Malformed GeoJSON geometry: {exc}") from exc
    if not geoms:
        raise AOIError("GeoJSON contains no geometries.")
    return from_geodataframe(
        gpd.GeoDataFrame(geometry=geoms, crs=crs), name=name or "geojson-aoi", source=source, limits=limits
    )


def from_shapefile(path: str | Path, *, name: str | None = None, limits: AOISettings | None = None) -> AOI:
    """Create an AOI from a shapefile. Requires .shp, .shx and .dbf; a .prj is needed to know the CRS."""
    path = Path(path)
    if path.suffix.lower() != ".shp":
        raise AOIError("Provide the path to the .shp file.")
    missing = [ext for ext in _SHAPEFILE_REQUIRED if not path.with_suffix(ext).exists()
               and not path.with_suffix(ext.upper()).exists()]
    if missing:
        raise AOIError(f"Shapefile is missing required sidecar file(s): {', '.join(missing)}")
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001 - driver errors vary
        raise AOIError(f"Could not read shapefile: {exc}") from exc
    return from_geodataframe(gdf, name=name or path.stem, source=f"shapefile:{path.name}", limits=limits)
