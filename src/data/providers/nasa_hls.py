"""NASA HLS (HLSL30 / HLSS30 v2.0) provider using NASA CMR via ``earthaccess``.

Catalog search is public and needs no credentials. Downloading data needs
NASA Earthdata Login and is handled separately (``src/data/download.py``).

Notes on the catalog metadata (verified against live CMR responses):
- ``GranuleUR`` looks like ``HLS.L30.T15TVG.2024186T165841.v2.0``.
- Layers are separate COG files; links are ``.../<GranuleUR>.<LAYER>.tif``.
- ``CLOUD_COVERAGE`` / ``SPATIAL_COVERAGE`` are percentages in ``AdditionalAttributes``.
- The footprint is the whole MGRS tile (~110 km), so a hit means the tile
  overlaps the AOI, not that the AOI itself is cloud-free.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable

from shapely.geometry import Polygon

from src import config
from src.data.providers.base import (
    AuthError,
    AuthStatus,
    DataProvider,
    Granule,
    ProviderError,
    SearchQuery,
    SearchResult,
)

log = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"\.v\d+\.\d+\.([A-Za-z0-9]+)\.tif$")
_ID_RE = re.compile(r"^HLS\.(?P<kind>[LS]30)\.(?P<tile>T\w{5})\.(?P<time>\d{7}T\d{6})\.v(?P<ver>[\d.]+)$")
_NUMERIC_ATTRS = {
    "CLOUD_COVERAGE": "cloud_cover",
    "SPATIAL_COVERAGE": "spatial_coverage",
    "REF_SCALE_FACTOR": "scale_factor",
    "ADD_OFFSET": "add_offset",
    "FILLVALUE": "fill_value",
    "QA_FILLVALUE": "qa_fill_value",
    "NCOLS": "ncols",
    "NROWS": "nrows",
    "ULX": "ulx",
    "ULY": "uly",
    "MEAN_SUN_ZENITH_ANGLE": "mean_sun_zenith",
}

SearchFn = Callable[..., list[Any]]


def _attrs(umm: dict[str, Any]) -> dict[str, str]:
    """Flatten ``AdditionalAttributes`` to ``{name: first value}``."""
    out: dict[str, str] = {}
    for a in umm.get("AdditionalAttributes", []) or []:
        values = a.get("Values") or []
        if a.get("Name") and values:
            out[a["Name"]] = str(values[0])
    return out


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _footprint(umm: dict[str, Any]) -> Polygon | None:
    try:
        pts = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
        poly = Polygon([(p["Longitude"], p["Latitude"]) for p in pts])
        return poly if poly.is_valid and not poly.is_empty else poly.buffer(0)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def parse_umm(umm: dict[str, Any], sensor_by_kind: dict[str, str] | None = None) -> Granule:
    """Convert one CMR UMM-G granule record to a :class:`Granule`.

    Raises :class:`ProviderError` when the record lacks the information needed
    to use it safely (id, acquisition time).
    """
    granule_id = umm.get("GranuleUR", "")
    m = _ID_RE.match(granule_id)
    if not m:
        raise ProviderError(f"Unrecognised HLS granule id: {granule_id!r}")
    kind = m["kind"]
    sensor = (sensor_by_kind or {"L30": "HLSL30", "S30": "HLSS30"})[kind]

    try:
        acquired = _parse_time(umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
    except (KeyError, ValueError) as exc:
        raise ProviderError(f"{granule_id}: missing or invalid acquisition time") from exc

    attrs = _attrs(umm)
    meta: dict[str, Any] = {}
    for key, name in _NUMERIC_ATTRS.items():
        if key in attrs and (val := _to_float(attrs[key])) is not None:
            meta[name] = val
    for key, name in (("HORIZONTAL_CS_CODE", "epsg"), ("HORIZONTAL_CS_NAME", "crs_name"),
                      ("MGRS_TILE_ID", "mgrs_tile"), ("SENSING_TIME", "sensing_time")):
        if key in attrs:
            meta[name] = attrs[key]

    assets: dict[str, str] = {}
    for rel in umm.get("RelatedUrls", []) or []:
        url = rel.get("URL", "")
        if rel.get("Type") == "GET DATA" and url.startswith("https://") and (lm := _LAYER_RE.search(url)):
            assets[lm.group(1)] = url

    return Granule(
        granule_id=granule_id,
        provider="nasa_hls",
        sensor=sensor,
        tile_id=m["tile"],
        acquired=acquired,
        cloud_cover=meta.pop("cloud_cover", None),
        spatial_coverage=meta.pop("spatial_coverage", None),
        footprint=_footprint(umm),
        assets=assets,
        metadata=meta,
    )


class NasaHLSProvider(DataProvider):
    """Discover HLS granules over an AOI.

    ``search_fn`` is injectable so tests run offline; by default it is
    ``earthaccess.search_data``.
    """

    name = "nasa_hls"

    def __init__(self, search_fn: SearchFn | None = None) -> None:
        self._search_fn = search_fn
        sensors = config.load_sensors()
        self._sensors = {k: v for k, v in sensors.items() if v.get("provider") == self.name}
        self._kind_to_sensor = {"L30": "HLSL30", "S30": "HLSS30"}

    @property
    def available_sensors(self) -> list[str]:
        return sorted(self._sensors)

    def _search(self, **kwargs: Any) -> list[Any]:
        if self._search_fn is not None:
            return self._search_fn(**kwargs)
        import earthaccess

        return earthaccess.search_data(**kwargs)

    def search(self, query: SearchQuery) -> SearchResult:
        sensors = list(query.sensors) if query.sensors else self.available_sensors
        unknown = [s for s in sensors if s not in self._sensors]
        if unknown:
            raise ProviderError(f"Unknown sensor(s) for {self.name}: {unknown}. Available: {self.available_sensors}")

        temporal = (f"{query.start.isoformat()}T00:00:00Z", f"{query.end.isoformat()}T23:59:59Z")
        result = SearchResult(granules=[])
        seen: set[str] = set()

        for sensor in sensors:
            short_name = self._sensors[sensor]["collection_short_name"]
            log.info("Searching %s bbox=%s temporal=%s limit=%d", short_name, query.aoi.bounds, temporal, query.max_results)
            try:
                items = self._search(short_name=short_name, bounding_box=query.aoi.bounds,
                                     temporal=temporal, count=query.max_results)
            except Exception as exc:  # noqa: BLE001 - network/catalog errors vary
                raise ProviderError(
                    f"Catalog search for {short_name} failed: {exc}. This is a catalog (CMR) problem, "
                    "not an authentication problem; check your internet connection and try again."
                ) from exc

            if len(items) >= query.max_results:
                result.truncated = True
                result.warnings.append(
                    f"{short_name}: result limit ({query.max_results}) reached; narrow the date range or raise the limit."
                )
            for item in items:
                try:
                    g = parse_umm(item["umm"], self._kind_to_sensor)
                except ProviderError as exc:
                    result.warnings.append(f"Skipped a record: {exc}")
                    continue
                if g.granule_id in seen:
                    continue
                seen.add(g.granule_id)
                if g.footprint is not None and not g.footprint.intersects(query.aoi.geometry):
                    continue  # bbox hit but tile footprint does not touch the AOI polygon
                if query.max_cloud_cover is not None and g.cloud_cover is not None and g.cloud_cover > query.max_cloud_cover:
                    continue
                result.granules.append(g)

        result.granules.sort(key=lambda g: (g.acquired, g.sensor))
        return result

    def auth_status(self, verify: bool = False) -> AuthStatus:
        creds = config.credential_status()
        has_token = creds["EARTHDATA_TOKEN"]
        has_pair = creds["EARTHDATA_USERNAME"] and creds["EARTHDATA_PASSWORD"]
        if not (has_token or has_pair):
            return AuthStatus(
                state="not_configured",
                message=("No Earthdata credentials in the environment. Catalog search works; downloads need "
                         "EARTHDATA_TOKEN (preferred) or EARTHDATA_USERNAME/EARTHDATA_PASSWORD in .env."),
                can_search=True, can_download=False,
            )
        how = "token" if has_token else "username/password"
        if not verify:
            return AuthStatus("unchecked", f"Credentials ({how}) are present but not verified.", True, False)
        try:
            import earthaccess

            auth = earthaccess.login(strategy="environment")
        except Exception as exc:  # noqa: BLE001
            return AuthStatus("failed", f"Earthdata login raised {type(exc).__name__}. "
                              "Check the credentials and that EULAs are accepted for HLS in your Earthdata profile.",
                              True, False)
        if getattr(auth, "authenticated", False):
            return AuthStatus("ok", f"Authenticated with Earthdata using {how}.", True, True)
        return AuthStatus("failed", f"Earthdata rejected the {how} credentials. Tokens expire; generate a new one at "
                          "https://urs.earthdata.nasa.gov/profile", True, False)


def require_login() -> None:
    """Log in for downloads or raise :class:`AuthError` with an actionable message."""
    status = NasaHLSProvider().auth_status(verify=True)
    if not status.can_download:
        raise AuthError(status.message)

