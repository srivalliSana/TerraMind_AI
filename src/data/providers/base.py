"""Provider-independent data-acquisition types.

Every satellite data source (NASA HLS, STAC catalogs, ...) implements
:class:`DataProvider` and returns :class:`Granule` objects, so the rest of the
pipeline never depends on a specific catalog's metadata layout.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from shapely.geometry.base import BaseGeometry

from src.geospatial.aoi import AOI


class ProviderError(RuntimeError):
    """A provider failed (network, catalog or parsing). Message is user-facing."""


class AuthError(ProviderError):
    """Authentication or authorisation failed (distinct from catalog-search errors)."""


@dataclass(frozen=True)
class SearchQuery:
    """What to look for. Always AOI- and time-bounded; there is no 'everything' query."""

    aoi: AOI
    start: date
    end: date
    sensors: tuple[str, ...] | None = None  # None = all sensors of the provider
    max_cloud_cover: float | None = None    # percent, catalog-level pre-filter
    max_results: int = 50

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("end date must not be before start date")
        if self.max_results < 1:
            raise ValueError("max_results must be >= 1")
        if self.max_cloud_cover is not None and not 0 <= self.max_cloud_cover <= 100:
            raise ValueError("max_cloud_cover must be within 0-100")


@dataclass(frozen=True)
class Granule:
    """One observation of one tile by one sensor, with downloadable layers."""

    granule_id: str
    provider: str
    sensor: str                              # e.g. HLSL30
    tile_id: str | None
    acquired: datetime                       # UTC; see provider docs for timing accuracy
    cloud_cover: float | None                # percent, as reported by the catalog
    spatial_coverage: float | None           # percent of tile with data
    footprint: BaseGeometry | None           # lon/lat polygon
    assets: dict[str, str] = field(default_factory=dict)      # layer name -> URL
    metadata: dict[str, Any] = field(default_factory=dict)    # scale, offset, fill, EPSG, ...

    @property
    def acquired_date(self) -> date:
        return self.acquired.date()


@dataclass
class SearchResult:
    """Granules plus the honest context needed to interpret them."""

    granules: list[Granule]
    truncated: bool = False                  # True if a result limit may have hidden granules
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AuthStatus:
    """Authentication diagnostics. Never contains secret values."""

    state: str            # 'ok' | 'not_configured' | 'failed' | 'unchecked'
    message: str
    can_search: bool      # catalog search availability
    can_download: bool    # data access availability


class DataProvider(ABC):
    """Interface every data source implements."""

    name: str

    @abstractmethod
    def search(self, query: SearchQuery) -> SearchResult:
        """Find granules for an AOI and time range. Must not download data."""

    @abstractmethod
    def auth_status(self, verify: bool = False) -> AuthStatus:
        """Report authentication state; ``verify=True`` may contact the auth server."""
