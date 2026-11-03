"""Provider registry and high-level search entry point."""
from __future__ import annotations

from datetime import date

from src import config
from src.data.providers.base import DataProvider, ProviderError, SearchQuery, SearchResult
from src.data.providers.nasa_hls import NasaHLSProvider
from src.geospatial.aoi import AOI

_PROVIDERS: dict[str, type[DataProvider]] = {"nasa_hls": NasaHLSProvider}


def get_provider(name: str | None = None) -> DataProvider:
    """Instantiate a provider by name (default from configuration)."""
    name = name or config.get_settings().search.default_provider
    try:
        return _PROVIDERS[name]()
    except KeyError as exc:
        raise ProviderError(f"Unknown provider {name!r}. Available: {sorted(_PROVIDERS)}") from exc


def search_granules(
    aoi: AOI,
    start: date,
    end: date,
    *,
    provider: str | None = None,
    sensors: tuple[str, ...] | None = None,
    max_cloud_cover: float | None = None,
    max_results: int | None = None,
) -> SearchResult:
    """Search for granules over an AOI, applying configured result limits."""
    settings = config.get_settings().search
    limit = min(max_results or settings.max_results, settings.max_results)
    query = SearchQuery(
        aoi=aoi, start=start, end=end, sensors=sensors,
        max_cloud_cover=settings.max_cloud_cover if max_cloud_cover is None else max_cloud_cover,
        max_results=limit,
    )
    return get_provider(provider).search(query)
