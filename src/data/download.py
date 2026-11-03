"""Selective, cached, verified download of granule layers.

Guarantees:
- Only the requested layers are fetched, and the QA layer (Fmask) is *always*
  included so cloud masking cannot be forgotten.
- Files are written to ``*.part`` and renamed only after integrity checks pass.
- Valid cached files (matching the manifest) are never downloaded again.
- Network errors are retried with backoff; authentication errors are not.
- Every file is recorded in a per-granule manifest (provenance).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

import rasterio
from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt, wait_exponential, wait_none

from src import config
from src.data.manifest import (
    FileRecord,
    GranuleManifest,
    load_manifest,
    save_manifest,
    utc_now_iso,
)
from src.data.providers.base import AuthError, Granule, ProviderError
from src.storage.cache import is_cached_valid, layer_path, relative_posix, sha256_file

log = logging.getLogger(__name__)

DEFAULT_BANDS = ("red", "nir", "swir1")


class IntegrityError(ProviderError):
    """A downloaded file failed validation."""


class Fetcher(Protocol):
    """Anything that can copy a URL to a local path. Injectable for tests."""

    def fetch(self, url: str, dest: Path) -> None: ...


class EarthdataFetcher:
    """Authenticated HTTPS streaming download from NASA Earthdata (LP DAAC)."""

    def __init__(self) -> None:
        self._session = None

    def _get_session(self):
        if self._session is None:
            import earthaccess

            hint = ("Set EARTHDATA_TOKEN (or EARTHDATA_USERNAME/EARTHDATA_PASSWORD) in .env, and accept the HLS "
                    "terms in your Earthdata profile. Run 'terramind auth-status --verify' to diagnose.")
            try:
                auth = earthaccess.login(strategy="environment")
            except Exception as exc:  # noqa: BLE001 - earthaccess raises its own exception types
                raise AuthError(f"Earthdata login unavailable ({type(exc).__name__}). {hint}") from exc
            if not getattr(auth, "authenticated", False):
                raise AuthError(f"Earthdata login failed. {hint}")
            self._session = earthaccess.get_requests_https_session()
        return self._session

    def fetch(self, url: str, dest: Path) -> None:
        import requests

        session = self._get_session()
        try:
            with session.get(url, stream=True, timeout=(10, 120)) as resp:
                if resp.status_code in (401, 403):
                    raise AuthError(
                        f"Download refused (HTTP {resp.status_code}). The catalog is public but data access needs a valid "
                        "Earthdata login and accepted HLS EULA. Run 'terramind auth-status --verify'."
                    )
                resp.raise_for_status()
                expected = resp.headers.get("Content-Length")
                written = 0
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        written += len(chunk)
            if expected is not None and written != int(expected):
                raise IntegrityError(f"Truncated download: got {written} of {expected} bytes")
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise ProviderError(f"Network error while downloading: {type(exc).__name__}") from exc
        except requests.HTTPError as exc:
            raise ProviderError(f"HTTP error while downloading: {exc.response.status_code}") from exc


def layers_for(sensor: str, band_names: tuple[str, ...] = DEFAULT_BANDS) -> list[str]:
    """Map semantic band names (red, nir, swir1 ...) to layer names, always adding the QA layer."""
    spec = config.load_sensors().get(sensor)
    if spec is None:
        raise ProviderError(f"Unknown sensor {sensor!r}")
    bands = spec["bands"]
    missing = [b for b in band_names if b not in bands]
    if missing:
        raise ProviderError(f"Sensor {sensor} has no band(s) {missing}. Available: {sorted(bands)}")
    layers = [bands[b] for b in band_names]
    qa = spec["qa_layer"]
    if qa not in layers:
        layers.append(qa)
    return layers


def verify_geotiff(path: Path) -> None:
    """Raise :class:`IntegrityError` unless the file is a readable, georeferenced raster."""
    try:
        with rasterio.open(path) as src:
            if src.count < 1 or src.width < 1 or src.height < 1:
                raise IntegrityError(f"{path.name}: empty raster")
            if src.crs is None:
                raise IntegrityError(f"{path.name}: no CRS")
    except rasterio.errors.RasterioError as exc:
        raise IntegrityError(f"{path.name}: not a readable GeoTIFF ({exc})") from exc


def _fetch_verified(fetcher: Fetcher, url: str, final: Path, retries: int, backoff: bool) -> None:
    part = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)

    def attempt() -> None:
        part.unlink(missing_ok=True)
        fetcher.fetch(url, part)
        if not part.is_file() or part.stat().st_size == 0:
            raise IntegrityError(f"{final.name}: empty download")
        verify_geotiff(part)

    try:
        for retry in Retrying(
            stop=stop_after_attempt(retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=20) if backoff else wait_none(),
            retry=retry_if_not_exception_type(AuthError),
            reraise=True,
            before_sleep=lambda rs: log.warning("Retrying %s (attempt %d failed)", final.name, rs.attempt_number),
        ):
            with retry:
                attempt()
        os.replace(part, final)
    finally:
        part.unlink(missing_ok=True)


def download_granule(
    granule: Granule,
    layers: list[str],
    *,
    raw_dir: Path,
    manifest_dir: Path,
    fetcher: Fetcher | None = None,
    retries: int | None = None,
    force: bool = False,
    backoff: bool = True,
) -> GranuleManifest:
    """Download ``layers`` of ``granule`` (skipping valid cached files) and update its manifest."""
    missing = [layer for layer in layers if layer not in granule.assets]
    if missing:
        raise ProviderError(f"{granule.granule_id}: catalog lists no file for layer(s) {missing}")
    retries = config.get_settings().processing.download_retries if retries is None else retries
    fetcher = fetcher or EarthdataFetcher()
    raw_dir = Path(raw_dir)

    manifest = load_manifest(manifest_dir, granule.granule_id) or GranuleManifest(
        granule_id=granule.granule_id, provider=granule.provider, sensor=granule.sensor,
        tile_id=granule.tile_id, acquired=granule.acquired.isoformat(),
        catalog_metadata={**granule.metadata, "cloud_cover": granule.cloud_cover,
                          "spatial_coverage": granule.spatial_coverage},
    )

    for layer in layers:
        path = layer_path(raw_dir, granule, layer)
        if not force and is_cached_valid(path, manifest.files.get(layer)):
            log.info("cache hit: %s", path.name)
            continue
        url = granule.assets[layer]
        log.info("downloading %s", path.name)
        _fetch_verified(fetcher, url, path, retries, backoff)
        manifest.files[layer] = FileRecord(
            layer=layer, url=url, path=relative_posix(raw_dir, path), size_bytes=path.stat().st_size,
            sha256=sha256_file(path), downloaded_at=utc_now_iso(),
        )
        save_manifest(manifest_dir, manifest)  # persist after each file so a crash keeps progress

    save_manifest(manifest_dir, manifest)
    return manifest
