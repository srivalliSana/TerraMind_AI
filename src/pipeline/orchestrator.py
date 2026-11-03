"""Series orchestration: search -> pick one granule per time slot -> download -> process.

Design notes:
- One granule per slot keeps downloads bounded and avoids near-duplicate observations.
- Catalog cloud cover is for the whole ~110 km tile, so it is only a rough ranking signal; the real
  AOI quality is measured after processing and unusable dates are dropped later.
- A failure on one granule (no AOI coverage, bad file) is recorded and the run continues, but an
  authentication failure aborts immediately because every later download would fail the same way.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

from src import config
from src.data.download import download_granule, layers_for
from src.data.manifest import load_manifest
from src.data.providers.base import AuthError, Granule, ProviderError
from src.data.search import search_granules
from src.geospatial.aoi import AOI
from src.geospatial.raster_io import RasterError
from src.pipeline.jobs import ProcessedObservation, ProcessingError, process_granule

log = logging.getLogger(__name__)


@dataclass
class SeriesReport:
    planned: list[Granule] = field(default_factory=list)
    processed: list[ProcessedObservation] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)      # (granule_id, reason)
    warnings: list[str] = field(default_factory=list)


def select_per_slot(granules: list[Granule], slot_days: int, min_spatial_coverage: float = 50.0) -> list[Granule]:
    """Keep the best granule per ``slot_days`` window: lowest tile cloud cover, then highest tile coverage."""
    if slot_days < 1:
        raise ValueError("slot_days must be >= 1")
    slots: dict[int, list[Granule]] = defaultdict(list)
    for g in granules:
        if g.spatial_coverage is not None and g.spatial_coverage < min_spatial_coverage:
            continue
        slots[g.acquired_date.toordinal() // slot_days].append(g)
    best = [min(gs, key=lambda g: (g.cloud_cover if g.cloud_cover is not None else 100.0,
                                   -(g.spatial_coverage or 0.0), g.acquired)) for gs in slots.values()]
    return sorted(best, key=lambda g: g.acquired)


def run_series(
    aoi: AOI,
    ranges: list[tuple[date, date]],
    *,
    raw_dir: Path,
    manifest_dir: Path,
    out_root: Path,
    sensors: tuple[str, ...] | None = None,
    bands: tuple[str, ...] = ("red", "nir", "swir1"),
    slot_days: int = 7,
    max_granules: int = 20,
    dry_run: bool = False,
    search: Callable[..., object] = search_granules,
    download: Callable[..., object] = download_granule,
    process: Callable[..., ProcessedObservation] = process_granule,
    fetcher=None,
) -> SeriesReport:
    """Plan (and unless ``dry_run`` execute) a multi-range download-and-process run."""
    report = SeriesReport()
    candidates: list[Granule] = []
    for start, end in ranges:
        res = search(aoi, start, end, sensors=sensors)
        report.warnings.extend(res.warnings)
        chosen = select_per_slot(res.granules, slot_days)
        log.info("%s..%s: %d granules found, %d selected", start, end, len(res.granules), len(chosen))
        candidates.extend(chosen)
    if len(candidates) > max_granules:
        report.warnings.append(f"{len(candidates)} slots selected; capped to {max_granules} (--max-granules).")
        candidates = sorted(candidates, key=lambda g: g.cloud_cover if g.cloud_cover is not None else 100.0)[:max_granules]
        candidates.sort(key=lambda g: g.acquired)
    report.planned = candidates
    if dry_run:
        return report

    for g in candidates:
        try:
            layers = layers_for(g.sensor, bands)
            kwargs = {"fetcher": fetcher} if fetcher is not None else {}
            download(g, layers, raw_dir=raw_dir, manifest_dir=manifest_dir, **kwargs)
            manifest = load_manifest(manifest_dir, g.granule_id)
            if manifest is None:
                raise ProcessingError("manifest missing after download")
            report.processed.append(process(manifest, aoi, raw_dir=raw_dir, out_root=out_root))
        except AuthError:
            raise                                    # every further download would fail identically
        except (ProviderError, ProcessingError, RasterError) as exc:
            log.warning("skipping %s: %s", g.granule_id, exc)
            report.failed.append((g.granule_id, str(exc)))
    return report
