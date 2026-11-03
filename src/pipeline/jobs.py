"""Processing job: downloaded HLS layers -> clipped, masked indices + provenance.

Order matters and is deliberate:
  read AOI window -> verify same grid -> scale (fill->NaN) -> decode Fmask ->
  mask unusable pixels -> compute indices -> summarise -> write GeoTIFFs + JSON.
Masking happens BEFORE indices so clouds/fill can never produce index values.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src import config
from src.data.manifest import GranuleManifest
from src.geospatial.alignment import assert_same_grid
from src.geospatial.aoi import AOI
from src.geospatial.raster_io import RasterMeta, read_band_clipped, write_geotiff
from src.indices.moisture import ndmi
from src.indices.vegetation import evi, ndvi
from src.preprocessing.cloud_mask import build_valid_mask, decode_fmask
from src.preprocessing.quality_control import QualityReport, compute_quality
from src.preprocessing.scaling import scale_reflectance
from src.storage.cache import is_cached_valid

log = logging.getLogger(__name__)
PIPELINE_VERSION = "0.1.0"
INDEX_CAVEAT = ("Spectral indices are indicators, not diagnoses: a decline can reflect season, planting, harvest, "
                "irrigation, residual cloud or sensor differences, not only crop stress.")


class ProcessingError(RuntimeError):
    """Processing cannot proceed (missing files, incompatible grids, no observed pixels...)."""


@dataclass
class ProcessedObservation:
    granule_id: str
    sensor: str
    acquired: str
    aoi_id: str
    output_dir: Path
    quality: QualityReport
    stats: dict[str, dict[str, float] | None]
    files: dict[str, str]
    usable_for_analysis: bool
    warnings: list[str] = field(default_factory=list)


def _summary_stats(arr: np.ndarray) -> dict[str, float] | None:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return None
    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    return {"count": int(v.size), "mean": float(v.mean()), "std": float(v.std()), "median": float(p50),
            "p10": float(p10), "p25": float(p25), "p75": float(p75), "p90": float(p90),
            "min": float(v.min()), "max": float(v.max())}


def _scale(raw: np.ndarray, meta: RasterMeta, catalog_scale: float | None, valid_range: tuple[float, float],
           warnings: list[str]) -> np.ndarray:
    """Scale using the file's own scale/offset; cross-check against the catalog value."""
    if meta.scale == 1.0 and catalog_scale is None:
        raise ProcessingError(f"{Path(meta.path).name}: no scale factor in file or catalog; refusing to guess")
    scale = meta.scale if meta.scale != 1.0 else catalog_scale
    if catalog_scale is not None and abs(scale - catalog_scale) > 1e-12:
        raise ProcessingError(f"Scale mismatch for {Path(meta.path).name}: file {scale} vs catalog {catalog_scale}")
    fill = meta.nodata if meta.nodata is not None else -9999
    try:
        return scale_reflectance(raw, scale=scale, offset=meta.offset, fill_value=fill, valid_range=valid_range)
    except ValueError as exc:
        raise ProcessingError(str(exc)) from exc


def process_granule(
    manifest: GranuleManifest,
    aoi: AOI,
    *,
    raw_dir: Path,
    out_root: Path,
    indices: tuple[str, ...] = ("ndvi", "ndmi"),
) -> ProcessedObservation:
    """Process one downloaded granule over an AOI and write outputs under ``out_root/<aoi_id>/<granule_id>/``."""
    t0 = time.time()
    st = config.get_settings().processing
    sensors = config.load_sensors()
    if manifest.sensor not in sensors:
        raise ProcessingError(f"Unknown sensor {manifest.sensor!r}")
    spec = sensors[manifest.sensor]
    need = {"red": spec["bands"]["red"], "nir": spec["bands"]["nir"], "swir1": spec["bands"]["swir1"]}
    if "evi" in indices:
        need["blue"] = spec["bands"]["blue"]
    layer_by_name = {**need, "fmask": spec["qa_layer"]}

    paths: dict[str, Path] = {}
    for name, layer in layer_by_name.items():
        rec = manifest.files.get(layer)
        if rec is None:
            raise ProcessingError(f"Layer {layer} ({name}) was not downloaded for {manifest.granule_id}; run download first")
        path = Path(raw_dir) / rec.path
        if not is_cached_valid(path, rec):
            raise ProcessingError(f"{path.name} is missing or does not match its manifest record; re-download with --force")
        paths[name] = path

    data: dict[str, np.ndarray] = {}
    metas: dict[str, RasterMeta] = {}
    inside = None
    for name, path in paths.items():
        arr, meta, ins = read_band_clipped(path, aoi)
        data[name], metas[name] = arr, meta
        inside = ins if inside is None else inside
    assert_same_grid(*metas.values())
    if inside is None or not inside.any():
        raise ProcessingError("AOI is smaller than one pixel at this resolution")

    warnings: list[str] = []
    cat = manifest.catalog_metadata
    refl = {n: _scale(data[n], metas[n], cat.get("scale_factor"), st.reflectance_valid_range, warnings)
            for n in need}

    fmask = data["fmask"]
    if fmask.dtype != np.uint8:
        raise ProcessingError(f"Fmask must be uint8, got {fmask.dtype}")
    flags = decode_fmask(fmask)
    valid = build_valid_mask(fmask, mask_cloud=st.mask_cloud, mask_adjacent=st.mask_adjacent,
                             mask_shadow=st.mask_shadow, mask_snow_ice=st.mask_snow_ice, mask_water=st.mask_water)
    quality = compute_quality(flags, valid, inside)
    if quality.observed_fraction == 0:
        raise ProcessingError(
            f"The AOI has no observed pixels in {manifest.granule_id} (outside the swath/tile coverage; "
            f"tile spatial coverage reported by catalog: {cat.get('spatial_coverage')}%)")

    usable_px = valid & inside
    for n in refl:
        refl[n] = np.where(usable_px, refl[n], np.nan).astype(np.float32)   # mask BEFORE computing indices

    results: dict[str, np.ndarray] = {}
    if "ndvi" in indices:
        results["ndvi"] = ndvi(refl["nir"], refl["red"])
    if "ndmi" in indices:
        results["ndmi"] = ndmi(refl["nir"], refl["swir1"])
    if "evi" in indices:
        results["evi"] = evi(refl["nir"], refl["red"], refl["blue"])
    if not results:
        raise ProcessingError(f"No supported indices requested: {indices}")

    usable_for_analysis = quality.valid_fraction >= st.min_valid_fraction
    if not usable_for_analysis:
        warnings.append(f"Only {quality.valid_fraction:.0%} of the AOI is valid (< {st.min_valid_fraction:.0%}); "
                        "excluded from analysis by default.")

    out_dir = Path(out_root) / aoi.id / manifest.granule_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(out_root) / aoi.id / "aoi.geojson").write_text(json.dumps(aoi.to_geojson()), encoding="utf-8")  # reload AOI by id
    meta0 = metas["red"]
    provenance = {"granule_id": manifest.granule_id, "sensor": manifest.sensor, "acquired": manifest.acquired,
                  "aoi_id": aoi.id, "pipeline_version": PIPELINE_VERSION, "caveat": INDEX_CAVEAT}
    files: dict[str, str] = {}
    stats: dict[str, dict[str, float] | None] = {}
    for name, arr in results.items():
        arr = np.where(inside, arr, np.nan).astype(np.float32)
        p = write_geotiff(out_dir / f"{name}.tif", arr, crs=meta0.crs, transform=meta0.transform, nodata=np.nan,
                          tags={**provenance, "index": name.upper(), "units": "unitless"})
        files[name] = p.name
        stats[name] = _summary_stats(arr)
        if stats[name] is None:
            warnings.append(f"{name.upper()} has no valid pixels inside the AOI.")
    vm = np.where(inside, usable_px, False).astype(np.uint8)
    files["valid_mask"] = write_geotiff(out_dir / "valid_mask.tif", vm, crs=meta0.crs, transform=meta0.transform,
                                        nodata=None, tags={**provenance, "meaning": "1=usable pixel inside AOI"}).name

    summary = {
        **provenance,
        "processed_at_unix": int(t0),
        "runtime_seconds": round(time.time() - t0, 2),
        "crs": meta0.crs, "resolution_m": meta0.resolution, "window_shape": list(meta0.shape),
        "layers": {n: layer for n, layer in layer_by_name.items()},
        "input_sha256": {rec.layer: rec.sha256 for rec in manifest.files.values() if rec.layer in layer_by_name.values()},
        "masking": {"cloud": st.mask_cloud, "adjacent": st.mask_adjacent, "shadow": st.mask_shadow,
                    "snow_ice": st.mask_snow_ice, "water": st.mask_water,
                    "reflectance_valid_range": list(st.reflectance_valid_range)},
        "catalog": {k: cat.get(k) for k in ("cloud_cover", "spatial_coverage")},
        "quality": quality.to_dict(),
        "usable_for_analysis": usable_for_analysis,
        "index_stats": stats,
        "warnings": warnings,
        "files": files,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files["summary"] = "summary.json"
    log.info("processed %s over %s: valid=%.1f%%", manifest.granule_id, aoi.id, quality.valid_fraction * 100)
    return ProcessedObservation(manifest.granule_id, manifest.sensor, manifest.acquired, aoi.id, out_dir,
                                quality, stats, files, usable_for_analysis, warnings)

