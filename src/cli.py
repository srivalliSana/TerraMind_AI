"""TerraMind AI command-line interface.

Run with ``python -m src.cli <command>`` (or ``terramind <command>`` once the
package is installed with ``pip install -e .``).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import logging
import sys
from pathlib import Path

from src import config
from src.logging_config import setup_logging

log = logging.getLogger(__name__)

REQUIRED_PACKAGES = [
    "numpy", "pandas", "scipy", "rasterio", "geopandas", "shapely", "pyproj",
    "xarray", "dask", "sklearn", "earthaccess", "pystac_client", "stackstac",
    "pydantic", "yaml", "plotly", "folium", "streamlit",
]
OPTIONAL_PACKAGES = ["fastapi", "uvicorn", "torch", "pytest"]


def _pkg_version(module: str) -> str | None:
    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - report any import failure
        return None
    dist = {"sklearn": "scikit-learn", "yaml": "PyYAML", "pystac_client": "pystac-client"}.get(module, module)
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "unknown"


def cmd_check_env(_: argparse.Namespace) -> int:
    """Check Python, packages, configuration, credentials and directories."""
    problems = 0
    print(f"Python        : {sys.version.split()[0]} ({sys.executable})")

    print("\nRequired packages")
    for name in REQUIRED_PACKAGES:
        ver = _pkg_version(name)
        print(f"  [{'ok' if ver else 'MISSING':>7}] {name} {ver or ''}")
        problems += ver is None
    print("Optional packages")
    for name in OPTIONAL_PACKAGES:
        ver = _pkg_version(name)
        print(f"  [{'ok' if ver else 'absent':>7}] {name} {ver or ''}")

    print("\nConfiguration")
    try:
        settings = config.load_settings()
        thresholds = config.load_thresholds()
        sensors = config.load_sensors()
        print(f"  [     ok] default.yaml, thresholds.yaml (validated={thresholds.validated}), "
              f"sensors.yaml ({', '.join(sensors)})")
    except config.ConfigError as exc:
        print(f"  [  ERROR] {exc}")
        return 1

    print("\nEarthdata credentials (presence only, values are never shown)")
    status = config.credential_status()
    for key, present in status.items():
        print(f"  [{'set' if present else 'unset':>7}] {key}")
    if not (status["EARTHDATA_TOKEN"] or (status["EARTHDATA_USERNAME"] and status["EARTHDATA_PASSWORD"])):
        print("  note: no credentials. Catalog search and demo mode still work; downloads need credentials "
              "(see README, 'NASA Earthdata authentication').")

    print("\nDirectories")
    for label, path in (("data", settings.data_dir), ("db parent", settings.db_path.parent)):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            print(f"  [     ok] {label}: {path}")
        except OSError as exc:
            print(f"  [  ERROR] {label}: {path} ({exc})")
            problems += 1

    print("\nResult:", "OK" if not problems else f"{problems} problem(s) found")
    return 1 if problems else 0


def _load_aoi(args: argparse.Namespace):
    from src.geospatial import aoi as aoi_mod

    limits = config.get_settings().aoi
    if args.bbox:
        return aoi_mod.from_bbox(*args.bbox, name=args.name or "cli-bbox", limits=limits)
    path = Path(args.aoi_file)
    if path.suffix.lower() == ".shp":
        return aoi_mod.from_shapefile(path, name=args.name, limits=limits)
    return aoi_mod.from_geojson(path, name=args.name, limits=limits)


def cmd_search(args: argparse.Namespace) -> int:
    """Search the catalog for observations over an AOI (no download, no credentials needed)."""
    import json
    from datetime import date

    from src.data.providers.base import ProviderError
    from src.data.search import search_granules
    from src.geospatial.aoi import AOIError

    try:
        aoi = _load_aoi(args)
        result = search_granules(
            aoi, date.fromisoformat(args.start), date.fromisoformat(args.end),
            sensors=tuple(args.sensors) if args.sensors else None,
            max_cloud_cover=args.max_cloud, max_results=args.limit,
        )
    except (AOIError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"AOI {aoi.id} '{aoi.name}': {aoi.area_km2:,.2f} km2, bounds={tuple(round(b, 4) for b in aoi.bounds)}")
    print(f"{len(result.granules)} granule(s) found (catalog metadata, nothing downloaded)\n")
    print(f"{'date':<11}{'sensor':<8}{'tile':<8}{'cloud%':>7}{'cover%':>8}  granule_id")
    for g in result.granules:
        cc = "-" if g.cloud_cover is None else f"{g.cloud_cover:.0f}"
        sc = "-" if g.spatial_coverage is None else f"{g.spatial_coverage:.0f}"
        print(f"{g.acquired_date.isoformat():<11}{g.sensor:<8}{g.tile_id or '-':<8}{cc:>7}{sc:>8}  {g.granule_id}")
    for w in result.warnings:
        print(f"warning: {w}")
    print("\nNote: cloud% is for the whole ~110 km tile, not your AOI. Exact AOI cloud cover is computed after processing.")

    if args.json_out:
        out = [{"granule_id": g.granule_id, "sensor": g.sensor, "tile": g.tile_id, "acquired": g.acquired.isoformat(),
                "cloud_cover": g.cloud_cover, "spatial_coverage": g.spatial_coverage, "assets": g.assets,
                "metadata": g.metadata} for g in result.granules]
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


def cmd_auth_status(args: argparse.Namespace) -> int:
    """Diagnose Earthdata authentication (search vs download), without printing secrets."""
    from src.data.search import get_provider

    s = get_provider().auth_status(verify=args.verify)
    print(f"state         : {s.state}\ncatalog search: {'available' if s.can_search else 'unavailable'}\n"
          f"downloads     : {'available' if s.can_download else 'unavailable'}\n{s.message}")
    return 0 if s.state in {"ok", "unchecked", "not_configured"} else 1


def _add_search_args(p: argparse.ArgumentParser) -> None:
    src_group = p.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    src_group.add_argument("--aoi-file", help="GeoJSON or .shp file")
    p.add_argument("--name", help="AOI name")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--sensors", nargs="+", help="e.g. HLSL30 HLSS30 (default: all)")
    p.add_argument("--max-cloud", type=float, help="max tile cloud cover percent (default from config)")
    p.add_argument("--limit", type=int, help="max results per sensor (capped by config)")


def cmd_download(args: argparse.Namespace) -> int:
    """Download only the needed layers for the least-cloudy matching granules."""
    from datetime import date

    from src.data.download import download_granule, layers_for
    from src.data.manifest import load_manifest
    from src.data.providers.base import AuthError, ProviderError
    from src.data.search import search_granules
    from src.geospatial.aoi import AOIError
    from src.storage.cache import is_cached_valid, layer_path

    settings = config.get_settings()
    raw_dir, manifest_dir = settings.data_dir / "raw", settings.data_dir / "manifests"
    try:
        aoi = _load_aoi(args)
        result = search_granules(
            aoi, date.fromisoformat(args.start), date.fromisoformat(args.end),
            sensors=tuple(args.sensors) if args.sensors else None,
            max_cloud_cover=args.max_cloud, max_results=args.limit,
        )
        chosen = sorted(result.granules, key=lambda g: (g.cloud_cover is None, g.cloud_cover or 0))[: args.max_granules]
        if not chosen:
            print("No matching granules; nothing to download.")
            return 0
        print(f"Plan: {len(chosen)} granule(s) (least cloudy first, cap {args.max_granules}); bands={args.bands} + Fmask")
        for g in chosen:
            layers = layers_for(g.sensor, tuple(args.bands))
            man = load_manifest(manifest_dir, g.granule_id)
            for layer in layers:
                cached = is_cached_valid(layer_path(raw_dir, g, layer), man.files.get(layer) if man else None)
                print(f"  {g.granule_id}  {layer:<6} {'cached' if cached else 'to download'}")
        if args.dry_run:
            print("\nDry run: nothing downloaded.")
            return 0
        for g in chosen:
            m = download_granule(g, layers_for(g.sensor, tuple(args.bands)), raw_dir=raw_dir,
                                 manifest_dir=manifest_dir, force=args.force)
            print(f"ok {g.granule_id}: {len(m.files)} file(s), manifest data/manifests/{g.granule_id}.json")
        return 0
    except AuthError as exc:
        print(f"Authentication problem (search works, download does not): {exc}", file=sys.stderr)
        return 3
    except (AOIError, ProviderError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def cmd_process(args: argparse.Namespace) -> int:
    """Clip, mask and compute indices for an already-downloaded granule over an AOI."""
    from src.data.manifest import load_manifest
    from src.geospatial.aoi import AOIError
    from src.geospatial.raster_io import RasterError
    from src.pipeline.jobs import ProcessingError, process_granule

    settings = config.get_settings()
    try:
        aoi = _load_aoi(args)
        manifest = load_manifest(settings.data_dir / "manifests", args.granule_id)
        if manifest is None:
            print(f"No manifest for {args.granule_id}. Download it first with the 'download' command.", file=sys.stderr)
            return 2
        res = process_granule(manifest, aoi, raw_dir=settings.data_dir / "raw", out_root=settings.data_dir / "processed",
                              indices=tuple(args.indices))
    except (AOIError, ProcessingError, RasterError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    q = res.quality
    print(f"{res.granule_id} over AOI {res.aoi_id} -> {res.output_dir}")
    print(f"  AOI pixels: {q.aoi_pixels}; observed {q.observed_fraction:.1%}, valid {q.valid_fraction:.1%} "
          f"(cloud {q.cloud_fraction:.1%}, adjacent {q.adjacent_fraction:.1%}, shadow {q.shadow_fraction:.1%}, "
          f"snow/ice {q.snow_ice_fraction:.1%}, water {q.water_fraction:.1%}, no-data {q.fill_fraction:.1%})")
    for name, s in res.stats.items():
        print(f"  {name.upper()}: " + ("no valid pixels" if s is None else
              f"mean {s['mean']:.3f}, median {s['median']:.3f}, std {s['std']:.3f}, p10-p90 {s['p10']:.3f}..{s['p90']:.3f} (n={s['count']})"))
    for w in res.warnings:
        print(f"  warning: {w}")
    print(f"  usable for analysis: {res.usable_for_analysis}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="terramind", description="TerraMind AI command-line tools")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check-env", help="verify environment, configuration and credentials")
    p.set_defaults(func=cmd_check_env)

    p = sub.add_parser("search", help="search satellite observations over an AOI")
    _add_search_args(p)
    p.add_argument("--json-out", help="write results to a JSON file")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("download", help="download selected bands (+ Fmask) for observations over an AOI")
    _add_search_args(p)
    p.add_argument("--bands", nargs="+", default=["red", "nir", "swir1"],
                   help="semantic band names (default: red nir swir1); Fmask is always added")
    p.add_argument("--max-granules", type=int, default=3, help="safety cap on granules to download (default 3)")
    p.add_argument("--dry-run", action="store_true", help="show the plan without downloading")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("process", help="clip, mask and compute indices for a downloaded granule")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    g.add_argument("--aoi-file", help="GeoJSON or .shp file")
    p.add_argument("--name", help="AOI name")
    p.add_argument("--granule-id", required=True)
    p.add_argument("--indices", nargs="+", default=["ndvi", "ndmi"], choices=["ndvi", "ndmi", "evi"])
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("auth-status", help="check NASA Earthdata authentication")
    p.add_argument("--verify", action="store_true", help="contact Earthdata to verify credentials")
    p.set_defaults(func=cmd_auth_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    config.load_dotenv_if_present()
    args = build_parser().parse_args(argv)
    try:
        settings = config.load_settings()
        setup_logging(settings.logging.level, settings.resolve(settings.logging.file) if settings.logging.file else None)
    except config.ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
