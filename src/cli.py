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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="terramind", description="TerraMind AI command-line tools")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check-env", help="verify environment, configuration and credentials")
    p.set_defaults(func=cmd_check_env)
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
