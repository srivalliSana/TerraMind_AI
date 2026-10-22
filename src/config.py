"""Typed, validated configuration for TerraMind AI.

Configuration comes from ``configs/*.yaml`` with a small set of environment
overrides (``TERRAMIND_*``). Secrets are never part of the settings object;
:func:`credential_status` only reports whether they are present.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(RuntimeError):
    """Raised when configuration files are missing or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppSettings(_Strict):
    name: str = "TerraMind AI"
    demo_mode: bool = True


class PathSettings(_Strict):
    data_dir: Path = Path("data")
    db_path: Path = Path("data/terramind.sqlite")


class LoggingSettings(_Strict):
    level: str = "INFO"
    file: Path | None = None

    @model_validator(mode="after")
    def _check_level(self) -> "LoggingSettings":
        self.level = self.level.upper()
        if self.level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {self.level}")
        return self


class SearchSettings(_Strict):
    default_provider: str = "nasa_hls"
    max_results: int = Field(50, ge=1, le=2000)
    page_size: int = Field(50, ge=1, le=2000)
    max_cloud_cover: float = Field(60, ge=0, le=100)


class AOISettings(_Strict):
    max_area_km2: float = Field(2500, gt=0)
    min_area_km2: float = Field(0.01, gt=0)
    default_crs: str = "EPSG:4326"

    @model_validator(mode="after")
    def _check_bounds(self) -> "AOISettings":
        if self.min_area_km2 >= self.max_area_km2:
            raise ValueError("min_area_km2 must be smaller than max_area_km2")
        return self


class ProcessingSettings(_Strict):
    min_valid_fraction: float = Field(0.30, ge=0, le=1)
    max_workers: int = Field(2, ge=1, le=64)
    download_retries: int = Field(3, ge=0, le=10)
    chunk_size: int = Field(1024, ge=64)


class TimeSeriesSettings(_Strict):
    min_observations_for_baseline: int = Field(5, ge=2)
    smoothing_window: int = Field(3, ge=1)
    rolling_window: int = Field(3, ge=1)


class Settings(_Strict):
    app: AppSettings = AppSettings()
    paths: PathSettings = PathSettings()
    logging: LoggingSettings = LoggingSettings()
    search: SearchSettings = SearchSettings()
    aoi: AOISettings = AOISettings()
    processing: ProcessingSettings = ProcessingSettings()
    timeseries: TimeSeriesSettings = TimeSeriesSettings()

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def db_path(self) -> Path:
        return self.resolve(self.paths.db_path)


class Thresholds(_Strict):
    """Risk thresholds. Engineering defaults, not validated scientific rules."""

    validated: bool = False
    anomaly: dict[str, float] = {}
    affected_area: dict[str, float] = {}
    quality: dict[str, float] = {}
    risk_categories: dict[str, float] = {}

    @model_validator(mode="after")
    def _check_ordering(self) -> "Thresholds":
        a = self.anomaly
        keys = ("zscore_watch", "zscore_elevated", "zscore_high")
        if all(k in a for k in keys) and not (a[keys[0]] > a[keys[1]] > a[keys[2]]):
            raise ValueError("anomaly z-score thresholds must become more negative: watch > elevated > high")
        f = self.affected_area
        fk = ("min_fraction_watch", "min_fraction_elevated", "min_fraction_high")
        if all(k in f for k in fk) and not (f[fk[0]] < f[fk[1]] < f[fk[2]]):
            raise ValueError("affected-area fractions must increase: watch < elevated < high")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Missing config file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return data


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    paths = raw.setdefault("paths", {})
    logging_cfg = raw.setdefault("logging", {})
    if v := os.environ.get("TERRAMIND_DATA_DIR"):
        paths["data_dir"] = v
    if v := os.environ.get("TERRAMIND_DB_PATH"):
        paths["db_path"] = v
    if v := os.environ.get("TERRAMIND_LOG_LEVEL"):
        logging_cfg["level"] = v
    return raw


def load_settings(config_dir: Path | None = None) -> Settings:
    """Load and validate ``default.yaml`` plus environment overrides."""
    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR
    raw = _apply_env_overrides(_read_yaml(cfg_dir / "default.yaml"))
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {cfg_dir / 'default.yaml'}:\n{exc}") from exc


def load_thresholds(config_dir: Path | None = None) -> Thresholds:
    """Load and validate ``thresholds.yaml``."""
    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR
    try:
        return Thresholds.model_validate(_read_yaml(cfg_dir / "thresholds.yaml"))
    except ValidationError as exc:
        raise ConfigError(f"Invalid thresholds in {cfg_dir / 'thresholds.yaml'}:\n{exc}") from exc


def load_sensors(config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the sensor registry from ``sensors.yaml``."""
    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR
    sensors = _read_yaml(cfg_dir / "sensors.yaml").get("sensors")
    if not isinstance(sensors, dict) or not sensors:
        raise ConfigError("sensors.yaml must define a non-empty 'sensors' mapping")
    return sensors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for application code."""
    return load_settings()


def credential_status() -> dict[str, bool]:
    """Report which Earthdata credentials are present. Never returns values."""
    return {
        "EARTHDATA_TOKEN": bool(os.environ.get("EARTHDATA_TOKEN")),
        "EARTHDATA_USERNAME": bool(os.environ.get("EARTHDATA_USERNAME")),
        "EARTHDATA_PASSWORD": bool(os.environ.get("EARTHDATA_PASSWORD")),
    }


def load_dotenv_if_present(path: Path | None = None) -> bool:
    """Load ``.env`` into the process environment if it exists (no override)."""
    from dotenv import load_dotenv

    env_path = path or (PROJECT_ROOT / ".env")
    return load_dotenv(env_path, override=False) if env_path.is_file() else False
