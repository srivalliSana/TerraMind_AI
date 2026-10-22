"""Tests for configuration loading and validation (no network, no credentials)."""
import logging
import shutil

import pytest

from src import config
from src.logging_config import RedactingFilter


def test_default_config_loads():
    s = config.load_settings()
    assert s.search.max_results >= 1
    assert s.aoi.min_area_km2 < s.aoi.max_area_km2
    assert s.data_dir.is_absolute()


def test_thresholds_are_marked_unvalidated():
    t = config.load_thresholds()
    assert t.validated is False


def test_sensors_registry_lists_hls():
    assert {"HLSL30", "HLSS30"} <= set(config.load_sensors())


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("TERRAMIND_LOG_LEVEL", "debug")
    monkeypatch.setenv("TERRAMIND_DATA_DIR", "custom_data")
    s = config.load_settings()
    assert s.logging.level == "DEBUG"
    assert s.paths.data_dir.name == "custom_data"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(config.ConfigError, match="Missing config file"):
        config.load_settings(tmp_path)


def test_invalid_values_rejected(tmp_path):
    shutil.copytree(config.CONFIG_DIR, tmp_path / "c")
    (tmp_path / "c" / "default.yaml").write_text("aoi:\n  min_area_km2: 10\n  max_area_km2: 5\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_settings(tmp_path / "c")


def test_unknown_keys_rejected(tmp_path):
    (tmp_path / "default.yaml").write_text("bogus: 1\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load_settings(tmp_path)


def test_threshold_ordering_validated(tmp_path):
    (tmp_path / "thresholds.yaml").write_text(
        "anomaly:\n  zscore_watch: -3\n  zscore_elevated: -2\n  zscore_high: -1\n", encoding="utf-8"
    )
    with pytest.raises(config.ConfigError):
        config.load_thresholds(tmp_path)


def test_credential_status_never_returns_values(monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "super-secret-token")
    status = config.credential_status()
    assert status["EARTHDATA_TOKEN"] is True
    assert "super-secret-token" not in str(status)


def test_log_redaction(monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "super-secret-token")
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "token=%s Authorization: Bearer abc.def", ("super-secret-token",), None)
    RedactingFilter().filter(rec)
    assert "super-secret-token" not in rec.getMessage()
    assert "abc.def" not in rec.getMessage()
