"""Pin the HLS v2.0 spec facts verified against the LP DAAC HLS User Guide V2.0.

If one of these fails, re-check the official documentation before changing the test.
"""
import yaml

from src.config import CONFIG_DIR, load_sensors


def _raw():
    return yaml.safe_load((CONFIG_DIR / "sensors.yaml").read_text(encoding="utf-8"))


def test_hlsl30_band_mapping():
    b = load_sensors()["HLSL30"]["bands"]
    assert (b["red"], b["nir"], b["swir1"], b["swir2"]) == ("B04", "B05", "B06", "B07")


def test_hlss30_band_mapping_uses_narrow_nir_and_distinct_swir():
    b = load_sensors()["HLSS30"]["bands"]
    assert (b["red"], b["nir"], b["swir1"], b["swir2"]) == ("B04", "B8A", "B11", "B12")
    assert b["nir"] != b["nir_broad"]


def test_scale_and_fill_values():
    r = _raw()["hls_v2"]
    assert r["reflectance"]["scale_factor"] == 0.0001
    assert r["reflectance"]["fill_value"] == -9999
    assert r["fmask"]["fill_value"] == 255


def test_fmask_bit_positions():
    bits = {k: v["name"] for k, v in _raw()["hls_v2"]["fmask"]["bits"].items()}
    assert bits[1] == "cloud"
    assert bits[2] == "adjacent_to_cloud_shadow"
    assert bits[3] == "cloud_shadow"
    assert bits[4] == "snow_ice"
    assert bits[5] == "water"


def test_sensors_flagged_verified():
    assert all(s["spec_verified"] for s in load_sensors().values())
