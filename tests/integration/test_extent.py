"""End-to-end affected-area analysis on SYNTHETIC processed outputs (not real satellite data)."""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.pipeline.assessment import assess_aoi
from src.pipeline.extent import compute_extent

AOI = "aoi_t"
SIZE = 40
RNG_SEED = 7


def write_obs(root, date, array, *, aoi=AOI, gid=None, valid=0.95, origin=(399960.0, 4700040.0), usable=True):
    gid = gid or f"HLS.S30.T15TVG.{date.replace('-', '')}T170000.v2.0"
    d = root / aoi / gid
    d.mkdir(parents=True, exist_ok=True)
    with rasterio.open(d / "ndvi.tif", "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype="float32", crs="EPSG:32615", transform=from_origin(*origin, 30, 30), nodata=np.nan) as dst:
        dst.write(array.astype("float32"), 1)
    med = float(np.nanmedian(array))
    (d / "summary.json").write_text(json.dumps({
        "granule_id": gid, "sensor": "HLSS30", "acquired": f"{date}T17:00:00+00:00", "usable_for_analysis": usable,
        "quality": {"valid_fraction": valid},
        "index_stats": {"ndvi": {"median": med, "std": 0.01, "count": int(np.isfinite(array).sum())},
                        "ndmi": {"median": 0.3, "std": 0.01, "count": int(np.isfinite(array).sum())}}}))
    return gid


def healthy(rng):
    return 0.8 + rng.normal(0, 0.01, (SIZE, SIZE))


@pytest.fixture
def root(tmp_path):
    rng = np.random.default_rng(RNG_SEED)
    for year in (2022, 2023):
        for day in ("07-10", "07-15", "07-20"):
            write_obs(tmp_path, f"{year}-{day}", healthy(rng))
    return tmp_path


def target_with_block(rng, root, rows=slice(0, 10), cols=slice(0, 10), drop=0.3, date="2024-07-15", **kw):
    arr = healthy(rng)
    arr[rows, cols] -= drop
    return write_obs(root, date, arr, **kw)


def test_block_decline_is_located_and_measured(root):
    gid = target_with_block(np.random.default_rng(1), root)
    r = compute_extent(root, AOI)
    assert r.ok and r.target_granule == gid and r.history_years == 2 and len(r.history_granules) == 6
    assert r.affected_pixels == 100 and r.patches == 1 and r.largest_patch_pixels == 100
    assert r.affected_fraction == pytest.approx(100 / 1600)
    assert r.area_ha == pytest.approx(9.0)                         # 100 px * 900 m2 = 9 ha
    d = root / AOI / gid
    with rasterio.open(d / "anomaly_z.tif") as s:
        z = s.read(1)
        assert s.crs.to_string() == "EPSG:32615" and np.nanmin(z) < -5
    with rasterio.open(d / "affected_mask.tif") as s:
        m = s.read(1)
    assert m.sum() == 100 and m[:10, :10].all() and not m[20:, 20:].any()
    saved = json.loads((d / "extent.json").read_text())
    assert saved["affected_pixels"] == 100 and saved["expected_by_chance"] == pytest.approx(0.02275, abs=1e-4)


def test_healthy_season_has_almost_no_affected_area(root):
    target_with_block(np.random.default_rng(2), root, drop=0.0)
    r = compute_extent(root, AOI)
    assert r.ok and r.affected_pixels == 0 and r.affected_fraction == 0.0


def test_isolated_pixels_and_small_patches_are_ignored(root):
    arr = healthy(np.random.default_rng(3))
    arr[5, 5] -= 0.4                       # single pixel
    arr[20:22, 20:22] -= 0.4               # 4-pixel patch (< 9)
    write_obs(root, "2024-07-15", arr)
    r = compute_extent(root, AOI)
    assert r.ok and r.raw_affected_pixels >= 5 and r.affected_pixels == 0 and r.patches == 0


def test_too_little_history_is_reported_not_faked(tmp_path):
    rng = np.random.default_rng(4)
    for day in ("07-05", "07-10", "07-15", "07-20", "07-25"):
        write_obs(tmp_path, f"2023-{day}", healthy(rng))          # one earlier year only
    write_obs(tmp_path, "2024-07-15", healthy(rng))
    r = compute_extent(tmp_path, AOI)
    assert r.status == "unavailable" and "earlier years" in r.reason and r.affected_fraction is None
    assert not list(tmp_path.rglob("anomaly_z.tif"))


def test_history_outside_the_seasonal_window_is_not_used(tmp_path):
    rng = np.random.default_rng(5)
    for year in (2022, 2023):
        for day in ("03-10", "03-15", "03-20"):                  # March history vs a July target
            write_obs(tmp_path, f"{year}-{day}", healthy(rng))
    write_obs(tmp_path, "2024-07-15", healthy(rng))
    r = compute_extent(tmp_path, AOI)
    assert r.status == "unavailable" and r.history_years == 0


def test_grid_mismatch_is_skipped_with_warning(root):
    rng = np.random.default_rng(6)
    bad = write_obs(root, "2023-07-25", healthy(rng), origin=(400020.0, 4700040.0))      # shifted by 2 pixels
    target_with_block(rng, root)
    r = compute_extent(root, AOI)
    assert r.ok and bad not in r.history_granules and any("pixel grid differs" in w for w in r.warnings)


def test_unusable_observations_are_excluded_from_history(root):
    rng = np.random.default_rng(8)
    cloudy = write_obs(root, "2023-07-12", healthy(rng) - 0.5, usable=False, valid=0.1)
    target_with_block(rng, root)
    assert cloudy not in compute_extent(root, AOI).history_granules


def test_no_observations_at_all(tmp_path):
    assert compute_extent(tmp_path, "aoi_none").status == "unavailable"


def test_assessment_includes_extent_component_and_evidence(root):
    target_with_block(np.random.default_rng(9), root)
    res = assess_aoi(root, AOI)
    assert res.extent.ok
    a = res.assessment
    assert a.affected_fraction == pytest.approx(100 / 1600)
    assert a.components["extent"] == pytest.approx(20 * (100 / 1600) / 0.30, rel=1e-3)      # 4.17 points
    assert any("connected patch" in e and "by chance" in e for e in a.evidence)
    assert not any("not assessed" in e for e in a.evidence)


def test_assessment_without_extent_still_works(root):
    target_with_block(np.random.default_rng(9), root)
    res = assess_aoi(root, AOI, with_extent=False)
    assert res.extent is None and res.assessment.components["extent"] is None
