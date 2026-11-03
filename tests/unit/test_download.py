"""Download manager tests. Rasters are tiny SYNTHETIC GeoTIFFs written by a fake fetcher."""
from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.data.download import IntegrityError, download_granule, layers_for
from src.data.manifest import load_manifest
from src.data.providers.base import AuthError, Granule, ProviderError

GID = "HLS.L30.T15TVG.2024186T165841.v2.0"


def make_granule(layers=("B04", "B05", "B06", "Fmask")):
    return Granule(
        granule_id=GID, provider="nasa_hls", sensor="HLSL30", tile_id="T15TVG",
        acquired=datetime(2024, 7, 4, 17, 12, tzinfo=timezone.utc), cloud_cover=28, spatial_coverage=99,
        footprint=None, assets={l: f"https://example.test/{GID}.{l}.tif" for l in layers},
        metadata={"scale_factor": 0.0001, "add_offset": 0.0, "fill_value": -9999},
    )


def write_tif(path, crs="EPSG:32615"):
    data = np.arange(16, dtype="int16").reshape(1, 4, 4)
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="int16",
                       crs=crs, transform=from_origin(399960, 4700040, 30, 30)) as dst:
        dst.write(data)


class FakeFetcher:
    def __init__(self, script=None):
        self.calls = []
        self.script = script or {}

    def fetch(self, url, dest):
        self.calls.append(url)
        behaviour = self.script.get(len(self.calls))
        if isinstance(behaviour, Exception):
            raise behaviour
        if behaviour == "garbage":
            dest.write_bytes(b"this is not a tiff")
            return
        write_tif(dest)


@pytest.fixture
def dirs(tmp_path):
    return {"raw_dir": tmp_path / "raw", "manifest_dir": tmp_path / "manifests"}


def run(granule, layers, fetcher, dirs, **kw):
    return download_granule(granule, layers, fetcher=fetcher, backoff=False, **dirs, **kw)


def test_layers_for_always_adds_qa_and_maps_bands():
    assert layers_for("HLSL30") == ["B04", "B05", "B06", "Fmask"]
    assert layers_for("HLSS30") == ["B04", "B8A", "B11", "Fmask"]   # narrow NIR on Sentinel-2
    assert layers_for("HLSL30", ("red",)) == ["B04", "Fmask"]


def test_layers_for_rejects_unknown():
    with pytest.raises(ProviderError):
        layers_for("HLSL30", ("rededge1",))   # not an OLI band
    with pytest.raises(ProviderError):
        layers_for("NOPE")


def test_download_writes_files_and_manifest(dirs):
    f = FakeFetcher()
    m = run(make_granule(), ["B04", "Fmask"], f, dirs)
    assert len(f.calls) == 2
    assert set(m.files) == {"B04", "Fmask"}
    rec = m.files["B04"]
    assert (dirs["raw_dir"] / rec.path).is_file()
    assert rec.path.startswith("nasa_hls/HLSL30/T15TVG/") and len(rec.sha256) == 64 and rec.size_bytes > 0
    stored = load_manifest(dirs["manifest_dir"], GID)
    assert stored.catalog_metadata["scale_factor"] == 0.0001 and stored.sensor == "HLSL30"
    assert not list(dirs["raw_dir"].rglob("*.part"))   # no leftovers


def test_cache_hit_skips_download_and_force_redownloads(dirs):
    g = make_granule()
    run(g, ["B04"], FakeFetcher(), dirs)
    f = FakeFetcher()
    run(g, ["B04"], f, dirs)
    assert f.calls == []
    run(g, ["B04"], f, dirs, force=True)
    assert len(f.calls) == 1


def test_modified_cached_file_is_redownloaded(dirs):
    g = make_granule()
    m = run(g, ["B04"], FakeFetcher(), dirs)
    (dirs["raw_dir"] / m.files["B04"].path).write_bytes(b"corrupted")   # size no longer matches manifest
    f = FakeFetcher()
    run(g, ["B04"], f, dirs)
    assert len(f.calls) == 1


def test_corrupt_download_is_retried_then_succeeds(dirs):
    f = FakeFetcher({1: "garbage"})
    m = run(make_granule(), ["B04"], f, dirs, retries=2)
    assert len(f.calls) == 2 and "B04" in m.files


def test_network_errors_exhaust_retries(dirs):
    f = FakeFetcher({1: ProviderError("net"), 2: ProviderError("net"), 3: ProviderError("net")})
    with pytest.raises(ProviderError):
        run(make_granule(), ["B04"], f, dirs, retries=2)
    assert len(f.calls) == 3
    assert load_manifest(dirs["manifest_dir"], GID) is None or "B04" not in load_manifest(dirs["manifest_dir"], GID).files
    assert not list(dirs["raw_dir"].rglob("*.part"))


def test_auth_error_is_not_retried(dirs):
    f = FakeFetcher({1: AuthError("denied")})
    with pytest.raises(AuthError):
        run(make_granule(), ["B04"], f, dirs, retries=3)
    assert len(f.calls) == 1


def test_missing_layer_in_catalog_fails_before_any_download(dirs):
    f = FakeFetcher()
    with pytest.raises(ProviderError, match="no file for layer"):
        run(make_granule(layers=("B04",)), ["B04", "Fmask"], f, dirs)
    assert f.calls == []


def test_raster_without_crs_is_rejected(dirs, tmp_path):
    class NoCrs(FakeFetcher):
        def fetch(self, url, dest):
            with rasterio.open(dest, "w", driver="GTiff", height=2, width=2, count=1, dtype="int16") as d:
                d.write(np.zeros((1, 2, 2), dtype="int16"))

    with pytest.raises(IntegrityError, match="no CRS"):
        run(make_granule(), ["B04"], NoCrs(), dirs, retries=0)


def test_missing_credentials_give_auth_error_not_traceback(monkeypatch, tmp_path):
    import earthaccess

    from src.data.download import EarthdataFetcher

    def no_creds(**kw):
        raise RuntimeError("environment variables not set")

    monkeypatch.setattr(earthaccess, "login", no_creds)
    with pytest.raises(AuthError, match="EARTHDATA_TOKEN"):
        EarthdataFetcher().fetch("https://example.test/x.tif", tmp_path / "x.part")


def test_partial_progress_survives_failure(dirs):
    f = FakeFetcher({2: ProviderError("net")})
    with pytest.raises(ProviderError):
        run(make_granule(), ["B04", "Fmask"], f, dirs, retries=0)
    assert set(load_manifest(dirs["manifest_dir"], GID).files) == {"B04"}   # first file kept
