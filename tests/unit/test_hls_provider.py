"""HLS provider tests. Records below are SYNTHETIC but mirror the structure of real CMR UMM-G
responses (verified against a live public search). No network, no credentials."""
from datetime import date

import pytest

from src.data.providers.base import ProviderError, SearchQuery
from src.data.providers.nasa_hls import NasaHLSProvider, parse_umm
from src.geospatial.aoi import from_bbox

BASE = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/HLSL30.020/{gid}/{gid}"


def make_umm(gid="HLS.L30.T15TVG.2024186T165841.v2.0", when="2024-07-04T17:12:10.868Z",
             cloud="28", box=(-94.2, 41.45, -93.1, 42.45), layers=("B04", "B05", "B06", "Fmask")):
    x0, y0, x1, y1 = box
    urls = []
    for layer in layers:
        urls.append({"Type": "GET DATA", "URL": f"{BASE.format(gid=gid)}.{layer}.tif"})
        urls.append({"Type": "GET DATA VIA DIRECT ACCESS", "URL": f"s3://x/{gid}.{layer}.tif"})
    urls.append({"Type": "VIEW RELATED INFORMATION", "URL": f"{BASE.format(gid=gid)}.cmr.xml"})
    return {"umm": {
        "GranuleUR": gid,
        "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": when}},
        "AdditionalAttributes": [
            {"Name": "CLOUD_COVERAGE", "Values": [cloud]},
            {"Name": "SPATIAL_COVERAGE", "Values": ["99"]},
            {"Name": "MGRS_TILE_ID", "Values": ["15TVG"]},
            {"Name": "REF_SCALE_FACTOR", "Values": ["0.0001"]},
            {"Name": "ADD_OFFSET", "Values": ["0"]},
            {"Name": "FILLVALUE", "Values": ["-9999"]},
            {"Name": "QA_FILLVALUE", "Values": ["255"]},
            {"Name": "HORIZONTAL_CS_CODE", "Values": ["EPSG:32615"]},
        ],
        "SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"GPolygons": [{"Boundary": {"Points": [
            {"Longitude": x0, "Latitude": y0}, {"Longitude": x1, "Latitude": y0},
            {"Longitude": x1, "Latitude": y1}, {"Longitude": x0, "Latitude": y1},
            {"Longitude": x0, "Latitude": y0}]}}]}}},
        "RelatedUrls": urls,
    }}


AOI = from_bbox(-93.7, 41.9, -93.6, 42.0)


def query(**kw):
    return SearchQuery(aoi=AOI, start=date(2024, 7, 1), end=date(2024, 7, 20), **kw)


def test_parse_umm_fields():
    g = parse_umm(make_umm()["umm"])
    assert g.sensor == "HLSL30" and g.tile_id == "T15TVG"
    assert g.cloud_cover == 28 and g.spatial_coverage == 99
    assert g.acquired.tzinfo is not None and g.acquired_date == date(2024, 7, 4)
    assert set(g.assets) == {"B04", "B05", "B06", "Fmask"}  # only https GET DATA .tif layers
    assert all(u.startswith("https://") for u in g.assets.values())
    assert g.metadata["scale_factor"] == 0.0001 and g.metadata["add_offset"] == 0
    assert g.metadata["fill_value"] == -9999 and g.metadata["epsg"] == "EPSG:32615"
    assert g.footprint is not None and g.footprint.is_valid


def test_sentinel_kind_maps_to_hlss30():
    assert parse_umm(make_umm(gid="HLS.S30.T15TVG.2024186T165849.v2.0")["umm"]).sensor == "HLSS30"


@pytest.mark.parametrize("bad", ["bogus", "HLS.X30.T15TVG.2024186T165841.v2.0", ""])
def test_parse_rejects_bad_ids(bad):
    with pytest.raises(ProviderError):
        parse_umm({"GranuleUR": bad})


def test_parse_requires_acquisition_time():
    u = make_umm()["umm"]
    del u["TemporalExtent"]
    with pytest.raises(ProviderError, match="acquisition time"):
        parse_umm(u)


def test_search_passes_aoi_bbox_dates_and_limit():
    calls = []

    def fake(**kw):
        calls.append(kw)
        return []

    NasaHLSProvider(fake).search(query(max_results=7))
    assert {c["short_name"] for c in calls} == {"HLSL30", "HLSS30"}
    c = calls[0]
    assert c["bounding_box"] == AOI.bounds and c["count"] == 7
    assert c["temporal"] == ("2024-07-01T00:00:00Z", "2024-07-20T23:59:59Z")  # end date inclusive


def test_search_sorts_dedupes_and_filters_cloud():
    items = [
        make_umm(gid="HLS.L30.T15TVG.2024190T165841.v2.0", when="2024-07-08T17:00:00Z", cloud="10"),
        make_umm(gid="HLS.L30.T15TVG.2024186T165841.v2.0", when="2024-07-04T17:00:00Z", cloud="80"),
        make_umm(gid="HLS.L30.T15TVG.2024190T165841.v2.0", when="2024-07-08T17:00:00Z", cloud="10"),  # duplicate
        make_umm(gid="HLS.L30.T15TVG.2024182T165841.v2.0", when="2024-06-30T17:00:00Z", cloud="5"),
    ]
    res = NasaHLSProvider(lambda **kw: items if kw["short_name"] == "HLSL30" else []).search(query(max_cloud_cover=50))
    assert [g.granule_id for g in res.granules] == [
        "HLS.L30.T15TVG.2024182T165841.v2.0", "HLS.L30.T15TVG.2024190T165841.v2.0"]  # 80% cloud dropped, dup removed, sorted


def test_footprint_not_touching_aoi_is_dropped():
    far = make_umm(box=(10.0, 10.0, 11.0, 11.0))
    assert NasaHLSProvider(lambda **kw: [far]).search(query(sensors=("HLSL30",))).granules == []


def test_truncation_is_reported():
    items = [make_umm(gid=f"HLS.L30.T15TVG.20241{i:02d}T165841.v2.0") for i in range(3)]
    res = NasaHLSProvider(lambda **kw: items).search(query(sensors=("HLSL30",), max_results=3))
    assert res.truncated and any("limit" in w for w in res.warnings)


def test_bad_record_skipped_with_warning():
    res = NasaHLSProvider(lambda **kw: [{"umm": {"GranuleUR": "junk"}}, make_umm()]).search(query(sensors=("HLSL30",)))
    assert len(res.granules) == 1 and any("Skipped" in w for w in res.warnings)


def test_catalog_failure_is_distinguished_from_auth():
    def boom(**kw):
        raise ConnectionError("dns")

    with pytest.raises(ProviderError, match="not an authentication problem"):
        NasaHLSProvider(boom).search(query(sensors=("HLSL30",)))


def test_unknown_sensor_rejected():
    with pytest.raises(ProviderError, match="Unknown sensor"):
        NasaHLSProvider(lambda **kw: []).search(query(sensors=("NOPE",)))


def test_query_validation():
    with pytest.raises(ValueError):
        SearchQuery(aoi=AOI, start=date(2024, 7, 2), end=date(2024, 7, 1))
    with pytest.raises(ValueError):
        query(max_cloud_cover=150)
    with pytest.raises(ValueError):
        query(max_results=0)


def test_auth_status_without_credentials(monkeypatch):
    for k in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    s = NasaHLSProvider().auth_status(verify=True)  # must not attempt a network login
    assert s.state == "not_configured" and s.can_search and not s.can_download


def test_auth_status_never_leaks_secret(monkeypatch):
    monkeypatch.setenv("EARTHDATA_TOKEN", "s3cr3t-token-value")
    s = NasaHLSProvider().auth_status(verify=False)
    assert s.state == "unchecked" and "s3cr3t" not in s.message
