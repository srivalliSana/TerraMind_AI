"""AOI tests. All geometries are synthetic, hand-made coordinates."""
import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from src.config import AOISettings
from src.geospatial import aoi as A

LIMITS = AOISettings(min_area_km2=0.01, max_area_km2=2500)


def test_bbox_basic_and_area():
    a = A.from_bbox(10.0, 50.0, 10.1, 50.1)
    assert a.bounds == (10.0, 50.0, 10.1, 50.1)
    # ~0.1° x 0.1° at 50°N is roughly 7.1 km x 11.1 km ≈ 79 km²
    assert 75 < a.area_km2 < 83
    assert a.id.startswith("aoi_")


def test_equator_one_degree_area():
    # 1°x1° at the equator ≈ 12,308 km² (geodesic)
    assert A.geodesic_area_km2(A.from_bbox(0, 0, 1, 1).geometry) == pytest.approx(12308, rel=0.01)


@pytest.mark.parametrize(
    "args",
    [(10, 50, 9, 51), (10, 50, 10, 51), (10, 51, 11, 50), (0, 0, 1, float("nan")), (0, 0, 1, float("inf"))],
)
def test_bbox_rejects_degenerate(args):
    with pytest.raises(A.AOIError):
        A.from_bbox(*args)


def test_bbox_out_of_range():
    with pytest.raises(A.AOIError, match="out of range"):
        A.from_bbox(170, 0, 190, 1)
    with pytest.raises(A.AOIError, match="out of range"):
        A.from_bbox(0, 80, 1, 95)


def test_area_limits():
    with pytest.raises(A.AOIError, match="exceeds"):
        A.from_bbox(0, 0, 2, 2, limits=LIMITS)
    with pytest.raises(A.AOIError, match="below"):
        A.from_bbox(0, 0, 0.00001, 0.00001, limits=LIMITS)
    assert A.from_bbox(0, 0, 0.1, 0.1, limits=LIMITS)


def test_id_is_stable_and_geometry_based():
    a = A.from_bbox(10, 50, 10.1, 50.1, name="one")
    b = A.from_bbox(10, 50, 10.1, 50.1, name="two")
    c = A.from_bbox(10, 50, 10.1, 50.2)
    assert a.id == b.id != c.id


def test_geojson_feature_collection_dissolves_and_roundtrips():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[10, 50], [10.1, 50], [10.1, 50.1], [10, 50.1], [10, 50]]]}},
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[10.1, 50], [10.2, 50], [10.2, 50.1], [10.1, 50.1], [10.1, 50]]]}},
        ],
    }
    a = A.from_geojson(fc)
    assert a.bounds == pytest.approx((10, 50, 10.2, 50.1))
    again = A.from_geojson(a.to_geojson())
    assert again.id == a.id


def test_geojson_from_file_and_string(tmp_path):
    feat = {"type": "Polygon", "coordinates": [[[1, 1], [1.1, 1], [1.1, 1.1], [1, 1.1], [1, 1]]]}
    p = tmp_path / "field.geojson"
    p.write_text(json.dumps(feat), encoding="utf-8")
    assert A.from_geojson(p).name == "field"
    assert A.from_geojson(str(p)).name == "field"
    assert A.from_geojson(json.dumps(feat)).source == "geojson"


@pytest.mark.parametrize("bad", ["{not json", {"foo": 1}, {"type": "FeatureCollection", "features": []}, {"type": "Polygon", "coordinates": "x"}])
def test_geojson_rejects_bad_input(bad):
    with pytest.raises(A.AOIError):
        A.from_geojson(bad)


def test_geojson_missing_file(tmp_path):
    with pytest.raises(A.AOIError, match="not found"):
        A.from_geojson(tmp_path / "nope.geojson")


def test_non_polygon_and_empty_rejected():
    with pytest.raises(A.AOIError, match="polygonal"):
        A.from_geometry(Point(1, 1))
    with pytest.raises(A.AOIError, match="polygonal"):
        A.from_geometry(LineString([(0, 0), (1, 1)]))
    with pytest.raises(A.AOIError):
        A.from_geometry(Polygon())


def test_invalid_bowtie_is_repaired():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    assert not bowtie.is_valid
    a = A.from_geometry(bowtie)
    assert a.geometry.is_valid and a.area_km2 > 0


def test_reprojection_from_utm():
    utm = gpd.GeoDataFrame(geometry=[Polygon([(500000, 5538000), (501000, 5538000), (501000, 5539000), (500000, 5539000)])], crs="EPSG:32632")
    a = A.from_geodataframe(utm)
    assert a.area_km2 == pytest.approx(1.0, rel=0.02)  # 1 km x 1 km
    assert -180 <= a.bounds[0] <= 180


def test_missing_crs_refused_unless_default_given():
    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (0.1, 0), (0.1, 0.1)])])
    with pytest.raises(A.AOIError, match="no CRS"):
        A.from_geodataframe(gdf)
    assert A.from_geodataframe(gdf, default_crs="EPSG:4326")


def test_swapped_lonlat_detected():
    # lat/lon swapped: "lon" 95 is fine but "lat" 150 is not
    with pytest.raises(A.AOIError, match="out of range"):
        A.from_bbox(95, 150, 96, 151)


def test_shapefile_roundtrip_and_missing_sidecar(tmp_path):
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(5, 5), (5.1, 5), (5.1, 5.1), (5, 5.1)])], crs="EPSG:4326")
    shp = tmp_path / "f.shp"
    gdf.to_file(shp)
    assert A.from_shapefile(shp).bounds == pytest.approx((5, 5, 5.1, 5.1))
    shp.with_suffix(".dbf").unlink()
    with pytest.raises(A.AOIError, match="sidecar"):
        A.from_shapefile(shp)
    with pytest.raises(A.AOIError, match=".shp"):
        A.from_shapefile(tmp_path / "f.dbf")


def test_to_geodataframe_is_wgs84():
    gdf = A.from_bbox(1, 1, 1.1, 1.1).to_geodataframe()
    assert gdf.crs.to_epsg() == 4326 and len(gdf) == 1
