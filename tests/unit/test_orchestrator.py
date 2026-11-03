"""Orchestrator tests: slot selection and failure handling. Uses SYNTHETIC granules and fake callables."""
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from src.data.providers.base import AuthError, Granule, ProviderError
from src.geospatial import aoi as A
from src.pipeline.jobs import ProcessingError
from src.pipeline.orchestrator import run_series, select_per_slot

AOI = A.from_bbox(-93.7, 41.9, -93.6, 42.0)


def g(day, cloud, cover=99, sensor="HLSS30"):
    d = datetime(2024, 7, day, 17, tzinfo=timezone.utc)
    return Granule(f"HLS.S30.T15TVG.2024{d.timetuple().tm_yday:03d}T170000.v2.0", "nasa_hls", sensor, "T15TVG", d,
                   cloud, cover, None, {"B04": "u", "B8A": "u", "B11": "u", "Fmask": "u"}, {})


def test_select_best_per_slot_prefers_low_cloud_then_coverage():
    gs = [g(1, 40), g(3, 10), g(5, 10, cover=60), g(12, 30), g(13, 5, cover=40)]  # last one below coverage floor
    # ordinals of Jul 1,3,5 may straddle a 7-day boundary; assert on invariants instead of exact slots
    out = select_per_slot(gs, slot_days=7, min_spatial_coverage=50)
    assert all(x.spatial_coverage >= 50 for x in out)
    assert out == sorted(out, key=lambda x: x.acquired)
    assert len({x.acquired_date.toordinal() // 7 for x in out}) == len(out)    # at most one per slot
    assert g(13, 5, cover=40).granule_id not in {x.granule_id for x in out}


def test_slot_selection_picks_lowest_cloud_within_one_slot():
    base = date(2024, 7, 1).toordinal()
    first =(date.fromordinal(base + (7 - base % 7) % 7))       # slot boundary on/after Jul 1
    a = Granule("HLS.S30.T15TVG.2024190T170000.v2.0", "nasa_hls", "HLSS30", "T15TVG",
                datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc), 50, 99, None, {}, {})
    b = Granule("HLS.S30.T15TVG.2024191T170000.v2.0", "nasa_hls", "HLSS30", "T15TVG",
                datetime.combine(date.fromordinal(first.toordinal() + 2), datetime.min.time(), tzinfo=timezone.utc), 12, 99, None, {}, {})
    assert [x.granule_id for x in select_per_slot([a, b], 7)] == [b.granule_id]


def test_slot_days_validation():
    with pytest.raises(ValueError):
        select_per_slot([], 0)


def fake_search_factory(granules):
    return lambda aoi, s, e, **kw: SimpleNamespace(granules=granules, warnings=[])


def test_dry_run_downloads_nothing():
    called = []
    rep = run_series(AOI, [(date(2024, 7, 1), date(2024, 7, 31))], raw_dir=None, manifest_dir=None, out_root=None,
                     dry_run=True, search=fake_search_factory([g(2, 10), g(20, 5)]),
                     download=lambda *a, **k: called.append(1))
    assert called == [] and len(rep.planned) == 2 and rep.processed == []


def test_failure_is_recorded_and_run_continues(tmp_path, monkeypatch):
    gs = [g(2, 10), g(20, 5)]
    seen = []

    def download(gr, layers, **kw):
        seen.append(gr.granule_id)

    def process(manifest, aoi, **kw):
        if manifest.granule_id == gs[0].granule_id:
            raise ProcessingError("no observed pixels")
        return SimpleNamespace(granule_id=manifest.granule_id)

    import src.pipeline.orchestrator as orch
    monkeypatch.setattr(orch, "load_manifest", lambda d, gid: SimpleNamespace(granule_id=gid))
    rep = run_series(AOI, [(date(2024, 7, 1), date(2024, 7, 31))], raw_dir=tmp_path, manifest_dir=tmp_path,
                     out_root=tmp_path, search=fake_search_factory(gs), download=download, process=process, slot_days=1)
    assert len(seen) == 2 and len(rep.processed) == 1
    assert rep.failed == [(gs[0].granule_id, "no observed pixels")]


def test_auth_error_aborts_run(tmp_path):
    def download(*a, **k):
        raise AuthError("denied")

    with pytest.raises(AuthError):
        run_series(AOI, [(date(2024, 7, 1), date(2024, 7, 31))], raw_dir=tmp_path, manifest_dir=tmp_path,
                   out_root=tmp_path, search=fake_search_factory([g(2, 10)]), download=download)


def test_provider_error_on_one_granule_is_not_fatal(tmp_path):
    def download(*a, **k):
        raise ProviderError("network")

    rep = run_series(AOI, [(date(2024, 7, 1), date(2024, 7, 31))], raw_dir=tmp_path, manifest_dir=tmp_path,
                     out_root=tmp_path, search=fake_search_factory([g(2, 10)]), download=download)
    assert rep.failed and not rep.processed


def test_cap_applies_with_warning():
    gs = [g(d, c) for d, c in [(1, 50), (9, 10), (17, 30), (25, 20)]]
    rep = run_series(AOI, [(date(2024, 7, 1), date(2024, 7, 31))], raw_dir=None, manifest_dir=None, out_root=None,
                     dry_run=True, search=fake_search_factory(gs), max_granules=2, slot_days=7)
    assert len(rep.planned) == 2 and any("capped" in w for w in rep.warnings)
    assert {x.cloud_cover for x in rep.planned} == {10, 20}      # least cloudy kept
