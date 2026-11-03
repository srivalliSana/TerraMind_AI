"""CLI smoke tests."""
from src.cli import main


def test_search_command_offline(capsys, monkeypatch, tmp_path):
    import json

    from src.data import search as search_mod
    from src.data.providers.nasa_hls import NasaHLSProvider
    from tests.unit.test_hls_provider import make_umm

    monkeypatch.setattr(search_mod, "get_provider",
                        lambda name=None: NasaHLSProvider(lambda **kw: [make_umm()] if kw["short_name"] == "HLSL30" else []))
    out = tmp_path / "r.json"
    code = main(["search", "--bbox", "-93.7", "41.9", "-93.6", "42.0", "--start", "2024-07-01",
                 "--end", "2024-07-20", "--json-out", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "1 granule(s) found" in text and "nothing downloaded" in text
    assert json.loads(out.read_text())[0]["granule_id"].startswith("HLS.L30")


def test_search_command_rejects_oversized_aoi(capsys):
    code = main(["search", "--bbox", "0", "0", "5", "5", "--start", "2024-07-01", "--end", "2024-07-20"])
    assert code == 2 and "exceeds" in capsys.readouterr().err


def test_check_env_runs(capsys, monkeypatch):
    for k in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    code = main(["check-env"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Required packages" in out
    assert "unset" in out  # credentials absent, reported without values
