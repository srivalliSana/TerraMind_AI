"""CLI smoke tests."""
from src.cli import main


def test_check_env_runs(capsys, monkeypatch):
    for k in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    code = main(["check-env"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Required packages" in out
    assert "unset" in out  # credentials absent, reported without values
