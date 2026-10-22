# TerraMind_AI

**TerraMind AI**: satellite-based crop stress detection and early warning platform.
Outputs are *preliminary stress indicators* derived from satellite data, not diagnoses.

> Status: under active development (foundation phase). Full documentation is added as features land.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then edit .env locally; never commit it
python -m src.cli check-env
python -m pytest
```

Earthdata credentials are optional for now: catalog search and demo mode do not need them.
