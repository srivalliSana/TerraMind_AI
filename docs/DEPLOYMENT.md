# Deployment

Current state: **local use only**. There is no Dockerfile, REST API or scheduler yet; this page documents what is actually supported.

## Run locally (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # add EARTHDATA_TOKEN to .env (gitignored)
python -m src.cli check-env
streamlit run app/dashboard.py
```

## Configuration
| What | Where |
|---|---|
| Secrets (`EARTHDATA_TOKEN`) | environment variable or `.env` (never committed) |
| Paths | `TERRAMIND_DATA_DIR`, `TERRAMIND_DB_PATH` environment variables |
| Settings | `configs/default.yaml`; local overrides in `configs/local.yaml` (written by the Settings page, validated, gitignored) |
| Thresholds | `configs/thresholds.yaml` |

## Data directories (all gitignored)
`data/raw` downloaded COGs, `data/processed/<AOI>/<granule>` outputs, the SQLite file, `logs/`.

## Typical workflow
1. `python -m src.cli search ...` to find granules (no login).
2. `python -m src.cli download ...` (token required).
3. `python -m src.cli process ...` or `run-series ...` to process them.
4. `python -m src.cli assess ...`, `extent ...`, `alerts evaluate ...`.
5. Open the dashboard; it reads the processed outputs and database.

Run `python -m src.cli <command> --help` for exact options.

## Before any shared or production deployment
- Run the secret scan on the git diff; confirm no `.env`, databases or rasters are tracked.
- Add authentication in front of the dashboard (Streamlit has none built in for public exposure).
- Pin dependencies and add a container build; schedule ingestion with an external scheduler.
- Complete the validation steps in [SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md) section 7.
