# TerraMind AI

Satellite-based crop stress detection and early warning, built on NASA HLS (Harmonized Landsat Sentinel-2) 30 m imagery.

> **Read this first.** Outputs are *preliminary potential stress indicators*: a pixel or field is "below its own seasonal normal".
> They are **not** diagnoses of drought, disease or yield loss. Thresholds are engineering defaults and have **not** been validated
> against ground truth. There is **no trained machine-learning model** in this project (see [docs/MODEL_CARD.md](docs/MODEL_CARD.md)).

## What works today

| Capability | State |
|---|---|
| Search NASA CMR for HLS granules (no login needed) | Implemented, tested offline; used live on a real AOI |
| Download granules (needs an Earthdata token) | Implemented; verified live once |
| Cloud/shadow/snow/water masking from Fmask, scaling, NDVI/NDMI/SAVI/EVI | Implemented; scale factors, fill values and QA bits taken from NASA docs |
| Seasonal baseline from earlier years, robust z-scores, persistence | Implemented, unit-tested on synthetic series |
| Per-pixel anomaly map and affected-area patches | Implemented; crop-rotation confounding is disclosed |
| Interpretable 0-100 risk score with evidence and confidence | Implemented; weights are **unvalidated** |
| Alert lifecycle in SQLite (dedupe, resolve, dismiss, export) | Implemented, tested |
| Streamlit dashboard (7 pages) with Real and Demo modes | Implemented; tested headlessly, not visually reviewed in a browser |
| Trained ML model | **Not built** (no labelled data) |
| Sensors other than HLS, RGB true-colour view, REST API | **Not built** |

Demo mode shows **synthetic** data under a permanent banner. It never mixes with real results.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # edit .env locally; it is gitignored
python -m src.cli check-env
python -m pytest                # no credentials or network needed
streamlit run app/dashboard.py  # starts in Demo mode if no real data exist
```

### Credentials
Downloads need a NASA Earthdata **token**. Put it in `.env` as `EARTHDATA_TOKEN=...` (never in `.env.example`, never in git).
`python -m src.cli auth-status` reports only whether a credential is present; add `--verify` to test it against NASA.
Catalog search, processing of already-downloaded files, tests and demo mode need no credentials.

## Command line

```powershell
python -m src.cli --help
```

Main commands: `search`, `download`, `process`, `run-series`, `analyze`, `assess`, `extent`, `alerts evaluate|list|dismiss|resolve|export`, `auth-status`, `check-env`.
Exit code 2 = error, 3 = authentication problem.

## Repository map

```
src/data/          provider interface, NASA HLS search, download, manifest
src/geospatial/    AOI model, raster IO, grid alignment checks
src/preprocessing/ scaling, Fmask decoding, quality control
src/indices/       vegetation and moisture indices
src/features/      temporal aggregation, per-pixel baselines
src/detection/     seasonal baseline, z-scores, anomaly classes
src/pipeline/      process_granule, orchestrator, analysis, extent, assessment
src/alerts/        risk scoring, alert lifecycle
src/storage/       SQLite schema, migrations, repositories
src/utils/         dashboard data layer, charts, demo data
app/               Streamlit dashboard (home + 7 pages)
configs/           default.yaml, sensors.yaml, thresholds.yaml
tests/             unit and integration tests
docs/              architecture, methodology, data sources, model card, deployment
```

## Documentation
- [Architecture](docs/ARCHITECTURE.md)
- [Scientific methodology and limitations](docs/SCIENTIFIC_METHODOLOGY.md)
- [Data sources](docs/DATA_SOURCES.md)
- [Model card](docs/MODEL_CARD.md)
- [Deployment](docs/DEPLOYMENT.md)

## Honest limitations
- Validated on one small Iowa test area (about 92 km2) with Sentinel-2-derived HLS only and two earlier baseline years.
  Confidence is capped at "medium" when fewer than three baseline years exist.
- A field planted late, or with a different crop than in earlier years, looks "below normal" without any stress.
- Cloud masking relies on the HLS Fmask; thin cloud and haze can survive it.
- Nothing here has been checked against agronomic ground truth.
