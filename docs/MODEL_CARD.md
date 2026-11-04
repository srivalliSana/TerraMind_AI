# Model card

## Summary
**There is no trained machine-learning model in TerraMind AI.** No labelled stress data exist in this repository, so training or
reporting accuracy, precision, recall or F1 would be fabrication. The dashboard's Model Performance page states this.

## What produces the risk score
A deterministic, rule-based statistical method (robust seasonal baseline, z-scores, persistence, affected fraction), documented in
[SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md). Parameters live in `configs/thresholds.yaml` with `validated: false`.

## Intended use
- Screening: prioritising which fields or regions to inspect with other evidence.
- Education and prototyping of a satellite monitoring workflow.

## Out of scope
- Diagnosing the cause of low greenness (drought, pests, disease, nutrient deficiency, planting date, crop type).
- Yield forecasting, insurance, regulatory or financial decisions.
- Regions, crops or sensors beyond what was exercised (one Iowa test area, Sentinel-2-derived HLS).

## Evaluation
None against ground truth. Automated tests check code behaviour on synthetic data only.

## Plan if labels become available
1. Define the target (e.g. field-season yield loss class) and obtain labels with dates and locations.
2. Build features from the existing indices and baselines; split by region and by year, never randomly by pixel.
3. Compare against the rule-based score as the baseline model; report uncertainty and failure cases.
4. Register the model in the existing `model_versions` / `evaluation_runs` tables so the dashboard shows real metrics.
