# Metrics

This directory contains frozen machine-readable outputs used by the selected
reports.

- `e1_raw/` and `e4_query_raw/`: Route 1 aggregate evaluation JSON.
- `route1_recomputed/`: endpoint and paired-bootstrap summaries.
- `route1_bootstrap/`: tile-level and calibration summaries.
- `figure_source_data/`: source tables for manuscript figures.
- `q2_x1_*`: X1 resolution raw metrics and summaries.
- `q2_x2_augmentation*`: X2 paired augmentation experiment.
- `q2_x3_raw/`: aggregate X3 source-expansion metrics.
- `q2_x3_tiles/`: per-tile X3 outputs used by spatial bootstrap.
- `q2_x3_bootstrap/`: component and merged-block X3 intervals.
- `q2_x3_loro_tiles/` and `q2_x3_loro_bootstrap/`: six-region LORO
  extension, retained as exploratory evidence.

No image pixels or checkpoint weights are stored in this directory.

## Provenance and superseded files

- `q2_x3_seed_stratified_bootstrap/` contains the current seed-stratified t
  intervals and the fixed-region bootstrap sensitivity.
- `figure_source_data/fig4_exposure_matched_effects.csv` is the current
  manuscript-facing endpoint table.
- `figure_source_data/fig4_exposure_matched_effects_route1_superseded.csv` is a
  historical Route 1 plotting input and must not be used for manuscript claims.

The compatibility file `figure_source_data/fig4_exposure_matched_causal_effects.csv` is retained only for older review links. It has SHA-256 `1cd6c768155a7ed3e657983569b737e343b224e04c7ba77aeb94566e71674838`.
