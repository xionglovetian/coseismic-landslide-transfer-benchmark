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
