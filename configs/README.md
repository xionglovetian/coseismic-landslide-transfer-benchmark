# Run Configurations

`records/` contains the archived `config.json` and, when present, `history.csv`
for each non-smoke run under the reference experiment root.

`RUN_INDEX.csv` records the run name, config path, history path, availability of
`metrics.json`, and checkpoint counts/sizes. Absolute paths inside a historical
config are records of the original run; portability comes from
`LANDSLIDE_PROJECT_ROOT` and the documented command sequence in
`../REPRODUCIBILITY.md`, not from editing these frozen records.