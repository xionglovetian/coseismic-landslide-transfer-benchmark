# Coseismic Landslide Transfer Benchmark

Reproducibility package for a retrospective, region-held-out benchmark of
coseismic landslide segmentation under cross-event domain shift.

This repository contains experiment code, frozen manifests and splits,
machine-readable metrics, and selected audit/protocol reports. It does not
contain the manuscript, source or target imagery, processed chips, model
checkpoints, or files restricted by the original data licenses.


## Which number is authoritative

The headline exposure-matched macro IoU effect reported in the manuscript is the
q2_x3 Table 6 / Table 8 value, with the seed-stratified interval taking
precedence for architecture-level inference (+0.0773 for ResUNet,
seed-stratified interval [+0.0514, +0.1031]). The value +0.0355 in
`metrics/route1_bootstrap/route1_e1_tile_bootstrap.csv` is a historical
tile-level bootstrap variant and is not the manuscript headline. Files with
`fig4` in their names are plotting inputs and may lag the manuscript tables. The
current authoritative sources are `reports/selected/e1_exposure_matched_contrasts.csv`
and `metrics/q2_x3_seed_stratified_bootstrap/q2_x3_seed_stratified_t_interval.csv`.

## Current benchmark scope

- Target benchmark: seven external regions and 12,324 positive 512-pixel chips.
- Six region-labelled held-out target units used for aggregate evaluation; physical
  independence is not asserted. Moxitaidi is reported separately as a
  source-adjacent diagnostic.
- Primary models: ResUNet and Bottleneck-LiteASK.
- Primary seeds: 42, 2026, and 777.
- Primary input size: 128 pixels unless an experiment explicitly states 256.
- Source expansion uses matched optimisation exposure: 1,120 steps, batch 32,
  and 35,840 exposures per run.
- Few-shot adaptation is evaluated as a bounded benchmark experiment, not as
  operational readiness.

## Main packaged results

- X2 augmentation selection selected `none` using source-validation IoU only.
  The paired source-validation delta for `none-current` was `+0.04315` on six
  paired model-seed pairs.
- X3 matched-exposure source expansion produced target-macro IoU deltas of
  `+0.07727` for ResUNet and `+0.04559` for Bottleneck-LiteASK using the
  seed-stratified t intervals with regions fixed. MCC increased in both
  architectures but the Bottleneck MCC result is margin-sensitive; balanced IoU
  is mixed across interval families.
- X1 resolution retraining did not show a consistent target gain at 256 pixels.
  The target-IoU difference for retrained 256 versus 128 was `-0.02431` for
  ResUNet and `+0.00287` for SegFormer-B0.
- Region dependence remained material: Hokkaido showed a large positive effect,
  while Lombok and Palu effects were much smaller and endpoint-dependent.

These numerical statements are tied to the frozen protocols and packaged raw
metrics under `metrics/`; see `REPRODUCIBILITY.md`.

## Repository layout

```text
src/                    core datasets, metrics, models
scripts/                training, evaluation, bootstrap and audit scripts
data_manifests/         frozen manifests and split assignments
metrics/                raw and summarized machine-readable results
reports/selected/       protocols, audits and final result summaries
scripts/checks/         environment, artifact and manifest validators
environment-full.txt    original exported software environment
requirements.txt        reduced pinned dependency set for the experiment code
```

## Quick validation

Python 3.10 is recommended.

```powershell
git clone https://github.com/xionglovetian/coseismic-landslide-transfer-benchmark.git
cd coseismic-landslide-transfer-benchmark
python scripts/checks/validate_release.py
python scripts/checks/verify_manifest.py
```

The release validator uses only the Python standard library. Full model
execution additionally requires the CUDA-enabled packages in
`requirements.txt`.

## Data and checkpoints

The raw image data are not redistributed. Obtain each source from its original
provider and comply with the terms recorded in `DATA_LICENSES.md`. The frozen
manifests contain image and mask path fields from the reference workstation;
`scripts/checks/rebase_manifest_paths.py` can rewrite them for a new data root.

Model checkpoints are not stored in this Git repository because they are large
binary artifacts and their release policy depends on the source-data terms.
The code can retrain all packaged configurations. A future checkpoint release
should be archived separately with an explicit license and persistent
identifier.

## Reproduction status

The repository supports two levels of reproduction:

1. **Metric-level reproduction:** rerun the packaged endpoint and bootstrap
   summaries from `metrics/` without imagery or checkpoints.
2. **Model-level reproduction:** obtain the licensed data, materialize the
   expected layout, retrain or place the corresponding checkpoints, then run
   the evaluation and bootstrap scripts.

The second level is hardware- and data-dependent. One complete Gate-D path was
re-executed from the published code: X3 ResUNet pooled seed 42. It reproduced
the `metrics.json` and the full 2,737-row tile-metric CSV bitwise, and produced
278 identical checkpoint tensors (24,455,423 elements, zero differences).
Details are in `reports/verification/GATE_D_X3_RESUNET_SEED42.md`. The other
multi-day queues are executable from the pinned data and configurations but
must still complete on the target machine before the whole study is called
independently reproduced.

## License

Code and documentation in this repository are released under the MIT License.
Image data and derived annotations remain governed by their original licenses.
See `DATA_LICENSES.md`.

## Repository URL

https://github.com/xionglovetian/coseismic-landslide-transfer-benchmark
