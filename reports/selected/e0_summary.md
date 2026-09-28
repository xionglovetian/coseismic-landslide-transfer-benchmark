# E0 Summary: Existing-Result Audit

Date: 2026-09-24

## 1. Exposure Matching

- CAS-only continuation: 1,120 optimizer steps and 35,840 image exposures.
- Pooled-source continuation: 7,100 optimizer steps and 227,200 image exposures.
- Ratio: 6.34x.
- `compute-matched` is not supported. Current comparison is epoch-matched and cannot causally isolate added source regions.

Artifacts:

- `reports/e0_exposure_ledger.md`
- `reports/e0_exposure_ledger.csv`

## 2. Threshold and Prevalence Robustness

At threshold 0.5, the seven-region prevalence-IoU association is Spearman rho = 0.929 and leave-one-region-out rho = 0.886-0.943. This remains an association, not a causal prevalence-dominance result.

The threshold sensitivity is region-dependent:

- IoU ranges over 0.3-0.7 are 0.0046 for Lombok to 0.1066 for Jiuzhai Valley.
- Palu's best IoU threshold is 0.7, while most regions prefer 0.3-0.5.
- MCC and balanced IoU are retained as prevalence-robust companions to IoU.

Artifacts:

- `reports/e0_threshold_metrics.csv`
- `reports/e0_prevalence_metrics.csv`
- `reports/e0_region_threshold_sensitivity.csv`
- `reports/e0_threshold_uncertainty_report.md`

## 3. Architecture Rank Stability

At threshold 0.5, across regions and seeds:

| Model | Mean rank | Probability rank 1 |
|---|---:|---:|
| SegFormer-B0 | 1.458 | 0.616 |
| Bottleneck-LiteASK | 2.370 | 0.300 |
| ResUNet | 2.698 | 0.081 |
| DeepLabV3+ | 3.474 | 0.003 |

Source-validation versus target-Macro Spearman rho varies from -0.8 to +0.4 across seeds, with leave-one-region-out values from -0.8 to +0.4. The in-domain/cross-event rank claim should be reported with intervals and sensitivity, not as a stable negative correlation.

Artifact:

- `reports/e0_architecture_rank.csv`

## 4. Transfer Delta and Uncertainty

For the three common fixed-source targets at threshold 0.5:

| Region | Mean delta IoU | Positive seeds | Seeds >= +0.01 |
|---|---:|---:|---:|
| Hokkaido | +0.0348 | 3/3 | 3/3 |
| Lombok | -0.0099 | 0/3 | 0/3 |
| Palu | +0.0067 | 3/3 | 0/3 |

Hierarchical seed/region bootstrap of the three-region mean:

- Mean: +0.0105
- 95% interval: [-0.0035, +0.0290]
- P(delta > 0) = 0.920
- P(delta >= +0.01) = 0.465

Threshold stability:

- Lombok is negative at all thresholds.
- Hokkaido is positive through 0.6 and near zero at 0.7.
- Palu is positive but below the +0.01 material threshold at 0.4-0.7.

Artifacts:

- `reports/e0_transfer_deltas.csv`
- `reports/e0_transfer_threshold_summary.csv`
- `reports/e0_threshold_uncertainty_report.md`

## 5. Physical Spatial Buffer

Current few-shot support-query splits guarantee non-overlap but do not guarantee a physical buffer:

- Minimum grid Chebyshev distance: 2.
- Minimum edge gap: 0 m because adjacent stride-256 chips touch.
- GSD: 3 m for Hokkaido, 5 m for Lombok and Palu.

Query retention under a 256 m edge buffer:

- Hokkaido: 85.6%-85.8%.
- Lombok: 60.7%-65.8%.
- Palu: 92.2%-93.4%.

A revised few-shot split should use an explicit GSD-aware spatial buffer. Source train/validation physical separation cannot be established from the current PNG manifest because source geolocation is absent.

Artifacts:

- `reports/e0_physical_buffer_report.md`
- `reports/e0_physical_buffer_summary.csv`
- `reports/e0_physical_buffer_counts.csv`

## 6. Reproducibility Inventory

The machine-readable inventory contains 125 artifacts covering code/manifests, raw zero-shot results, continuation controls, fixed-source transfer, and 90 few-shot run results.

Artifacts:

- `reports/e0_reproducibility_report.md`
- `reports/e0_reproducibility_manifest.csv`

## 7. E0 Decision

1. Do not claim compute matching or added-region causality from the current continuation experiment.
2. Treat the prevalence association and transfer signs as uncertainty-aware exploratory results.
3. Do not aggregate the four-region LORO target set with the three-region fixed-source target set.
4. Keep IoU only as one metric; report threshold curves, MCC, balanced accuracy, and physical buffers.
5. Proceed to E1 exposure-matched source expansion as the first new training experiment.

## 8. E1 Starting Conditions

- Primary model: Bottleneck-LiteASK.
- Seeds: 42, 2026, 777.
- Regimes: CAS-only single-stream, CAS-only multi-stream replay placebo, pooled-source multi-domain.
- Common budget: 1,120 optimizer steps and 35,840 image exposures per run.
- Step-based LR schedule and checkpoints every 280 steps.
- Same target set for every regime: Hokkaido, Lombok, Palu.
- Validation: CAS-only validation.
