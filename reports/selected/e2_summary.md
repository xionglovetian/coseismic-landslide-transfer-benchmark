# E2 Summary: Early-vs-Late Source Checkpoint Adaptation

Date: 2026-09-24

## Protocol

- Model: ResUNet.
- Seed: 42 screening.
- Source initialisation: epoch 20, 30, 50.
- Adaptation: 20-shot, 400 steps, full or decoder-only.
- Targets: Hokkaido, Lombok, Palu.
- Epoch-50 results reuse the existing P3 adaptation runs; epochs 20 and 30 are 12 new runs.

## Results

| Mode | Init epoch | Macro IoU | Delta vs zero-shot | Macro BF1@2 | Macro HD95 |
|---|---:|---:|---:|---:|---:|
| full | 20 | 0.1681 | +0.1216 | 0.2733 | 76.82 |
| full | 30 | 0.1883 | +0.1418 | 0.2730 | 77.05 |
| full | 50 | 0.1821 | +0.1356 | 0.2727 | 75.73 |
| decoder-only | 20 | 0.1777 | +0.1312 | 0.2367 | 84.06 |
| decoder-only | 30 | 0.1818 | +0.1353 | 0.2406 | 82.59 |
| decoder-only | 50 | 0.1843 | +0.1378 | 0.2518 | 80.20 |

## Early-minus-Epoch50

| Mode | Early epoch | Hokkaido | Lombok | Palu | Macro |
|---|---:|---:|---:|---:|---:|
| full | 20 | -0.0173 | -0.0204 | -0.0043 | -0.0140 |
| full | 30 | +0.0179 | -0.0007 | +0.0016 | +0.0062 |
| decoder-only | 20 | -0.0029 | -0.0037 | -0.0129 | -0.0065 |
| decoder-only | 30 | -0.0007 | -0.0052 | -0.0014 | -0.0025 |

## Decision

No early checkpoint passes the pre-specified material criterion:

- At least two of three target regions improve;
- Macro IoU gain >= 0.01 relative to epoch 50.

Epoch-30 full is the only early setting above epoch 50 in Macro IoU, but only by +0.0062, with the gain concentrated in Hokkaido. Lombok is unchanged and Palu gains only +0.0016. Decoder-only early initialisation is worse than epoch 50 on all three regions.

## Interpretation

The E2 result does not support the causal claim that early stopping improves few-shot adaptation in the tested protocol. It weakens the mechanistic statement that source overfitting materially harms target adaptation.

The source-overfitting evidence remains valid for zero-shot transfer and source-target divergence, but it cannot be extended to claim that a later source checkpoint worsens few-shot adaptation.

## Artifacts

- `reports/e2_early_late_report.md`
- `reports/e2_early_late_per_region.csv`
- `reports/e2_early_late_macro.csv`
- `reports/e2_early_late_contrasts.csv`
- `outputs/bench_v2_e2_epoch20_*`
- `outputs/bench_v2_e2_epoch30_*`
