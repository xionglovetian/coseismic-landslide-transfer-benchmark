# E4 Summary: Few-Shot Robustness and Operational Checks

Date: 2026-09-25

## Scope

- Model: ResUNet.
- Main adaptation: 20-shot, 400 steps.
- Modes: full and decoder-only.
- Targets: Hokkaido, Lombok, Palu.
- E4 includes physical-buffer rerun, threshold/calibration robustness, support-draw replication, and stitched-map evaluation.

## E4-A: Physical Buffer

The same source and adapted checkpoints were evaluated under:

- existing guard query;
- GSD-aware 256 m physical buffer;
- GSD-aware 512 m physical buffer.

Adaptation gain persisted:

| Mode | Buffer | Mean Delta IoU | 95% interval | Positive regions |
|---|---:|---:|---:|---:|
| full | 0 m | +0.1316 | [+0.0559, +0.2187] | 3/3 |
| full | 256 m | +0.1303 | [+0.0493, +0.2195] | 3/3 |
| full | 512 m | +0.1303 | [+0.0513, +0.2196] | 3/3 |
| decoder-only | 0 m | +0.1313 | [+0.0374, +0.2326] | 3/3 |
| decoder-only | 256 m | +0.1268 | [+0.0385, +0.2281] | 2/3 |
| decoder-only | 512 m | +0.1268 | [+0.0366, +0.2286] | 2/3 |

The few-shot gain is not explained by direct support-query overlap.

## E4-B: Threshold and Calibration

- Adaptation gain remains positive across thresholds 0.3-0.7.
- The effect is strongest at 0.3-0.5 and weaker but still positive at 0.6-0.7.
- Full fine-tuning gains are material in all three regions at 0.5 under all buffers.
- High-confidence error decreases substantially after adaptation:
  - full: 0.0716 -> 0.0131 at 0 m; 0.0709 -> 0.0127 at 256/512 m.
  - decoder-only: 0.0716 -> 0.0117 at 0 m; 0.0709 -> 0.0117 at 256/512 m.

## E4-C: Support-Set Draws

Source initialization was fixed to ResUNet seed42. Three deterministic support draws were compared.

All mode-region cells improved in at least 2/3 draws:

- Full: Hokkaido +0.2955 +/- 0.0262; Lombok +0.0155 +/- 0.0115; Palu +0.0603 +/- 0.0087.
- Decoder-only: Hokkaido +0.3396 +/- 0.0147; Lombok +0.0078 +/- 0.0017; Palu +0.0559 +/- 0.0016.

Lombok decoder-only remains below the 0.01 material threshold, but no support draw is negative.

## E4-D: Stitched Map Check

Seed-42 stitched map evaluation is consistent with tile-level results:

| Mode | Region | Source IoU | Stitched adapted IoU | Delta IoU |
|---|---|---:|---:|---:|
| full | Hokkaido | 0.0409 | 0.3794 | +0.3384 |
| full | Lombok | 0.0379 | 0.0728 | +0.0349 |
| full | Palu | 0.0133 | 0.0685 | +0.0553 |
| decoder-only | Hokkaido | 0.0409 | 0.4356 | +0.3947 |
| decoder-only | Lombok | 0.0379 | 0.0492 | +0.0113 |
| decoder-only | Palu | 0.0133 | 0.0484 | +0.0352 |

The map-level direction agrees with tile-level evaluation. Lombok and decoder-only Palu show a precision-recall tradeoff, so the gain is not uniform in error type.

## Overall Decision

E4 supports the following bounded claim:

`Twenty supporting tiles improved the tested ResUNet benchmark after adaptation, and the improvement persisted under physical buffers, independent support draws, multiple thresholds, calibration analysis, and stitched-map evaluation.`

The claim remains bounded by:

- one adapted architecture for E4 robustness;
- three target regions;
- 20-shot and 400-step budget;
- low absolute IoU in Lombok;
- boundary-metric tradeoffs in Palu;
- no established physical spatial separation for source train/validation.

The unsupported operational language is:

`20 tiles are sufficient for deployed emergency mapping.`

## Artifacts

- `reports/e4_query_robustness_report.md`
- `reports/e4_support_draws_report.md`
- `reports/e4_stitched_report.md`
- `reports/e4_query_deltas.csv`
- `reports/e4_support_draws_summary.csv`
- `reports/e4_stitched_summary.csv`
