# E1 Summary: Exposure-Matched Source Expansion

Date: 2026-09-24

## Protocol

- Model: Bottleneck-LiteASK.
- Seeds: 42, 2026, 777.
- Regimes: CAS-only single-stream, CAS-only multi-stream replay placebo, pooled-source multi-domain.
- Each run: 1,120 optimizer steps, batch size 32, 35,840 image exposures, five domain slots for placebo and pooled regimes, step-based cosine schedule.
- Checkpoints: 0, 280, 560, 840, 1,120 steps.
- Target set: Hokkaido, Lombok, Palu. Validation: CAS-only validation.

## Completion

- 9/9 training runs completed without failure.
- 45/45 checkpoint-evaluation JSONs completed.
- Pooled and placebo regimes have matched optimizer steps, samples seen, and domain slots.

## Final Results

| Regime | CAS val IoU | Target Macro IoU | Hokkaido IoU | Lombok IoU | Palu IoU |
|---|---:|---:|---:|---:|---:|
| CAS-only single-stream | 0.7366 | 0.0562 | 0.0826 | 0.0556 | 0.0305 |
| CAS multi-stream replay placebo | 0.7356 | 0.0566 | 0.0828 | 0.0570 | 0.0299 |
| Pooled-source | 0.6418 | 0.0921 | 0.1832 | 0.0565 | 0.0365 |

## Causal Contrast

Primary contrast: `pooled - CAS multi-stream replay placebo`.

| Region | Mean Delta IoU | 95% interval | P(delta>0) | P(delta>=0.01) |
|---|---:|---:|---:|---:|
| Hokkaido | +0.1004 | [+0.0635, +0.1472] | 1.000 | 1.000 |
| Lombok | -0.0005 | [-0.0026, +0.0036] | 0.256 | 0.000 |
| Palu | +0.0066 | [+0.0051, +0.0093] | 1.000 | 0.000 |
| Macro | +0.0355 | [+0.0242, +0.0513] | 1.000 | 1.000 |

Per-seed Macro deltas: seed42 +0.0242, seed2026 +0.0513, seed777 +0.0310.

## Decision

The exposure-matched experiment supports a causal added-region benefit in aggregate, but not a uniform cross-event gain:

- Hokkaido: large material positive transfer.
- Palu: small positive effect below the pre-specified 0.01 material threshold.
- Lombok: no detectable positive effect.
- The positive Macro effect is therefore driven mostly by Hokkaido.

The supported claim is:

`Adding source regions can improve aggregate target Macro IoU under matched exposure, but the effect is strongly region-dependent rather than uniformly positive.`

The unsupported claim is:

`Adding source regions universally improves cross-event segmentation.`

## Artifacts

- `reports/e1_exposure_matched_report.md`
- `reports/e1_exposure_matched_per_run.csv`
- `reports/e1_exposure_matched_summary.csv`
- `reports/e1_exposure_matched_contrasts.csv`
- `reports/e1_exposure_matched_bootstrap.csv`
- `reports/e1_raw/`
- `outputs/bench_v2_e1_*/`

## Next Step

Proceed to E2: early-vs-late source checkpoint adaptation to test whether source overfitting causally reduces few-shot adaptation performance.

- Model: ResUNet.
- Initial checkpoints: epochs 20, 30, 50.
- Adaptation: 20-shot, full and decoder-only.
- Targets: Hokkaido, Lombok, Palu.
- Seed 42 screening; epoch-50 results already exist.
