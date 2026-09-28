# E1: Exposure-Matched Source Expansion

All regimes use 1,120 optimizer steps, batch size 32, 35,840 image exposures per run, and a step-based cosine schedule. Pooled and CAS multi-stream replay both use five domain slots.

## Final-Step Summary

| Regime | Seed count | CAS val IoU | Target Macro IoU | Hokkaido IoU | Lombok IoU | Palu IoU |
|---|---:|---:|---:|---:|---:|---:|
| cas-single | 3 | 0.7366 | 0.0562 | 0.0826 | 0.0556 | 0.0305 |
| cas-multistream | 3 | 0.7356 | 0.0566 | 0.0828 | 0.0570 | 0.0299 |
| pooled | 3 | 0.6418 | 0.0921 | 0.1832 | 0.0565 | 0.0365 |

## Final-Step Contrasts

| Contrast | Mean Delta Macro IoU | 95% Interval | P(delta>0) | P(delta>=0.01) |
|---|---:|---:|---:|---:|
| pooled-cas-multistream | +0.0355 | [+0.0242, +0.0513] | 1.000 | 1.000 |
| pooled-cas-single | +0.0359 | [+0.0264, +0.0474] | 1.000 | 1.000 |

## Final-Step Per-Region Contrasts

| Contrast | Region | Mean Delta IoU | 95% Interval | P(delta>0) | P(delta>=0.01) |
|---|---|---:|---:|---:|---:|
| pooled-cas-multistream | hokkaido_iburi_tobu | +0.1004 | [+0.0635, +0.1472] | 1.000 | 1.000 |
| pooled-cas-multistream | lombok | -0.0005 | [-0.0026, +0.0036] | 0.256 | 0.000 |
| pooled-cas-multistream | palu | +0.0066 | [+0.0051, +0.0093] | 1.000 | 0.000 |
| pooled-cas-multistream | target_macro | +0.0355 | [+0.0242, +0.0513] | 1.000 | 1.000 |
| pooled-cas-single | hokkaido_iburi_tobu | +0.1006 | [+0.0676, +0.1436] | 1.000 | 1.000 |
| pooled-cas-single | lombok | +0.0009 | [-0.0054, +0.0051] | 0.737 | 0.000 |
| pooled-cas-single | palu | +0.0060 | [+0.0040, +0.0078] | 1.000 | 0.000 |
| pooled-cas-single | target_macro | +0.0359 | [+0.0264, +0.0474] | 1.000 | 1.000 |

## Per-Seed Added-Region Deltas

| Seed | Hokkaido | Lombok | Palu | Macro |
|---:|---:|---:|---:|---:|
| 42 | +0.0635 | +0.0036 | +0.0056 | +0.0242 |
| 2026 | +0.1472 | -0.0026 | +0.0093 | +0.0513 |
| 777 | +0.0904 | -0.0026 | +0.0051 | +0.0310 |

## Decision

The exposure-matched pooled-source regime has a positive Macro IoU effect relative to the CAS multi-stream replay placebo. However, the effect is strongly region-dependent: Hokkaido has a large material gain, Palu has a small positive gain below the pre-specified 0.01 material threshold, and Lombok is effectively unchanged. The causal conclusion is therefore `added regions can improve aggregate transfer in this benchmark, but the effect is not uniformly positive across target regions`.

## Interpretation Rule

The causal added-region contrast is `pooled - cas-multistream`, because both have five domain slots and identical exposure. `pooled - cas-single` measures the combined effect of added regions and multi-stream exposure.

Do not claim a stable added-region effect unless the pooled-minus-placebo contrast has a positive interval, consistent event signs, and an effect of practical size.
