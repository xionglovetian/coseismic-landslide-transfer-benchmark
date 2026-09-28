# E4-C: Support-Set Draw Robustness

The source initialization is fixed to ResUNet seed42. Three deterministic 20-shot support draws are compared: model/support seed42, support seed2026, and support seed777.

| Mode | Region | Delta IoU mean +/- SD | Range | Positive draws | Material draws | Delta BF1@2 | Delta HD95 |
|---|---|---:|---:|---:|---:|---:|---:|
| full | Hokkaido | +0.2955 +/- 0.0262 | [+0.2653, +0.3123] | 3/3 | 3/3 | +0.3776 | -11.24 |
| full | Lombok | +0.0155 +/- 0.0115 | [+0.0048, +0.0277] | 3/3 | 2/3 | +0.0814 | -2.06 |
| full | Palu | +0.0603 +/- 0.0087 | [+0.0549, +0.0703] | 3/3 | 3/3 | +0.1305 | +30.51 |
| decoder-only | Hokkaido | +0.3396 +/- 0.0147 | [+0.3239, +0.3529] | 3/3 | 3/3 | +0.3770 | -12.25 |
| decoder-only | Lombok | +0.0078 +/- 0.0017 | [+0.0060, +0.0094] | 3/3 | 0/3 | +0.0819 | -2.28 |
| decoder-only | Palu | +0.0559 +/- 0.0016 | [+0.0544, +0.0576] | 3/3 | 3/3 | +0.0799 | +38.99 |

## Decision

Support-draw robustness criterion: every mode-region cell has a positive gain in at least 2/3 draws. Result: supported.

This experiment isolates support-set sampling from model initialization because all adapted runs start from the same ResUNet seed42 source checkpoint.
