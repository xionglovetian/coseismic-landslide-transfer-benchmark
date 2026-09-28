# E4-D: Stitched Full-Map Check

Seed 42, ResUNet. Overlapping 512-pixel query chips were stitched with averaged probabilities according to the reconstructed stride-256 grid. Metrics are computed on the union of query-chip pixels, excluding support and guard chips.

| Mode | Region | Source IoU | Stitched adapted IoU | Delta IoU | Delta precision | Delta recall |
|---|---|---:|---:|---:|---:|---:|
| full | Hokkaido | 0.0409 | 0.3794 | +0.3384 | +0.5087 | +0.4270 |
| full | Lombok | 0.0379 | 0.0728 | +0.0349 | +0.1402 | -0.0589 |
| full | Palu | 0.0133 | 0.0685 | +0.0553 | +0.4003 | +0.0099 |
| decoder-only | Hokkaido | 0.0409 | 0.4356 | +0.3947 | +0.5176 | +0.5150 |
| decoder-only | Lombok | 0.0379 | 0.0492 | +0.0113 | +0.2027 | -0.1077 |
| decoder-only | Palu | 0.0133 | 0.0484 | +0.0352 | +0.2896 | -0.0115 |

## Decision

The stitched-map check supports the same direction as the tile-level evaluation if the adapted map improves IoU without a precision collapse. It remains a query-covered stitched evaluation, not a claim about unsurveyed terrain.
