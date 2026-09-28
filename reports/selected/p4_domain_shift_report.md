# P4-F: Lightweight Domain-Shift Perturbations

Checkpoint: ResUNet seed 42, epoch 50, 128x128. Each perturbation is applied after resizing to 128; the target mask is unchanged.

| Region | Perturbation | IoU | Delta IoU | Dice |
|---|---|---:|---:|---:|
| Hokkaido | none | 0.0712 | +0.0000 | 0.1330 |
| Hokkaido | brightness_low | 0.0729 | +0.0017 | 0.1359 |
| Hokkaido | brightness_high | 0.0677 | -0.0035 | 0.1268 |
| Hokkaido | blur1 | 0.0567 | -0.0145 | 0.1073 |
| Hokkaido | blur2 | 0.0605 | -0.0108 | 0.1140 |
| Hokkaido | jpeg30 | 0.0671 | -0.0041 | 0.1258 |
| Hokkaido | jpeg60 | 0.0719 | +0.0006 | 0.1341 |
| Hokkaido | downscale64 | 0.0572 | -0.0140 | 0.1082 |
| Lombok | none | 0.0525 | +0.0000 | 0.0998 |
| Lombok | brightness_low | 0.0599 | +0.0074 | 0.1131 |
| Lombok | brightness_high | 0.0398 | -0.0127 | 0.0765 |
| Lombok | blur1 | 0.0525 | +0.0000 | 0.0998 |
| Lombok | blur2 | 0.0491 | -0.0034 | 0.0936 |
| Lombok | jpeg30 | 0.0488 | -0.0037 | 0.0931 |
| Lombok | jpeg60 | 0.0509 | -0.0016 | 0.0969 |
| Lombok | downscale64 | 0.0536 | +0.0011 | 0.1017 |
| Palu | none | 0.0214 | +0.0000 | 0.0420 |
| Palu | brightness_low | 0.0258 | +0.0044 | 0.0503 |
| Palu | brightness_high | 0.0212 | -0.0003 | 0.0415 |
| Palu | blur1 | 0.0258 | +0.0044 | 0.0503 |
| Palu | blur2 | 0.0238 | +0.0024 | 0.0466 |
| Palu | jpeg30 | 0.0241 | +0.0027 | 0.0471 |
| Palu | jpeg60 | 0.0218 | +0.0004 | 0.0427 |
| Palu | downscale64 | 0.0260 | +0.0045 | 0.0506 |

Interpretation: large degradation indicates sensitivity to acquisition or preprocessing shifts that are not represented by the source training distribution. Small changes are not evidence of robustness beyond these tested transforms.
