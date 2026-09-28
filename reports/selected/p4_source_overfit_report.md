# P4-A: Epoch-wise Source Overfitting Diagnostic

Model: ResUNet, seed 42, 128x128, PolyGHMDiceLoss. Checkpoints evaluated every 10 epochs on CAS validation and the full Hokkaido/Lombok/Palu target regions.

## Curve Summary

| Epoch | CAS Val IoU | Hokkaido IoU | Lombok IoU | Palu IoU |
|---:|---:|---:|---:|---:|
| 10 | 0.4602 | 0.0993 | 0.0526 | 0.0280 |
| 20 | 0.4642 | 0.1186 | 0.0482 | 0.0227 |
| 30 | 0.6324 | 0.0813 | 0.0550 | 0.0295 |
| 40 | 0.6439 | 0.0876 | 0.0572 | 0.0238 |
| 50 | 0.6771 | 0.0712 | 0.0525 | 0.0214 |

## Trend Diagnostic

| Region | Target peak epoch | Peak IoU | Final IoU | First-to-final delta | Peak-to-final drop | Corr(source val, target) |
|---|---:|---:|---:|---:|---:|---:|
| Hokkaido | 20 | 0.1186 | 0.0712 | -0.0281 | +0.0474 | -0.894 |
| Lombok | 40 | 0.0572 | 0.0525 | -0.0001 | +0.0047 | +0.664 |
| Palu | 30 | 0.0295 | 0.0214 | -0.0065 | +0.0081 | -0.190 |

CAS validation peak: epoch 50 with IoU 0.6771.

A source-overfitting pattern is supported only when source validation continues to improve while one or more target-domain curves peak earlier and then decline. With only five checkpoints, correlations and peak timing are descriptive, not inferential tests.

Figure: D:\landslide_unet_project\figures\p4_source_overfit_resunet_seed42.png
