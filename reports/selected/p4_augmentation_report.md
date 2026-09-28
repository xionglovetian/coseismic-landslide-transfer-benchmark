# P4-C: Source Augmentation Ablation

Model: ResUNet, seed 42, 128x128, 100% source data. `current` is the original geometric augmentation used by the main benchmark. `strong` adds brightness/contrast, Gaussian blur, and scale perturbation.

| Augmentation | Epoch-50 CAS IoU | Epoch-50 target Macro | Source-best epoch | Target-best epoch | Epoch-50 peak drop | Source-target gap |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.7044 | 0.0661 | 50 | 40 | +0.0031 | 0.6383 |
| current | 0.6771 | 0.0484 | 50 | 20 | +0.0148 | 0.6288 |
| strong | 0.6129 | 0.0562 | 50 | 20 | +0.0237 | 0.5567 |

Interpretation: strong augmentation is useful only if it improves target-domain peak or epoch-50 performance and reduces the post-peak drop without causing a large source-validation collapse.

Figure: D:\landslide_unet_project\figures\p4_augmentation_resunet_seed42.png
