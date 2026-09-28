# P4 Core Diagnostic Summary

Date: 2026-09-24

Model scope: ResUNet, seed 42, 128x128. This is a diagnostic replication on the selected best model, not a new full model matrix.

## P4-A Epoch-wise Source Overfitting

| Epoch | CAS Val IoU | Hokkaido IoU | Lombok IoU | Palu IoU |
|---:|---:|---:|---:|---:|
| 10 | 0.4602 | 0.0993 | 0.0526 | 0.0280 |
| 20 | 0.4642 | 0.1186 | 0.0482 | 0.0227 |
| 30 | 0.6324 | 0.0813 | 0.0550 | 0.0295 |
| 40 | 0.6439 | 0.0876 | 0.0572 | 0.0238 |
| 50 | 0.6771 | 0.0712 | 0.0525 | 0.0214 |

CAS validation peaks at epoch 50. Hokkaido peaks at epoch 20 and Palu at epoch 30, then both decline. This is direct epoch-wise evidence of source-domain overfitting and target-domain negative transfer.

## P4-B Source Data Fraction

| Source fraction | Epoch-50 CAS IoU | Epoch-50 target Macro IoU | Target-best epoch |
|---:|---:|---:|---:|
| 25% | 0.5372 | 0.0553 | 10 |
| 50% | 0.5819 | 0.0537 | 10 |
| 75% | 0.6373 | 0.0555 | 10 |
| 100% | 0.6771 | 0.0484 | 20 |

More source data steadily improves source validation, but target Macro IoU does not improve. Source scaling is not by itself a cross-event generalization intervention.

## P4-D Confidence Calibration

| Domain | Error rate | Mean confidence | High-confidence fraction | High-confidence error rate |
|---|---:|---:|---:|---:|
| CAS validation | 0.0590 | 0.9222 | 0.7451 | 0.0069 |
| Hokkaido | 0.2612 | 0.8793 | 0.6071 | 0.1208 |
| Lombok | 0.1389 | 0.9362 | 0.7926 | 0.0709 |
| Palu | 0.1394 | 0.8980 | 0.6712 | 0.0233 |

Target-domain errors are frequently assigned high confidence. Raw model confidence is not a deployable reliability signal without target-domain calibration.

## P4-E Spatial Leakage and Duplicate Audit

- Source train/validation: 1795/513 images, zero exact image or mask overlap, zero strong perceptual near-duplicates.
- Low-threshold aHash candidates exist but have low SSIM and mostly disjoint masks.
- Hokkaido, Lombok, and Palu have zero duplicate images and zero duplicate masks.
- For all three P3 seeds and regions, the minimum support-query Chebyshev distance is 2 in the reconstructed stride-256 grid, with zero distance <=1 pairs.
- Remaining limitation: spatial adjacency in the variable-sized source train/validation images cannot be reconstructed from filenames alone.

## P4-F Domain-Shift Perturbations

Macro IoU across the three target regions:

| Perturbation | Macro IoU | Delta IoU |
|---|---:|---:|
| None | 0.0484 | +0.0000 |
| Brightness x0.75 | 0.0529 | +0.0045 |
| Brightness x1.25 | 0.0429 | -0.0055 |
| Gaussian blur sigma 1 | 0.0450 | -0.0034 |
| Gaussian blur sigma 2 | 0.0445 | -0.0039 |
| JPEG quality 30 | 0.0467 | -0.0017 |
| JPEG quality 60 | 0.0482 | -0.0002 |
| Downscale to 64 and back | 0.0456 | -0.0028 |

High brightness, blur, and resolution loss degrade target transfer; JPEG and moderate intensity shifts are comparatively secondary in this checkpoint.

## P4-C Augmentation Ablation

| Augmentation | Epoch-50 CAS IoU | Epoch-50 target Macro IoU | Target-best epoch | Epoch-50 peak drop | Source-target gap |
|---|---:|---:|---:|---:|---:|
| none | 0.7044 | 0.0661 | 40 | +0.0031 | 0.6383 |
| current | 0.6771 | 0.0484 | 20 | +0.0148 | 0.6288 |
| strong | 0.6129 | 0.0562 | 20 | +0.0237 | 0.5567 |

Strong augmentation increases the target-domain peak but does not remove the post-peak decline. The no-augmentation run has the best epoch-50 target Macro IoU and the smallest post-peak drop, but also the highest source validation IoU. The tested augmentation schedule therefore does not provide a clean solution to source overfitting.

## Overall Conclusion

The P4 evidence supports a source-overfitting diagnosis: source validation continues to improve while target curves peak early and decline, additional source data does not improve target Macro IoU, target errors remain highly confident, and the tested augmentation schedules do not eliminate the source-target gap. The P3 few-shot query split is spatially isolated, so the measured few-shot gains are not explained by direct support-query chip overlap.

## Remaining Limits

- Epoch-wise replication on a second seed or model is not required for the current diagnostic, but would strengthen the temporal trend estimate.
- Source-image spatial adjacency remains unresolved.

## Artifacts

- `reports/p4_source_overfit_report.md`
- `reports/p4_source_fraction_report.md`
- `reports/p4_confidence_calibration_report.md`
- `reports/spatial_leakage_audit.md`
- `reports/p4_domain_shift_report.md`
- `reports/p4_augmentation_report.md`
- `figures/p4_source_overfit_resunet_seed42.png`
- `figures/p4_source_fraction_resunet_seed42.png`
- `figures/p4_augmentation_resunet_seed42.png`
