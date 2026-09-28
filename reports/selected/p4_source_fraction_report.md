# P4-B: Source-Domain Data Fraction Ablation

Model: ResUNet, seed 42, 128x128. Training fractions use the same source train root and validation set; subsets are deterministic random samples with subset seed 42.

| Source fraction | Epoch-50 CAS IoU | Epoch-50 Hokkaido | Epoch-50 Lombok | Epoch-50 Palu | Epoch-50 target macro | Source-best epoch | Target-best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.5372 | 0.0831 | 0.0567 | 0.0259 | 0.0553 | 50 | 10 |
| 50% | 0.5819 | 0.0767 | 0.0561 | 0.0285 | 0.0537 | 50 | 10 |
| 75% | 0.6373 | 0.0899 | 0.0555 | 0.0212 | 0.0555 | 40 | 10 |
| 100% | 0.6771 | 0.0712 | 0.0525 | 0.0214 | 0.0484 | 50 | 20 |

Interpretation rule: if source validation improves with fraction but target macro does not, additional source data changes source fit more than cross-event transfer. If both improve, source scaling is also a transfer intervention.

Figure: D:\landslide_unet_project\figures\p4_source_fraction_resunet_seed42.png
