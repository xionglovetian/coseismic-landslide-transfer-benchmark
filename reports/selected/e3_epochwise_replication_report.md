# E3: Model and Seed Replication of Epoch-wise Source Overfitting

The prespecified pattern is: CAS validation peaks at epoch 50, target Macro IoU peaks at epoch 20 or 30, and the target then declines by at least 0.005.

## Per-Run Diagnostics

| Model | Seed | Source peak epoch | Target peak epoch | Target peak IoU | Epoch-50 target IoU | Peak-to-final drop | Pattern supported |
|---|---:|---:|---:|---:|---:|---:|---|
| ResUNet | 42 | 50 | 20 | 0.0632 | 0.0484 | +0.0148 | True |
| ResUNet | 2026 | 50 | 20 | 0.0550 | 0.0455 | +0.0095 | True |
| ResUNet | 777 | 50 | 50 | 0.0559 | 0.0559 | +0.0000 | False |
| SegFormer-B0 | 42 | 50 | 20 | 0.0597 | 0.0578 | +0.0019 | False |
| SegFormer-B0 | 2026 | 50 | 50 | 0.0551 | 0.0551 | +0.0000 | False |
| SegFormer-B0 | 777 | 50 | 30 | 0.0582 | 0.0576 | +0.0006 | False |

## Decision

The pattern is supported in 2/6 model-seed runs. The prespecified reviewer-response threshold was at least 4/6 runs.

The prespecified threshold is not met. The temporal source-overfitting pattern is not replicated across models and seeds. Downgrade this mechanism claim to exploratory and report model/seed heterogeneity.

Figure: D:\landslide_unet_project\figures\e3_epochwise_replication.png
