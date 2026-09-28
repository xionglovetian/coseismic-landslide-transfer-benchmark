# E3 Summary: Model/Seed Replication of Temporal Source Overfitting

Date: 2026-09-25

## Protocol

- Models: ResUNet and SegFormer-B0.
- Seeds: 42, 2026, 777.
- Input: 128x128.
- Training: 50 epochs, same source data and optimizer protocol as the main benchmark, with checkpoint interval 10.
- Evaluated checkpoints: 20, 30, 50 on CAS validation and Hokkaido/Lombok/Palu.
- ResUNet seed42 reused the existing P4 epoch-wise run; five new runs were trained.\n- ResUNet seed2026/777 reruns matched the original main checkpoints exactly. SegFormer reruns were protocol-matched independent trajectories with minor training nondeterminism; the trend analysis is within each run.

## Prespecified Pattern

A run supports temporal source overfitting when:

1. CAS validation peaks at epoch 50;
2. target Macro IoU peaks at epoch 20 or 30;
3. target Macro IoU then declines by at least 0.005.

## Results

| Model | Seed | Source peak epoch | Target peak epoch | Target peak IoU | Epoch-50 target IoU | Peak-to-final drop | Pattern supported |
|---|---:|---:|---:|---:|---:|---:|---|
| ResUNet | 42 | 50 | 20 | 0.0632 | 0.0484 | +0.0148 | Yes |
| ResUNet | 2026 | 50 | 20 | 0.0550 | 0.0455 | +0.0095 | Yes |
| ResUNet | 777 | 50 | 50 | 0.0559 | 0.0559 | 0.0000 | No |
| SegFormer-B0 | 42 | 50 | 20 | 0.0597 | 0.0578 | +0.0019 | No |
| SegFormer-B0 | 2026 | 50 | 50 | 0.0551 | 0.0551 | 0.0000 | No |
| SegFormer-B0 | 777 | 50 | 30 | 0.0582 | 0.0576 | +0.0006 | No |

Source validation peaks at epoch 50 in 6/6 runs, but the target-side early peak with material decline occurs in only 2/6 runs.

## Decision

The prespecified reviewer-response threshold was at least 4/6 runs. E3 fails this threshold.

Therefore:

- The temporal source-overfitting pattern is not replicated across models and seeds.
- It remains an exploratory, ResUNet-specific observation.
- It cannot be presented as a stable mechanism of cross-event failure.
- This is consistent with E2, where early source checkpoints did not improve few-shot adaptation.

## Supported Claims

- Source validation continues to rise while target performance remains flat or declines, but the timing and magnitude are model/seed dependent.
- Source validation is not a reliable predictor of target performance.
- Cross-event performance heterogeneity remains robust.

## Unsupported Claims

- Target performance systematically peaks before source validation across models and seeds.
- Source overfitting is the dominant general mechanism of cross-event failure.
- Early stopping is generally beneficial for target adaptation.

## Artifacts

- `reports/e3_epochwise_replication_report.md`
- `reports/e3_epochwise_per_run.csv`
- `reports/e3_epochwise_diagnostics.csv`
- `figures/e3_epochwise_replication.png`
- `outputs/bench_v2_e3_epochwise_*`
