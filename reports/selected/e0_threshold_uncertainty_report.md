# E0.2-E0.3: Threshold, Prevalence, Transfer, and Rank Audit

All metrics use the existing seed-42/2026/777 benchmark-v2 raw JSON. Threshold metrics are recomputed into MCC, balanced accuracy, and balanced IoU from precision, recall, accuracy, and the known 128x128 pixel count; no new model inference was run.

## Prevalence Association at Threshold 0.5

| Region | Mean foreground fraction | Mean zero-shot IoU | Mean MCC | Mean balanced IoU |
|---|---:|---:|---:|---:|
| Wenchuan | 0.0455 | 0.1097 | 0.1568 | 0.5139 |
| Jiuzhai Valley | 0.2374 | 0.5203 | 0.6021 | 0.6798 |
| Moxitaidi | 0.1053 | 0.2604 | 0.3389 | 0.5581 |
| Longxi River | 0.1453 | 0.1173 | 0.0786 | 0.4419 |
| Hokkaido | 0.1025 | 0.0796 | 0.0306 | 0.4198 |
| Lombok | 0.0307 | 0.0554 | 0.0757 | 0.4585 |
| Palu | 0.0121 | 0.0282 | 0.0502 | 0.4727 |

Spearman correlation across seven regions: rho=0.929, p=0.0025. Leave-one-region-out rho range: 0.886 to 0.943.

This is an association, not evidence that prevalence causes regional difficulty. It must not be reported as a causal mechanism.

## Threshold Sensitivity

| Region | IoU@0.3 | IoU@0.5 | IoU@0.7 | IoU range | Best IoU threshold | MCC range |
|---|---:|---:|---:|---:|---:|---:|
| Wenchuan | 0.1159 | 0.1097 | 0.0871 | 0.0289 | 0.3 | 0.0311 |
| Jiuzhai Valley | 0.5427 | 0.5203 | 0.4361 | 0.1066 | 0.3 | 0.0554 |
| Moxitaidi | 0.2509 | 0.2604 | 0.2315 | 0.0289 | 0.5 | 0.0160 |
| Longxi River | 0.1391 | 0.1173 | 0.0903 | 0.0488 | 0.3 | 0.0204 |
| Hokkaido | 0.0989 | 0.0796 | 0.0598 | 0.0391 | 0.3 | 0.0233 |
| Lombok | 0.0570 | 0.0554 | 0.0525 | 0.0046 | 0.3 | 0.0209 |
| Palu | 0.0234 | 0.0282 | 0.0330 | 0.0097 | 0.7 | 0.0097 |

## Architecture Rank at Threshold 0.5

| Model | Mean rank | Probability rank 1 |
|---|---:|---:|
| Bottleneck-LiteASK | 2.370 | 0.300 |
| ResUNet | 2.698 | 0.081 |
| DeepLabV3+ | 3.474 | 0.003 |
| SegFormer-B0 | 1.458 | 0.616 |

| Seed | Spearman source-vs-target rho | p | LOO-region rho min | LOO-region rho max |
|---:|---:|---:|---:|---:|
| 42 | -0.200 | 0.800 | -0.400 | -0.200 |
| 2026 | 0.400 | 0.600 | -0.200 | 0.400 |
| 777 | -0.800 | 0.200 | -0.800 | -0.400 |

## Source-Expansion Transfer Delta at Threshold 0.5

| Region | Delta IoU mean | SD across seeds | Positive seeds | Seeds >= +0.01 |
|---|---:|---:|---:|---:|
| Hokkaido | +0.0348 | 0.0239 | 3/3 | 3/3 |
| Lombok | -0.0099 | 0.0077 | 0/3 | 0/3 |
| Palu | +0.0067 | 0.0041 | 3/3 | 0/3 |

## Transfer Sign Across Thresholds

| Region | Threshold | Mean delta IoU | Positive seeds | Seeds >= +0.01 |
|---|---:|---:|---:|---:|
| Hokkaido | 0.3 | +0.0638 | 3/3 | 3/3 |
| Hokkaido | 0.4 | +0.0505 | 3/3 | 3/3 |
| Hokkaido | 0.5 | +0.0348 | 3/3 | 3/3 |
| Hokkaido | 0.6 | +0.0184 | 3/3 | 2/3 |
| Hokkaido | 0.7 | +0.0016 | 2/3 | 0/3 |
| Lombok | 0.3 | -0.0077 | 0/3 | 0/3 |
| Lombok | 0.4 | -0.0085 | 0/3 | 0/3 |
| Lombok | 0.5 | -0.0099 | 0/3 | 0/3 |
| Lombok | 0.6 | -0.0114 | 0/3 | 0/3 |
| Lombok | 0.7 | -0.0130 | 0/3 | 0/3 |
| Palu | 0.3 | +0.0074 | 3/3 | 1/3 |
| Palu | 0.4 | +0.0075 | 3/3 | 1/3 |
| Palu | 0.5 | +0.0067 | 3/3 | 0/3 |
| Palu | 0.6 | +0.0050 | 2/3 | 0/3 |
| Palu | 0.7 | +0.0014 | 2/3 | 0/3 |

Hierarchical seed/region bootstrap of the three-region mean delta: +0.0105 [-0.0035, +0.0290], P(delta>0)=0.920, P(delta>=0.01)=0.465.

The transfer delta is reported here only as an exploratory epoch-matched result because E0.1 showed a 6.34x exposure mismatch. It cannot support the causal added-region claim.

## Required Claim Changes

- Replace the prevalence-dominance claim with an association claim.
- Report threshold curves and MCC/balanced metrics alongside IoU.
- Report per-seed paired transfer deltas and intervals.
- Replace the retention-ratio headline with absolute paired deltas.
- Do not combine the four-region LORO target set with the three-region fixed-source target set in one aggregate.
