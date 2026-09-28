# E2: Early-vs-Late Source Checkpoint Adaptation

Model: ResUNet, seed 42, 20-shot adaptation, 400 optimizer steps. The epoch-50 results are the same P3 runs; epoch-20 and epoch-30 results are the new E2 runs.

## Macro Results

| Mode | Init epoch | Macro IoU | Delta vs zero-shot | Macro BF1@2 | Macro HD95 |
|---|---:|---:|---:|---:|---:|
| full | 20 | 0.1681 | +0.1216 | 0.2733 | 76.82 |
| full | 30 | 0.1883 | +0.1418 | 0.2730 | 77.05 |
| full | 50 | 0.1821 | +0.1356 | 0.2727 | 75.73 |
| decoder-only | 20 | 0.1777 | +0.1312 | 0.2367 | 84.06 |
| decoder-only | 30 | 0.1818 | +0.1353 | 0.2406 | 82.59 |
| decoder-only | 50 | 0.1843 | +0.1378 | 0.2518 | 80.20 |

## Early-minus-Late Contrasts

| Mode | Contrast | Region | Delta IoU |
|---|---|---|---:|
| full | epoch20-epoch50 | hokkaido_iburi_tobu | -0.0173 |
| full | epoch20-epoch50 | lombok | -0.0204 |
| full | epoch20-epoch50 | palu | -0.0043 |
| full | epoch20-epoch50 | macro | -0.0140 |
| full | epoch30-epoch50 | hokkaido_iburi_tobu | +0.0179 |
| full | epoch30-epoch50 | lombok | -0.0007 |
| full | epoch30-epoch50 | palu | +0.0016 |
| full | epoch30-epoch50 | macro | +0.0062 |
| decoder-only | epoch20-epoch50 | hokkaido_iburi_tobu | -0.0029 |
| decoder-only | epoch20-epoch50 | lombok | -0.0037 |
| decoder-only | epoch20-epoch50 | palu | -0.0129 |
| decoder-only | epoch20-epoch50 | macro | -0.0065 |
| decoder-only | epoch30-epoch50 | hokkaido_iburi_tobu | -0.0007 |
| decoder-only | epoch30-epoch50 | lombok | -0.0052 |
| decoder-only | epoch30-epoch50 | palu | -0.0014 |
| decoder-only | epoch30-epoch50 | macro | -0.0025 |

## Decision

No early checkpoint meets the pre-specified criterion of improving at least two of three target regions and reaching a +0.01 Macro IoU gain relative to epoch 50. Epoch-30 full fine-tuning is slightly higher on Macro IoU (+0.0062), but the gain is driven by Hokkaido while Lombok is unchanged and Palu is +0.0016. Decoder-only early initialization does not beat epoch 50. The current evidence therefore does not show that early stopping improves few-shot adaptation; it weakens the causal claim that source overfitting materially harms target adaptation.

## Decision Rule

Source overfitting is causally harmful for adaptation only if an early checkpoint improves at least two of three target regions and reaches a Macro IoU gain of at least 0.01 relative to epoch 50 under the same adaptation protocol.

This is a seed-42 screening experiment; a positive result requires seed2026/777 confirmation before making a general claim.
