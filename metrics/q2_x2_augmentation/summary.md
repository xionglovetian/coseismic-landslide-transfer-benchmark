# Q2 X2 Augmentation Results

Primary configuration selection is based on source validation only. Target results are reported after that decision.

## Per-model means

| Model | Augmentation | Seeds | Source val IoU | Target 6-region Macro IoU |
|---|---:|---:|---:|---:|
| ResUNet | current | 3 | 0.6822 +/- 0.0067 | 0.1511 +/- 0.0124 |
| ResUNet | none | 3 | 0.7130 +/- 0.0094 | 0.1648 +/- 0.0038 |
| ResUNet | strong | 3 | 0.6225 +/- 0.0103 | 0.1631 +/- 0.0124 |
| BottleneckLiteASKUNetPlusPlus | current | 3 | 0.7105 +/- 0.0039 | 0.1400 +/- 0.0161 |
| BottleneckLiteASKUNetPlusPlus | none | 3 | 0.7660 +/- 0.0061 | 0.1506 +/- 0.0082 |
| BottleneckLiteASKUNetPlusPlus | strong | 3 | 0.6598 +/- 0.0027 | 0.1698 +/- 0.0038 |

## Paired deltas

| Comparison | Pairs | Source val delta | Target macro delta |
|---|---:|---:|---:|
| none-current | 6 | +0.0432 +/- 0.0158 | +0.0122 +/- 0.0108 |
| strong-current | 6 | -0.0552 +/- 0.0059 | +0.0210 +/- 0.0152 |

## Prespecified selection rule

`none` replaces `current` only if it is non-inferior for both architectures at the -0.010 source-validation IoU margin and the overall six-pair source delta is non-negative.

Decision: **none**.
