# P4-D: Confidence Calibration and High-Confidence Errors

Checkpoint: ResUNet seed 42, epoch 50, 128x128. Confidence is max(p, 1-p); high confidence is >= 0.90.

| Domain | Error rate | Mean confidence | High-confidence fraction | High-confidence error rate |
|---|---:|---:|---:|---:|
| CAS validation | 0.0590 | 0.9222 | 0.7451 | 0.0069 |
| Hokkaido | 0.2612 | 0.8793 | 0.6071 | 0.1208 |
| Lombok | 0.1389 | 0.9362 | 0.7926 | 0.0709 |
| Palu | 0.1394 | 0.8980 | 0.6712 | 0.0233 |

## Confidence Bins

| Domain | Confidence bin | Pixel fraction | Error rate | Mean confidence |
|---|---|---:|---:|---:|
| CAS validation | [0.50, 0.70) | 0.0853 | 0.3724 | 0.6031 |
| CAS validation | [0.70, 0.90) | 0.1696 | 0.1301 | 0.8214 |
| CAS validation | [0.90, 0.99) | 0.2771 | 0.0166 | 0.9583 |
| CAS validation | [0.99, 1.00) | 0.4680 | 0.0011 | 0.9956 |
| Hokkaido | [0.50, 0.70) | 0.1557 | 0.4796 | 0.5993 |
| Hokkaido | [0.70, 0.90) | 0.2371 | 0.4772 | 0.8141 |
| Hokkaido | [0.90, 0.99) | 0.2869 | 0.1865 | 0.9568 |
| Hokkaido | [0.99, 1.00) | 0.3203 | 0.0620 | 0.9944 |
| Lombok | [0.50, 0.70) | 0.0612 | 0.4585 | 0.6038 |
| Lombok | [0.70, 0.90) | 0.1462 | 0.3739 | 0.8236 |
| Lombok | [0.90, 0.99) | 0.2794 | 0.1658 | 0.9591 |
| Lombok | [0.99, 1.00) | 0.5132 | 0.0193 | 0.9955 |
| Palu | [0.50, 0.70) | 0.1309 | 0.4532 | 0.5995 |
| Palu | [0.70, 0.90) | 0.1979 | 0.3257 | 0.8138 |
| Palu | [0.90, 0.99) | 0.2636 | 0.0456 | 0.9591 |
| Palu | [0.99, 1.00) | 0.4076 | 0.0090 | 0.9952 |

Interpretation: the source domain has a much lower high-confidence error rate than the target domains. Hokkaido is particularly overconfident, so confidence cannot be used as a deployment reliability signal without target-domain calibration.
