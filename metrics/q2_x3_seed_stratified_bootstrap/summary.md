# Seed-stratified reanalysis of the fixed three-region X3 contrast

Regions fixed: Hokkaido, Lombok and Palu. Seeds resampled within each region, then connected components and tiles resampled within each selected seed. 10,000 draws.

| Architecture | Scheme | Endpoint | Delta | 95% CI | P(delta>0) |
|---|---|---|---:|---:|---:|
| ResUNet | component | iou | +0.0773 | [+0.0345, +0.0849] | 1.000 |
| ResUNet | component | balanced_iou | +0.0603 | [+0.0415, +0.0683] | 1.000 |
| ResUNet | component | mcc | +0.1573 | [+0.0718, +0.1663] | 1.000 |
| ResUNet | component | precision | +0.1784 | [+0.0475, +0.1842] | 1.000 |
| ResUNet | component | recall | +0.1075 | [+0.0315, +0.1328] | 0.999 |
| ResUNet | merged30 | iou | +0.0773 | [+0.0639, +0.0912] | 1.000 |
| ResUNet | merged30 | balanced_iou | +0.0603 | [+0.0489, +0.0728] | 1.000 |
| ResUNet | merged30 | mcc | +0.1573 | [+0.1397, +0.1750] | 1.000 |
| ResUNet | merged30 | precision | +0.1784 | [+0.1637, +0.1923] | 1.000 |
| ResUNet | merged30 | recall | +0.1075 | [+0.0850, +0.1351] | 1.000 |
| Bottleneck-LiteASK | component | iou | +0.0456 | [+0.0097, +0.0620] | 1.000 |
| Bottleneck-LiteASK | component | balanced_iou | +0.0383 | [+0.0160, +0.0533] | 1.000 |
| Bottleneck-LiteASK | component | mcc | +0.0990 | [+0.0228, +0.1254] | 1.000 |
| Bottleneck-LiteASK | component | precision | +0.1182 | [+0.0126, +0.1496] | 1.000 |
| Bottleneck-LiteASK | component | recall | +0.0549 | [-0.0281, +0.0946] | 0.856 |
| Bottleneck-LiteASK | merged30 | iou | +0.0456 | [+0.0250, +0.0713] | 1.000 |
| Bottleneck-LiteASK | merged30 | balanced_iou | +0.0383 | [+0.0217, +0.0586] | 1.000 |
| Bottleneck-LiteASK | merged30 | mcc | +0.0990 | [+0.0657, +0.1379] | 1.000 |
| Bottleneck-LiteASK | merged30 | precision | +0.1182 | [+0.0776, +0.1647] | 1.000 |
| Bottleneck-LiteASK | merged30 | recall | +0.0549 | [+0.0207, +0.0907] | 1.000 |
