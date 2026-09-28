# Seed-stratified t intervals for the fixed three-region X3 contrast

Regions are fixed to Hokkaido, Lombok and Palu. Each seed effect is the unweighted mean of the three region-level paired effects. The 95% interval uses Student t with 2 degrees of freedom across the three training seeds.

| Architecture | Endpoint | Delta | SD across seeds | 95% seed-stratified CI |
|---|---|---:|---:|---:|
| ResUNet | iou | +0.0773 | 0.0104 | [+0.0514, +0.1031] |
| ResUNet | balanced_iou | +0.0603 | 0.0100 | [+0.0354, +0.0852] |
| ResUNet | mcc | +0.1573 | 0.0067 | [+0.1406, +0.1739] |
| ResUNet | precision | +0.1784 | 0.0078 | [+0.1590, +0.1977] |
| ResUNet | recall | +0.1075 | 0.0055 | [+0.0939, +0.1211] |
| Bottleneck-LiteASK | iou | +0.0456 | 0.0206 | [-0.0056, +0.0968] |
| Bottleneck-LiteASK | balanced_iou | +0.0383 | 0.0119 | [+0.0087, +0.0678] |
| Bottleneck-LiteASK | mcc | +0.0990 | 0.0302 | [+0.0240, +0.1740] |
| Bottleneck-LiteASK | precision | +0.1182 | 0.0406 | [+0.0174, +0.2190] |
| Bottleneck-LiteASK | recall | +0.0549 | 0.0215 | [+0.0016, +0.1082] |
