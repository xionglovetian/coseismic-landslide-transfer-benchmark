# CAS Landslide Benchmark v2: Seven-Region Zero-Shot Results

All models are trained on the CAS source split and evaluated on the same seven held-out regions.

| Model | Seeds | Macro IoU | Worst region | Worst IoU | Macro Dice | Macro BF1@2 | Macro BF1@4 | Macro HD95 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Bottleneck-LiteASK | 3 | 0.1649 | Palu | 0.0303 | 0.2558 | 0.1428 | 0.2163 | 61.43 |
| DeepLabV3+ | 3 | 0.1567 | Palu | 0.0275 | 0.2468 | 0.1131 | 0.1846 | 69.43 |
| ResUNet | 3 | 0.1670 | Palu | 0.0237 | 0.2554 | 0.1399 | 0.2151 | 64.14 |
| SegFormer-B0 | 3 | 0.1805 | Palu | 0.0316 | 0.2753 | 0.1433 | 0.2215 | 64.56 |

## Per-Region IoU

| Model | Region | Seeds | IoU mean +/- SD | Dice | BF1@2 | BF1@4 | HD95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Bottleneck-LiteASK | Wenchuan | 3 | 0.0993 +/- 0.0158 | 0.1803 | 0.1745 | 0.2541 | 52.05 |
| Bottleneck-LiteASK | Jiuzhai Valley | 3 | 0.4885 +/- 0.0983 | 0.6523 | 0.2549 | 0.3868 | 54.90 |
| Bottleneck-LiteASK | Moxitaidi | 3 | 0.2961 +/- 0.0087 | 0.4569 | 0.2208 | 0.3187 | 64.00 |
| Bottleneck-LiteASK | Longxi River | 3 | 0.1014 +/- 0.0124 | 0.1840 | 0.0878 | 0.1408 | 60.04 |
| Bottleneck-LiteASK | Hokkaido | 3 | 0.0842 +/- 0.0063 | 0.1553 | 0.0957 | 0.1582 | 53.50 |
| Bottleneck-LiteASK | Lombok | 3 | 0.0545 +/- 0.0049 | 0.1034 | 0.0859 | 0.1320 | 79.42 |
| Bottleneck-LiteASK | Palu | 3 | 0.0303 +/- 0.0035 | 0.0587 | 0.0798 | 0.1233 | 66.08 |
| DeepLabV3+ | Wenchuan | 3 | 0.0979 +/- 0.0039 | 0.1783 | 0.1564 | 0.2482 | 62.92 |
| DeepLabV3+ | Jiuzhai Valley | 3 | 0.4766 +/- 0.0749 | 0.6433 | 0.2082 | 0.3420 | 56.94 |
| DeepLabV3+ | Moxitaidi | 3 | 0.2428 +/- 0.0115 | 0.3907 | 0.1542 | 0.2437 | 67.76 |
| DeepLabV3+ | Longxi River | 3 | 0.1165 +/- 0.0046 | 0.2086 | 0.0777 | 0.1307 | 59.01 |
| DeepLabV3+ | Hokkaido | 3 | 0.0854 +/- 0.0065 | 0.1572 | 0.0976 | 0.1645 | 55.52 |
| DeepLabV3+ | Lombok | 3 | 0.0506 +/- 0.0044 | 0.0963 | 0.0543 | 0.0889 | 102.46 |
| DeepLabV3+ | Palu | 3 | 0.0275 +/- 0.0010 | 0.0535 | 0.0434 | 0.0743 | 81.41 |
| ResUNet | Wenchuan | 3 | 0.1008 +/- 0.0111 | 0.1829 | 0.1886 | 0.2709 | 56.63 |
| ResUNet | Jiuzhai Valley | 3 | 0.5428 +/- 0.0405 | 0.7030 | 0.2696 | 0.4176 | 52.09 |
| ResUNet | Moxitaidi | 3 | 0.2624 +/- 0.0102 | 0.4156 | 0.1993 | 0.2927 | 65.68 |
| ResUNet | Longxi River | 3 | 0.1131 +/- 0.0152 | 0.2030 | 0.0865 | 0.1403 | 59.68 |
| ResUNet | Hokkaido | 3 | 0.0723 +/- 0.0122 | 0.1347 | 0.0993 | 0.1617 | 57.54 |
| ResUNet | Lombok | 3 | 0.0538 +/- 0.0020 | 0.1021 | 0.0728 | 0.1127 | 92.71 |
| ResUNet | Palu | 3 | 0.0237 +/- 0.0026 | 0.0462 | 0.0632 | 0.1096 | 64.67 |
| SegFormer-B0 | Wenchuan | 3 | 0.1410 +/- 0.0031 | 0.2472 | 0.2335 | 0.3413 | 51.16 |
| SegFormer-B0 | Jiuzhai Valley | 3 | 0.5735 +/- 0.0128 | 0.7289 | 0.2515 | 0.4019 | 48.55 |
| SegFormer-B0 | Moxitaidi | 3 | 0.2403 +/- 0.0100 | 0.3874 | 0.1583 | 0.2469 | 68.87 |
| SegFormer-B0 | Longxi River | 3 | 0.1381 +/- 0.0157 | 0.2425 | 0.1214 | 0.1897 | 59.32 |
| SegFormer-B0 | Hokkaido | 3 | 0.0764 +/- 0.0011 | 0.1420 | 0.0981 | 0.1525 | 61.39 |
| SegFormer-B0 | Lombok | 3 | 0.0628 +/- 0.0023 | 0.1182 | 0.0765 | 0.1225 | 83.21 |
| SegFormer-B0 | Palu | 3 | 0.0316 +/- 0.0039 | 0.0612 | 0.0637 | 0.0959 | 79.42 |
