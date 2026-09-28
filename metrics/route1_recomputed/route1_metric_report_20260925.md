# Route 1 Recomputed Metrics from Existing Raw JSON

Protocol: route1-q2-freeze-v1-20260925

No model inference or training was run. E1 and E4 endpoint tables were reconstructed from existing aggregate raw JSON. Intervals use paired cluster bootstrap over target regions and seeds at threshold 0.5.

## E1 Exposure-Matched Primary Contrast

Contrast: pooled source minus CAS multi-stream replay placebo.

| Region | Endpoint | Delta | 95% interval | P(delta>0) |
|---|---|---:|---:|---:|
| Target Macro | iou | +0.0355 | [+0.0002, +0.0971] | 0.978 |
| Target Macro | balanced_iou | +0.0426 | [-0.0163, +0.1255] | 0.849 |
| Target Macro | mcc | +0.0850 | [+0.0029, +0.2289] | 0.999 |
| Target Macro | precision | +0.1012 | [-0.0046, +0.2884] | 0.889 |
| Target Macro | recall | +0.0151 | [-0.0290, +0.0610] | 0.741 |
| Target Macro | bf1_2 | +0.0832 | [+0.0440, +0.1405] | 1.000 |
| Target Macro | bf1_4 | +0.0894 | [+0.0381, +0.1665] | 1.000 |
| Target Macro | hd95 | -3.4683 | [-16.1619, +4.4282] | 0.292 |
| Hokkaido | iou | +0.1004 | [+0.0635, +0.1472] | 1.000 |
| Hokkaido | balanced_iou | +0.1290 | [+0.1060, +0.1658] | 1.000 |
| Hokkaido | mcc | +0.2412 | [+0.1801, +0.3277] | 1.000 |
| Hokkaido | precision | +0.2996 | [+0.2152, +0.4303] | 1.000 |
| Hokkaido | recall | +0.0090 | [-0.0492, +0.0420] | 0.733 |
| Hokkaido | bf1_2 | +0.1462 | [+0.0986, +0.1957] | 1.000 |
| Hokkaido | bf1_4 | +0.1726 | [+0.1177, +0.2278] | 1.000 |
| Hokkaido | hd95 | +3.0310 | [-1.4832, +8.1267] | 0.850 |
| Lombok | iou | -0.0005 | [-0.0026, +0.0036] | 0.259 |
| Lombok | balanced_iou | -0.0172 | [-0.0203, -0.0116] | 0.000 |
| Lombok | mcc | +0.0030 | [-0.0015, +0.0114] | 0.707 |
| Lombok | precision | -0.0052 | [-0.0076, -0.0011] | 0.000 |
| Lombok | recall | +0.0615 | [+0.0572, +0.0678] | 1.000 |
| Lombok | bf1_2 | +0.0542 | [+0.0390, +0.0633] | 1.000 |
| Lombok | bf1_4 | +0.0541 | [+0.0351, +0.0662] | 1.000 |
| Lombok | hd95 | -16.3532 | [-19.8866, -10.1473] | 0.000 |
| Palu | iou | +0.0066 | [+0.0051, +0.0093] | 1.000 |
| Palu | balanced_iou | +0.0159 | [+0.0114, +0.0195] | 1.000 |
| Palu | mcc | +0.0109 | [+0.0037, +0.0188] | 1.000 |
| Palu | precision | +0.0093 | [+0.0076, +0.0120] | 1.000 |
| Palu | recall | -0.0251 | [-0.0609, -0.0032] | 0.000 |
| Palu | bf1_2 | +0.0492 | [+0.0308, +0.0743] | 1.000 |
| Palu | bf1_4 | +0.0415 | [+0.0273, +0.0699] | 1.000 |
| Palu | hd95 | +2.9174 | [-0.0174, +5.1403] | 0.963 |

## E4 Few-Shot Primary Contrasts

Contrast: adapted ResUNet minus its same-seed source checkpoint. Full fine-tuning is primary; decoder-only is secondary.

| Mode | Buffer m | Region | IoU delta | Balanced IoU delta | MCC delta | Precision delta | Recall delta | BF1@2 delta | HD95 delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0 | Target Macro | +0.1316 | +0.1301 | +0.2397 | +0.3387 | +0.0031 | +0.1839 | +7.0830 |
| full | 0 | Hokkaido | +0.3034 | +0.2368 | +0.5024 | +0.5149 | +0.2955 | +0.3629 | -10.5192 |
| full | 0 | Lombok | +0.0249 | +0.0651 | +0.0783 | +0.2050 | -0.1485 | +0.0801 | -0.3025 |
| full | 0 | Palu | +0.0665 | +0.0885 | +0.1385 | +0.2962 | -0.1375 | +0.1087 | +32.0708 |
| full | 256 | Target Macro | +0.1303 | +0.1306 | +0.2350 | +0.3358 | -0.0114 | +0.1784 | +8.8367 |
| full | 256 | Hokkaido | +0.3045 | +0.2379 | +0.5040 | +0.5177 | +0.2955 | +0.3623 | -9.9697 |
| full | 256 | Lombok | +0.0169 | +0.0631 | +0.0617 | +0.1950 | -0.1826 | +0.0617 | +4.0752 |
| full | 256 | Palu | +0.0694 | +0.0907 | +0.1392 | +0.2948 | -0.1469 | +0.1112 | +32.4046 |
| full | 512 | Target Macro | +0.1303 | +0.1306 | +0.2350 | +0.3358 | -0.0114 | +0.1784 | +8.8367 |
| full | 512 | Hokkaido | +0.3045 | +0.2379 | +0.5040 | +0.5177 | +0.2955 | +0.3623 | -9.9697 |
| full | 512 | Lombok | +0.0169 | +0.0631 | +0.0617 | +0.1950 | -0.1826 | +0.0617 | +4.0752 |
| full | 512 | Palu | +0.0694 | +0.0907 | +0.1392 | +0.2948 | -0.1469 | +0.1112 | +32.4046 |
| decoder-only | 0 | Target Macro | +0.1313 | +0.1296 | +0.2235 | +0.2806 | +0.0117 | +0.1682 | +6.1005 |
| decoder-only | 0 | Hokkaido | +0.3337 | +0.2532 | +0.5328 | +0.5215 | +0.3441 | +0.3529 | -12.4782 |
| decoder-only | 0 | Lombok | +0.0084 | +0.0552 | +0.0392 | +0.1266 | -0.1637 | +0.0773 | -5.8793 |
| decoder-only | 0 | Palu | +0.0519 | +0.0803 | +0.0984 | +0.1937 | -0.1452 | +0.0744 | +36.6591 |
| decoder-only | 256 | Target Macro | +0.1268 | +0.1277 | +0.2139 | +0.2733 | -0.0051 | +0.1623 | +7.8806 |
| decoder-only | 256 | Hokkaido | +0.3298 | +0.2514 | +0.5293 | +0.5230 | +0.3362 | +0.3501 | -11.9317 |
| decoder-only | 256 | Lombok | -0.0023 | +0.0503 | +0.0158 | +0.1059 | -0.1948 | +0.0592 | -1.1296 |
| decoder-only | 256 | Palu | +0.0528 | +0.0815 | +0.0966 | +0.1909 | -0.1566 | +0.0776 | +36.7032 |
| decoder-only | 512 | Target Macro | +0.1268 | +0.1277 | +0.2139 | +0.2733 | -0.0051 | +0.1623 | +7.8806 |
| decoder-only | 512 | Hokkaido | +0.3298 | +0.2514 | +0.5293 | +0.5230 | +0.3362 | +0.3501 | -11.9317 |
| decoder-only | 512 | Lombok | -0.0023 | +0.0503 | +0.0158 | +0.1059 | -0.1948 | +0.0592 | -1.1296 |
| decoder-only | 512 | Palu | +0.0528 | +0.0815 | +0.0966 | +0.1909 | -0.1566 | +0.0776 | +36.7032 |

## Interpretation Guardrails

- Positive IoU must be checked against balanced IoU and MCC before retaining a general segmentation-improvement claim.
- HD95 and boundary deltas require separate interpretation and can disagree with pixel-overlap gains.
- E4 remains a benchmark-tile result. It is not a full-map or operational validation result.
