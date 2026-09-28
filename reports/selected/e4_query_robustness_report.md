# E4-A/B: Query-Buffer, Threshold, and Calibration Robustness

The same ResUNet source and 20-shot adapted checkpoints were evaluated under the existing guard split and GSD-aware 256 m/512 m physical buffers.

## Query-Buffer Adaptation Gain

| Mode | Buffer (m) | Mean Delta IoU | 95% interval | P(delta>0) | Positive regions | Material regions |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | +0.1316 | [+0.0559, +0.2187] | 1.000 | 3/3 | 3/3 |
| full | 256 | +0.1303 | [+0.0493, +0.2195] | 1.000 | 3/3 | 3/3 |
| full | 512 | +0.1303 | [+0.0513, +0.2196] | 1.000 | 3/3 | 3/3 |
| decoder-only | 0 | +0.1313 | [+0.0374, +0.2326] | 1.000 | 3/3 | 2/3 |
| decoder-only | 256 | +0.1268 | [+0.0385, +0.2281] | 0.999 | 2/3 | 2/3 |
| decoder-only | 512 | +0.1268 | [+0.0366, +0.2286] | 0.999 | 2/3 | 2/3 |

## Threshold Robustness

| Mode | Buffer (m) | Threshold | Mean Delta IoU | Positive cases | Material cases |
|---|---:|---:|---:|---:|---:|
| full | 0 | 0.3 | +0.1428 | 9/9 | 9/9 |
| full | 0 | 0.4 | +0.1571 | 9/9 | 9/9 |
| full | 0 | 0.5 | +0.1316 | 9/9 | 9/9 |
| full | 0 | 0.6 | +0.0818 | 7/9 | 5/9 |
| full | 0 | 0.7 | +0.0264 | 3/9 | 3/9 |
| full | 256 | 0.3 | +0.1417 | 9/9 | 9/9 |
| full | 256 | 0.4 | +0.1559 | 9/9 | 9/9 |
| full | 256 | 0.5 | +0.1303 | 8/9 | 7/9 |
| full | 256 | 0.6 | +0.0806 | 7/9 | 6/9 |
| full | 256 | 0.7 | +0.0263 | 3/9 | 3/9 |
| full | 512 | 0.3 | +0.1417 | 9/9 | 9/9 |
| full | 512 | 0.4 | +0.1559 | 9/9 | 9/9 |
| full | 512 | 0.5 | +0.1303 | 8/9 | 7/9 |
| full | 512 | 0.6 | +0.0806 | 7/9 | 6/9 |
| full | 512 | 0.7 | +0.0263 | 3/9 | 3/9 |
| decoder-only | 0 | 0.3 | +0.1395 | 9/9 | 9/9 |
| decoder-only | 0 | 0.4 | +0.1497 | 9/9 | 9/9 |
| decoder-only | 0 | 0.5 | +0.1313 | 9/9 | 7/9 |
| decoder-only | 0 | 0.6 | +0.0868 | 6/9 | 5/9 |
| decoder-only | 0 | 0.7 | +0.0313 | 3/9 | 3/9 |
| decoder-only | 256 | 0.3 | +0.1354 | 9/9 | 8/9 |
| decoder-only | 256 | 0.4 | +0.1444 | 8/9 | 8/9 |
| decoder-only | 256 | 0.5 | +0.1268 | 7/9 | 7/9 |
| decoder-only | 256 | 0.6 | +0.0845 | 6/9 | 5/9 |
| decoder-only | 256 | 0.7 | +0.0312 | 3/9 | 3/9 |
| decoder-only | 512 | 0.3 | +0.1354 | 9/9 | 8/9 |
| decoder-only | 512 | 0.4 | +0.1444 | 8/9 | 8/9 |
| decoder-only | 512 | 0.5 | +0.1268 | 7/9 | 7/9 |
| decoder-only | 512 | 0.6 | +0.0845 | 6/9 | 5/9 |
| decoder-only | 512 | 0.7 | +0.0312 | 3/9 | 3/9 |

## Calibration

| Mode | Buffer (m) | Source high-confidence error | Adapted high-confidence error | Delta |
|---|---:|---:|---:|---:|
| full | 0 | 0.0716 | 0.0131 | -0.0585 |
| full | 256 | 0.0709 | 0.0127 | -0.0582 |
| full | 512 | 0.0709 | 0.0127 | -0.0582 |
| decoder-only | 0 | 0.0716 | 0.0117 | -0.0599 |
| decoder-only | 256 | 0.0709 | 0.0117 | -0.0592 |
| decoder-only | 512 | 0.0709 | 0.0117 | -0.0592 |

## Decision Rule

Adaptation robustness is supported only if the positive IoU gain persists under 256/512 m buffers, remains sign-consistent across most region-seed cases, and does not come with a systematic calibration deterioration.
