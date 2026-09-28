# B1 Pooled Source Composition

Date: 2026-09-27

| Domain | Samples | Expected exposures |
|---|---:|---:|
| CAS | 1,795 | 7,168 |
| Jiuzhai Valley | 5,925 | 7,168 |
| Longxi River | 2,504 | 7,168 |
| Moxitaidi | 980 | 7,168 |
| Wenchuan | 178 | 7,168 |
| Total | 11,382 | 35,840 |

Sampler: `WeightedRandomSampler`, domain mass `0.2` per slot.

With batch size 32 and `drop_last=True`, one full traversal gives `355 * 32 = 11,360` exposures and drops 22 samples. This explains the previously unclear `227,200 / 20 = 11,360` arithmetic from the older epoch-matched regime.

## Action

Manuscript Methods updated.
