# E0.1: Continuation Exposure Ledger

This audit uses the actual train counts parsed from the original launch logs. All runs use batch size 32 with `drop_last=True`.

| Regime | Train images | Steps/epoch | Total optimizer steps | Image exposures | Step ratio vs CAS | Exposure ratio vs CAS |
|---|---:|---:|---:|---:|---:|---:|
| CAS-only continuation | 1795 | 56 | 1120 | 35840 | 1.00x | 1.00x |
| pooled-source continuation | 11382 | 355 | 7100 | 227200 | 6.34x | 6.34x |
| pooled-source continuation | 11382 | 355 | 7100 | 227200 | 6.34x | 6.34x |

## Conclusion

The current CAS-only continuation and pooled-source continuation are epoch-matched, not optimizer-step matched and not exposure matched. The pooled-source regime receives approximately 6.34 times as many optimizer steps and image exposures.

Therefore, the existing contrast cannot causally isolate the effect of adding source regions from the effect of additional optimization and sample exposure. It must be relabeled as an epoch-matched exploratory comparison until E1 is completed.
