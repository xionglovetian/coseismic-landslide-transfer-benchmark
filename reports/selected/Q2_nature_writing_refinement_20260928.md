# Nature-style refinement: claim-evidence map

Date: 2026-09-28

## Core claim

Under source-validation-only configuration selection and matched optimisation exposure, source expansion improves average three-region macro IoU and MCC for ResUNet and Bottleneck-LiteASK, but the effect remains region-, model- and endpoint-dependent.

## Strongly supported claims

| Claim | Evidence | Writing stance |
|---|---|---|
| No augmentation is the prespecified primary configuration | Paired 2 x 3 x 3 X2 experiment; source-validation selection rule | Direct: `selected`, `unambiguously` |
| Source expansion improves average macro IoU | X3 component-bootstrap intervals exclude zero for both architectures | Direct: `improved`, `produced a positive average benchmark effect` |
| Source expansion improves MCC | Component-bootstrap intervals exclude zero for both architectures | Direct: `increased MCC in both` |
| Hokkaido responds strongly and consistently | Large positive intervals under both architectures | Direct: `improved strongly` |
| Independent 256-pixel retraining does not improve target IoU | X1 three-arm experiment; ResUNet -0.0243, SegFormer +0.0029 | Direct: `did not improve` |
| 128-trained weights at 256 mainly diagnose weight-scale mismatch | Target IoU rises, but source-validation IoU collapses to 0.4146 and 0.3211 | Direct with boundary: `cannot support a deployment claim` |

## Defensive claims

| Boundary | Why defensive | Manuscript wording |
|---|---|---|
| Lombok | Small estimate and interval crossing or near zero | `uncertain`, `effectively unchanged` |
| Balanced IoU | Intervals cross zero for both architectures | `remained mixed` |
| Palu under Bottleneck-LiteASK | Positive point estimate but lower interval marginally crosses zero | `smaller positive estimate` |
| Six-region LORO extension | Placebo controls use earlier augmentation setting | `exploratory only` |
| Few-shot adaptation | Precision-led; recall and Palu HD95 trade-offs | `bounded`, `benchmark gain` |
| Complete-map and operational readiness | No complete-event mosaics or independent annotation | `not assessable`, `not operational readiness` |

## Structural choices

- Abstract leads with the controlled intervention and states the average gain before listing heterogeneity.
- Contributions distinguish configuration selection, matched-exposure intervention and resolution attribution.
- Results separate average effect from region and endpoint boundaries.
- Discussion interprets the protocol as the contribution and avoids claiming a universal source-diversity law.
- Limitations derive from specific unresolved inference boundaries rather than a generic list.

## Data integrity

Only prose, captions and narrative structure were refined in this pass. No experimental value, table cell or figure data was changed.
