# Table 1 Count and Source-Adjacency Audit

## Count Audit

- 1484 + 5925 + 436 + 2504 + 980 + 817 + 178 = 12324.
- Six-region held-out total = 11344.
- Source-adjacent Moxitaidi total = 980.
- The Moxitaidi archive contains 984 pairs; four are excluded in prepare_external_regions.py with reason pHASH distance 8 to CAS training image, leaving 980.

## Source Relationship

- Source training contains moxizheng_0.2m_UAV* and moxizheng_1m_UAV* files.
- Official CAS Table 2 lists Moxitaidi UAV 0.6 m, Moxi town 0.2 m and Moxi town 1 m with the same acquisition period 2022.09-2022.10 and the same provider.
- Official table: https://www.nature.com/articles/s41597-023-02847-z/tables/2
- Therefore Moxitaidi is a source-adjacent cross-GSD diagnostic, not a held-out region.

## Held-Out Six-Region Recalculation

Spearman prevalence-IoU among six held-out regions: rho=0.943.

Leave-one-region-out rho: wenchuan=1.000, jiuzhai_valley=0.900, longxi_river=0.900, hokkaido_iburi_tobu=1.000, lombok=0.900, palu=0.900.

Including Moxitaidi gives rho=0.929, but this is not a held-out-region correlation.

| Model | Source val IoU | Held-out6 macro IoU | Retention | Moxitaidi diagnostic |
|---|---:|---:|---:|---:|
| Bottleneck-LiteASK | 0.7190 | 0.1430 +/- 0.0223 | 0.199 | 0.2961 +/- 0.0087 |
| DeepLabV3+ | 0.6110 | 0.1424 +/- 0.0114 | 0.233 | 0.2428 +/- 0.0115 |
| ResUNet | 0.6820 | 0.1511 +/- 0.0124 | 0.221 | 0.2624 +/- 0.0102 |
| SegFormer-B0 | 0.5930 | 0.1706 +/- 0.0044 | 0.288 | 0.2403 +/- 0.0100 |

## Required Manuscript Changes

- Replace seven-region held-out language with six independent held-out regions plus the source-adjacent Moxitaidi cross-GSD diagnostic.
- Exclude Moxitaidi from held-out macro, prevalence correlations and source-expansion mechanism summaries.
- State explicitly that Moxitaidi and the Moxi-town source subset share area and acquisition period.
- Keep 12,324 only as the sum of all target tiles, not as the held-out benchmark size.
- Replace the previous seven-region macro and retention values with the held-out6 values.
