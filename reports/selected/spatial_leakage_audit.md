# Spatial Leakage and Duplicate Audit

## Source train/validation

- Train images: 1795; validation images: 513.
- Exact image overlaps: 0.
- Exact mask overlaps: 0.
- Validation images with average-hash distance <= 4 to any train image: 20.
- Strong perceptual near-duplicate candidates (aHash <= 2, SSIM >= 0.7, mask IoU >= 0.5): 0.

| Validation ID | Nearest Train ID | aHash Hamming | Grayscale SSIM | Mask IoU |
|---|---|---:|---:|---:|
| moxizheng_0.2m_UAV0021 | moxizheng_0.2m_UAV0089 | 4 | 0.156 | 0.000 |
| moxizheng_0.2m_UAV0092 | moxizheng_0.2m_UAV1341 | 3 | 0.118 | 0.000 |
| moxizheng_0.2m_UAV0319 | moxizheng_0.2m_UAV0257 | 4 | 0.137 | 0.000 |
| moxizheng_0.2m_UAV0525 | moxizheng_1m_UAV080 | 2 | 0.409 | 0.000 |
| moxizheng_0.2m_UAV0530 | moxizheng_0.2m_UAV1522 | 3 | 0.112 | 0.000 |
| moxizheng_0.2m_UAV0855 | moxizheng_0.2m_UAV0598 | 4 | 0.105 | 0.435 |
| moxizheng_0.2m_UAV0899 | moxizheng_0.2m_UAV1522 | 4 | 0.118 | 0.046 |
| moxizheng_0.2m_UAV1401 | moxizheng_1m_UAV061 | 4 | 0.186 | 0.000 |
| moxizheng_0.2m_UAV1407 | moxizheng_1m_UAV080 | 4 | 0.154 | 0.000 |
| moxizheng_0.2m_UAV1531 | moxizheng_0.2m_UAV0702 | 4 | 0.146 | 0.834 |
| moxizheng_0.2m_UAV1563 | moxizheng_0.2m_UAV1522 | 3 | 0.142 | 0.000 |
| moxizheng_0.2m_UAV1573 | moxizheng_0.2m_UAV0827 | 4 | 0.163 | 0.017 |
| moxizheng_0.2m_UAV1600 | moxizheng_0.2m_UAV0832 | 4 | 0.212 | 0.146 |
| moxizheng_0.2m_UAV1616 | moxizheng_1m_UAV160 | 0 | 0.513 | 0.000 |
| moxizheng_1m_UAV060 | moxizheng_0.2m_UAV0493 | 1 | 0.315 | 0.000 |
| moxizheng_1m_UAV090 | moxizheng_0.2m_UAV0844 | 4 | 0.132 | 0.000 |
| moxizheng_1m_UAV114 | moxizheng_0.2m_UAV1347 | 0 | 0.479 | 0.000 |
| qxg046 | df018 | 3 | 0.203 | 0.290 |
| qxg115 | qx053 | 2 | 0.112 | 0.749 |
| zj117 | moxizheng_0.2m_UAV0677 | 3 | 0.189 | 0.000 |

## Target exact duplicates

| Region | Samples | Duplicate images | Duplicate masks |
|---|---:|---:|---:|
| hokkaido_iburi_tobu | 1484 | 0 | 0 |
| jiuzhai_valley | 5925 | 0 | 80 |
| lombok | 436 | 0 | 0 |
| longxi_river | 2504 | 0 | 0 |
| moxitaidi | 980 | 0 | 0 |
| palu | 817 | 0 | 0 |
| wenchuan | 178 | 0 | 0 |

## Few-shot support/query separation

| Seed | Region | Support | Query | Minimum Chebyshev distance | Pairs with distance <= 1 |
|---:|---|---:|---:|---:|---:|
| 42 | hokkaido_iburi_tobu | 20 | 1341 | 2 | 0 |
| 42 | lombok | 20 | 327 | 2 | 0 |
| 42 | palu | 20 | 734 | 2 | 0 |
| 2026 | hokkaido_iburi_tobu | 20 | 1346 | 2 | 0 |
| 2026 | lombok | 20 | 323 | 2 | 0 |
| 2026 | palu | 20 | 730 | 2 | 0 |
| 777 | hokkaido_iburi_tobu | 20 | 1345 | 2 | 0 |
| 777 | lombok | 20 | 330 | 2 | 0 |
| 777 | palu | 20 | 730 | 2 | 0 |

Interpretation: the few-shot splits enforce distance >= 2 between support chips and common-query chips in the reconstructed stride-256 grid, so they do not overlap spatially. Source train/validation has no exact duplicate images or masks and no strong perceptual near-duplicates after joint aHash, SSIM, and mask-IoU review. Low-threshold aHash candidates are reported for audit only; most have low SSIM and disjoint masks. Spatial adjacency in the variable-sized source images cannot be reconstructed from filenames alone.
