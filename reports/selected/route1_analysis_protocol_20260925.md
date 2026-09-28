# Route 1 Analysis Protocol

Protocol ID: `route1-q2-freeze-v2-20260925`

Status: **FROZEN_AFTER_SOURCE_ADJACENCY_CORRECTION**

Frozen: 2026-09-25T03:20:45.954Z

## Source-Adjacency Amendment

Moxitaidi is reclassified from a held-out target to a source-adjacent cross-GSD diagnostic. Official CAS Table 2 places Moxitaidi UAV 0.6 m and Moxi town 0.2/1 m in the same area and acquisition period, and the source split contains Moxi-town imagery. Held-out aggregates therefore use six independent regions and 11,344 tiles. The full target total remains 12,324, including 980 Moxitaidi diagnostic tiles.

## Scope

Retrospective, region-held-out tile benchmark; no prospective, complete-map, operational, or independent-validation claim.

The primary scientific estimand is the exposure-matched change from pooled-source training relative to the source-only replay placebo. There is no primary operational estimand because complete event maps and independently annotated validation labels are unavailable.

## Frozen Populations

- Source-only benchmark: 4 architectures x 3 seeds x 6 independent held-out regions. Moxitaidi is source-adjacent and is reported separately.
- Exposure-matched source expansion: Bottleneck-LiteASK x seeds 42/2026/777 x Hokkaido/Lombok/Palu x 1120 steps x batch 32 x 35840 exposures.
- Few-shot adaptation: ResUNet x seeds 42/2026/777 x Hokkaido/Lombok/Palu x 20 support tiles x 400 steps; full fine-tuning primary, decoder-only secondary.

## Endpoints and Uncertainty

- Co-primary endpoints: IoU, balanced IoU and MCC.
- Secondary endpoints: precision, recall, boundary F1 at 2 and 4 pixels, HD95, Dice, accuracy, high-confidence error, ECE, Brier score and risk-coverage.
- Contrasts are paired by seed, architecture, target region and checkpoint budget.
- Uncertainty uses 10,000 paired cluster-bootstrap resamples over target region and seed; tile clusters will be resampled within those strata after eval-only inference.
- Pixels are never treated as independent observations.

## Material Effect Rule

- The reporting marker is an absolute IoU difference of 0.01 on the bounded 0-1 IoU scale.
- This is not a deployment threshold.
- A headline claim also requires an interval excluding zero and directional support from at least one prevalence-robust co-primary endpoint.
- The manuscript legacy 0.02 statement is superseded and must be removed or reconciled.

## Confirmatory Rules

- Confirmatory source-expansion claim is limited to the exact tested architecture, targets, budget, and treatment package.
- A broad homogeneity or universal-transfer claim is prohibited.
- Few-shot claims are limited to benchmark tiles and query-covered stitched maps.
- Negative or failed replication results remain visible.

## Prohibited Claims

- prospective or next-event generalization
- complete-event-map performance
- false-positive area or omitted area per complete event
- emergency readiness or deployment sufficiency
- independent validation language for the query split
- source spatial independence
- stable source-overfitting mechanism
- universal positive source transfer
- held-out or independent-region language for Moxitaidi
- 20-tile operational sufficiency
- native-resolution performance without a native-scale run

## Analysis Sequence

1. Freeze protocol and claim matrix.
2. Recompute all endpoints from existing E1 and E4 raw JSON.
3. Run eval-only inference on existing checkpoints for tile-level metrics, ECE, Brier, and risk-coverage.
4. Publish the pooled-source composition and sampling weights.
5. Publish code, split manifests, metric code, logs, and machine-readable outputs.
6. Rewrite manuscript claims and limitations to match this frozen protocol.
7. Decide separately whether to run the optional all-region architecture and budget expansion.

## Frozen Input Hashes

- `D:/landslide_unet_project/data/processed/benchmark_v2_regions_512/manifest_benchmark_v2.csv`: `0e63fa0647d0ba1052c29c6c17d8da990a6fa28d6d39c11d6326aff766d1f9cf`
- `D:/landslide_unet_project/data/processed/fewshot_target_splits/fewshot_support_seed42.csv`: `4dcea50cc77ce63b5b5f09b1aa9563794351f4a15ed758925d45bf40ae5aaf30`
- `D:/landslide_unet_project/data/processed/fewshot_target_splits/fewshot_support_seed2026_777.csv`: `977c8e8382cef363ea1f39956b706fb061df3a60b784cc6ab72eae6472e00a45`
- `D:/landslide_unet_project/reports/e1_exposure_matched_bootstrap.csv`: `7893722570e9a6d62e6fcc5c57f46a8f98d24bd8c27dc95a0817ba35d34c9f46`
- `D:/landslide_unet_project/reports/e4_query_summary.csv`: `9d43fc3e763e01b76fea35b5f42287ac360bd5eb5f284064f48380b36bff2333`
- `D:/landslide_unet_project/reports/e3_epochwise_replication_report.md`: `074255d1e8927f8d9e4de2690527d7120363416997b2f3ec282e7db37c9813a2`
- `D:/landslide_unet_project/reports/data_asset_readiness_audit_20260925.md`: `70aa27074455f34c613d9e559bc0e826c760639b0893f0bfcceb6db86515e0c7`
- `D:/landslide_unet_project/reports/route1_table1_source_adjacency_audit_20260925.md`: `7a438b066415c661e900ed4b3d3fa9c80963094090cc3bf45667e83346849eca`

## Claim Matrix

See `route1_claim_matrix_20260925.csv`. Confirmatory findings, supporting findings and prohibited claims are now separated. Future summaries must cite a claim ID and may not add a new headline claim without revising the freeze file.
