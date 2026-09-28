# Numerical provenance

This file defines the authoritative packaged source for each manuscript-level
number. A `fig4` plotting input is never authoritative when it disagrees with a
selected report or a q2_x3 summary.

| Manuscript object | Authoritative file | Column or rule |
|---|---|---|
| Table 1 evidence grades | `reports/selected/route1_claim_matrix_20260925.csv` | Evidence level and supported direction |
| Table 2 region metadata | `data_manifests/manifest_benchmark_v2.csv` | Region counts after frozen split construction |
| Table 4 zero-shot matrix | `reports/selected/benchmark_v2_zero_shot_region_summary.csv` | Three-seed model-region means |
| Table 5 augmentation decision | `metrics/q2_x2_augmentation/primary_paired_summary.csv` | Paired source-validation and target summaries |
| Table 6 regional source expansion | `reports/selected/e1_exposure_matched_contrasts.csv` | Paired region means and intervals |
| Table 8 pooled-source contrast | `metrics/q2_x3_seed_stratified_bootstrap/q2_x3_seed_stratified_t_interval.csv` and `metrics/q2_x3_bootstrap/x3_spatial_cluster_bootstrap.csv` | Seed-stratified t intervals and component-bootstrap intervals |
| Table 9 resolution arms | `metrics/q2_x1_summary/grouped.csv` | Three-seed arm means |
| Threshold, prevalence and calibration tables | `metrics/route1_recomputed/` and `metrics/route1_bootstrap/` | Frozen Route 1 outputs |
| Figure 4 seed-stratified endpoints | `metrics/q2_x3_seed_stratified_bootstrap/q2_x3_seed_stratified_t_interval.csv` | Mean, 95% t interval, df = 2 |

## Superseded data

`metrics/figure_source_data/fig4_exposure_matched_effects_route1_superseded.csv`
is retained for history only. It used the earlier Route 1 aggregation pipeline,
whose point estimates differ from the q2_x3 manuscript pipeline. Do not use it
for the manuscript Table 6 or Table 8.

## Reproduction command

```powershell
python scripts/q2_x3_seed_stratified_reanalysis.py --draws 10000
```

This regenerates the seed-level macro effects, the seed-stratified t intervals,
and the fixed-region component-bootstrap sensitivity.
