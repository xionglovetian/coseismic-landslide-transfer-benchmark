# Q2 X5 Spatial Bootstrap Protocol

Date frozen: 2026-09-27
Protocol ID: q2-x5-spatial-20260927-v1

## Purpose

Correct the variance underestimation produced by resampling individual overlapping
tiles as if they were independent.

## Primary resampling schemes

1. tile: current metric-level unit, retained as the historical comparison.
2. component: connected spatial component reconstructed from overlapping chip
   content. Components are treated as scene/parent-image proxies.
3. merged30: deterministic spatial blocks constructed inside each connected
   component. Initial blocks use a 6-chip grid. Blocks smaller than 30 chips are
   repeatedly merged with the nearest neighbouring block in the same component
   until all possible blocks contain at least 30 chips. A component whose total
   size is below 30 remains one intrinsic cluster because merging across
   disconnected components would not preserve spatial dependence.

## Bootstrap procedure

- Two-stage cluster bootstrap: resample spatial clusters with replacement, then
  resample tiles with replacement within each selected cluster.
- Region-level contrasts resample seed and spatial clusters within that region.
- Macro contrasts resample regions, then seeds, then spatial clusters within the
  selected region-seed strata.
- Endpoints: IoU, balanced IoU, MCC, precision and recall.
- Primary three-region X3 analysis: 10,000 draws.
- Six-region LORO extension: 5,000 draws, explicitly labelled as an extension.

## Reporting rule

- Report cluster counts and the exact size distribution for tile, component and
  merged30 schemes.
- If an interval changes coverage of zero, report the change and lower the
  corresponding evidence grade; do not choose the scheme based on significance.
