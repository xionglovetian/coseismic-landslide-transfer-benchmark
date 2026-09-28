# Data Asset Inventory and Route 2 Readiness Audit

Generated: 2026-09-25T11:02:08+08:00

## Decision

**No-go for the Route 2 108-run matrix under the current data assets.** Source geospatial independence, complete target maps/negative areas, and independent target validation labels are not locally available. The defensible fallback is the Route A/R1 tile-benchmark paper with bounded cross-event and few-shot claims.

## Inventory

- Target benchmark: 12,324 image-mask pairs; images missing=0, masks missing=0; all declared 512x512=True.
- Target zero-foreground chips: 0; the benchmark is positive-chip filtered.
- Source ready-data masks: 9,270 manifest rows from 2,727 base tiles; 2,301 rows and 684 base tiles have zero foreground.
- Source rasters: 11,181 TIFs; CRS present=0, non-identity transforms=0.
- Geospatial vector/sidecar files in the project: 0.
- Few-shot split: support rows=180, evaluation rows=7206, ID overlap=0; this is a label holdout, not an independent annotation partition.
- The target manifest has region-level acquisition/source/sensor/GSD metadata but no scene ID, CRS, geotransform, coordinates, annotator, or agreement fields.

## Target Regions

| Region | N | Zero FG | Min FG | Acquisition | Sensor | GSD (m) |
|---|---:|---:|---:|---|---|---:|
| hokkaido_iburi_tobu | 1,484 | 0 | 0.001057 | 2018.09-2018.10 | Satellite | 3 |
| jiuzhai_valley | 5,925 | 0 | 0.001045 | 2017.08-2017.09 | UAV | 0.2 |
| lombok | 436 | 0 | 0.001003 | 2019.05-2019.12 | WorldView-2/3 | 5 |
| longxi_river | 2,504 | 0 | 0.001022 | 2011.03-2011.05 | UAV | 0.5 |
| moxitaidi | 980 | 0 | 0.001080 | 2022.09-2022.10 | UAV | 0.6 |
| palu | 817 | 0 | 0.001003 | 2021.01-2021.11 | WorldView-2/3 | 5 |
| wenchuan | 178 | 0 | 0.001286 | 2008.11-2008.12 | Landsat | 5 |

## Gate Results

| Gate | Status | Evidence | Consequence |
|---|---|---|---|
| Source geospatial / defensible spatial split | **FAIL** | All 11181 local TIFs have no CRS (0 with CRS) and identity transforms; 0 vector/geospatial-sidecar files; source manifests have no scene/coordinate/CRS fields. | Gate 4 cannot pass; source train/validation independence and source retention cannot be verified. |
| Complete target mosaics with negatives | **FAIL** | Target benchmark has 12324 positive 512-pixel chips and 0 zero-foreground chips; target source archives are absent locally; only cropped tiles are retained. | Gate 3 cannot pass; complete-map precision, false-positive area, omitted area, and operational recommendation are unavailable. |
| Independent target validation labels | **FAIL** | Support and evaluation ID overlap is 0, but both come from the same target manifest and component inventory; no annotator, agreement, or validation-source fields are present. | Adaptation is a within-label-inventory holdout result, not an independently annotated validation result. |
| Prospective or defensible chronological holdout | **PARTIAL** | Target acquisition periods exist at region level, but source training imagery has no acquisition date and target labels are not independent. | A next-earthquake/prospective claim remains unsupported; only a bounded retrospective comparison may be considered. |
| Public reproducibility package | **PARTIAL** | Scripts, split manifests, reports, and outputs exist locally, but the project has no Git repository/public identifier. | Gate 5 cannot yet be fully satisfied. |
| Expanded exposure-matched 108-run package | **BLOCKED** | The three hard data gates currently fail; more runs would estimate an outcome that cannot yet be spatially or operationally validated. | Do not launch the 108-run matrix. |

## What Is Feasible Now

1. Finish a reproducible tile-benchmark paper using the existing E0-E4 evidence and bounded claims.
2. Implement tile-level object metrics, area metrics, hierarchical bootstrap, native sliding-window code, and cost/latency reporting on the existing tile data.
3. Build the prior-work matrix and public repository without new model runs.
4. Request source georeferencing and the original 1536x1536 rasters from the documented source-data contact.
5. Request or reconstruct complete target mosaics and create a genuinely independent target annotation partition.

## Not Feasible From Current Assets

- A defensible source spatial-independence claim.
- Complete-map precision/recall, false-positive area per km2, or omitted landslide area.
- Unbiased operational validation of few-shot adaptation.
- A next-earthquake or prospective generalization claim.
- The expanded 108-run Route 2 experiment before the data gates pass.

## Primary Blockers

- `GEO-SPLIT`: source TIFs are non-georeferenced 224x224 crops; original source size is documented as 1536x1536 but is not present.
- `COMPLETE-MAPS`: target benchmark contains only positive 512x512 chips; local target source archives are absent.
- `INDEPENDENT-LABELS`: support/evaluation tiles are disjoint samples but share the same annotation inventory and provenance.

## Artifacts

- `D:\landslide_unet_project\reports\data_asset_audit_20260925.json`
- `D:\landslide_unet_project\reports\data_asset_gate_matrix_20260925.csv`
- `D:\landslide_unet_project\reports\data_asset_readiness_audit_20260925.md`
