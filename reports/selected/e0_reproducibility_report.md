# E0.5: Reproducibility Inventory

The inventory separates verified local artifacts from missing publication requirements. It does not include model checkpoints because they are large and may be subject to redistribution constraints.

## Included

| Role | Files |
|---|---:|
| code_or_manifest | 14 |
| extended_source_result | 6 |
| few_shot_result | 90 |
| raw_continuation_control | 3 |
| raw_zero_shot_result | 12 |

## Verified Artifacts

- Seven-region benchmark manifest with hashes and region metadata.
- Target spatial index reconstructed from exact chip half-overlap.
- Deterministic few-shot support/query split files for seeds 42, 2026, and 777.
- Unified training and evaluation code.
- Raw zero-shot results for four architectures and three seeds.
- Continuation-control and fixed-source transfer results.
- Few-shot adaptation results across all formal runs.
- Environment lock file and experiment reports.

## Still Missing Before Public Release

- Source geolocation or geospatial polygons required for physical source train/validation block separation.
- A complete checkpoint archive, if dataset and model licensing permit redistribution.
- A frozen copy of the final manuscript table-generation scripts.
- A formal data and code availability statement covering the CC BY-NC 4.0 dataset license.

Machine-readable inventory: `D:\landslide_unet_project\reports\e0_reproducibility_manifest.csv`.
