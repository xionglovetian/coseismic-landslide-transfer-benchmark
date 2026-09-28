# Reproducibility Guide

## 1. Reference environment

- OS used for the experiments: Windows 11
- Python: 3.10.21
- CUDA runtime: 12.6
- PyTorch: 2.14.0+cu126
- GPU used for the reported runs: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB

`environment-full.txt` is the original environment export. `requirements.txt`
is a reduced pinned set for the benchmark code.

## 2. Path configuration

The scripts use `LANDSLIDE_PROJECT_ROOT` and default to the repository root.

```powershell
$env:LANDSLIDE_PROJECT_ROOT = (Get-Location).Path
$env:LANDSLIDE_PYTHON = (Get-Command python).Source
```

The reference data layout is:

```text
$LANDSLIDE_PROJECT_ROOT/
  repos/AS-UNet/inputs/data_sum_moxizhen+bijie/
  data/processed/benchmark_v2_regions_512/
  data/processed/external_regions_512/
  data/processed/fewshot_target_splits/
  outputs/
  reports/
  logs/
```

The image data are not included. Place licensed data at the layout above or set
`LANDSLIDE_PROJECT_ROOT` to an existing local experiment root.

The packaged manifest paths point to the original workstation. Rebase a copy
without modifying the frozen source CSV:

```powershell
python scripts/checks/rebase_manifest_paths.py `
  --input data_manifests/manifest_benchmark_v2.csv `
  --output data/processed/benchmark_v2_regions_512/manifest_benchmark_v2.csv `
  --old-root "D:/landslide_unet_project" `
  --new-root $env:LANDSLIDE_PROJECT_ROOT
```

Use the same command for `manifest_external.csv` and the few-shot split CSVs.

## 3. Package checks

```powershell
python scripts/checks/validate_release.py
python scripts/checks/verify_manifest.py
python scripts/checks/check_environment.py
```

`validate_release.py` checks manifest counts, required metric files, reported
numerical anchors, and the absence of raw imagery/checkpoints.
`verify_manifest.py` verifies SHA-256 hashes for every file listed in
`MANIFEST_SHA256.csv`.

## 4. Metric-level reproduction

This level requires no imagery or checkpoint files.

- X2 augmentation selection:
  `metrics/q2_x2_augmentation/`
- X1 resolution three-arm results:
  `metrics/q2_x1_resolution_raw/`,
  `metrics/q2_x1_a1c_raw/`,
  `metrics/q2_x1_128ref_eval_raw/`, and
  `metrics/q2_x1_summary/`
- X3 matched-exposure and spatial bootstrap:
  `metrics/q2_x3_raw/`, `metrics/q2_x3_tiles/`, and
  `metrics/q2_x3_bootstrap/`
- Six-region LORO extension:
  `metrics/q2_x3_loro_tiles/` and `metrics/q2_x3_loro_bootstrap/`
- Route 1 frozen contrast and calibration outputs:
  `metrics/route1_recomputed/` and `metrics/route1_bootstrap/`

The packaged CSVs and JSON files are the frozen inputs for the numerical claims
in the accompanying manuscript.

## 5. Model-level workflow

After obtaining licensed data and materializing the expected layout:

1. Inspect available arguments:
   `python scripts/train_unified.py --help`
   `python scripts/train_exposure_matched.py --help`
   `python scripts/evaluate_q2_checkpoint.py --help`
   `python scripts/q2_x3_spatial_bootstrap.py --help`
2. Run X2 first, because `none` was selected before target evaluation:
   `powershell -ExecutionPolicy Bypass -File scripts/run_q2_x2_augmentation.ps1`
3. Run X1 resolution arms:
   `powershell -ExecutionPolicy Bypass -File scripts/run_q2_x1_resolution.ps1`
4. Run the two-architecture X3 source-expansion experiment:
   one invocation per architecture using
   `scripts/run_q2_x3_resunet.ps1` or the corresponding Bottleneck queue.
5. Run tile-level evaluation and the component bootstrap:
   `python scripts/evaluate_q2_tile_metrics.py --help`
   `python scripts/q2_x3_spatial_bootstrap.py --help`
6. Run the exploratory six-region LORO extension only after preserving the
   primary three-region contrast.

All queue scripts are restartable: they skip completed runs unless `-Force` is
specified.

## 6. Numerical anchors

The release validator checks these anchors:

| Object | Anchor |
|---|---:|
| X2 `none-current` source-validation IoU delta | `+0.0431505334` |
| X3 ResUNet target-macro IoU delta | `+0.0772718654` |
| X3 Bottleneck target-macro IoU delta | `+0.0455875690` |
| X1 ResUNet retrained-256 minus 128 target IoU | `-0.0243143426` |
| X1 SegFormer-B0 retrained-256 minus 128 target IoU | `+0.0028660398` |

## 7. Evidence boundaries

The package supports a retrospective benchmark over positive image chips. It
does not establish prospective next-event performance, complete-map
false-positive area, operational readiness, spatially independent source
splitting, or independently annotated target validation. Those boundaries are
retained in the selected reports and must remain in downstream claims.

## 8. Known verification gap

The package was assembled from existing experiment outputs. Model-level
training and inference were not rerun during packaging, and checkpoints are not
included. Metrics and manifests were validated structurally and numerically.
For a formal archival release, record the exact Git commit and a Zenodo DOI,
then perform one complete end-to-end rerun from licensed data.
