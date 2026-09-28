# Full Reproduction Runbook

This runbook is the operational definition of full reproduction. It starts from
the public source repository and the exact public data release, rebuilds the
primary experiment outputs, and validates the final reported artifacts.

## Phase 0 - Environment

```powershell
conda env create -f environment-windows.yml
conda activate landslide
$env:LANDSLIDE_PROJECT_ROOT = (Get-Location).Path
$env:LANDSLIDE_PYTHON = (Get-Command python).Source
```

The original environment used Python 3.10.21, PyTorch 2.14.0+cu126, CUDA 12.6,
and an NVIDIA RTX 4060 Laptop GPU with 8 GB memory.

## Phase 1 - Obtain and verify inputs

```powershell
.\scripts\setup\download_inputs.ps1 -ProjectRoot $env:LANDSLIDE_PROJECT_ROOT
.\scripts\setup\prepare_primary_data.ps1 -ProjectRoot $env:LANDSLIDE_PROJECT_ROOT -Python $env:LANDSLIDE_PYTHON
python scripts\checks\validate_release.py
python scripts\checks\verify_manifest.py
python scripts\checks\check_inputs.py --verify-hashes
```

The input script pins the AS-UNet commit to
`1df78b3ea315fc9744db6de9f2917f9f309c7218` and verifies every downloaded CAS
archive against the Zenodo API checksum before processing.

## Phase 2 - Main Q2 experiment

This queue performs the final primary sequence:

1. X2: 18 paired augmentation runs.
2. X1: resolution three-arm experiment after X2 selection.
3. X3: two-architecture, three-region, three-seed matched-exposure source
   expansion.
4. X3 component/merged-block spatial bootstrap with 10,000 draws.
5. Six-region LORO extension and its 5,000-draw spatial bootstrap.

```powershell
.\scripts\run_q2_major_revision_queue_v3.ps1
```

Run this on a machine with enough disk for `outputs/` (reference run
checkpoints total about 20 GiB for the Q2 block) and enough GPU memory for the
configured batch sizes. The queue is restartable and skips completed runs.

## Phase 3 - Legacy Route 1 evidence

The manuscript also uses Route 1 endpoints and few-shot adaptation. Run the
corresponding queues after Phase 2 when the manuscript package must be rebuilt
from scratch:

```powershell
.\scripts\run_cas_baselines_3seeds.ps1
.\scripts\run_benchmark_v2_extended_resunet.ps1
.\scripts\run_benchmark_v2_extended_multisource.ps1
.\scripts\run_cas_retrain20_control.ps1
.\scripts\run_p1_zero_shot_evaluation.ps1
.\scripts\run_p4_augmentation.ps1
.\scripts\run_p4_augmentation_eval.ps1
.\scripts\run_p4_source_fraction.ps1
.\scripts\run_p4_source_fraction_eval.ps1
.\scripts\run_bottleneck_lite_seeds.ps1
.\scripts\run_e1_exposure_matched.ps1
.\scripts\run_e1_exposure_matched_eval.ps1
.\scripts\run_e2_early_late_adaptation.ps1
.\scripts\run_e3_epochwise_replication.ps1
.\scripts\run_benchmark_v2_fewshot128_seed42.ps1
.\scripts\run_benchmark_v2_fewshot128_stage2.ps1
.\scripts\run_e4_support_draws.ps1
.\scripts\run_e4_query_evaluation.ps1
```

The queue entry points are intentionally retained as the executable record of
the experiment. Each script writes stdout/stderr and queue state below `logs/`
and artifacts below `outputs/` and `reports/`.

## Phase 4 - Independent reproduction check

The repository includes a verified Gate-D anchor for the X3 ResUNet pooled
seed-42 arm. See `reports/verification/GATE_D_X3_RESUNET_SEED42.md`. It records:

- exact source-validation IoU reproduction;
- bitwise-identical `metrics.json`;
- bitwise-identical 2,737-row tile-metric CSV;
- tensor-by-tensor checkpoint equality across 278 tensors and 24,455,423
  parameters.

To re-run the structural checks without GPUs:

```powershell
python scripts\checks\validate_release.py
python scripts\checks\verify_manifest.py
python scripts\checks\verify_checkpoint_index.py --project-root $env:LANDSLIDE_PROJECT_ROOT
```

## Hardware and determinism boundary

The reported runs used a single RTX 4060 Laptop GPU and CUDA 12.6. Training
seeds and deterministic cuDNN settings are fixed in the executable scripts.
Exact bitwise reproduction was verified on the same software/hardware stack.
Different GPU architectures, CUDA kernels, or library versions may produce
small numerical differences even when the data, configuration, and seed are
identical.

## Full package acceptance rule

A new machine reproduces the package only when all of the following are true:

1. `check_inputs.py --verify-hashes` passes.
2. The main Q2 queue completes without missing expected runs.
3. The generated X2/X1/X3 tables reproduce the packaged metric anchors.
4. The X3 component bootstrap reproduces the packaged confidence intervals.
5. The Gate-D anchor reproduces the recorded metrics and tensor hash.
6. The final manuscript numbers are generated only from fresh outputs, not from
   stale files.