param(
  [string]$ProjectRoot = 'D:\landslide_unet_project',
  [string]$Python = '',
  [string]$RunRoot = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $Python) {
  if ($env:LANDSLIDE_PYTHON) { $Python = $env:LANDSLIDE_PYTHON } else { $Python = 'python' }
}
if (-not $RunRoot) { $RunRoot = Join-Path $ProjectRoot 'runs\gate_d\q2_x3_resunet_pooled_seed42' }
if (Test-Path -LiteralPath $RunRoot) { throw "Gate-D output already exists: $RunRoot" }
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
$env:LANDSLIDE_PROJECT_ROOT = $ProjectRoot

$InitCheckpoint = Join-Path $ProjectRoot 'outputs\q2_x2_aug_none_resunet_seed42\best_model.pth'
if (-not (Test-Path -LiteralPath $InitCheckpoint)) { throw "Missing X2 initialization checkpoint: $InitCheckpoint" }
$TrainOut = Join-Path $RunRoot 'train'
$TilesOut = Join-Path $RunRoot 'q2_x3_resunet_pooled_seed42.tiles.csv'

& $Python (Join-Path $RepoRoot 'scripts\train_exposure_matched.py') `
  --regime pooled --model ResUNet --seed 42 --steps 1120 --checkpoint-interval 280 `
  --batch-size 32 --input-size 128 --lr 0.0001 --min-lr 0.000001 --weight-decay 0.0001 `
  --num-workers 4 --amp --init-checkpoint $InitCheckpoint --output-dir $TrainOut
if ($LASTEXITCODE -ne 0) { throw 'Gate-D training failed' }

& $Python (Join-Path $RepoRoot 'scripts\evaluate_q2_tile_metrics.py') `
  --model ResUNet --checkpoint (Join-Path $TrainOut 'final_model.pth') --output $TilesOut `
  --regions hokkaido_iburi_tobu lombok palu --input-size 128 --batch-size 16 --num-workers 4 --amp `
  --run-id q2_x3_resunet_pooled_seed42 --regime pooled --role intervention --seed 42 --step 1120
if ($LASTEXITCODE -ne 0) { throw 'Gate-D tile evaluation failed' }

$expectedIou = 0.518395166665147
$actual = Get-Content -Raw -LiteralPath (Join-Path $TrainOut 'metrics.json') | ConvertFrom-Json
if ([math]::Abs([double]$actual.source_validation.iou - $expectedIou) -gt 1e-15) {
  throw "IoU mismatch: $($actual.source_validation.iou) vs $expectedIou"
}

$referenceTiles = Join-Path $RepoRoot 'metrics\q2_x3_tiles\q2_x3_resunet_pooled_seed42.tiles.csv'
$referenceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $referenceTiles).Hash
$actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $TilesOut).Hash
if ($actualHash -ne $referenceHash) { throw "Tile CSV mismatch: $actualHash vs $referenceHash" }

$originalCheckpoint = Join-Path $ProjectRoot 'outputs\q2_x3_resunet_pooled_seed42\final_model.pth'
$tensorCheck = 'SKIPPED_ORIGINAL_CHECKPOINT_NOT_FOUND'
if (Test-Path -LiteralPath $originalCheckpoint) {
  $tensorCheck = (& $Python (Join-Path $RepoRoot 'scripts\checks\compare_checkpoint_tensors.py') $originalCheckpoint (Join-Path $TrainOut 'final_model.pth')) -join "`n"
  if ($LASTEXITCODE -ne 0) { throw 'Checkpoint tensor comparison failed' }
}

$result = [ordered]@{
  protocol = 'gate-d-q2-x3-resunet-seed42'
  project_root = $ProjectRoot
  run_root = $RunRoot
  expected_source_val_iou = $expectedIou
  actual_source_val_iou = [double]$actual.source_validation.iou
  reference_tile_sha256 = $referenceHash
  reproduced_tile_sha256 = $actualHash
  tensor_check = $tensorCheck
}
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $RunRoot 'gate_d_result.json') -Encoding utf8
Write-Host 'GATE-D PASS'
Write-Host "Result: $(Join-Path $RunRoot 'gate_d_result.json')"