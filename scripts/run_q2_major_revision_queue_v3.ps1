$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$logs = Join-Path $project 'logs'
$log = Join-Path $logs 'q2_major_revision_v3_master.log'
function Log([string]$Message) { Add-Content -LiteralPath $log -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date),$Message) -Encoding UTF8 }

Log 'START amendment-v2 Bottleneck grid from scratch'
$bottleneckRunner = Join-Path $project 'scripts\run_q2_x2_augmentation.ps1'
& $bottleneckRunner -Models BottleneckLiteASKUNetPlusPlus -Seeds 42,2026,777 -Augmentations current,none,strong
if (-not $?) { throw 'Bottleneck X2 grid failed' }

Log 'Summarize 18-run X2 grid and select augmentation'
$summarizer = Join-Path $project 'scripts\summarize_q2_x2_augmentation.py'
& $python $summarizer | Add-Content -LiteralPath $log -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'X2 summarization failed' }

$rawFiles = Get-ChildItem -LiteralPath (Join-Path $project 'reports\q2_x2_augmentation_raw') -Filter 'q2_x2_aug_*.json' -File
if ($rawFiles.Count -ne 18) { throw "X2 incomplete: $($rawFiles.Count)/18 result files" }
$grouped = Import-Csv -LiteralPath (Join-Path $project 'reports\q2_x2_augmentation\grouped.csv')
$primaryPairs = Import-Csv -LiteralPath (Join-Path $project 'reports\q2_x2_augmentation\primary_paired_summary.csv')
$nonePairs = $primaryPairs | Where-Object { $_.comparison -eq 'none-current' } | Select-Object -First 1
if (-not $nonePairs) { throw 'Missing primary none-current summary' }
$nonInferior = $true
foreach ($model in @('ResUNet','BottleneckLiteASKUNetPlusPlus')) {
  $current = $grouped | Where-Object { $_.model -eq $model -and $_.augmentation -eq 'current' } | Select-Object -First 1
  $none = $grouped | Where-Object { $_.model -eq $model -and $_.augmentation -eq 'none' } | Select-Object -First 1
  if (-not $current -or -not $none) { $nonInferior = $false; continue }
  $delta = [double]$none.source_val_iou_mean - [double]$current.source_val_iou_mean
  Log "SELECTION model=$model source-none-minus-current=$delta"
  if ($delta -lt -0.010) { $nonInferior = $false }
}
$overallDelta = [double]$nonePairs.source_val_iou_delta_mean
$augmentation = if ($nonInferior -and $overallDelta -ge 0) { 'none' } else { 'current' }
Log "SELECTED primary augmentation=$augmentation overall-source-delta=$overallDelta"

$x1 = Join-Path $project 'scripts\run_q2_x1_resolution.ps1'
Log "START X1 augmentation=$augmentation"
& $x1 -Augmentation $augmentation
if (-not $?) { throw 'X1 queue failed' }
Log 'DONE X1'

$x3 = Join-Path $project 'scripts\run_q2_x3_resunet.ps1'
Log "START X3 ResUNet augmentation=$augmentation"
& $x3 -Model ResUNet -Slug resunet -Augmentation $augmentation
if (-not $?) { throw 'X3 ResUNet failed' }
Log "START X3 Bottleneck augmentation=$augmentation"
& $x3 -Model BottleneckLiteASKUNetPlusPlus -Slug bottleneckliteask -Augmentation $augmentation
if (-not $?) { throw 'X3 Bottleneck failed' }
Log 'DONE X3 both architectures'

$x3Bootstrap = Join-Path $project 'scripts\q2_x3_spatial_bootstrap.py'
Log 'START X3 10000-draw spatial bootstrap'
& $python $x3Bootstrap --draws 10000 | Add-Content -LiteralPath $log -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'X3 spatial bootstrap failed' }
Log 'DONE X3 spatial bootstrap'

$loroRunner = Join-Path $project 'scripts\run_q2_x3_loro_all6.ps1'
Log "START six-region LORO augmentation=$augmentation"
& $loroRunner -Augmentation $augmentation
if (-not $?) { throw 'LORO six-region runner failed' }
Log 'DONE six-region LORO training/evaluation'

$loroBootstrap = Join-Path $project 'scripts\loro_all6_spatial_bootstrap.py'
Log 'START six-region LORO 5000-draw spatial bootstrap'
& $python $loroBootstrap --draws 5000 | Add-Content -LiteralPath $log -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'LORO spatial bootstrap failed' }
Log 'Q2_MAJOR_REVISION_V2_COMPLETE'
