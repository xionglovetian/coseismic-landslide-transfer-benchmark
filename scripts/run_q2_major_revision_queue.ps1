param(
  [Parameter(Mandatory=$true)][int]$X2QueuePid
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$logs = Join-Path $project 'logs'
$masterLog = Join-Path $logs 'q2_major_revision_master.log'

function Log([string]$Message) {
  Add-Content -LiteralPath $masterLog -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date),$Message) -Encoding UTF8
}

Log "WAIT X2 pid=$X2QueuePid"
while (Get-Process -Id $X2QueuePid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
Log "X2 queue process exited"

$summarizer = Join-Path $project 'scripts\summarize_q2_x2_augmentation.py'
& $python $summarizer | Add-Content -LiteralPath $masterLog -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'X2 summarization failed' }

$rawFiles = Get-ChildItem -LiteralPath (Join-Path $project 'reports\q2_x2_augmentation_raw') -Filter 'q2_x2_aug_*.json' -File
if ($rawFiles.Count -ne 18) { throw "X2 incomplete: $($rawFiles.Count)/18 result files" }
$grouped = Import-Csv -LiteralPath (Join-Path $project 'reports\q2_x2_augmentation\grouped.csv')
$pairs = Import-Csv -LiteralPath (Join-Path $project 'reports\q2_x2_augmentation\paired_summary.csv')
$nonePairs = $pairs | Where-Object { $_.comparison -eq 'none-current' } | Select-Object -First 1
if (-not $nonePairs) { throw 'Missing none-current paired summary' }
$nonInferior = $true
foreach ($model in @('ResUNet','SegFormerB0')) {
  $current = $grouped | Where-Object { $_.model -eq $model -and $_.augmentation -eq 'current' } | Select-Object -First 1
  $none = $grouped | Where-Object { $_.model -eq $model -and $_.augmentation -eq 'none' } | Select-Object -First 1
  if (-not $current -or -not $none) { $nonInferior = $false; continue }
  $delta = [double]$none.source_val_iou_mean - [double]$current.source_val_iou_mean
  Log "SELECTION model=$model source-none-minus-current=$delta"
  if ($delta -lt -0.010) { $nonInferior = $false }
}
$overallDelta = [double]$nonePairs.source_val_iou_delta_mean
$augmentation = if ($nonInferior -and $overallDelta -ge 0) { 'none' } else { 'current' }
Log "SELECTED augmentation=$augmentation overall-source-delta=$overallDelta"

$x1 = Join-Path $project 'scripts\run_q2_x1_resolution.ps1'
Log "START X1 augmentation=$augmentation"
& $x1 -Augmentation $augmentation
if ($LASTEXITCODE -ne 0) { throw 'X1 queue failed' }
Log 'DONE X1'

$x3 = Join-Path $project 'scripts\run_q2_x3_resunet.ps1'
Log 'START X3 ResUNet'
& $x3
if ($LASTEXITCODE -ne 0) { throw 'X3 queue failed' }
Log 'DONE X3'

$x5 = Join-Path $project 'scripts\route1_spatial_cluster_bootstrap.py'
Log 'START X5 full 10000-draw spatial bootstrap'
& $python $x5 --draws 10000 | Add-Content -LiteralPath $masterLog -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'X5 bootstrap failed' }
Log 'DONE X5'
Log 'Q2_MAJOR_REVISION_EXPERIMENTS_COMPLETE'
