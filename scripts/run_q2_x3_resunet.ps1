param(
  [string]$Model = 'ResUNet',
  [string]$Slug = 'resunet',
  [ValidateSet('current','none','strong')]
  [string]$Augmentation = 'none',
  [int[]]$Seeds = @(42,2026,777),
  [ValidateSet('cas-single','cas-multistream','pooled')]
  [string[]]$Regimes = @('cas-single','cas-multistream','pooled'),
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer = Join-Path $project 'scripts\train_exposure_matched.py'
$evaluator = Join-Path $project 'scripts\evaluate_q2_checkpoint.py'
$tileEvaluator = Join-Path $project 'scripts\evaluate_q2_tile_metrics.py'
$outputs = Join-Path $project 'outputs'
$logs = Join-Path $project 'logs'
$rawRoot = Join-Path $project 'reports\q2_x3_raw'
$tileRoot = Join-Path $project 'reports\q2_x3_tiles'
$queueLog = Join-Path $logs 'q2_x3_both_architectures_queue.log'
New-Item -ItemType Directory -Path $rawRoot -Force | Out-Null
New-Item -ItemType Directory -Path $tileRoot -Force | Out-Null

function Invoke-LoggedProcess {
  param([string]$FilePath, [string[]]$Arguments, [string]$Stdout, [string]$Stderr)
  $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $project `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "Process failed with exit code $($process.ExitCode): $FilePath" }
}

foreach ($seed in $Seeds) {
  $initCheckpoint = Join-Path $outputs ("q2_x2_aug_{0}_{1}_seed{2}\best_model.pth" -f $Augmentation, $Slug, $seed)
  if (-not (Test-Path -LiteralPath $initCheckpoint)) { throw "Missing source checkpoint: $initCheckpoint" }
  foreach ($regime in $Regimes) {
    $runName = "q2_x3_${Slug}_${regime}_seed${seed}"
    $outDir = Join-Path $outputs $runName
    $metricsPath = Join-Path $outDir 'metrics.json'
    $checkpointPath = Join-Path $outDir 'final_model.pth'
    $rawPath = Join-Path $rawRoot "$runName.json"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    if ($Force -or -not (Test-Path -LiteralPath $metricsPath) -or -not (Test-Path -LiteralPath $checkpointPath)) {
      $stdout = Join-Path $logs "$runName.stdout.log"
      $stderr = Join-Path $logs "$runName.stderr.log"
      $trainArgs = @(
        $trainer,'--regime',$regime,'--model',$Model,'--init-checkpoint',$initCheckpoint,
        '--seed',"$seed",'--steps','1120','--checkpoint-interval','280','--batch-size','32',
        '--input-size','128','--lr','0.0001','--min-lr','0.000001','--weight-decay','0.0001',
        '--num-workers','4','--amp','--output-dir',$outDir
      )
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$runName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $trainArgs -Stdout $stdout -Stderr $stderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-TRAIN {1}" -f (Get-Date),$runName) -Encoding UTF8
    } else { Add-Content -LiteralPath $queueLog -Value "SKIP-TRAIN $runName" -Encoding UTF8 }

    if ($Force -or -not (Test-Path -LiteralPath $rawPath)) {
      $evalStdout = Join-Path $logs "$runName.eval.stdout.log"
      $evalStderr = Join-Path $logs "$runName.eval.stderr.log"
      $evalArgs = @(
        $evaluator,'--model',$Model,'--checkpoint',$checkpointPath,'--output',$rawPath,
        '--input-size','128','--batch-size','16','--num-workers','4','--amp',
        '--target-regions','hokkaido_iburi_tobu','lombok','palu'
      )
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $evalArgs -Stdout $evalStdout -Stderr $evalStderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
    } else { Add-Content -LiteralPath $queueLog -Value "SKIP-EVAL $runName" -Encoding UTF8 }

    $tilePath = Join-Path $tileRoot "$runName.tiles.csv"
    if ($Force -or -not (Test-Path -LiteralPath $tilePath)) {
      $role = if ($regime -eq 'pooled') { 'intervention' } elseif ($regime -eq 'cas-multistream') { 'placebo' } else { 'single-stream' }
      $tileStdout = Join-Path $logs "$runName.tile.stdout.log"
      $tileStderr = Join-Path $logs "$runName.tile.stderr.log"
      $tileArgs = @($tileEvaluator,'--model',$Model,'--checkpoint',$checkpointPath,'--output',$tilePath,'--regions','hokkaido_iburi_tobu','lombok','palu','--input-size','128','--batch-size','16','--num-workers','4','--amp','--run-id',$runName,'--regime',$regime,'--role',$role,'--seed',"$seed",'--step','1120')
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-TILE-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $tileArgs -Stdout $tileStdout -Stderr $tileStderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-TILE-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
    }  }
}
Add-Content -LiteralPath $queueLog -Value 'Q2_X3_BOTH_ARCHITECTURES_COMPLETE' -Encoding UTF8
$global:LASTEXITCODE = 0
