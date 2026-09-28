param(
  [ValidateSet('current','none','strong')]
  [string]$Augmentation = 'none',
  [string[]]$Targets = @('hokkaido_iburi_tobu','lombok','palu','wenchuan','longxi_river','jiuzhai_valley'),
  [int[]]$Seeds = @(42,2026,777),
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer = Join-Path $project 'scripts\train_exposure_matched_loro.py'
$tileEvaluator = Join-Path $project 'scripts\evaluate_q2_tile_metrics.py'
$outputs = Join-Path $project 'outputs'
$logs = Join-Path $project 'logs'
$tileRoot = Join-Path $project 'reports\q2_x3_loro_tiles'
$queueLog = Join-Path $logs 'q2_x3_loro_all6_queue.log'
New-Item -ItemType Directory -Path $tileRoot -Force | Out-Null

function Invoke-LoggedProcess {
  param([string]$FilePath, [string[]]$Arguments, [string]$Stdout, [string]$Stderr)
  $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $project -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "Process failed with exit code $($process.ExitCode): $FilePath" }
}

foreach ($seed in $Seeds) {
  foreach ($control in @(
    @{Regime='cas-multistream'; Role='placebo'},
    @{Regime='cas-single'; Role='single-stream'}
  )) {
    $regime = $control.Regime
    $runName = "e1__{0}__seed{1}" -f $regime, $seed
    $checkpoint = Join-Path $outputs ("q2_x3_bottleneckliteask_{0}_seed{1}\final_model.pth" -f $regime, $seed)
    $tilePath = Join-Path $tileRoot "$runName.tiles.csv"
    if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing control checkpoint: $checkpoint" }
    if ($Force -or -not (Test-Path -LiteralPath $tilePath)) {
      $stdout = Join-Path $logs "$runName.tile.stdout.log"
      $stderr = Join-Path $logs "$runName.tile.stderr.log"
      $args = @($tileEvaluator,'--model','BottleneckLiteASKUNetPlusPlus','--checkpoint',$checkpoint,'--output',$tilePath,'--input-size','128','--batch-size','16','--num-workers','4','--amp','--run-id',$runName,'--regime',$regime,'--role',$control.Role,'--seed',"$seed",'--step','1120')
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $args -Stdout $stdout -Stderr $stderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
    }
  }
}

foreach ($target in $Targets) {
  foreach ($seed in $Seeds) {
    $runName = "q2_x3_loro_{0}_seed{1}" -f $target, $seed
    $outDir = Join-Path $outputs $runName
    $metricsPath = Join-Path $outDir 'metrics.json'
    $checkpointPath = Join-Path $outDir 'final_model.pth'
    $initCheckpoint = Join-Path $outputs ("q2_x2_aug_{0}_bottleneckliteask_seed{1}\best_model.pth" -f $Augmentation, $seed)
    if (-not (Test-Path -LiteralPath $initCheckpoint)) { throw "Missing init checkpoint: $initCheckpoint" }
    if ($Force -or -not (Test-Path -LiteralPath $metricsPath) -or -not (Test-Path -LiteralPath $checkpointPath)) {
      $stdout = Join-Path $logs "$runName.stdout.log"
      $stderr = Join-Path $logs "$runName.stderr.log"
      $args = @($trainer,'--target-region',$target,'--model','BottleneckLiteASKUNetPlusPlus','--init-checkpoint',$initCheckpoint,'--seed',"$seed",'--steps','1120','--checkpoint-interval','280','--batch-size','32','--input-size','128','--lr','0.0001','--min-lr','0.000001','--weight-decay','0.0001','--num-workers','4','--amp','--output-dir',$outDir)
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-TRAIN {1}" -f (Get-Date),$runName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $args -Stdout $stdout -Stderr $stderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-TRAIN {1}" -f (Get-Date),$runName) -Encoding UTF8
    }
    $tilePath = Join-Path $tileRoot "$runName.tiles.csv"
    if ($Force -or -not (Test-Path -LiteralPath $tilePath)) {
      $stdout = Join-Path $logs "$runName.tile.stdout.log"
      $stderr = Join-Path $logs "$runName.tile.stderr.log"
      $args = @($tileEvaluator,'--model','BottleneckLiteASKUNetPlusPlus','--checkpoint',$checkpointPath,'--output',$tilePath,'--regions',$target,'--input-size','128','--batch-size','16','--num-workers','4','--amp','--run-id',$runName,'--regime','pooled_loro','--role','intervention','--seed',"$seed",'--step','1120')
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $args -Stdout $stdout -Stderr $stderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
    }
  }
}
Add-Content -LiteralPath $queueLog -Value 'Q2_X3_LORO_ALL6_COMPLETE' -Encoding UTF8
$global:LASTEXITCODE = 0
