param(
  [ValidateSet('current','none','strong')]
  [string]$Augmentation = 'none',
  [string[]]$Models = @('ResUNet','SegFormerB0'),
  [int[]]$Seeds = @(42,2026,777),
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer = Join-Path $project 'scripts\train_unified.py'
$evaluator = Join-Path $project 'scripts\evaluate_q2_checkpoint.py'
$trainRoot = Join-Path $project 'repos\AS-UNet\inputs\data_sum_moxizhen+bijie'
$outputs = Join-Path $project 'outputs'
$logs = Join-Path $project 'logs'
$rawRoot = Join-Path $project 'reports\q2_x1_resolution_raw'
$queueLog = Join-Path $logs 'q2_x1_resolution_queue.log'
New-Item -ItemType Directory -Path $rawRoot -Force | Out-Null
$slugByModel = @{ 'ResUNet' = 'resunet'; 'SegFormerB0' = 'segformerb0' }

function Invoke-LoggedProcess {
  param([string]$FilePath, [string[]]$Arguments, [string]$Stdout, [string]$Stderr)
  $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $project `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -Wait -PassThru
  if ($process.ExitCode -ne 0) { throw "Process failed with exit code $($process.ExitCode): $FilePath" }
}

foreach ($model in $Models) {
  if (-not $slugByModel.ContainsKey($model)) { throw "Unsupported model: $model" }
  $slug = $slugByModel[$model]
  foreach ($seed in $Seeds) {
    $runName = "q2_x1_256_${Augmentation}_${slug}_seed${seed}"
    $outDir = Join-Path $outputs $runName
    $metricsPath = Join-Path $outDir 'metrics.json'
    $bestPath = Join-Path $outDir 'best_model.pth'
    $rawPath = Join-Path $rawRoot "$runName.json"
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    if ($Force -or -not (Test-Path -LiteralPath $metricsPath) -or -not (Test-Path -LiteralPath $bestPath)) {
      $stdout = Join-Path $logs "$runName.stdout.log"
      $stderr = Join-Path $logs "$runName.stderr.log"
      $trainArgs = @(
        $trainer,'--model',$model,'--loss','PolyGHMDiceLoss','--train-root',$trainRoot,
        '--output-dir',$outDir,'--epochs','50','--batch-size','8','--grad-accum-steps','4',
        '--input-size','256','--checkpoint-interval','10','--lr','0.0001','--weight-decay','0.0001',
        '--num-workers','4','--seed',"$seed",'--val-interval','5','--amp','--skip-external-eval',
        '--augmentation',$Augmentation
      )
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$runName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $trainArgs -Stdout $stdout -Stderr $stderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-TRAIN {1}" -f (Get-Date),$runName) -Encoding UTF8
    } else { Add-Content -LiteralPath $queueLog -Value "SKIP-TRAIN $runName" -Encoding UTF8 }

    if ($Force -or -not (Test-Path -LiteralPath $rawPath)) {
      $evalStdout = Join-Path $logs "$runName.eval.stdout.log"
      $evalStderr = Join-Path $logs "$runName.eval.stderr.log"
      $evalArgs = @(
        $evaluator,'--model',$model,'--checkpoint',$bestPath,'--output',$rawPath,
        '--input-size','256','--batch-size','8','--num-workers','4','--amp'
      )
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $evalArgs -Stdout $evalStdout -Stderr $evalStderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
    } else { Add-Content -LiteralPath $queueLog -Value "SKIP-EVAL $runName" -Encoding UTF8 }
  }
}

# A1-c: 128-trained weights evaluated at 256 input.
$refRawRoot = Join-Path $project 'reports\q2_x1_a1c_raw'
New-Item -ItemType Directory -Path $refRawRoot -Force | Out-Null
foreach ($model in $Models) {
  $slug = $slugByModel[$model]
  foreach ($seed in $Seeds) {
    $refCheckpoint = ''
    if ($model -eq 'ResUNet') {
      $refCheckpoint = Join-Path $outputs ("q2_x2_aug_{0}_resunet_seed{1}\best_model.pth" -f $Augmentation, $seed)
    } elseif ($model -eq 'SegFormerB0') {
      $refRunName = ("q2_x1_128ref_{0}_segformerb0_seed{1}" -f $Augmentation, $seed)
      $refDir = Join-Path $outputs $refRunName
      $refCheckpoint = Join-Path $refDir 'best_model.pth'
      $refMetrics = Join-Path $refDir 'metrics.json'
      New-Item -ItemType Directory -Path $refDir -Force | Out-Null
      if ($Force -or -not (Test-Path -LiteralPath $refMetrics) -or -not (Test-Path -LiteralPath $refCheckpoint)) {
        $refStdout = Join-Path $logs "$refRunName.stdout.log"
        $refStderr = Join-Path $logs "$refRunName.stderr.log"
        $refTrainArgs = @($trainer,'--model',$model,'--loss','PolyGHMDiceLoss','--train-root',$trainRoot,'--output-dir',$refDir,'--epochs','50','--batch-size','32','--input-size','128','--checkpoint-interval','10','--lr','0.0001','--weight-decay','0.0001','--num-workers','4','--seed',"$seed",'--val-interval','5','--amp','--skip-external-eval','--augmentation',$Augmentation)
        Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-A1C-REF-TRAIN {1}" -f (Get-Date),$refRunName) -Encoding UTF8
        Invoke-LoggedProcess -FilePath $python -Arguments $refTrainArgs -Stdout $refStdout -Stderr $refStderr
        Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-A1C-REF-TRAIN {1}" -f (Get-Date),$refRunName) -Encoding UTF8
      }
    } else {
      throw "Unsupported A1-c reference model: $model"
    }
    if (-not (Test-Path -LiteralPath $refCheckpoint)) { throw "Missing A1-c reference checkpoint: $refCheckpoint" }
    $a1cName = ("q2_x1_a1c_{0}_{1}_seed{2}" -f $Augmentation, $slug, $seed)
    $a1cPath = Join-Path $refRawRoot "$a1cName.json"
    if ($Force -or -not (Test-Path -LiteralPath $a1cPath)) {
      $a1cStdout = Join-Path $logs "$a1cName.stdout.log"
      $a1cStderr = Join-Path $logs "$a1cName.stderr.log"
      $a1cArgs = @($evaluator,'--model',$model,'--checkpoint',$refCheckpoint,'--output',$a1cPath,'--input-size','256','--batch-size','8','--num-workers','4','--amp')
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-A1C {1}" -f (Get-Date),$a1cName) -Encoding UTF8
      Invoke-LoggedProcess -FilePath $python -Arguments $a1cArgs -Stdout $a1cStdout -Stderr $a1cStderr
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-A1C {1}" -f (Get-Date),$a1cName) -Encoding UTF8
    }
  }
}

Add-Content -LiteralPath $queueLog -Value 'Q2_X1_RESOLUTION_COMPLETE' -Encoding UTF8
$global:LASTEXITCODE = 0
