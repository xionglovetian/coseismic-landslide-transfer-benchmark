param(
  [string[]]$Models = @('ResUNet','BottleneckLiteASKUNetPlusPlus'),
  [int[]]$Seeds = @(42,2026,777),
  [ValidateSet('current','none','strong')]
  [string[]]$Augmentations = @('current','none','strong'),
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
$rawRoot = Join-Path $project 'reports\q2_x2_augmentation_raw'
$queueLog = Join-Path $logs 'q2_x2_augmentation_queue.log'

New-Item -ItemType Directory -Path $logs -Force | Out-Null
New-Item -ItemType Directory -Path $rawRoot -Force | Out-Null

$slugByModel = @{
  'ResUNet' = 'resunet'
  'SegFormerB0' = 'segformerb0'
  'BottleneckLiteASKUNetPlusPlus' = 'bottleneckliteask'
}

function Invoke-LoggedProcess {
  param(
    [string]$FilePath,
    [string[]]$Arguments,
    [string]$Stdout,
    [string]$Stderr
  )
  $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $project `
    -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -Wait -PassThru
  if ($process.ExitCode -ne 0) {
    throw "Process failed with exit code $($process.ExitCode): $FilePath"
  }
}

foreach ($model in $Models) {
  if (-not $slugByModel.ContainsKey($model)) { throw "Unsupported model: $model" }
  $slug = $slugByModel[$model]
  foreach ($seed in $Seeds) {
    foreach ($aug in $Augmentations) {
      $runName = "q2_x2_aug_${aug}_${slug}_seed${seed}"
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
          '--output-dir',$outDir,'--epochs','50','--batch-size','32','--input-size','128',
          '--checkpoint-interval','10','--lr','0.0001','--weight-decay','0.0001',
          '--num-workers','4','--seed',"$seed",'--val-interval','5','--amp','--skip-external-eval',
          '--augmentation',$aug
        )
        Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$runName) -Encoding UTF8
        Invoke-LoggedProcess -FilePath $python -Arguments $trainArgs -Stdout $stdout -Stderr $stderr
        Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-TRAIN {1}" -f (Get-Date),$runName) -Encoding UTF8
      } else {
        Add-Content -LiteralPath $queueLog -Value "SKIP-TRAIN $runName" -Encoding UTF8
      }

      if ($runName -eq 'q2_x2_aug_current_resunet_seed42') {
        $metric = Get-Content -LiteralPath $metricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $gateDelta = [math]::Abs([double]$metric.cas_val.iou - 0.6771664317674468)
        Add-Content -LiteralPath $queueLog -Value ("GATE-D source-val={0:F6} delta={1:F6}" -f [double]$metric.cas_val.iou,$gateDelta) -Encoding UTF8
        if ($gateDelta -gt 0.014) {
          throw "Gate-D failed for ${runName}: source val delta $gateDelta exceeds 0.014"
        }
      }

      if ($Force -or -not (Test-Path -LiteralPath $rawPath)) {
        $evalStdout = Join-Path $logs "$runName.eval.stdout.log"
        $evalStderr = Join-Path $logs "$runName.eval.stderr.log"
        $evalArgs = @(
          $evaluator,'--model',$model,'--checkpoint',$bestPath,'--output',$rawPath,
          '--input-size','128','--batch-size','16','--num-workers','4','--amp'
        )
        Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
        Invoke-LoggedProcess -FilePath $python -Arguments $evalArgs -Stdout $evalStdout -Stderr $evalStderr
        Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE-EVAL {1}" -f (Get-Date),$runName) -Encoding UTF8
      } else {
        Add-Content -LiteralPath $queueLog -Value "SKIP-EVAL $runName" -Encoding UTF8
      }
    }
  }
}

Add-Content -LiteralPath $queueLog -Value 'Q2_X2_AUGMENTATION_COMPLETE' -Encoding UTF8
$global:LASTEXITCODE = 0
