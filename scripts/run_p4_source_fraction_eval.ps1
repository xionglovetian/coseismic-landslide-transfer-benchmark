param([int]$InitialQueuePid=0)
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$evaluator=Join-Path $project 'scripts\evaluate_epochwise_overfit.py'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$rawRoot=Join-Path $project 'reports\p4_source_fraction_raw'
$queueLog=Join-Path $logs 'p4_source_fraction_eval.log'
if($InitialQueuePid -gt 0){ while(Get-Process -Id $InitialQueuePid -ErrorAction SilentlyContinue){ Start-Sleep -Seconds 60 } }
foreach($pct in @(25,50,75)){
  $run="bench_v2_p4_sourcefrac_${pct}_resunet_seed42"
  $checkpointDir=Join-Path $outputs $run
  $rawDir=Join-Path $rawRoot "frac_${pct}"
  New-Item -ItemType Directory -Path $rawDir -Force | Out-Null
  $args=@($evaluator,'--model','ResUNet','--checkpoint-dir',$checkpointDir,'--epochs','10','20','30','40','50','--input-size','128','--batch-size','16','--num-workers','4','--amp','--output-dir',$rawDir)
  Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$run) -Encoding UTF8
  & $python @args *>> $queueLog
  if($LASTEXITCODE -ne 0){ Add-Content -LiteralPath $queueLog -Value "FAIL $run exit=$LASTEXITCODE" -Encoding UTF8 } else { Add-Content -LiteralPath $queueLog -Value "DONE $run" -Encoding UTF8 }
}
Add-Content -LiteralPath $queueLog -Value 'P4_SOURCE_FRACTION_EVAL_COMPLETE' -Encoding UTF8
