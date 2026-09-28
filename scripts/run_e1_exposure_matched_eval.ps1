param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$evaluator=Join-Path $project 'scripts\evaluate_exposure_matched.py'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$rawRoot=Join-Path $project 'reports\e1_raw'
$queueLog=Join-Path $logs 'e1_exposure_matched_eval.log'
$failures=@()
foreach($seed in @(42,2026,777)){
  foreach($regime in @('cas-single','cas-multistream','pooled')){
    $name="bench_v2_e1_${regime}_seed${seed}"
    $run=Join-Path $outputs $name
    $raw=Join-Path $rawRoot $name
    New-Item -ItemType Directory -Path $raw -Force | Out-Null
    $complete=$true
    foreach($step in @(0,280,560,840,1120)){ if(-not (Test-Path -LiteralPath (Join-Path $raw ("step_{0:D4}.json" -f $step)))){ $complete=$false; break } }
    if($complete){ Add-Content -LiteralPath $queueLog -Value "SKIP $name" -Encoding UTF8; continue }
    $args=@($evaluator,'--run-dir',$run,'--steps','0','280','560','840','1120','--input-size','128','--batch-size','16','--num-workers','4','--amp','--output-dir',$raw)
    Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$name) -Encoding UTF8
    & $python @args *>> $queueLog
    if($LASTEXITCODE -ne 0){ Add-Content -LiteralPath $queueLog -Value "FAIL $name exit=$LASTEXITCODE" -Encoding UTF8; $failures += $name } else { Add-Content -LiteralPath $queueLog -Value "DONE $name" -Encoding UTF8 }
  }
}
if($failures.Count -eq 0){ Add-Content -LiteralPath $queueLog -Value 'E1_EXPOSURE_MATCHED_EVAL_COMPLETE no failures' -Encoding UTF8 } else { Add-Content -LiteralPath $queueLog -Value "E1_EXPOSURE_MATCHED_EVAL_COMPLETE failures=$($failures -join ',')" -Encoding UTF8 }
