param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer=Join-Path $project 'scripts\train_exposure_matched.py'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$queueLog=Join-Path $logs 'e1_exposure_matched_queue.log'
$failures=@()
New-Item -ItemType Directory -Path $logs -Force | Out-Null
foreach($seed in @(42,2026,777)){
  foreach($regime in @('cas-single','cas-multistream','pooled')){
    $name="bench_v2_e1_${regime}_seed${seed}"
    $out=Join-Path $outputs $name
    if((Test-Path -LiteralPath (Join-Path $out 'metrics.json')) -and (Test-Path -LiteralPath (Join-Path $out 'checkpoints\step_1120.pth'))){ Add-Content -LiteralPath $queueLog -Value "SKIP $name" -Encoding UTF8; continue }
    New-Item -ItemType Directory -Path $out -Force | Out-Null
    $stdout=Join-Path $logs "$name.stdout.log"
    $stderr=Join-Path $logs "$name.stderr.log"
    $args=@($trainer,'--regime',$regime,'--seed',"$seed",'--steps','1120','--checkpoint-interval','280','--batch-size','32','--input-size','128','--lr','0.0001','--min-lr','0.000001','--weight-decay','0.0001','--num-workers','4','--amp','--output-dir',$out)
    Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$name) -Encoding UTF8
    $p=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
    if($p.ExitCode -ne 0){ Add-Content -LiteralPath $queueLog -Value "FAIL $name exit=$($p.ExitCode) stderr=$stderr" -Encoding UTF8; $failures += $name } else { Add-Content -LiteralPath $queueLog -Value "DONE $name" -Encoding UTF8 }
  }
}
if($failures.Count -eq 0){ Add-Content -LiteralPath $queueLog -Value 'E1_EXPOSURE_MATCHED_COMPLETE no failures' -Encoding UTF8 } else { Add-Content -LiteralPath $queueLog -Value "E1_EXPOSURE_MATCHED_COMPLETE failures=$($failures -join ',')" -Encoding UTF8 }
