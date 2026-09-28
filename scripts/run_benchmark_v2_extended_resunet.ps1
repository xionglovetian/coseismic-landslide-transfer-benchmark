param([int]$InitialQueuePid = 0)
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer=Join-Path $project 'scripts\train_multisource_dg.py'
$manifest=Join-Path $project 'data\processed\benchmark_v2_regions_512\manifest_benchmark_v2.csv'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$queueLog=Join-Path $logs 'benchmark_v2_extended_resunet_queue.log'
if($InitialQueuePid -gt 0){ while(Get-Process -Id $InitialQueuePid -ErrorAction SilentlyContinue){ Start-Sleep -Seconds 60 } }
foreach($seed in @(42,2026,777)){
  $runName="bench_v2_extended_resunet_seed${seed}"
  $runDir=Join-Path $outputs $runName
  $init=Join-Path $outputs "bench_v2_resunet_seed${seed}\best_model.pth"
  if(-not (Test-Path -LiteralPath $init)){ throw "Missing init checkpoint: $init" }
  if((Test-Path -LiteralPath (Join-Path $runDir 'best_model.pth')) -and (Test-Path -LiteralPath (Join-Path $runDir 'target_metrics.json'))){ Add-Content -LiteralPath $queueLog -Value "SKIP $runName" -Encoding UTF8; continue }
  $args=@($trainer,'--target-region','hokkaido_iburi_tobu','--exclude-regions','hokkaido_iburi_tobu','lombok','palu','--eval-regions','hokkaido_iburi_tobu','lombok','palu','--validation-mode','cas_only','--manifest',$manifest,'--split-name',"benchmark_v2_fixed_sources_resunet_${seed}",'--model','ResUNet','--init-checkpoint',$init,'--epochs','20','--batch-size','32','--input-size','128','--lr','0.0001','--weight-decay','0.0001','--num-workers','4','--seed',"$seed",'--alignment-weight','0.0','--sampling','uniform','--output-dir',$runDir)
  Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$runName) -Encoding UTF8
  $p=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project -RedirectStandardOutput (Join-Path $logs "${runName}.stdout.log") -RedirectStandardError (Join-Path $logs "${runName}.stderr.log") -WindowStyle Hidden -Wait -PassThru
  if($p.ExitCode -ne 0){ Add-Content -LiteralPath $queueLog -Value "FAIL $runName exit=$($p.ExitCode)" -Encoding UTF8 } else { Add-Content -LiteralPath $queueLog -Value "DONE $runName" -Encoding UTF8 }
}
Add-Content -LiteralPath $queueLog -Value 'EXTENDED_RESUNET_QUEUE_COMPLETE' -Encoding UTF8
