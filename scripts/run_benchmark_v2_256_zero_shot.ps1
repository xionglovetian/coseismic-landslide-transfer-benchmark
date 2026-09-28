param([int]$InitialQueuePid=0)
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }; $python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }; $trainer=Join-Path $project 'scripts\train_unified.py'; $casRoot=Join-Path $project 'repos\AS-UNet\inputs\data_sum_moxizhen+bijie'; $logs=Join-Path $project 'logs'; $outputs=Join-Path $project 'outputs'; $queueLog=Join-Path $logs 'benchmark_v2_256_zero_shot_queue.log'
if($InitialQueuePid -gt 0){ while(Get-Process -Id $InitialQueuePid -ErrorAction SilentlyContinue){ Start-Sleep -Seconds 60 } }
$models=@(@{Model='ResUNet';Slug='resunet'},@{Model='DeepLabV3Plus';Slug='deeplabv3plus'},@{Model='SegFormerB0';Slug='segformerb0'},@{Model='BottleneckLiteASKUNetPlusPlus';Slug='bottleneckliteask'})
foreach($m in $models){ foreach($seed in @(42,2026,777)){
  $run="bench_v2_256_$($m.Slug)_seed${seed}"; $out=Join-Path $outputs $run; if((Test-Path -LiteralPath (Join-Path $out 'best_model.pth')) -and (Test-Path -LiteralPath (Join-Path $out 'metrics.json'))){ Add-Content -LiteralPath $queueLog -Value "SKIP $run" -Encoding UTF8; continue }
  $args=@($trainer,'--model',$m.Model,'--loss','PolyGHMDiceLoss','--train-root',$casRoot,'--output-dir',$out,'--epochs','50','--batch-size','8','--input-size','256','--grad-accum-steps','4','--lr','0.0001','--weight-decay','0.0001','--num-workers','4','--seed',"$seed",'--val-interval','5','--amp','--skip-external-eval')
  Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$run) -Encoding UTF8
  $p=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project -RedirectStandardOutput (Join-Path $logs "${run}.stdout.log") -RedirectStandardError (Join-Path $logs "${run}.stderr.log") -WindowStyle Hidden -Wait -PassThru
  if($p.ExitCode -ne 0){ Add-Content -LiteralPath $queueLog -Value "FAIL $run exit=$($p.ExitCode)" -Encoding UTF8 } else { Add-Content -LiteralPath $queueLog -Value "DONE $run" -Encoding UTF8 }
}}
Add-Content -LiteralPath $queueLog -Value 'BENCHMARK_V2_256_ZERO_SHOT_COMPLETE' -Encoding UTF8
