param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer=Join-Path $project 'scripts\train_unified.py'
$root=Join-Path $project 'repos\AS-UNet\inputs\data_sum_moxizhen+bijie'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$queueLog=Join-Path $logs 'p4_source_fraction_queue.log'
$failures=@()
$fractions=@(0.25,0.50,0.75)
foreach($fraction in $fractions){
  $pct=[int]($fraction*100)
  $name="bench_v2_p4_sourcefrac_${pct}_resunet_seed42"
  $out=Join-Path $outputs $name
  if(Test-Path -LiteralPath (Join-Path $out 'metrics.json')){ Add-Content -LiteralPath $queueLog -Value "SKIP $name" -Encoding UTF8; continue }
  New-Item -ItemType Directory -Path $out -Force | Out-Null
  $stdout=Join-Path $logs "$name.stdout.log"
  $stderr=Join-Path $logs "$name.stderr.log"
  $args=@($trainer,'--model','ResUNet','--loss','PolyGHMDiceLoss','--train-root',$root,'--output-dir',$out,'--epochs','50','--batch-size','32','--input-size','128','--checkpoint-interval','10','--lr','0.0001','--weight-decay','0.0001','--num-workers','4','--seed','42','--val-interval','5','--amp','--skip-external-eval','--train-fraction',"$fraction",'--subset-seed','42')
  Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$name) -Encoding UTF8
  $p=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
  if($p.ExitCode -ne 0){ Add-Content -LiteralPath $queueLog -Value "FAIL $name exit=$($p.ExitCode)" -Encoding UTF8; $failures += $name } else { Add-Content -LiteralPath $queueLog -Value "DONE $name" -Encoding UTF8 }
}
if($failures.Count -eq 0){ Add-Content -LiteralPath $queueLog -Value 'P4_SOURCE_FRACTION_COMPLETE no failures' -Encoding UTF8 } else { Add-Content -LiteralPath $queueLog -Value "P4_SOURCE_FRACTION_COMPLETE failures=$($failures -join ',')" -Encoding UTF8 }
