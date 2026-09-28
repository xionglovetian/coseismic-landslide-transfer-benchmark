param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer=Join-Path $project 'scripts\train_unified.py'
$root=Join-Path $project 'repos\AS-UNet\inputs\data_sum_moxizhen+bijie'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$queueLog=Join-Path $logs 'e3_epochwise_replication_queue.log'
$failures=@()
$runs=@(
  @{Model='ResUNet';Slug='resunet';Seed=2026},
  @{Model='ResUNet';Slug='resunet';Seed=777},
  @{Model='SegFormerB0';Slug='segformerb0';Seed=42},
  @{Model='SegFormerB0';Slug='segformerb0';Seed=2026},
  @{Model='SegFormerB0';Slug='segformerb0';Seed=777}
)
foreach($r in $runs){
  $name="bench_v2_e3_epochwise_$($r.Slug)_seed$($r.Seed)"
  $out=Join-Path $outputs $name
  if((Test-Path -LiteralPath (Join-Path $out 'metrics.json')) -and (Test-Path -LiteralPath (Join-Path $out 'epoch_050.pth'))){ Add-Content -LiteralPath $queueLog -Value "SKIP $name" -Encoding UTF8; continue }
  New-Item -ItemType Directory -Path $out -Force | Out-Null
  $stdout=Join-Path $logs "$name.stdout.log"
  $stderr=Join-Path $logs "$name.stderr.log"
  $args=@($trainer,'--model',$r.Model,'--loss','PolyGHMDiceLoss','--train-root',$root,'--output-dir',$out,'--epochs','50','--batch-size','32','--input-size','128','--checkpoint-interval','10','--lr','0.0001','--weight-decay','0.0001','--num-workers','4','--seed',"$($r.Seed)",'--val-interval','5','--amp','--skip-external-eval')
  Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$name) -Encoding UTF8
  $p=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
  if($p.ExitCode -ne 0){ Add-Content -LiteralPath $queueLog -Value "FAIL $name exit=$($p.ExitCode) stderr=$stderr" -Encoding UTF8; $failures += $name } else { Add-Content -LiteralPath $queueLog -Value "DONE $name" -Encoding UTF8 }
}
if($failures.Count -eq 0){ Add-Content -LiteralPath $queueLog -Value 'E3_EPOCHWISE_REPLICATION_COMPLETE no failures' -Encoding UTF8 } else { Add-Content -LiteralPath $queueLog -Value "E3_EPOCHWISE_REPLICATION_COMPLETE failures=$($failures -join ',')" -Encoding UTF8 }
