param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$runner=Join-Path $project 'scripts\run_target_fewshot.py'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$support=Join-Path $project 'data\processed\fewshot_target_splits\fewshot_support_seed42.csv'
$eval=Join-Path $project 'data\processed\fewshot_target_splits\fewshot_eval_seed42.csv'
$queueLog=Join-Path $logs 'e2_early_late_adaptation_queue.log'
$failures=@()
foreach($epoch in @(20,30)){
  $init=Join-Path $project ("outputs\bench_v2_p4_epochwise_resunet_seed42\epoch_{0:D3}.pth" -f $epoch)
  foreach($region in @('hokkaido_iburi_tobu','lombok','palu')){
    foreach($mode in @('full','decoder-only')){
      $name="bench_v2_e2_epoch${epoch}_resunet_${region}_20shot_${mode}_seed42"
      $out=Join-Path $outputs $name
      if(Test-Path -LiteralPath (Join-Path $out 'metrics.json')){ Add-Content -LiteralPath $queueLog -Value "SKIP $name" -Encoding UTF8; continue }
      New-Item -ItemType Directory -Path $out -Force | Out-Null
      $stdout=Join-Path $logs "$name.stdout.log"
      $stderr=Join-Path $logs "$name.stderr.log"
      $args=@($runner,'--model','ResUNet','--mode',$mode,'--region',$region,'--shot','20','--seed','42','--max-steps','400','--lr','5e-5','--min-lr','1e-6','--batch-size','4','--amp','--num-workers','2','--init-checkpoint',$init,'--support-csv',$support,'--eval-csv',$eval,'--output-dir',$out)
      Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} START {1}" -f (Get-Date),$name) -Encoding UTF8
      $p=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
      if($p.ExitCode -ne 0){ Add-Content -LiteralPath $queueLog -Value "FAIL $name exit=$($p.ExitCode) stderr=$stderr" -Encoding UTF8; $failures += $name } else { Add-Content -LiteralPath $queueLog -Value "DONE $name" -Encoding UTF8 }
    }
  }
}
if($failures.Count -eq 0){ Add-Content -LiteralPath $queueLog -Value 'E2_EARLY_LATE_COMPLETE no failures' -Encoding UTF8 } else { Add-Content -LiteralPath $queueLog -Value "E2_EARLY_LATE_COMPLETE failures=$($failures -join ',')" -Encoding UTF8 }
