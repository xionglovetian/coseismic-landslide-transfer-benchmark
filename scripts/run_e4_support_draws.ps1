param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$runner=Join-Path $project 'scripts\run_target_fewshot.py'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$failures=@()
foreach($supportSeed in @(2026,777)){
  $support=Join-Path $project 'data\processed\fewshot_target_splits\fewshot_support_seed2026_777.csv'
  $eval=Join-Path $project 'data\processed\fewshot_target_splits\fewshot_eval_seed2026_777.csv'
  foreach($region in @('hokkaido_iburi_tobu','lombok','palu')){
    foreach($mode in @('full','decoder-only')){
      $name="bench_v2_e4_supportdraw${supportSeed}_resunet_${region}_20shot_${mode}_init42"
      $out=Join-Path $outputs $name
      if(Test-Path -LiteralPath (Join-Path $out 'metrics.json')){ continue }
      New-Item -ItemType Directory -Path $out -Force | Out-Null
      $stdout=Join-Path $logs "$name.stdout.log"
      $stderr=Join-Path $logs "$name.stderr.log"
      $args=@($runner,'--model','ResUNet','--mode',$mode,'--region',$region,'--shot','20','--seed','42','--support-seed',"$supportSeed",'--eval-seed',"$supportSeed",'--max-steps','400','--lr','5e-5','--min-lr','1e-6','--batch-size','4','--amp','--num-workers','2','--support-csv',$support,'--eval-csv',$eval,'--output-dir',$out)
      $p=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
      if($p.ExitCode -ne 0){ $failures += $name }
    }
  }
}
if($failures.Count -eq 0){ 'E4_SUPPORT_DRAWS_COMPLETE no failures' } else { "E4_SUPPORT_DRAWS_COMPLETE failures=$($failures -join ',')" }
