$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$eval=Join-Path $project 'scripts\evaluate_q2_checkpoint.py'
$outRoot=Join-Path $project 'reports\q2_x1_128ref_eval_raw'
New-Item -ItemType Directory -Path $outRoot -Force | Out-Null
foreach($seed in @(42,2026,777)){
  $run="q2_x1_128ref_none_segformerb0_seed$seed"
  $ckpt=Join-Path $project ("outputs\{0}\best_model.pth" -f $run)
  $out=Join-Path $outRoot ("{0}_input128.json" -f $run)
  if(-not (Test-Path -LiteralPath $ckpt)){throw "Missing checkpoint: $ckpt"}
  if(Test-Path -LiteralPath $out){continue}
  $args=@($eval,'--model','SegFormerB0','--checkpoint',$ckpt,'--output',$out,'--input-size','128','--batch-size','16','--num-workers','4','--amp')
  $p=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project -RedirectStandardOutput (Join-Path $project "logs\$run.128eval.stdout.log") -RedirectStandardError (Join-Path $project "logs\$run.128eval.stderr.log") -WindowStyle Hidden -Wait -PassThru
  if($p.ExitCode -ne 0){throw "128 ref eval failed for $run"}
}
