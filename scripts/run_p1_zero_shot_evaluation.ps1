param([int]$WaitPid = 0)
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$evaluator=Join-Path $project 'scripts\evaluate_external_benchmark.py'
$summarizer=Join-Path $project 'scripts\summarize_benchmark_v2_zero_shot.py'
$manifest=Join-Path $project 'data\processed\benchmark_v2_regions_512\manifest_benchmark_v2.csv'
$rawDir=Join-Path $project 'reports\benchmark_v2_zero_shot_raw'
$log=Join-Path $project 'logs\p1_zero_shot_evaluation.log'
if($WaitPid -gt 0){ while(Get-Process -Id $WaitPid -ErrorAction SilentlyContinue){ Start-Sleep -Seconds 60 } }
$args=@($evaluator,'--manifest',$manifest,'--raw-dir',$rawDir,'--input-size','128','--batch-size','32','--num-workers','4','--amp','--models','BottleneckLiteASKUNetPlusPlus','ResUNet','DeepLabV3Plus','SegFormerB0','--seeds','42','2026','777')
$old=$ErrorActionPreference; $ErrorActionPreference='Continue'; & $python @args *> $log; $code=$LASTEXITCODE; $ErrorActionPreference=$old
if($code -eq 0){ & $python $summarizer 2>&1 | ForEach-Object { Add-Content -LiteralPath $log -Value $_ -Encoding UTF8 } }
Add-Content -LiteralPath $log -Value ("{0:yyyy-MM-dd HH:mm:ss} P1_EVALUATION_COMPLETE exit=$code" -f (Get-Date)) -Encoding UTF8
