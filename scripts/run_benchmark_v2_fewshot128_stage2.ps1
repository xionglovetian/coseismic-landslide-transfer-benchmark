param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$runner=Join-Path $project 'scripts\run_target_fewshot.py'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$queueLog=Join-Path $logs 'benchmark_v2_fewshot128_stage2_queue.log'
$splitDir=Join-Path $project 'data\processed\fewshot_target_splits'
$failures=@()
New-Item -ItemType Directory -Path $logs -Force | Out-Null

function Write-Log([string]$Message){
  Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date),$Message) -Encoding UTF8
}
function Invoke-Run([string]$Name,[string[]]$Arguments){
  $out=Join-Path $outputs $Name
  if(Test-Path -LiteralPath (Join-Path $out 'metrics.json')){ Write-Log "SKIP $Name"; return }
  New-Item -ItemType Directory -Path $out -Force | Out-Null
  $stdout=Join-Path $logs "$Name.stdout.log"
  $stderr=Join-Path $logs "$Name.stderr.log"
  Write-Log "START $Name"
  $process=Start-Process -FilePath $python -ArgumentList $Arguments -WorkingDirectory $project -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
  if($process.ExitCode -ne 0){ Write-Log "FAIL $Name exit=$($process.ExitCode) stderr=$stderr"; $script:failures += $Name } else { Write-Log "DONE $Name" }
}

$models=@(
  @{Name='ResUNet';Slug='resunet'},
  @{Name='SegFormerB0';Slug='segformerb0'}
)
$regions=@('hokkaido_iburi_tobu','lombok','palu')
$seeds=@(2026,777)
foreach($seed in $seeds){
  $support=Join-Path $splitDir 'fewshot_support_seed2026_777.csv'
  $eval=Join-Path $splitDir 'fewshot_eval_seed2026_777.csv'
  foreach($m in $models){
    foreach($region in $regions){
      $name="bench_v2_fewshot128_$($m.Slug)_${region}_zeroshot_seed$seed"
      $args=@($runner,'--model',$m.Name,'--mode','zero-shot','--region',$region,'--seed',"$seed",'--support-csv',$support,'--eval-csv',$eval,'--amp','--num-workers','2','--output-dir',(Join-Path $outputs $name))
      Invoke-Run -Name $name -Arguments $args
    }
  }
}
$configs=@(
  @{Model='ResUNet';Slug='resunet';Mode='full';Shot=5},
  @{Model='ResUNet';Slug='resunet';Mode='full';Shot=10},
  @{Model='ResUNet';Slug='resunet';Mode='full';Shot=20},
  @{Model='ResUNet';Slug='resunet';Mode='decoder-only';Shot=20},
  @{Model='SegFormerB0';Slug='segformerb0';Mode='full';Shot=10},
  @{Model='SegFormerB0';Slug='segformerb0';Mode='full';Shot=20}
)
foreach($seed in $seeds){
  $support=Join-Path $splitDir 'fewshot_support_seed2026_777.csv'
  $eval=Join-Path $splitDir 'fewshot_eval_seed2026_777.csv'
  foreach($cfg in $configs){
    foreach($region in $regions){
      $name="bench_v2_fewshot128_$($cfg.Slug)_${region}_$($cfg.Shot)shot_$($cfg.Mode)_seed$seed"
      $args=@($runner,'--model',$cfg.Model,'--mode',$cfg.Mode,'--region',$region,'--shot',"$($cfg.Shot)",'--seed',"$seed",'--support-csv',$support,'--eval-csv',$eval,'--max-steps','400','--lr','5e-5','--min-lr','1e-6','--batch-size','4','--amp','--num-workers','2','--output-dir',(Join-Path $outputs $name))
      Invoke-Run -Name $name -Arguments $args
    }
  }
}
if($failures.Count -eq 0){ Write-Log 'FEWSHOT128_STAGE2_COMPLETE no failures' } else { Write-Log "FEWSHOT128_STAGE2_COMPLETE failures=$($failures -join ',')" }
