param()
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$runner=Join-Path $project 'scripts\run_target_fewshot.py'
$logs=Join-Path $project 'logs'
$outputs=Join-Path $project 'outputs'
$queueLog=Join-Path $logs 'benchmark_v2_fewshot128_seed42_queue.log'
$failures=@()
New-Item -ItemType Directory -Path $logs -Force | Out-Null

function Write-Log([string]$Message){
  Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date),$Message) -Encoding UTF8
}

function Invoke-Run([string]$Name,[string[]]$Arguments){
  $out=Join-Path $outputs $Name
  if(Test-Path -LiteralPath (Join-Path $out 'metrics.json')){
    Write-Log "SKIP $Name"
    return
  }
  New-Item -ItemType Directory -Path $out -Force | Out-Null
  $stdout=Join-Path $logs "$Name.stdout.log"
  $stderr=Join-Path $logs "$Name.stderr.log"
  Write-Log "START $Name"
  $process=Start-Process -FilePath $python -ArgumentList $Arguments -WorkingDirectory $project -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
  if($process.ExitCode -ne 0){
    Write-Log "FAIL $Name exit=$($process.ExitCode) stderr=$stderr"
    $script:failures += $Name
  } else {
    Write-Log "DONE $Name"
  }
}

$models=@(
  @{Name='ResUNet';Slug='resunet'},
  @{Name='SegFormerB0';Slug='segformerb0'}
)
$regions=@('hokkaido_iburi_tobu','lombok','palu')
$shots=@(5,10,20)
$modes=@('full','decoder-only')

foreach($m in $models){
  foreach($region in $regions){
    $name="bench_v2_fewshot128_$($m.Slug)_${region}_zeroshot_seed42"
    $args=@($runner,'--model',$m.Name,'--mode','zero-shot','--region',$region,'--seed','42','--amp','--num-workers','2','--output-dir',(Join-Path $outputs $name))
    Invoke-Run -Name $name -Arguments $args
  }
}
foreach($m in $models){
  foreach($region in $regions){
    foreach($shot in $shots){
      foreach($mode in $modes){
        $name="bench_v2_fewshot128_$($m.Slug)_${region}_${shot}shot_${mode}_seed42"
        $args=@($runner,'--model',$m.Name,'--mode',$mode,'--region',$region,'--shot',"$shot",'--seed','42','--max-steps','400','--lr','5e-5','--min-lr','1e-6','--batch-size','4','--amp','--num-workers','2','--output-dir',(Join-Path $outputs $name))
        Invoke-Run -Name $name -Arguments $args
      }
    }
  }
}
if($failures.Count -eq 0){ Write-Log 'FEWSHOT128_SEED42_COMPLETE no failures' } else { Write-Log "FEWSHOT128_SEED42_COMPLETE failures=$($failures -join ',')" }
