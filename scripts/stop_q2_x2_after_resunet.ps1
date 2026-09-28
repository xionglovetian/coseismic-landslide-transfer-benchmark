param([Parameter(Mandatory=$true)][int]$X2Pid)
$ErrorActionPreference='Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$log=Join-Path $project 'logs\q2_x2_extra_segformer_stop.log'
$queue=Join-Path $project 'logs\q2_x2_augmentation_queue.log'
function Log([string]$m){Add-Content -LiteralPath $log -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date),$m) -Encoding UTF8}
Log "WAIT ResUNet final eval pid=$X2Pid"
while($true){
  if(Get-Process -Id $X2Pid -ErrorAction SilentlyContinue){
    $text=Get-Content -LiteralPath $queue -Raw -Encoding UTF8
    if($text -match 'DONE-EVAL q2_x2_aug_strong_resunet_seed777'){break}
    Start-Sleep -Seconds 10
  } else { Log 'X2 parent exited before stop watcher'; exit 0 }
}
Start-Sleep -Seconds 5
$children=Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'q2_x2_aug_.*segformerb0' }
foreach($child in $children){Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue; Log "STOPPED SEGFORMER_CHILD=$($child.ProcessId)"}
if(Get-Process -Id $X2Pid -ErrorAction SilentlyContinue){Stop-Process -Id $X2Pid -Force; Log "STOPPED X2_PARENT=$X2Pid"}
Log 'RESUNET_X2_DONE_SEGFORMER_X2_STOPPED'
