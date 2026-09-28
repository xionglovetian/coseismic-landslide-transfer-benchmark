param([Parameter(Mandatory=$true)][int]$MasterPid)
$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$masterScript = Join-Path $project 'scripts\run_q2_major_revision_queue_v3.ps1'
$masterLog = Join-Path $project 'logs\q2_major_revision_v3_master.log'
$watchLog = Join-Path $project 'logs\q2_major_revision_watchdog.log'
$pidFile = Join-Path $project 'logs\q2_major_revision_v3_master.pid'
$currentPid = $MasterPid
function Log([string]$m) { Add-Content -LiteralPath $watchLog -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date),$m) -Encoding UTF8 }
Log "WATCH START master=$currentPid"
for ($attempt = 1; $attempt -le 10; $attempt++) {
  while (Get-Process -Id $currentPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
  $text = if (Test-Path -LiteralPath $masterLog) { Get-Content -LiteralPath $masterLog -Raw -Encoding UTF8 } else { '' }
  if ($text -match 'Q2_MAJOR_REVISION_V2_COMPLETE') { Log 'MASTER COMPLETE'; break }
  Log "MASTER EXIT WITHOUT COMPLETE marker attempt=$attempt"
  while (Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'q2_x2|q2_x1|train_exposure_matched|q2_x3|loro_all6' }) { Start-Sleep -Seconds 60 }
  Start-Sleep -Seconds 30
  $p = Start-Process -FilePath 'pwsh' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$masterScript) -WorkingDirectory $project -RedirectStandardOutput (Join-Path $project 'logs\q2_major_revision_v3_master.stdout.log') -RedirectStandardError (Join-Path $project 'logs\q2_major_revision_v3_master.stderr.log') -WindowStyle Hidden -PassThru
  $currentPid = $p.Id
  Set-Content -LiteralPath $pidFile -Value $currentPid -Encoding ASCII
  Log "RESTARTED MASTER pid=$currentPid attempt=$attempt"
}
