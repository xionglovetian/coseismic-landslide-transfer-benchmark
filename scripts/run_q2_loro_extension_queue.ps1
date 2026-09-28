param([Parameter(Mandatory=$true)][int]$MasterPid)
$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$log = Join-Path $project 'logs\q2_loro_extension_master.log'
function Log([string]$Message) { Add-Content -LiteralPath $log -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date),$Message) -Encoding UTF8 }
Log "WAIT main queue pid=$MasterPid"
while (Get-Process -Id $MasterPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
$mainLog = Get-Content -LiteralPath (Join-Path $project 'logs\q2_major_revision_master.log') -Raw -Encoding UTF8
if ($mainLog -notmatch 'Q2_MAJOR_REVISION_EXPERIMENTS_COMPLETE') { throw 'Main Q2 queue did not complete successfully' }
Log 'Main Q2 queue complete; start all-six spatial index'
$indexScript = Join-Path $project 'scripts\build_target_spatial_index.py'
& $python $indexScript --regions hokkaido_iburi_tobu lombok palu wenchuan longxi_river jiuzhai_valley --output (Join-Path $project 'data\processed\benchmark_v2_regions_512\target_spatial_index_all6.csv') | Add-Content -LiteralPath $log -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'All-six spatial index failed' }
Log 'Spatial index complete; start LORO training'
$loroRunner = Join-Path $project 'scripts\run_q2_x3_loro_all6.ps1'
& $loroRunner
if ($LASTEXITCODE -ne 0) { throw 'All-six LORO runner failed' }
Log 'LORO training complete; start spatial bootstrap'
$bootstrap = Join-Path $project 'scripts\loro_all6_spatial_bootstrap.py'
& $python $bootstrap --draws 5000 | Add-Content -LiteralPath $log -Encoding UTF8
if ($LASTEXITCODE -ne 0) { throw 'LORO spatial bootstrap failed' }
Log 'Q2_LORO_ALL6_EXTENSION_COMPLETE'
