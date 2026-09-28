param(
  [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$ArchiveRoot = '',
  [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
if (-not $ArchiveRoot) { $ArchiveRoot = Join-Path $ProjectRoot 'data\raw\cas_archives' }
if (-not $Python) {
  if ($env:LANDSLIDE_PYTHON) { $Python = $env:LANDSLIDE_PYTHON } else { $Python = 'python' }
}
$env:LANDSLIDE_PROJECT_ROOT = $ProjectRoot
$env:LANDSLIDE_CAS_ARCHIVE_ROOT = $ArchiveRoot

function Invoke-Step([string]$Script, [string[]]$Arguments) {
  Write-Host "RUN $Script $($Arguments -join ' ')"
  & $Python (Join-Path $ProjectRoot $Script) @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Step failed: $Script" }
}

Invoke-Step 'scripts\prepare_external_regions.py' @('--source-root', $ArchiveRoot, '--output-root', (Join-Path $ProjectRoot 'data\processed\external_regions_512'))
Invoke-Step 'scripts\prepare_benchmark_v2.py' @('--archive-root', $ArchiveRoot, '--existing-manifest', (Join-Path $ProjectRoot 'data\processed\external_regions_512\manifest_external.csv'), '--output-root', (Join-Path $ProjectRoot 'data\processed\benchmark_v2_regions_512'))
Invoke-Step 'scripts\build_target_spatial_index.py' @()
Invoke-Step 'scripts\make_fewshot_target_splits.py' @('--seeds','42')
Invoke-Step 'scripts\make_fewshot_target_splits.py' @('--seeds','2026','777')
Invoke-Step 'scripts\checks\check_inputs.py' @('--verify-hashes')

Write-Host 'PRIMARY DATA PREPARATION COMPLETE'