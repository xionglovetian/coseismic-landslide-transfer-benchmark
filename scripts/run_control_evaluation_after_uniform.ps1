param([int]$UniformQueuePid = 0)
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$eval = Join-Path $project 'scripts\evaluate_control_checkpoints.py'
$log = Join-Path $project 'logs\control_evaluation_after_uniform.log'
if ($UniformQueuePid -gt 0) {
    while (Get-Process -Id $UniformQueuePid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
}
Add-Content -LiteralPath $log -Value ("{0:yyyy-MM-dd HH:mm:ss} START control evaluation" -f (Get-Date)) -Encoding UTF8
& $python $eval --amp 2>&1 | ForEach-Object { Add-Content -LiteralPath $log -Value $_ -Encoding UTF8 }
if ($LASTEXITCODE -eq 0) {
    Add-Content -LiteralPath $log -Value ("{0:yyyy-MM-dd HH:mm:ss} DONE control evaluation" -f (Get-Date)) -Encoding UTF8
} else {
    Add-Content -LiteralPath $log -Value ("{0:yyyy-MM-dd HH:mm:ss} FAIL control evaluation exit=$LASTEXITCODE" -f (Get-Date)) -Encoding UTF8
}
