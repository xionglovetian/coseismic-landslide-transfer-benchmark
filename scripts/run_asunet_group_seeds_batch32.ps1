$ErrorActionPreference = 'Stop'
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$trainer = Join-Path $project 'scripts\train_unified.py'
$external = Join-Path $project 'data\processed\resunet_bfa_grouped_224'
$logDir = Join-Path $project 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$seeds = 42, 2026, 777
foreach ($seed in $seeds) {
    $output = Join-Path $project "outputs\group_asunet32_seed$seed"
    $metrics = Join-Path $output 'metrics.json'
    $log = Join-Path $logDir "group_asunet32_seed$seed.log"
    if (Test-Path -LiteralPath $metrics) {
        Write-Host "[$(Get-Date -Format s)] SKIP completed AS_UNet batch32 seed=$seed"
        continue
    }
    Write-Host "[$(Get-Date -Format s)] START AS_UNet batch32 seed=$seed"
    "[$(Get-Date -Format s)] START model=AS_UNet seed=$seed batch=32 epochs=50" | Out-File -LiteralPath $log -Encoding utf8
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    & $python $trainer --model AS_UNet --loss PolyGHMDiceLoss --epochs 50 --batch-size 32 --input-size 128 --lr 0.0001 --weight-decay 0.0001 --num-workers 4 --seed $seed --val-interval 5 --amp --external-root $external --output-dir $output *>> $log
    $code = $LASTEXITCODE
    $timer.Stop()
    "[$(Get-Date -Format s)] END model=AS_UNet seed=$seed exit=$code elapsed_min=$([math]::Round($timer.Elapsed.TotalMinutes,2))" | Out-File -LiteralPath $log -Append -Encoding utf8
    if ($code -ne 0) {
        Write-Host "[$(Get-Date -Format s)] FAILED AS_UNet batch32 seed=$seed exit=$code"
        exit $code
    }
    Write-Host "[$(Get-Date -Format s)] DONE AS_UNet batch32 seed=$seed elapsed_min=$([math]::Round($timer.Elapsed.TotalMinutes,2))"
}
"[$(Get-Date -Format s)] ALL DONE" | Out-File -LiteralPath (Join-Path $logDir 'group_asunet32_seeds.status') -Encoding utf8
Write-Host "[$(Get-Date -Format s)] ALL DONE AS_UNet batch32"
