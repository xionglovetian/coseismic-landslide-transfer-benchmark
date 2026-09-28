$ErrorActionPreference = 'Stop'
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$trainer = Join-Path $project 'scripts\train_unified.py'
$external = Join-Path $project 'data\processed\resunet_bfa_grouped_224'
$logDir = Join-Path $project 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$seeds = 42, 2026, 777
foreach ($seed in $seeds) {
    $output = Join-Path $project "outputs\group_asunet_seed$seed"
    $log = Join-Path $logDir "group_asunet_seed$seed.log"
    "[$(Get-Date -Format s)] START seed=$seed" | Out-File -LiteralPath $log -Encoding utf8
    & $python $trainer --model AS_UNet --loss PolyGHMDiceLoss --epochs 50 --batch-size 16 --input-size 128 --lr 0.0001 --num-workers 4 --seed $seed --external-root $external --output-dir $output *>> $log
    $code = $LASTEXITCODE
    "[$(Get-Date -Format s)] END seed=$seed exit=$code" | Out-File -LiteralPath $log -Append -Encoding utf8
    if ($code -ne 0) { exit $code }
}
"[$(Get-Date -Format s)] ALL DONE" | Out-File -LiteralPath (Join-Path $logDir 'group_asunet_seeds.status') -Encoding utf8
