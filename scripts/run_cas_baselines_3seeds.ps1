param(
    [int]$InitialQueuePid = 0
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer = Join-Path $project 'scripts\train_unified.py'
$evaluator = Join-Path $project 'scripts\evaluate_external_benchmark.py'
$casRoot = Join-Path $project 'repos\AS-UNet\inputs\data_sum_moxizhen+bijie'
$manifest = Join-Path $project 'data\processed\benchmark_v2_regions_512\manifest_benchmark_v2.csv'
$logs = Join-Path $project 'logs'
$outputs = Join-Path $project 'outputs'
$rawDir = Join-Path $project 'reports\benchmark_v2_zero_shot_raw'
$queueLog = Join-Path $logs 'cas_baselines_3seeds_queue.log'
$failures = @()

New-Item -ItemType Directory -Path $logs -Force | Out-Null
New-Item -ItemType Directory -Path $rawDir -Force | Out-Null

function Write-QueueLog([string]$Message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $queueLog -Value $line -Encoding UTF8
}

function Wait-ForQueue([int]$ProcessId) {
    if ($ProcessId -le 0) { return }
    while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Write-QueueLog "Initial queue process $ProcessId has exited; starting CAS baseline queue."
}

function Invoke-Baseline([string]$Model, [string]$Slug, [int]$Seed) {
    $runName = "bench_v2_${Slug}_seed${Seed}"
    $runDir = Join-Path $outputs $runName
    $bestModel = Join-Path $runDir 'best_model.pth'
    $metrics = Join-Path $runDir 'metrics.json'
    if ((Test-Path -LiteralPath $bestModel) -and (Test-Path -LiteralPath $metrics)) {
        Write-QueueLog "SKIP $runName (completed)"
        return $true
    }

    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $stdout = Join-Path $logs "${runName}.stdout.log"
    $stderr = Join-Path $logs "${runName}.stderr.log"
    $arguments = @(
        $trainer,
        '--model', $Model,
        '--loss', 'PolyGHMDiceLoss',
        '--train-root', $casRoot,
        '--output-dir', $runDir,
        '--epochs', '50',
        '--batch-size', '32',
        '--input-size', '128',
        '--lr', '0.0001',
        '--weight-decay', '0.0001',
        '--num-workers', '4',
        '--seed', "$Seed",
        '--val-interval', '5',
        '--amp',
        '--skip-external-eval'
    )

    Write-QueueLog "START $runName"
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $project `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        Write-QueueLog "FAIL $runName exit=$($process.ExitCode) stderr=$stderr"
        $script:failures += $runName
        return $false
    }
    Write-QueueLog "DONE $runName"
    return $true
}

Wait-ForQueue $InitialQueuePid
Write-QueueLog 'CAS baseline queue started.'

$models = @(
    @{ Model = 'ResUNet';       Slug = 'resunet' },
    @{ Model = 'DeepLabV3Plus'; Slug = 'deeplabv3plus' },
    @{ Model = 'SegFormerB0';   Slug = 'segformerb0' }
)
$seeds = @(42, 2026, 777)
foreach ($model in $models) {
    foreach ($seed in $seeds) {
        Invoke-Baseline -Model $model.Model -Slug $model.Slug -Seed $seed | Out-Null
    }
}

$evalArgs = @(
    $evaluator,
    '--manifest', $manifest,
    '--raw-dir', $rawDir,
    '--input-size', '128',
    '--batch-size', '32',
    '--num-workers', '4',
    '--amp',
    '--models', 'BottleneckLiteASKUNetPlusPlus', 'ResUNet', 'DeepLabV3Plus', 'SegFormerB0',
    '--seeds', '42', '2026', '777'
)
Write-QueueLog 'START benchmark v2 zero-shot evaluation'
$evalLog = Join-Path $logs 'benchmark_v2_zero_shot_evaluation.stdout.log'
$oldErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python @evalArgs *> $evalLog
$evalExitCode = $LASTEXITCODE
$ErrorActionPreference = $oldErrorAction
Get-Content -LiteralPath $evalLog -ErrorAction SilentlyContinue | ForEach-Object { Write-QueueLog $_ }
if ($evalExitCode -ne 0) {
    Write-QueueLog "FAIL benchmark v2 evaluation exit=$evalExitCode"
    $script:failures += 'benchmark_v2_evaluation'
} else {
    & $python $summarizer 2>&1 | ForEach-Object { Write-QueueLog $_ }
    if ($LASTEXITCODE -ne 0) {
        Write-QueueLog "FAIL benchmark v2 summary exit=$LASTEXITCODE"
        $script:failures += 'benchmark_v2_summary'
    }
}

if ($failures.Count -eq 0) {
    Write-QueueLog 'CAS_BASELINE_QUEUE_COMPLETE no failures'
} else {
    Write-QueueLog "CAS_BASELINE_QUEUE_COMPLETE failures=$($failures -join ',')"
}
