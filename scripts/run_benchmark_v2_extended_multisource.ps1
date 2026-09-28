param(
    [int]$InitialQueuePid = 0
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer = Join-Path $project 'scripts\train_multisource_dg.py'
$manifest = Join-Path $project 'data\processed\benchmark_v2_regions_512\manifest_benchmark_v2.csv'
$logs = Join-Path $project 'logs'
$outputs = Join-Path $project 'outputs'
$queueLog = Join-Path $logs 'benchmark_v2_extended_multisource_queue.log'
$failures = @()

New-Item -ItemType Directory -Path $logs -Force | Out-Null

function Write-QueueLog([string]$Message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $queueLog -Value $line -Encoding UTF8
}

function Wait-ForQueue([int]$ProcessId) {
    if ($ProcessId -le 0) { return }
    while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Write-QueueLog "Initial queue process $ProcessId has exited; starting extended multisource queue."
}

function Invoke-ExtendedRun([int]$Seed) {
    $initCheckpoint = Join-Path $project "outputs\group_bottleneckliteaskunetpp_seed${Seed}\best_model.pth"
    $runName = "bench_v2_extended_uniform_seed${Seed}"
    $runDir = Join-Path $outputs $runName
    $bestModel = Join-Path $runDir 'best_model.pth'
    $targetMetrics = Join-Path $runDir 'target_metrics.json'
    if ((Test-Path -LiteralPath $bestModel) -and (Test-Path -LiteralPath $targetMetrics)) {
        Write-QueueLog "SKIP $runName (completed)"
        return $true
    }

    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $stdout = Join-Path $logs "${runName}.stdout.log"
    $stderr = Join-Path $logs "${runName}.stderr.log"
    $arguments = @(
        $trainer,
        '--target-region', 'hokkaido_iburi_tobu',
        '--exclude-regions', 'hokkaido_iburi_tobu', 'lombok', 'palu',
        '--eval-regions', 'hokkaido_iburi_tobu', 'lombok', 'palu',
        '--validation-mode', 'cas_only',
        '--manifest', $manifest,
        '--split-name', 'benchmark_v2_fixed_sources',
        '--model', 'BottleneckLiteASKUNetPlusPlus',
        '--init-checkpoint', $initCheckpoint,
        '--epochs', '20',
        '--batch-size', '32',
        '--input-size', '128',
        '--lr', '0.0001',
        '--weight-decay', '0.0001',
        '--num-workers', '4',
        '--seed', "$Seed",
        '--alignment-weight', '0.0',
        '--sampling', 'uniform',
        '--output-dir', $runDir
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
Write-QueueLog 'Extended multisource queue started.'
foreach ($seed in @(42, 2026, 777)) {
    Invoke-ExtendedRun -Seed $seed | Out-Null
}
if ($failures.Count -eq 0) {
    Write-QueueLog 'EXTENDED_MULTISOURCE_QUEUE_COMPLETE no failures'
} else {
    Write-QueueLog "EXTENDED_MULTISOURCE_QUEUE_COMPLETE failures=$($failures -join ',')"
}
