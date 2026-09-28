param(
    [Alias('InitialPid')]
    [int]$InitialQueuePid = 0
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer = Join-Path $project 'scripts\train_multisource_dg.py'
$summarizer = Join-Path $project 'scripts\summarize_multisource_loro.py'
$logs = Join-Path $project 'logs'
$outputs = Join-Path $project 'outputs'
$queueLog = Join-Path $logs 'multisource_loro_uniform_queue.log'
$failures = @()

New-Item -ItemType Directory -Path $logs -Force | Out-Null

function Write-QueueLog([string]$Message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $queueLog -Value $line -Encoding UTF8
}

function Wait-ForInitialProcess([int]$ProcessId) {
    if ($ProcessId -le 0) { return }
    while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-QueueLog "Initial process $ProcessId has exited; starting queue."
}

function Invoke-UniformRun([string]$Region, [string]$RunRegion, [int]$Seed) {
    $initCheckpoint = Join-Path $outputs "group_bottleneckliteaskunetpp_seed${Seed}\best_model.pth"
    $runName = "dg_${RunRegion}_uniform_noalign_seed${Seed}"
    $runDir = Join-Path $outputs $runName
    $targetMetrics = Join-Path $runDir 'target_metrics.json'
    $bestModel = Join-Path $runDir 'best_model.pth'

    $validTargetMetrics = $false
    if (Test-Path -LiteralPath $targetMetrics) {
        try {
            $summary = Get-Content -LiteralPath $targetMetrics -Raw -Encoding UTF8 | ConvertFrom-Json
            $validTargetMetrics = -not [bool]$summary.invalidated
        } catch {
            $validTargetMetrics = $false
        }
    }
    if ((Test-Path -LiteralPath $bestModel) -and $validTargetMetrics) {
        Write-QueueLog "SKIP $runName (valid completed)"
        return $true
    }

    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $stdout = Join-Path $logs "${runName}.stdout.log"
    $stderr = Join-Path $logs "${runName}.stderr.log"
    $arguments = @(
        $trainer,
        '--target-region', $Region,
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
        '--validation-mode', 'cas_only',
        '--split-name', "benchmark_v1_casval_uniform_$Region",
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
    & $python $summarizer | ForEach-Object { Write-QueueLog $_ }
    if ($LASTEXITCODE -ne 0) {
        Write-QueueLog "FAIL summarizer after $runName exit=$LASTEXITCODE"
        $script:failures += "$runName-summary"
    }
    return $true
}

Wait-ForInitialProcess $InitialQueuePid
Write-QueueLog "Queue started."

$experiments = @(
    @{ Region = 'wenchuan';       RunRegion = 'wenchuan';       Seeds = @(42, 2026, 777) },
    @{ Region = 'jiuzhai_valley'; RunRegion = 'jiuzhai_valley'; Seeds = @(42, 2026, 777) },
    @{ Region = 'moxitaidi';      RunRegion = 'moxitaidi';      Seeds = @(42, 2026, 777) },
    @{ Region = 'longxi_river';   RunRegion = 'longxi';         Seeds = @(42, 2026, 777) }
)

foreach ($experiment in $experiments) {
    foreach ($seed in $experiment.Seeds) {
        Invoke-UniformRun -Region $experiment.Region -RunRegion $experiment.RunRegion -Seed $seed | Out-Null
    }
}

& $python $summarizer | ForEach-Object { Write-QueueLog $_ }
if ($failures.Count -eq 0) {
    Write-QueueLog 'QUEUE_COMPLETE no failures'
} else {
    Write-QueueLog "QUEUE_COMPLETE failures=$($failures -join ',')"
}
