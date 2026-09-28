param(
    [int]$InitialQueuePid = 0
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer = Join-Path $project 'scripts\train_multisource_dg.py'
$summarizer = Join-Path $project 'scripts\summarize_multisource_loro.py'
$logs = Join-Path $project 'logs'
$outputs = Join-Path $project 'outputs'
$initCheckpoint = Join-Path $outputs 'group_bottleneckliteaskunetpp_seed42\best_model.pth'
$queueLog = Join-Path $logs 'multisource_loro_secondary_queue.log'
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
    Write-QueueLog "Initial queue process $ProcessId has exited; starting secondary benchmark queue."
}

function Invoke-Experiment([string]$Region, [string]$RunRegion, [string]$MethodKey, [int]$Seed, [string]$Sampling, [double]$AlignmentWeight) {
    $runName = "dg_${RunRegion}_${MethodKey}_seed${Seed}"
    $runDir = Join-Path $outputs $runName
    $targetMetrics = Join-Path $runDir 'target_metrics.json'
    $allRegionMetrics = Join-Path $runDir 'all_region_metrics.json'
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
        '--alignment-weight', "$AlignmentWeight",
        '--sampling', $Sampling,
        '--validation-mode', 'cas_only',
        '--split-name', "benchmark_v1_casval_${MethodKey}_${Region}",
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

Wait-ForQueue $InitialQueuePid
Write-QueueLog "Secondary queue started."

$experiments = @(
    @{ Region = 'wenchuan';       RunRegion = 'wenchuan';       MethodKey = 'balanced_noalign'; Sampling = 'balanced'; AlignmentWeight = 0.0 },
    @{ Region = 'jiuzhai_valley'; RunRegion = 'jiuzhai_valley'; MethodKey = 'balanced_noalign'; Sampling = 'balanced'; AlignmentWeight = 0.0 },
    @{ Region = 'moxitaidi';      RunRegion = 'moxitaidi';      MethodKey = 'balanced_noalign'; Sampling = 'balanced'; AlignmentWeight = 0.0 },
    @{ Region = 'longxi_river';   RunRegion = 'longxi';         MethodKey = 'balanced_noalign'; Sampling = 'balanced'; AlignmentWeight = 0.0 },
    @{ Region = 'wenchuan';       RunRegion = 'wenchuan';       MethodKey = 'balanced_mmd';     Sampling = 'balanced'; AlignmentWeight = 0.2 },
    @{ Region = 'jiuzhai_valley'; RunRegion = 'jiuzhai_valley'; MethodKey = 'balanced_mmd';     Sampling = 'balanced'; AlignmentWeight = 0.2 },
    @{ Region = 'moxitaidi';      RunRegion = 'moxitaidi';      MethodKey = 'balanced_mmd';     Sampling = 'balanced'; AlignmentWeight = 0.2 },
    @{ Region = 'longxi_river';   RunRegion = 'longxi';         MethodKey = 'balanced_mmd';     Sampling = 'balanced'; AlignmentWeight = 0.2 }
)

foreach ($experiment in $experiments) {
    Invoke-Experiment -Region $experiment.Region -RunRegion $experiment.RunRegion -MethodKey $experiment.MethodKey `
        -Seed 42 -Sampling $experiment.Sampling -AlignmentWeight $experiment.AlignmentWeight | Out-Null
}

& $python $summarizer | ForEach-Object { Write-QueueLog $_ }
if ($failures.Count -eq 0) {
    Write-QueueLog 'SECONDARY_QUEUE_COMPLETE no failures'
} else {
    Write-QueueLog "SECONDARY_QUEUE_COMPLETE failures=$($failures -join ',')"
}
