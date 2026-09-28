param(
    [int]$InitialQueuePid = 0
)

$ErrorActionPreference = 'Stop'
$project = if ($env:LANDSLIDE_PROJECT_ROOT) { $env:LANDSLIDE_PROJECT_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..')).Path }
$python = if ($env:LANDSLIDE_PYTHON) { $env:LANDSLIDE_PYTHON } else { 'python' }
$trainer = Join-Path $project 'scripts\train_unified.py'
$casRoot = Join-Path $project 'repos\AS-UNet\inputs\data_sum_moxizhen+bijie'
$logs = Join-Path $project 'logs'
$outputs = Join-Path $project 'outputs'
$queueLog = Join-Path $logs 'cas_retrain20_control_queue.log'

function Write-QueueLog([string]$Message) {
    Add-Content -LiteralPath $queueLog -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $Message) -Encoding UTF8
}

if ($InitialQueuePid -gt 0) {
    while (Get-Process -Id $InitialQueuePid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
}

foreach ($seed in @(42, 2026, 777)) {
    $runName = "bench_v2_cas_retrain20_seed${seed}"
    $runDir = Join-Path $outputs $runName
    $initCheckpoint = Join-Path $outputs "group_bottleneckliteaskunetpp_seed${seed}\best_model.pth"
    $bestModel = Join-Path $runDir 'best_model.pth'
    $metrics = Join-Path $runDir 'metrics.json'
    if ((Test-Path -LiteralPath $bestModel) -and (Test-Path -LiteralPath $metrics)) {
        Write-QueueLog "SKIP $runName (completed)"
        continue
    }
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    $stdout = Join-Path $logs "${runName}.stdout.log"
    $stderr = Join-Path $logs "${runName}.stderr.log"
    $arguments = @(
        $trainer,
        '--model', 'BottleneckLiteASKUNetPlusPlus',
        '--loss', 'PolyGHMDiceLoss',
        '--train-root', $casRoot,
        '--init-checkpoint', $initCheckpoint,
        '--output-dir', $runDir,
        '--epochs', '20',
        '--batch-size', '32',
        '--input-size', '128',
        '--lr', '0.0001',
        '--weight-decay', '0.0001',
        '--num-workers', '4',
        '--seed', "$seed",
        '--val-interval', '5',
        '--amp',
        '--skip-external-eval'
    )
    Write-QueueLog "START $runName init=$initCheckpoint"
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $project -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        Write-QueueLog "FAIL $runName exit=$($process.ExitCode) stderr=$stderr"
    } else {
        Write-QueueLog "DONE $runName"
    }
}
Write-QueueLog 'CAS_RETRAIN20_CONTROL_QUEUE_COMPLETE'
