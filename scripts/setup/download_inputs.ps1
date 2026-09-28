param(
  [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$ArchiveRoot = '',
  [string]$AsUnetRoot = '',
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
if (-not $ArchiveRoot) { $ArchiveRoot = Join-Path $ProjectRoot 'data\raw\cas_archives' }
if (-not $AsUnetRoot) { $AsUnetRoot = Join-Path $ProjectRoot 'repos\AS-UNet' }
New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $AsUnetRoot) -Force | Out-Null

$AsUnetUrl = 'https://github.com/Youhaoran1512/AS-UNet.git'
$AsUnetCommit = '1df78b3ea315fc9744db6de9f2917f9f309c7218'
if (-not (Test-Path -LiteralPath (Join-Path $AsUnetRoot '.git'))) {
  New-Item -ItemType Directory -Path $AsUnetRoot -Force | Out-Null
  git -C $AsUnetRoot init
  git -C $AsUnetRoot remote add origin $AsUnetUrl
  git -C $AsUnetRoot fetch --depth 1 origin $AsUnetCommit
  git -C $AsUnetRoot checkout --detach FETCH_HEAD
}
$actualAsUnet = (git -C $AsUnetRoot rev-parse HEAD).Trim()
if ($actualAsUnet -ne $AsUnetCommit) {
  git -C $AsUnetRoot fetch --depth 1 origin $AsUnetCommit
  git -C $AsUnetRoot checkout --detach $AsUnetCommit
  $actualAsUnet = (git -C $AsUnetRoot rev-parse HEAD).Trim()
}
if ($actualAsUnet -ne $AsUnetCommit) { throw "AS-UNet commit mismatch: $actualAsUnet" }

$RecordId = 10294997
$record = Invoke-RestMethod -Uri "https://zenodo.org/api/records/$RecordId" -TimeoutSec 60
$requiredNames = @(
  'Hokkaido Iburi-Tobu.zip',
  'Lombok.zip',
  'palu.zip',
  'Wenchuan.zip',
  'Jiuzhai valley (UAV-0.2m).zip',
  'Moxitaidi (UAV-0.6m).zip',
  'Longxi River（UAV）.zip',
  'CAS Landslide Dataset README.html'
)
foreach ($name in $requiredNames) {
  $entry = $record.files | Where-Object { $_.key -eq $name } | Select-Object -First 1
  if (-not $entry) { throw "Zenodo file not found: $name" }
  $destination = Join-Path $ArchiveRoot $name
  if ($Force -or -not (Test-Path -LiteralPath $destination)) {
    $temporary = "$destination.download"
    Write-Host "Downloading $name"
    Invoke-WebRequest -Uri $entry.links.self -OutFile $temporary -TimeoutSec 3600
    Move-Item -LiteralPath $temporary -Destination $destination -Force
  }
  $actualSize = (Get-Item -LiteralPath $destination).Length
  if ($actualSize -ne [long]$entry.size) {
    throw "Size mismatch for $name`: expected $($entry.size), got $actualSize"
  }
  if ($entry.checksum -like 'md5:*') {
    $expectedMd5 = $entry.checksum.Substring(4).ToUpperInvariant()
    $actualMd5 = (Get-FileHash -LiteralPath $destination -Algorithm MD5).Hash.ToUpperInvariant()
    if ($actualMd5 -ne $expectedMd5) { throw "MD5 mismatch for $name" }
  }
}

Write-Host 'INPUT DOWNLOAD COMPLETE'
Write-Host "AS-UNet commit: $AsUnetCommit"
Write-Host "CAS archives:  $ArchiveRoot"