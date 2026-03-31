# tools/build_windows.ps1
# Windows build: PyInstaller (onedir) -> trim VC++ runtime DLLs -> copy assets -> zip
# Usage:
#   .\tools\build_windows.ps1
#   .\tools\build_windows.ps1 -IncludeModels $false

param(
  [string]$Name = "SpeechSummarizer",
  [string]$Entry = "launcher.py",
  [string]$DistDir = "dist",
  [string]$BuildDir = "build",
  [string]$OutDir = "release",
  [bool]$IncludeModels = $true
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Exists([string]$Path) {
  if (!(Test-Path $Path)) { throw "Not found: $Path" }
}

function Wait-UntilFileUnlocked([string]$Path, [int]$TimeoutSeconds = 30) {
  if (!(Test-Path $Path)) { return }

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $stream = [System.IO.File]::Open($Path, 'Open', 'Read', 'None')
      $stream.Close()
      return
    }
    catch [System.IO.IOException] {
      Start-Sleep -Milliseconds 500
    }
  }

  throw "Timed out waiting for file to unlock: $Path"
}

function Remove-TreeWithRetry([string]$Path, [int]$MaxAttempts = 5) {
  if (!(Test-Path $Path)) { return }

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Remove-Item $Path -Recurse -Force
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Write-Host "[clean] retry $attempt/$MaxAttempts after file lock: $Path"
      Start-Sleep -Seconds 2
    }
  }
}

function Compress-ArchiveWithRetry(
  [string]$SourcePath,
  [string]$DestinationPath,
  [int]$MaxAttempts = 5
) {
  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Compress-Archive -Path $SourcePath -DestinationPath $DestinationPath -Force
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Write-Host "[zip] retry $attempt/$MaxAttempts after file lock"
      if (Test-Path $DestinationPath) {
        Remove-Item $DestinationPath -Force -ErrorAction SilentlyContinue
      }
      Start-Sleep -Seconds 2
    }
  }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Push-Location $RepoRoot

try {
  Assert-Exists $Entry

  $py = (Get-Command python -ErrorAction Stop).Source
  Write-Host "[build] python=$py"

  Remove-TreeWithRetry $DistDir
  Remove-TreeWithRetry $BuildDir

  Write-Host "[build] PyInstaller onedir..."
  python -m PyInstaller `
    --noconfirm --clean `
    --onedir `
    --windowed `
    --name $Name `
    --add-data "tools\analysis_tools\static;tools\analysis_tools\static" `
    --add-data "tools\analysis_tools\templates;tools\analysis_tools\templates" `
    --add-data "tools\correction_tool\static;tools\correction_tool\static" `
    --add-data "tools\correction_tool\templates;tools\correction_tool\templates" `
    --add-data "tools\so_labeler\static;tools\so_labeler\static" `
    --add-data "tools\so_labeler\templates;tools\so_labeler\templates" `
    $Entry

  $AppDir = Join-Path $DistDir $Name
  Assert-Exists $AppDir

  $internalDir = Join-Path $AppDir "_internal"
  if (Test-Path $internalDir) {
    $vcDlls = @(
      "msvcp140.dll",
      "MSVCP140_1.dll",
      "vcruntime140.dll",
      "vcruntime140_1.dll",
      "concrt140.dll"
    )

    foreach ($dll in $vcDlls) {
      $target = Join-Path $internalDir $dll
      if (Test-Path $target) {
        Remove-Item $target -Force
        Write-Host "[fix] removed VC runtime DLL: $dll"
      }
    }
  }

  Write-Host "[pack] Copy assets into $AppDir"

  $itemsToCopy = @(
    "config.json.sample",
    "corrections.json.sample",
    "static",
    "certs"
  )

  if ($IncludeModels) { $itemsToCopy += "models" }

  # Do not ship config.json. The app creates it from config.json.sample
  # only when missing, so user settings are not overwritten on upgrade.
  $packagedConfig = Join-Path $AppDir "config.json"
  if (Test-Path $packagedConfig) {
    Remove-Item $packagedConfig -Force
    Write-Host "[pack] removed: config.json"
  }

  foreach ($it in $itemsToCopy) {
    if (Test-Path $it) {
      Copy-Item $it -Destination $AppDir -Recurse -Force
      Write-Host ("[pack] copied: " + $it)
    }
    else {
      Write-Host ("[pack] skip (not found): " + $it)
    }
  }

  if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $modelTag = if ($IncludeModels) { "" } else { "-NoModels" }
  $zipName = "$Name$modelTag-win64-onedir-$stamp.zip"
  $zipPath = Join-Path $OutDir $zipName
  $baseLibraryZip = Join-Path $internalDir "base_library.zip"

  if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

  Wait-UntilFileUnlocked -Path $baseLibraryZip
  Write-Host "[zip] create: $zipPath"
  Compress-ArchiveWithRetry -SourcePath $AppDir -DestinationPath $zipPath

  Write-Host "[done] $zipPath"
}
finally {
  Pop-Location
}
