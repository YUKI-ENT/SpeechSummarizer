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
  [string]$PythonExe = "",
  [bool]$IncludeModels = $true
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Exists([string]$Path) {
  if (!(Test-Path $Path)) { throw "Not found: $Path" }
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
  [int]$MaxAttempts = 15
) {
  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Compress-Archive -Path $SourcePath -DestinationPath $DestinationPath -Force
      return
    }
    catch {
      if ($attempt -eq $MaxAttempts) { throw }
      Write-Host "[zip] retry $attempt/$MaxAttempts after error: $($_.Exception.Message)"
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

  $repoRootPath = [System.IO.Path]::GetFullPath([string]$RepoRoot)
  if ($PythonExe) {
    $pythonCandidate = if ([System.IO.Path]::IsPathRooted($PythonExe)) {
      $PythonExe
    }
    else {
      Join-Path $repoRootPath $PythonExe
    }
  }
  else {
    $activePythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $activePython = if ($activePythonCommand) { $activePythonCommand.Source } else { $null }
    $localPythonCandidates = @(
      (Join-Path $repoRootPath "venv312\Scripts\python.exe"),
      (Join-Path $repoRootPath "venv\Scripts\python.exe")
    )
    $pythonCandidate = $null
    if ($activePython) {
      $activePythonPath = [System.IO.Path]::GetFullPath($activePython)
      $repoPrefix = $repoRootPath.TrimEnd('\') + '\'
      if ($activePythonPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        $pythonCandidate = $activePythonPath
      }
    }
    if (!$pythonCandidate) {
      $pythonCandidate = $localPythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    }
  }

  if (!$pythonCandidate -or !(Test-Path $pythonCandidate)) {
    throw "SpeechSummarizer用Pythonが見つかりません。-PythonExe で仮想環境のpython.exeを指定してください。"
  }
  $buildPython = (Resolve-Path $pythonCandidate).Path
  Write-Host "[build] python=$buildPython"

  Write-Host "[build] checking required packages..."
  & $buildPython -c "import struct; from importlib.metadata import version; import PyInstaller, faster_whisper, ctranslate2; assert struct.calcsize('P') * 8 == 64, '64-bit Python is required'; print('PyInstaller=' + version('pyinstaller')); print('faster-whisper=' + version('faster-whisper')); print('ctranslate2=' + version('ctranslate2'))"
  if ($LASTEXITCODE -ne 0) {
    throw "ビルド用Pythonに必須packageがありません。requirements.txt と PyInstaller をインストールしてください: $buildPython"
  }

  Remove-TreeWithRetry $DistDir
  Remove-TreeWithRetry $BuildDir

  Write-Host "[build] PyInstaller onedir..."
  & $buildPython -m PyInstaller `
    --noconfirm --clean `
    --onedir `
    --windowed `
    --name $Name `
    --distpath $DistDir `
    --workpath $BuildDir `
    --add-data "tools\analysis_tools\static;tools\analysis_tools\static" `
    --add-data "tools\analysis_tools\templates;tools\analysis_tools\templates" `
    --add-data "tools\correction_tool\static;tools\correction_tool\static" `
    --add-data "tools\correction_tool\templates;tools\correction_tool\templates" `
    --add-data "tools\so_labeler\static;tools\so_labeler\static" `
    --add-data "tools\so_labeler\templates;tools\so_labeler\templates" `
    $Entry
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
  }

  $warningFile = Join-Path (Join-Path $BuildDir $Name) "warn-$Name.txt"
  if (Test-Path $warningFile) {
    $missingRequired = Select-String -Path $warningFile -Pattern "missing module named (faster_whisper|ctranslate2)"
    if ($missingRequired) {
      throw "PyInstallerが必須ASR moduleを収集できませんでした: $($missingRequired.Line -join '; ')"
    }
  }

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
    "memo_templates.json.sample",
    "static",
    "certs"
  )

  if ($IncludeModels) { $itemsToCopy += "models" }

  # Do not ship config.json. The app creates it from config.json.sample
  # only when missing, so user settings are not overwritten on upgrade.
  $packagedConfig = Join-Path $AppDir "config.json"
  $packagedMemoTemplates = Join-Path $AppDir "memo_templates.json"
  if (Test-Path $packagedConfig) {
    Remove-Item $packagedConfig -Force
    Write-Host "[pack] removed: config.json"
  }
  if (Test-Path $packagedMemoTemplates) {
    Remove-Item $packagedMemoTemplates -Force
    Write-Host "[pack] removed: memo_templates.json"
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

  Write-Host "[check] packaged server imports..."
  $packagedExe = Join-Path $AppDir "$Name.exe"
  $importCheck = Start-Process `
    -FilePath $packagedExe `
    -ArgumentList "--check-server-imports" `
    -WorkingDirectory $AppDir `
    -WindowStyle Hidden `
    -PassThru
  if (!$importCheck.WaitForExit(30000)) {
    Stop-Process -Id $importCheck.Id -Force -ErrorAction SilentlyContinue
    throw "配布EXEのserver import確認が30秒以内に完了しませんでした。"
  }
  if ($importCheck.ExitCode -ne 0) {
    throw "配布EXEのserver import確認に失敗しました (exit=$($importCheck.ExitCode))"
  }
  if (Test-Path $packagedConfig) {
    Remove-Item $packagedConfig -Force
    Write-Host "[check] removed config.json created by import check"
  }
  if (Test-Path $packagedMemoTemplates) {
    Remove-Item $packagedMemoTemplates -Force
    Write-Host "[check] removed memo_templates.json created by import check"
  }

  if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $modelTag = if ($IncludeModels) { "" } else { "-NoModels" }
  $zipName = "$Name$modelTag-win64-onedir-$stamp.zip"
  $zipPath = Join-Path $OutDir $zipName

  if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

  Write-Host "[zip] create: $zipPath"
  Compress-ArchiveWithRetry -SourcePath $AppDir -DestinationPath $zipPath

  Write-Host "[done] $zipPath"
}
finally {
  Pop-Location
}
