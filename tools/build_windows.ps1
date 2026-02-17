# tools/build_windows.ps1
# Windows 用: venv作成 → 依存導入 → PyInstaller(onedir) → VC++ DLL削除 → zip作成
# 使い方:
#   powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1
# オプション:
#   -PythonExe "py" -PythonVersion "3.11" -Clean -NoZip

param(
  [string]$PythonExe = "py",
  [string]$PythonVersion = "3.11",
  [switch]$Clean,
  [switch]$NoZip
)

$ErrorActionPreference = "Stop"

# プロジェクトルートへ移動（このps1の2階層上を想定：tools/配下）
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Resolve-Path (Join-Path $SCRIPT_DIR "..")
Set-Location $ROOT

# ---- 設定 ----
$VenvDir = ".venv-win"
$AppPy = "app.py"
$AppName = "SpeechSummarizer"
$DistDir = Join-Path $ROOT "dist"
$BuildDir = Join-Path $ROOT "build"
$ReleaseDir = Join-Path $ROOT "release"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutZip = Join-Path $ReleaseDir "${AppName}_win_x64_${Stamp}.zip"

# requirements
$ReqFile = "requirements-win.txt"
if (-not (Test-Path $ReqFile)) {
  # fallback
  $ReqFile = "requirements.txt"
}

# 同梱データ（onedirの実行フォルダ基準で読む想定）
$AddDataArgs = @(
  "--add-data", "static;static",
  "--add-data", "config.json;."
)

# collect対象（faster-whisper/ct2）
$CollectArgs = @(
  "--collect-all", "faster_whisper",
  "--collect-all", "ctranslate2",
  "--collect-all", "tokenizers"
)

# VC++ DLL（同梱するとクラッシュするケースがあったため削除対象）
$BadVCDlls = @(
  "MSVCP140.dll",
  "VCRUNTIME140.dll",
  "VCRUNTIME140_1.dll",
  "CONCRT140.dll"
)

# ---- 事前チェック ----
if (-not (Test-Path $AppPy)) {
  throw "Not found: $AppPy (project root: $ROOT)"
}
if (-not (Test-Path "static")) {
  Write-Warning "static/ not found. If your app needs it, build will succeed but UI may be broken."
}
if (-not (Test-Path "config.json")) {
  Write-Warning "config.json not found. Build will succeed but runtime will fail."
}

# ---- Clean ----
if ($Clean) {
  Write-Host "[CLEAN] removing dist/, build/, $VenvDir/"
  if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
  if (Test-Path $BuildDir) { Remove-Item $BuildDir -Recurse -Force }
  if (Test-Path $VenvDir) { Remove-Item $VenvDir -Recurse -Force }
}

# ---- venv ----
if (-not (Test-Path $VenvDir)) {
  Write-Host "[VENV] create $VenvDir using Python $PythonVersion"
  & $PythonExe "-$PythonVersion" "-m" "venv" $VenvDir
}

$VenvPython = Join-Path $ROOT "$VenvDir\Scripts\python.exe"
$VenvPip = Join-Path $ROOT "$VenvDir\Scripts\pip.exe"

if (-not (Test-Path $VenvPython)) {
  throw "venv python not found: $VenvPython"
}

Write-Host "[PIP] upgrade pip/wheel"
& $VenvPython "-m" "pip" "install" "-U" "pip" "wheel" | Out-Host

Write-Host "[PIP] install requirements: $ReqFile"
& $VenvPython "-m" "pip" "install" "-r" $ReqFile | Out-Host

Write-Host "[PIP] install pyinstaller"
& $VenvPython "-m" "pip" "install" "pyinstaller" | Out-Host

# ---- PyInstaller build ----
Write-Host "[BUILD] pyinstaller onedir: $AppName"
$Args = @(
  $AppPy,
  "--name", $AppName,
  "--noconfirm",
  "--clean",
  "--onedir"
) + $CollectArgs + $AddDataArgs

# console=True が良い間はこのまま。配布で黒窓不要なら "--noconsole" を追加してください。
# $Args += @("--noconsole")

& $VenvPython "-m" "PyInstaller" @Args | Out-Host

$ExeDir = Join-Path $DistDir $AppName
$InternalDir = Join-Path $ExeDir "_internal"
$ExePath = Join-Path $ExeDir "$AppName.exe"

if (-not (Test-Path $ExePath)) {
  throw "Build failed: exe not found: $ExePath"
}

# ---- Post build cleanup (VC++ DLL remove) ----
if (Test-Path $InternalDir) {
  Write-Host "[POST] remove bundled VC++ DLLs from $InternalDir"
  foreach ($dll in $BadVCDlls) {
    $p = Join-Path $InternalDir $dll
    if (Test-Path $p) {
      Remove-Item $p -Force
      Write-Host "  removed $dll"
    }
  }
} else {
  Write-Warning "_internal not found: $InternalDir"
}

# ---- Release folder ----
if (-not (Test-Path $ReleaseDir)) {
  New-Item -ItemType Directory -Path $ReleaseDir | Out-Null
}

# ---- Zip ----
if (-not $NoZip) {
  if (Test-Path $OutZip) { Remove-Item $OutZip -Force }
  Write-Host "[ZIP] create $OutZip from $ExeDir"
  Compress-Archive -Path (Join-Path $ExeDir "*") -DestinationPath $OutZip
}

Write-Host ""
Write-Host "[OK] EXE: $ExePath"
if (-not $NoZip) {
  Write-Host "[OK] ZIP: $OutZip"
}
Write-Host ""
Write-Host "Note: VC++ 2015-2022 (x64) Redistributable should be installed on target PCs."
