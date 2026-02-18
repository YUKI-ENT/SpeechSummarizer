# tools/build_windows.ps1
# Windows 用: venv作成 → 依存導入 → PyInstaller(onedir) → VC++ DLL削除 → zip作成
# 使い方:
#   powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1
# models を同梱しないなら：
#   powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1  -IncludeModels:$false
# オプション:
#   -PythonExe "py" -PythonVersion "3.12" -Clean -NoZip

param(
  [string]$Name = "SpeechSummarizer",
  [string]$Entry = "app.py",
  [string]$DistDir = "dist",
  [string]$BuildDir = "build",
  [string]$OutDir = "release",
  [switch]$IncludeModels = $true   # models/を同梱しないなら -IncludeModels:$false
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-Exists([string]$Path){
  if (!(Test-Path $Path)) { throw "Not found: $Path" }
}

# どこから実行されても repo ルートに合わせる
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "..")
Push-Location $RepoRoot

try {
  Assert-Exists $Entry

  # 依存の前提：venvはユーザーが手動で有効化済み、pyinstallerも入っていること
  $py = (Get-Command python -ErrorAction Stop).Source
  Write-Host "[build] python=$py"

  # 既存成果物を掃除（onedirはdist/$Name配下が本体なので、そこを消す）
  if (Test-Path $DistDir)  { Remove-Item $DistDir  -Recurse -Force }
  if (Test-Path $BuildDir) { Remove-Item $BuildDir -Recurse -Force }

  # --- PyInstaller: onedir ---
  # 既に spec を使っているなら、この行を spec 呼び出しに置き換えてOK
  # 例: python -m PyInstaller --noconfirm --clean "$Name.spec"
  Write-Host "[build] PyInstaller onedir..."
  python -m PyInstaller `
    --noconfirm --clean `
    --onedir `
    --name $Name `
    $Entry

  $AppDir = Join-Path $DistDir $Name
  Assert-Exists $AppDir

  # --- 配布用ファイルを exe と同階層にコピー ---
  # ここが今回の修正ポイント
  Write-Host "[pack] Copy assets into $AppDir"

  $itemsToCopy = @(
    "config.json",
    "corrections.json",
    "static",
    "certs"
  )

  if ($IncludeModels) { $itemsToCopy += "models" }

  foreach ($it in $itemsToCopy) {
    if (Test-Path $it) {
      Copy-Item $it -Destination $AppDir -Recurse -Force
      Write-Host ("[pack] copied: " + $it)
    } else {
      Write-Host ("[pack] skip (not found): " + $it)
    }
  }

  # --- ZIP作成 ---
  if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $zipName = "$Name-win64-onedir-$stamp.zip"
  $zipPath = Join-Path $OutDir $zipName

  if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

  Write-Host "[zip] create: $zipPath"
  Compress-Archive -Path $AppDir -DestinationPath $zipPath -Force

  Write-Host "[done] $zipPath"
}
finally {
  Pop-Location
}
