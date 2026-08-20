# Windows GUI版のビルド方法

`tools\build_windows.ps1` は、SpeechSummarizerのWindows GUI版を
PyInstallerのOneFolder形式でビルドし、配布用ZIPを作成するスクリプトです。

このスクリプトは、次の処理をまとめて行います。

1. 既存のビルド結果を削除
2. `launcher.py` をPyInstallerでGUIアプリ化
3. Web UI用のstatic/templatesを同梱
4. 設定サンプル、証明書、必要に応じて音声認識モデルをコピー
5. 配布先との競合を避けるため、同梱された一部のVC++ Runtime DLLを削除
6. `release` フォルダに配布用ZIPを作成

## 必要な環境

- Windows 10/11 x64
- Python 3.11 x64（推奨）
- PowerShell
- Microsoft Visual C++ 2015-2022 Redistributable (x64)
  - ビルドしたアプリを実行するPCにも必要です
  - https://aka.ms/vs/17/release/vc_redist.x64.exe

Windows用EXEはWindows上でビルドしてください。LinuxやWSL上からこのスクリプトを
実行しても、通常のWindows用EXEは作成できません。

## Qwen3-ASRをGUIランチャーから利用する

QwenASRはSpeechSummarizerとは別のPython仮想環境に配置します。GUIランチャーのASRタブでproviderを`qwen3-asr`にし、「Windows GUIランチャーでQwenASRを起動・停止」を有効にして、次の3ファイルを指定してください。

- QwenASRの`.venv\Scripts\python.exe`
- QwenASRの`server.py`
- QwenASRの`config.json`

「サーバー起動」を押すと、ランチャーはQwenASRを起動して`/ready`を待ち、その後にSpeechSummarizer本体を起動します。「サーバー停止」またはランチャー終了時には、ランチャー自身が起動したQwenASRも停止します。すでに手動起動されたQwenASRがreadyの場合はそのプロセスを利用し、ランチャーから停止しません。

PythonからSpeechSummarizerの`app.py`を直接実行した場合、この自動管理は動作しません。その場合は従来どおりQwenASRを別途起動し、`config.json`でproviderとAPI URLを指定します。

## 初回のみ行う準備

PowerShellでリポジトリのルートに移動します。

```powershell
cd C:\path\to\SpeechSummarizer
```

仮想環境を作成して有効化します。

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

`Set-ExecutionPolicy -Scope Process` の設定は、現在開いているPowerShellだけに適用されます。
PowerShellを閉じると元に戻るため、Windows全体の設定は変更しません。

依存パッケージとPyInstallerをインストールします。

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

> `build_windows.ps1` 自体は仮想環境の作成やパッケージのインストールを行いません。
> 実行前に、ビルドに使う仮想環境を有効化してください。

現在使用されているPythonは、次のコマンドで確認できます。

```powershell
python -c "import sys; print(sys.executable)"
```

## 基本的なビルド方法

リポジトリのルートで次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1
```

PowerShellの実行ポリシーを変更済みの場合は、次の形式でも実行できます。

```powershell
.\tools\build_windows.ps1
```

正常終了すると、最後に次のようなメッセージが表示されます。

```text
[done] release\SpeechSummarizer-win64-onedir-20260619_120000.zip
```

## よく使うビルド例

### モデルを含めてビルドする

デフォルトの動作です。リポジトリの `models` フォルダを配布物に含めます。

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1
```

または、明示的に指定します。

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -IncludeModels $true
```

生成されるZIPの例：

```text
release\SpeechSummarizer-win64-onedir-20260619_120000.zip
```

### モデルを含めずにビルドする

配布ファイルを小さくしたい場合に使用します。

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -IncludeModels $false
```

生成されるZIPには `NoModels` が付きます。

```text
release\SpeechSummarizer-NoModels-win64-onedir-20260619_120000.zip
```

この配布物を使う場合は、実行先で別途ASRモデルを配置し、`config.json` から
モデルの場所を指定してください。

### アプリ名を変更する

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -Name "SpeechSummarizer-Test"
```

実行ファイル名、配布フォルダ名、ZIP名が指定した名前になります。

### 出力先を変更する

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 `
  -DistDir "dist-win" `
  -BuildDir "build-win" `
  -OutDir "release-win"
```

### 複数のオプションを組み合わせる

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 `
  -Name "SpeechSummarizer-Test" `
  -OutDir "release-test" `
  -IncludeModels $false
```

## オプション一覧

| オプション | 型 | デフォルト値 | 説明 |
|---|---|---|---|
| `-Name` | 文字列 | `SpeechSummarizer` | アプリ名。EXE、配布フォルダ、ZIPの名前に使われます |
| `-Entry` | 文字列 | `launcher.py` | PyInstallerでビルドする起動スクリプト |
| `-DistDir` | 文字列 | `dist` | PyInstallerの完成品を置くフォルダ |
| `-BuildDir` | 文字列 | `build` | PyInstallerの一時ビルドフォルダ |
| `-OutDir` | 文字列 | `release` | 配布用ZIPを置くフォルダ |
| `-IncludeModels` | 真偽値 | `$true` | `$true` なら `models` フォルダを配布物に含めます |

すべて指定する場合の例：

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 `
  -Name "SpeechSummarizer" `
  -Entry "launcher.py" `
  -DistDir "dist" `
  -BuildDir "build" `
  -OutDir "release" `
  -IncludeModels $true
```

通常は `-IncludeModels` だけ変更すれば十分です。`-Entry` は別の起動スクリプトを
試す場合など、開発用途を想定したオプションです。

## 生成されるファイル

デフォルト設定では、次の場所に生成されます。

```text
dist\
└─ SpeechSummarizer\
   ├─ SpeechSummarizer.exe
   ├─ _internal\
   ├─ config.json.sample
   ├─ corrections.json.sample
   ├─ static\
   ├─ certs\
   └─ models\                 # IncludeModelsがtrueの場合

release\
└─ SpeechSummarizer-win64-onedir-YYYYMMDD_HHMMSS.zip
```

配布・動作確認には、`SpeechSummarizer.exe` だけでなく
`SpeechSummarizer` フォルダ全体が必要です。

`config.json` は配布物には含めません。初回起動時に
`config.json.sample` を基に作成されるため、更新版を上書き展開しても利用者の設定を
不用意に置き換えない構成になっています。

## ビルド時の注意

- ビルド開始時に、指定した `DistDir` と `BuildDir` は削除されます。
- `OutDir` は削除されません。日時付きのZIPが追加されます。
- `static`、`certs`、`models` など、存在しない任意ファイルは
  `[pack] skip (not found)` と表示されてスキップされます。
- `Entry` に指定したファイルが存在しない場合、ビルドはエラーで停止します。
- PyInstallerは `--windowed` で実行されるため、通常起動時にコンソール画面は
  表示されません。
- 配布形式は `--onedir` です。単一EXE形式ではありません。

## 古いオプションについて

過去のバージョンには、次のオプションが存在しました。

```text
-PythonExe
-PythonVersion
-Clean
-NoZip
```

これらは現在の `build_windows.ps1` では使用できません。

現在は、あらかじめ有効化した仮想環境の `python` を使用します。また、
`dist` と `build` のクリーンアップおよびZIP作成は常に実行されます。

## トラブルシューティング

### 「このシステムではスクリプトの実行が無効」と表示される

現在開いているPowerShellだけスクリプト実行を許可してから、仮想環境を有効化します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

プロンプトの先頭に `(venv)` が表示されれば有効化できています。

```text
(venv) PS D:\work\SpeechSummarizer>
```

次の方法では、仮想環境は現在のPowerShellに引き継がれません。

```powershell
powershell -ExecutionPolicy Bypass -File .\venv\Scripts\Activate.ps1
```

これは別のPowerShellを一時的に起動し、その中で仮想環境を有効化した直後に終了するためです。

実行ポリシーを変更したくない場合は、仮想環境を有効化せず、Pythonを直接指定することも
できます。

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install pyinstaller
```

ただし、ビルドスクリプトは現在の `python` コマンドを使用するため、ビルド時は
仮想環境を有効化する方法を推奨します。

### `No module named PyInstaller` と表示される

仮想環境を有効化してからPyInstallerをインストールします。

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install pyinstaller
```

### ファイルが使用中で削除・ZIP作成できない

起動中の `SpeechSummarizer.exe`、エクスプローラーのプレビュー、ウイルス対策ソフト
などがビルド結果を使用していないか確認してください。

スクリプトはZIP作成時のエラーを最大15回再試行しますが、ロックが解除されなければ
停止します。Windows Defenderなどがビルド直後のファイルを一時的に検査している場合は、
少し待ってからもう一度ビルドしてください。

以前のスクリプトで次のエラーが出た場合は、最新の
`tools\build_windows.ps1` に更新してください。

```text
Timed out waiting for file to unlock:
dist\SpeechSummarizer\_internal\base_library.zip
```

これはビルド失敗ではなく、ZIP作成前の排他ロック確認がWindows Defenderなどの
ファイル参照をロックと判定したものです。現在のスクリプトでは、この不要な事前確認を
行わず、実際のZIP作成を再試行するように変更されています。

### 別のPythonが使われている

ビルド開始時に次のように使用中のPythonが表示されます。

```text
[build] python=C:\path\to\venv\Scripts\python.exe
```

意図した仮想環境のPythonでない場合は、仮想環境を有効化し直してから実行してください。

### ビルドしたアプリが起動しない

まずMicrosoft Visual C++ 2015-2022 Redistributable (x64)をインストールしてください。

https://aka.ms/vs/17/release/vc_redist.x64.exe

それでも起動しない場合は、PowerShellまたはコマンドプロンプトからEXEを起動して、
表示されるエラーを確認してください。
