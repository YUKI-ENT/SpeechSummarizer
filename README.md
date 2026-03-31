# SpeechSummarizer
## 概要
SpeechSummarizer は、医療現場向けに設計されたリアルタイム音声認識＋AI要約システムです。
- Faster-Whisper による音声認識（ASR）
- GPU / CPU 両対応
- Ollama 連携による SOAP 形式などの要約生成
- Windows EXE / Python 実行 両対応
- ローカルモデル運用（インターネット不要）
- 難聴モード（大字幕表示）
- 誤変換補正機能：correction ルールによる自動補正
- 電子カルテ：ダイナミクスと連動して患者IDごとに履歴保存
  
等の機能を持ち、診察会話の文字起こし、SOAP生成、カルテ補助を目的にしてます

## 🎥 紹介動画

[![SpeechSummarizer Demo](https://img.youtube.com/vi/ujOWRbY5qK4/maxresdefault.jpg)](https://youtu.be/ujOWRbY5qK4)

## インストール方法

### 【Windows版（EXE）】

1. release の zip をダウンロード（ファイル名に`NoModels`が入っているものはASRモデルなし、無印のものはsmallのModelファイルが同梱されています）
2. 任意のフォルダに展開

  フォルダ構成例：
```
SpeechSummarizer/
├─ SpeechSummarizer.exe
├─ _internal/
├─ config.json.sample
├─ corrections.json.sample
├─ static/
├─ cert/
└─ models/
```

### 【Python版】
1. Python (ver3.12推奨) をインストール
2. インストール先でgit cloneする
```
git clone https://github.com/YUKI-ENT/SpeechSummarizer.git
```
3. 仮想環境作成
```
python -m venv venv
venv\Scripts\activate （Windows）
source venv/bin/activate （Linux）
```
3. 依存パッケージインストール
```
pip install -r requirements.txt
```

※ Linux環境では、CUDAランタイムをシステムにインストールするとドライババージョンや他のアプリケーションと干渉し起動できなくなることがあるので、下記の[【補足】GPU (CUDA) 利用方法（Linux）](https://github.com/YUKI-ENT/SpeechSummarizer/edit/main/README.md#%E8%A3%9C%E8%B6%B3gpu-cuda-%E5%88%A9%E7%94%A8%E6%96%B9%E6%B3%95linux) を推奨します。

## config.jsonの設定（Windows版ではGUIから編集可能）
インストールフォルダに有る`config.json.sample`を`config.json`に名前を変えるかコピーし、環境に合わせて編集します。`corrections.json` も初回起動時に `corrections.json.sample` から自動生成されます。

必須の項目は、
- \"asr\": セクション
  - model_id:
    
      デフォルトのモデル名を指定します。下記のモデル一覧に存在するものを指定してください。
  - modelsセクション
    - デフォルトではsmallのみ同梱してますが、Huggingfaec等からダウンロードして、そのフォルダを指定すればASRに利用できます。
    - フォルダ指定時は、パス区切りは **￥でなく、/でお願いします**
  - device:
    Nvidia GPUがあれば、\"cuda\" を指定することでGPUが利用できます。このとき下記の\"compute_type\" も \"float16\"に変更してください。
  - \"compute_type\":
    GPUがあれば、\"float16\"、なければ\"int8\"を指定してください
- \"dyna_watch_dir\":
  ここに指定したフォルダに、ダイナミクスの他社連携からカルテ番号を出力することで電子カルテと連動します。ダイナミクス側は、枝番なしで出力してください。
- \"outputs_dir\", \"wav_dir\", \"llm_outputs_dir\"
  認識結果、録音音声データ、LLM問い合わせ結果の保存先です。
- `vad`: セクション
  - 音声区間判定（VAD）の設定です。配布先のマイク差や周囲雑音の差が大きい場合は、ここを調整します。
  - `mode`
    - `auto`: 推奨。接続ごとに無音寄りの区間からバックグラウンドノイズを推定し、`noise_floor + margin_db` を閾値として自動調整します。
    - `manual`: `manual_threshold_db` を固定閾値として使います。環境が一定で、すでに安定している場合はこちらでも構いません。
  - `manual_threshold_db`
    - `manual` モード時の固定閾値です。値を小さくすると拾いやすくなり、大きくすると誤検出しにくくなります。
  - `calibration_sec`
    - `auto` モード開始直後に、無音寄り区間を集める秒数です。短すぎると不安定、長すぎると録音開始直後の反応が遅くなります。
  - `margin_db`
    - 推定したノイズ床に足す余裕幅です。診察室の雑音で誤検出が多い場合は少し上げ、声を拾いにくい場合は少し下げてください。
  - `min_threshold_db`, `max_threshold_db`
    - 自動計算された閾値の上下限です。想定外に閾値が上がりすぎたり下がりすぎたりするのを防ぎます。
  - `noise_window_sec`
    - ノイズ床推定に使う履歴の長さです。長いほど安定し、短いほど環境変化に追従しやすくなります。
  - `update_margin_db`
    - 現在の閾値より十分に静かなフレームだけをノイズ床更新に使うための余裕です。大きくすると会話音や突発音に引っ張られにくくなります。
  - `quiet_percentile`
    - ノイズ床推定に使う分位点です。通常は既定値のままで構いません。
  - `start_voice_frames`, `end_silence_frames`
    - 何フレーム連続で音声が続いたら開始、何フレーム連続で無音になったら終了とみなすかを決めます。誤起動が多いときは `start_voice_frames` を増やし、切れやすいときは `end_silence_frames` を増やします。
  - `pre_roll_ms`
    - 発話開始直前の音をどれだけ巻き戻して含めるかです。語頭欠けが気になる場合に増やします。
  - `min_sec`, `max_sec`
    - 1セグメントの最短秒数と最長秒数です。短すぎるノイズ断片を捨てたり、長すぎる発話を分割したりするために使います。
  - まず試す設定
    - 通常は `mode: "auto"` のままで開始してください。
    - 雑音で反応しすぎる場合は `margin_db` を 2〜4 程度上げます。
    - 声を拾いにくい場合は `margin_db` を 2〜4 程度下げるか、`manual` にして `manual_threshold_db` を調整します。
    - 子供の泣き声や周囲の会話のような「人声に近い大きな音」は、`auto` でも完全には避けられません。その場合は `start_voice_frames` を少し増やしてください。
- \"ssl\":セクション
  -  **SpeechSummarizer** 実行PC（サーバー）とWebクライアントが同一(localhost:8000でアクセス)の場合は
      ```
      "ssl": {
        "enabled": false
      },
      ```
   とし、http://localhost:8000 でアクセスします。
  - サーバーとクライアントを別PCにする場合は、httpではマイクの許可ができないので、httpsを有効にします。以下のようにしてください。ただし、オレオレ自己証明書ですので接続時に安全でないサイトの警告が出ます。
    ```
     "ssl": {
        "enabled": true,
        "certfile": "certs/cert.pem",
        "keyfile": "certs/key.pem"
      },
    ```
- llmセクション
  - host:
    ollamaの稼働しているアドレスを指定します
  - model_default:
    デフォルトで使用するllmモデル名を指定します
  - default_prompt_id：
    デフォルトで使用するプロンプト名を下記の一覧にあるものを指定します
  - prompts セクション
    - 自由に追加できます。こちらの例を参考に追加してみてください。
      ```
      "prompts": {
        "soap_v1": {
          "label": "SOAP(発熱重視)",
          "template": "以下は医者と患者の診察室での会話（主に医者の発言）です。音声認識のため同音異義語(「咳」→「席」、「鼻」→「花」など)が混ざります。それを考慮して、SOAP形式に要約してください。\n\n重要な制約:\n- 勝手に情報を追加しない（会話に無いことは書かない）\n- 不明なことは「不明」とし、推測しない\n- 可能なら箇条書き、特にS)の部分はなるべく時系列で簡潔に\n -体温と思われる情報は必ずSに記載 \n出力はSOAPのみ（前置き不要）。\n\n【会話テキスト】\n{asr_text}\n"
        },
        "soap_v1_short": {
          "label": "SOAP(短め)",
          "template": "以下は診察会話で主に医者の発言部分のテキストです。推測せずSOAPで簡潔に要約してください。会話に無い情報は書かない。\n\n【会話テキスト】\n{asr_text}\n"
        },
        "yomi_correct": {
          "label": "誤変換訂正",
          "template": "以下は診察会話で主に医者の発言部分のテキストですが、音声認識のため変換ミスが見られます。「咳」→「席」、「鼻」→「花」など文脈から間違いと思われるものを修正し、質問に対する返答の「うん」「はい」「ええ」等は必ず残し、言ってもないことを追加することなく出力してください。【会話テキスト】\n{asr_text}\n"
        }
      }
      ```

## ASRモデルのダウンロード方法
- 初期状態ではsmallモデルを同梱しておりますが、以下の手順でより高性能モデルを利用できます。
- Huggingfaceのサイト： 

  https://huggingface.co/collections/Systran/faster-whisper や
  
  https://huggingface.co/RoachLin/kotoba-whisper-v2.2-faster

  の `Files and versions`タブにあるファイルすべてを、指定フォルダにダウンロードします。
- そのフォルダを、`config.json`の`"models": `セクションに追加し、サーバーを再起動してください。
  ```
    例：
    
    models/
    └─ kotoba-whisper-v2.2-faster/
      ├─ config.json
      ├─ model.bin
      └─ tokenizer.json
  ```

## 起動方法
### Windows Exe版
- SpeechSummarizer.exeをクリックし、起動に成功すると下記のような表示になります
  <img width="1042" height="852" alt="image" src="https://github.com/user-attachments/assets/1eed7ffc-6148-4ae4-839c-3d32d1d61bab" />

- 設定をした後、`サーバー起動`クリックでサーバーが起動します
- この画面が表示されない場合は、**コマンドプロンプト**や **powershell**から実行してみて、エラーメッセージを確認してみてください。
- Visual C++ 再頒布可能パッケージが必要になるケースもあります。その場合は、https://aka.ms/vc14/vc_redist.x64.exe からダウンロード、インストールを行ってください。

### Python版
- venvを有効にします
  ```
  venv\Scripts\activate （Windows）
  source venv/bin/activate （Linux）
  ```
- 実行
  ```
  python app.py
  ```
  
  ※Linux環境では、システムにCUDAランタイムを入れるとドライバのバージョンアップに伴い動作不良を起こすことがあるので、下の **【補足】GPU (CUDA) 利用方法（Linux)** の章を参考に、venv内にCUDAランタイムを入れて、`SpeechSummarizer.sh`で起動するようにしてください。

## 利用方法
- クライアントからローカルの場合は

  http://localhost:8000

  サーバーが別PCの場合は

  https://(サーバーアドレス):8000

  で画面が立ち上がります。
  
  <img width="918" height="758" alt="名称未設定" src="https://github.com/user-attachments/assets/d33551cd-f395-49a0-adeb-9d233ac5112e" />

- VADを `auto` にしている場合は、録音開始直後の数秒で周囲の静かな区間を使って閾値を整えます。録音開始直後から大きな音が続くと推定が安定しにくいため、できれば最初の1〜2秒は通常の環境音のままにしてください。
- 録音中のレベル表示は入力音量の目安です。必要に応じてランチャーの `VAD` 設定で `margin_db` や `start_voice_frames` を調整してください。
- `auto` モードの現在閾値と推定ノイズ床は内部で保持され、保存される JSONL / WAV メタ情報にも反映されます。録音結果が不安定な場合は、その値を見て調整すると原因を追いやすくなります。

# 【補足】GPU (CUDA) 利用方法（Linux）  
  
SpeechSummarizer は GPU を使用した高速音声認識（faster-whisper）に対応しています。  
  
Linux では **NVIDIAドライバだけではGPU認識は動作しません**。  
CUDA / cuBLAS / cuDNN のランタイムが必要です。  
  
本プロジェクトでは **CUDAをシステムではなく Python 仮想環境（venv）にインストールする方式**を採用しています。  
  
## この方式のメリット  
  
- システム全体の CUDA を汚さない  
- 他のソフトと CUDA バージョンが衝突しない  
- プロジェクト単位で GPU 環境を再現できる  
  
---  
  
# 1. NVIDIA Driver の確認  
  
まず GPU ドライバが動いていることを確認します。  
  
```bash  
nvidia-smi
```
GPU が表示されれば OK です。

----------

# 2. CUDA ランタイムをインストール

仮想環境を有効化します。
```
source .venv/bin/activate
```
CUDA runtime / cuBLAS / cuDNN をインストールします。
```
pip install \  
 nvidia-cuda-runtime-cu12 \  
 nvidia-cublas-cu12 \  
 nvidia-cudnn-cu12
```

# 3. アプリ起動

アプリケーション起動用のスクリプトを用意していますので、Linux環境では
```
./SpeechSummarizer.sh
```
で起動します。

正常に動作するとログに
```
[ASR] loading model ... device=cuda
```
と表示されます。

----------


# CPU モード

GPU が無い環境では`config.json`の設定で
```
device=cpu
```
にすると CPU で動作します。
