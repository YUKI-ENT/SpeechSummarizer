# SpeechSummarizer
## 概要
SpeechSummarizer は、医療現場向けに設計されたリアルタイム音声認識＋AI要約システムです。
- Faster-Whisper による音声認識（ASR）
- GPU / CPU 両対応
- OpenAI互換API連携による SOAP 形式などの要約生成（LM Studio、FreeTokensなど）
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
├─ memo_templates.json.sample
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
インストールフォルダに有る`config.json.sample`を`config.json`に名前を変えるかコピーし、環境に合わせて編集します。`corrections.json` と `memo_templates.json` も初回起動時に各 `.sample` から自動生成されます。メモ定型文はWindows GUIの「メモ定型文」タブで追加・削除・並べ替え・編集できます。

メモ画面の「AI送信」で使用する処理一覧とプロンプトは、`config.json` の `llm.memo_prompts` と `llm.memo_default_prompt_id` で設定します。Windows GUIの「メモAI」タブからも、追加・削除・並べ替え・既定値・本文を編集できます。プロンプト本文にはASR本文の差し込み位置として `{text}` が必要です。

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
  - server / port:
    OpenAI互換APIのサーバーとポートを指定します。`server: "127.0.0.1"`, `port: 11434` なら、内部では `http://127.0.0.1:11434/v1` に接続します。
  - use_https:
    HTTPSを使う外部APIでは `true` にします。ローカルのOllamaやLM Studioでは通常 `false` です。
  - api_key:
    APIがBearer認証を要求する場合のAPIキーです。LM Studioで認証を有効にしていなければ空欄で構いません。
  - api_key_env:
    APIキーを直接保存したくない場合の環境変数名です。`api_key` が空のときだけ参照します。
  - model_default:
    デフォルトで使用するllmモデル名を指定します
  - default_prompt_id：
    デフォルトで使用するプロンプト名を下記の一覧にあるものを指定します

  設定例:

  ```json
  "llm": {
    "server": "127.0.0.1",
    "port": 11434,
    "use_https": false,
    "api_key": "",
    "api_key_env": "",
    "model_default": "使用するモデルID"
  }
  ```

  APIキーが必要なサービスでは `api_key` に設定するか、たとえば `api_key_env` を `OPENAI_API_KEY` にして、その環境変数へキーを設定してください。

- hearing_translation セクション
  難聴字幕画面の「翻訳」を有効にすると、ASRの確定セグメントごとにLLMへ問い合わせ、上段に日本語、下段に翻訳字幕を表示します。日本語字幕と翻訳字幕はそれぞれ追記され、最新部分へ自動スクロールします。翻訳はASRと分離した専用workerで受信順に処理されます。翻訳結果は保存されません。
  `model` は通常の要約用 `llm.model_default` とは独立しています。接続設定を省略するとメインの `llm` と同じOpenAI互換APIを使うため、同じOllamaから別のローカルモデルやOllama Cloudモデルを呼ぶ場合は `model` だけ指定します。別サーバーを使う場合に限り `base_url` を追加してください。認証も分ける場合は `api_key` または `api_key_env` を追加できます。

  ```json
  "hearing_translation": {
    "enabled": true,
    "model": "翻訳に使用するOllamaモデルID",
    "timeout": 30,
    "default_language": "en",
    "languages": [
      { "id": "en", "label": "英語", "name": "English" },
      { "id": "zh", "label": "中国語", "name": "Simplified Chinese" },
      { "id": "ko", "label": "韓国語", "name": "Korean" }
    ]
  }
  ```

  クライアントの難聴字幕画面で翻訳用LLMを選択できます。候補は翻訳用の接続先から取得し、選択すると `config.json` の `hearing_translation.model` に保存して、以後の翻訳へ即時反映します。保存したモデルは次回起動時も使用します。

  `languages` の `id` は画面/API用識別子、`label` は画面表示、`name` はLLMへの翻訳先指定です。

  インドネシア語を追加する場合は、ランチャーの「難聴翻訳」→「言語を追加」で、IDに `id`、表示名に `インドネシア語`、LLM向け言語名に `Indonesian` を入力して保存します。JSONを直接編集する場合は `languages` 配列へ次の項目を追加してください。言語一覧の変更後はサーバーを再起動し、クライアント画面を再読み込みします。既定にする場合は `default_language` も `id` にします。翻訳の対応・精度は選択したLLMに依存します。

  ```json
  { "id": "id", "label": "インドネシア語", "name": "Indonesian" }
  ```

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

## ASR providerの切り替え

既定の `faster-whisper` はそのまま利用できます。別プロセスで稼働するQwen3-ASR HTTP APIを使う場合は、`config.json` の `asr.provider` を変更してSpeechSummarizerを再起動します。VADとWAV分割はどちらのproviderでもSpeechSummarizer側の同じ処理を使います。

Whisperを使う設定:

```json
"asr": {
  "provider": "whisper"
}
```

localhostのQwen3-ASR APIを使う設定:

```json
"asr": {
  "provider": "qwen3-asr",
  "qwen": {
    "base_url": "http://127.0.0.1:8010",
    "timeout_sec": 35,
    "language": "Japanese",
    "context": "日本の医療現場の会話。聞こえたとおりに書き起こす。推測で補完しない。"
  }
}
```

起動前に `curl http://127.0.0.1:8010/ready` でQwen3-ASR側が `status: ready` を返すことを確認してください。Qwen側のモデルはAPIサーバー起動時に固定されます。SpeechSummarizerは`/ready`と認識応答から実際のモデル名を取得するため、Qwenのモデル変更時にSpeechSummarizer側の設定を同期する必要はありません。クライアントのモデル名は`asr.provider`と実モデル名を組み合わせ、たとえば`qwen3-asr:1.7b`と表示します。

localhostのVibeVoice-ASR APIを使う設定:

```json
"asr": {
  "provider": "vibevoice-asr",
  "vibevoice": {
    "base_url": "http://127.0.0.1:8020",
    "timeout_sec": 135,
    "language": "Japanese",
    "context": "日本の医療現場の会話。",
    "hotwords": ["滲出性中耳炎", "鼓膜切開"],
    "include_segments": true
  }
}
```

起動前に `curl http://127.0.0.1:8020/ready` でReady状態を確認してください。`include_segments` を有効にすると、APIが返した話者・timestamp付きの`segments`を各ASRレコードの`meta.asr.segments`へ保存します。通常の画面表示とLLM入力には、従来どおりAPIの`text`を使用します。

`/ready`の`backend`が`vibeasr-cpp`、`model`が`bitnet`の場合、公式モデルカードの明示対応言語に日本語は含まれません。日本語音声にはTransformers版（`backend: transformers`、通常はmodel alias `7b`または`hf`）を使用してください。

Windows GUIランチャーから利用する場合は、ASRタブでproviderを`qwen3-asr`にし、API URL、言語、Contextを設定できます。API稼働状況には`/ready`から取得したReady状態、モデルサイズ（0.6b/1.7b等）、model ID、device、queue、API versionが表示されます。「Windows GUIランチャーでQwenASRを起動・停止」を有効にして`QwenASR-Server.exe`、`config.json`、起動モデル（0.6b/1.7b）を指定すると、ランチャーは`--config`と`--model`を付けてQwenASRを先に起動し、`/ready`を確認してからSpeechSummarizer本体を起動します。QwenASRの設定ファイルは書き換えません。「Qwen再起動」でモデル変更を反映できます。ランチャーが起動したQwenASRは、サーバー停止時とランチャー終了時に一緒に停止します。すでに同じURL・選択モデルでQwenASRがreadyの場合は外部プロセスとして利用し、ランチャーからは停止しません。

Pythonから`app.py`を直接起動する場合、ランチャーによるプロセス管理は行われません。`config.json`のproviderとAPI URLを使い、QwenASRまたはVibeVoiceASRを別途起動してください。

JSONLにはprovider名、engine、実測timing、providerが実際に返したmetricsだけを保存します。`avg_logprob`、`no_speech_prob`、`compression_ratio` はWhisper結果にだけ含まれ、外部API用の代替値は生成しません。外部API使用時の品質判定は既存の音声メタデータとテキスト検査だけで行います。

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

サーバーを起動したら、ブラウザーで以下のアドレスを開きます。初回はブラウザーのマイク使用を許可してください。

- 同じPCで利用する場合： http://localhost:8000
- 別のPCから利用する場合： https://(サーバーアドレス):8000 （HTTPSの設定が必要です）

### 1. メイン画面：マイクボタンからカルテ記録まで

**自動LLM送信と電子カルテ連携を設定しておけば、マイクボタンを押すだけで、音声認識・LLMへの送信・次の患者のカルテへの遷移まで自動で動作します。** 診察のたびに文字起こしや要約を手動で送信する手間を減らせます。

1. メイン画面のマイクボタンを押して録音を開始します。会話が自動で認識され、文字起こしが表示されます。
2. 電子カルテ（ダイナミクス）で次の患者へ切り替えると、それまでの診察内容がLLMへ自動送信され、SOAP形式など、設定した形式で要約されます。
3. SpeechSummarizerの画面も次の患者のカルテへ自動で切り替わり、そのまま録音を続けられます。文字起こしや要約は患者IDごとに履歴として保存されます。

診察が終わったら、もう一度マイクボタンを押して録音を停止します。最後の患者の診察内容も、録音停止時にLLMへ自動送信されます。

自動送信には `auto_llm` と送信先モデル・プロンプトの設定、カルテの自動切り替えにはダイナミクスとの連携設定が必要です。連携しない場合は、画面上のカルテNoを手動で指定できます。

<img width="918" height="758" alt="SpeechSummarizerのメイン画面" src="https://github.com/user-attachments/assets/d33551cd-f395-49a0-adeb-9d233ac5112e" />

<details>
<summary>録音が安定しない場合の調整</summary>

- VADを `auto` にしている場合は、録音開始直後の数秒で周囲の静かな区間を使って閾値を整えます。録音開始直後から大きな音が続くと推定が安定しにくいため、できれば最初の1〜2秒は通常の環境音のままにしてください。
- 録音中のレベル表示は入力音量の目安です。必要に応じてランチャーの `VAD` 設定で `margin_db` や `start_voice_frames` を調整してください。
- `auto` モードの現在閾値と推定ノイズ床は内部で保持され、保存される JSONL / WAV メタ情報にも反映されます。録音結果が不安定な場合は、その値を見て調整すると原因を追いやすくなります。

</details>

### 2. 字幕（難聴）モード：大きな文字で会話を表示・翻訳

**カルテ用の記録を続けながら、会話の内容を大きな文字で表示できます。** 聞き取りにくい方に画面を見せながら説明するときに便利です。

- メイン画面の耳のアイコンから「字幕（難聴）モード」に切り替えると、認識された会話が大きな字幕で表示されます。
- 「翻訳」を有効にして翻訳先の言語を選ぶと、認識された発話ごとに逐次翻訳し、リアルタイムに翻訳字幕を表示できます。相手に合わせて翻訳先を切り替え、多言語でのやり取りに活用できます。
- 翻訳用のLLMモデルや利用する言語は、Windows GUIの「難聴翻訳」タブで設定できます。

[![翻訳機能のデモ動画](https://img.youtube.com/vi/Z-p6Bf8G47Y/hqdefault.jpg)](https://youtu.be/Z-p6Bf8G47Y?si=f7-s5y5hDU-Clp28)

[▶ 翻訳機能のデモ動画をYouTubeで見る](https://youtu.be/Z-p6Bf8G47Y?si=f7-s5y5hDU-Clp28)

### 3. メモ機能：カルテ記録とは別に音声入力とLLM補正

**カルテ記録とは別に、音声でメモを作成し、LLMで文章を補正・整形できます。** 備忘録や説明文の下書きなどに利用できます。

1. メイン画面の鉛筆アイコンから「音声メモ」を開き、マイクボタンで音声入力します。
2. 認識された文章をそのままメモに反映するか、処理内容とLLMモデルを選んで「AI送信」し、文章を補正・整形します。
3. 結果を確認して「メモを置き換える」または「メモ末尾に追加」で反映します。メモは手入力でも編集でき、自動保存されます。

よく使う文章は定型文として登録できます。定型文はWindows GUIの「メモ定型文」タブ、LLMに依頼する処理やプロンプトは「メモAI」タブで編集できます。

[![メモ機能の音声変換デモ動画](https://img.youtube.com/vi/jg21wgJTzrI/hqdefault.jpg)](https://youtu.be/jg21wgJTzrI?si=NjOsRpbS2-W6quw0)

[▶ メモ機能の音声変換デモ動画をYouTubeで見る](https://youtu.be/jg21wgJTzrI?si=NjOsRpbS2-W6quw0)

### 4. 補正辞書：よくある誤認識を登録して自動補正

**聞き間違えやすい用語を補正辞書に登録すると、音声認識結果を自動で補正できます。** 医療用語など、繰り返し同じ誤変換が起こる言葉の修正に役立ちます。

- 「誤変換補正ツール」で診察会話の記録を確認し、誤認識された表記と正しい表記の組み合わせを登録します。
- 手入力のほか、LLMで補正候補を抽出し、確認して辞書に追加することもできます。
- ASRモデルごとの辞書を作成できるため、モデルごとの誤認識の傾向に合わせて調整できます。補正ルールは `corrections.json` に保存されます。

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
