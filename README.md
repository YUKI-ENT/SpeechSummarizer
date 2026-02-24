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

[![SpeechSummarizer Demo](https://img.youtube.com/vi/pZM9rbGqX3A/maxresdefault.jpg)](https://youtu.be/pZM9rbGqX3A)

## インストール方法

### 【Windows版（EXE）】

1. release の zip をダウンロード
2. 任意のフォルダに展開

  フォルダ構成例：
```
SpeechSummarizer/
├─ SpeechSummarizer.exe
├─ _internal/
├─ config.json.sample
├─ correct.json
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

## config.jsonの設定
インストールフォルダに有る`config.json.sample`を`config.json`に名前を変えるかコピーし、環境に合わせて編集します。

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
  ![スクリーンショット 2026-02-18 115919](https://github.com/user-attachments/assets/831438fd-9726-43ac-b048-bab42003ec99)
- この画面が表示されない場合は、`config.json` 等の設定を見直し、一度**コマンドプロンプト**や **powershell**から実行してみて、エラーメッセージを確認してみてください。
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

## 利用方法
- クライアントからローカルの場合は

  http://localhost:8000

  サーバーが別PCの場合は

  https://(サーバーアドレス):8000

  で画面が立ち上がります。

  ![スクリーンショット 2026-02-18 120922](https://github.com/user-attachments/assets/f72fad23-cc97-4266-aaa9-fa281dba0e8f)
