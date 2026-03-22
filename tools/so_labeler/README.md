# S/O Boundary Reviewer

SpeechSummarizer の既存 jsonl から、問診(S)から診察(O)へ切り替わる境界候補をレビューし、トリガー句辞書を作るためのツールです。

## いまの流れ
- data フォルダ配下の jsonl 一覧取得
- `patient_data.jsonl` を除外して `type=asr` イベントを読む
- 発話の境界ごとに S→O 候補を生成
- Ollama または stub で境界の仮判定と trigger phrase を作成
- 1件ずつレビューして `human_is_boundary` と `human_trigger_phrases` を保存
- 保存済み JSONL から、S→O 境界トリガー辞書を抽出

## 主なファイル
- `app.py` FastAPI エントリ
- `jsonl_loader.py` jsonl 一覧取得 / ASR 読み込み
- `segmenter.py` 発話境界候補の生成
- `llm_client.py` Ollama / stub による境界判定
- `reviewer_store.py` 教師データ JSONL 保存
- `rule_extractor.py` trigger phrase 辞書生成
- `templates/index.html` 画面
- `static/app.js` フロント処理
- `static/style.css` スタイル

## 保存される教師データ
- `llm_is_boundary`
- `llm_trigger_phrases`
- `human_is_boundary`
- `human_trigger_phrases`
- `human_note`

## 出力される辞書
```json
{
  "rule_type": "s_to_o_boundary_triggers",
  "triggers": {
    "見ていきますね": {
      "score": 3.0,
      "examples": [
        {
          "source_file": "...",
          "prev_text": "...",
          "next_text": "..."
        }
      ]
    }
  }
}
```
