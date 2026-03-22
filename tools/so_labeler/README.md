# S/O Boundary Reviewer

SpeechSummarizer の既存 jsonl を 1 診察単位でレビューし、S(問診)から O(診察・所見)へ切り替わる位置を確定するためのツールです。

## いまの流れ
- data フォルダ配下の jsonl 一覧取得
- `patient_data.jsonl` を除外して `type=asr` イベントを読む
- 1 jsonl = 1 診察として会話全文を表示
- Ollama または stub で「最初の切り替わり位置」を 1 箇所だけ提案
- 人手で区切り位置と trigger phrase を確定して保存
- 保存済み JSONL から、S→O 境界トリガー辞書を抽出

## LLM 設定
- Ollama モデルは UI から選択
- prompt は `config.json` の `llm.so_labeler_prompts` から選択
- 既定 prompt は `llm.so_labeler_default_prompt_id`
