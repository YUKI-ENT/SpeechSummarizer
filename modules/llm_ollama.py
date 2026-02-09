# llm_ollama.py
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "gemma3:12b"
    timeout_sec: float = 120.0


class OllamaError(RuntimeError):
    pass


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # ```json ... ``` や ``` ... ``` を除去
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _extract_first_json_object(s: str) -> str:
    """
    文字列から最初に出てくるJSONオブジェクト { ... } を雑に抜き出す。
    Ollamaが前後に説明文を混ぜたりしても耐えるため。
    """
    s = _strip_code_fences(s)
    # 最初の '{' から最後の '}' までを最大貪欲で取る（荒いが実務で強い）
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise OllamaError(f"JSONオブジェクトが見つかりません: {s[:200]!r}")
    return m.group(0)


def parse_ollama_json(response_text: str) -> Dict[str, Any]:
    """
    Ollamaの返答からJSONを安全に取り出す。
    """
    raw = _extract_first_json_object(response_text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise OllamaError(f"JSONパース失敗: {e}\n---raw---\n{raw[:5000]}") from e


def build_s_extraction_prompt(asr_text: str) -> str:
    """
    “S(症状/経過)”向けの抽出JSONを返すプロンプト。
    - 推測禁止
    - time.textは原文から抜き出しのみ
    - 時間は正規化しない（並べ替え用 rank のみ）
    """
    return f"""あなたは医療会話の文字起こしから、S(症状/経過)の「事実」だけを抽出します。
次のルールを厳守して JSON のみを返してください。

# ルール
- 入力に書いてない情報を追加しない（推測禁止）
- 症状名は入力に出てきた表現を基本にする（不明なら "unknown"）
- 時間表現は必ず入力からそのまま抜き出して time.text に入れる（作らない）
- 時間表現が無ければ time は null
- 次の type のどれかに分類する:
  - relative_day (例: 昨日, 3日前, 今日)
  - relative_week (例: 先週)
  - relative_weeks (例: 2週間前)
  - relative_months (例: 1ヶ月前, 数ヶ月前)
  - relative_year (例: 去年, 何年も前)
  - weekday (例: 月曜, 水曜日)
  - calendar_date (例: 2/7, 7日, 2026年2月7日)
  - seasonal_recurring (例: 毎年春, 毎冬)
  - unknown
- 並べ替え用に time.rank を付ける（小さいほど古い）。目安:
  - relative_year: 20
  - relative_months: 40
  - relative_weeks: 55
  - relative_week: 60
  - weekday: 75〜84（例: 月曜=80, 水曜=82 など）
  - relative_day: 80〜95（例: 3日前=85, 昨日=90, 今日=95）
  - calendar_date: 85（不明確ならこのあたり）
  - seasonal_recurring: rank は null、bucket は "seasonality"
- 「〜くらい」「〜およそ」など曖昧なら approx=true
- 自信がなければ uncertain=true を付ける
- 出力は次の JSON スキーマに従うこと

# JSONスキーマ
{{
  "events": [
    {{
      "symptom": string,
      "polarity": "+|-|unknown",
      "time": {{
        "text": string,
        "type": string,
        "rank": number|null,
        "approx": boolean,
        "bucket": "timeline"|"seasonality",
        "uncertain": boolean
      }} | null
    }}
  ],
  "context": [
    {{"item": string, "text": string, "bucket": "seasonality"}}
  ]
}}

# 入力
<<<
{asr_text}
>>>
"""


def ollama_generate_json(
    cfg: OllamaConfig,
    prompt: str,
    *,
    temperature: float = 0.2,
    top_p: float = 0.9,
    num_ctx: int = 4096,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Ollama /api/generate を使って JSON を返す。
    - stream: false
    - format: json を要求（モデルが守らない場合もあるので parse_ollama_jsonで保険）
    戻り値: (parsed_json, raw_response_dict)
    """
    url = cfg.host.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
        },
    }

    t0 = time.time()
    try:
        with httpx.Client(timeout=cfg.timeout_sec) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise OllamaError(f"Ollama呼び出し失敗: {e}") from e

    # Ollamaの返答は {"response":"...","done":true,...} 形式が多い
    response_text = data.get("response", "")
    if not isinstance(response_text, str) or not response_text.strip():
        raise OllamaError(f"Ollama responseが空です: keys={list(data.keys())}")

    parsed = parse_ollama_json(response_text)
    data["_elapsed_sec"] = round(time.time() - t0, 3)
    return parsed, data
