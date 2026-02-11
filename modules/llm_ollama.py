# llm_ollama.py
from __future__ import annotations

import json
import re
import time
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import httpx


@dataclass
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "gemma3:12b"
    timeout_sec: float = 120.0
    temperature: float = 0.0
    top_p: float = 0.9


class OllamaError(RuntimeError):
    pass



def list_ollama_models(host: str, timeout_sec: float = 10.0) -> Tuple[list[str], float]:
    """
    Ollama の /api/tags からモデル一覧を取得する。
    戻り値: (models, elapsed_sec)
    """
    url = host.rstrip("/") + "/api/tags"
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            r = client.get(url)
            r.raise_for_status()
            j = r.json()
    except Exception as e:
        raise OllamaError(f"/api/tags 取得失敗: {e}") from e

    models: list[str] = []
    for it in (j.get("models") or []):
        name = it.get("name") or it.get("model")  # 念のため
        if name:
            models.append(str(name))
    # 文字列配列の形式で返してくる実装もあるため保険
    if not models and isinstance(j.get("models"), list) and all(isinstance(x, str) for x in j["models"]):
        models = list(j["models"])

    models = sorted(set(models))
    return models, (time.time() - t0)



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


def build_soap_prompt(asr_text: str, *, style: str = "soap_v1") -> str:
    """
    SOAP形式の要約（推測禁止・追記禁止）。
    style:
      - soap_v1: 標準（推測禁止）
      - soap_v1_short: 短め（重要点のみ）
    """
    style = (style or "soap_v1").strip()

    common = """# 重要ルール
- 勝手に内容を追加しない（推測禁止）
- 喋った内容に沿って記述し、わからないことは書かない
- 内容が不明確な箇所は「不明」としてよい（創作しない）
"""

    if style == "soap_v1_short":
        extra = """# 出力の長さ
- 1項目あたり最大3行程度。重要点のみ。
"""
    else:
        extra = ""

    return f"""以下は医者と患者の診察室での会話で主に医者の発言部分ですが、音声認識で一部同音異義語の書き違いがあります。
これを考慮して、SOAP形式にまとめてください。

{common}
{extra}
# 出力形式
次の見出しで、読みやすい箇条書きでまとめてください。

SOAP記録

Subjective（主観）
Objective（客観）
Assessment（評価）
Plan（計画）

# 入力
<<<
{asr_text}
>>>
"""



def ollama_generate_text(
    cfg: OllamaConfig,
    prompt: str,
    *,
    num_ctx: int = 4096,
) -> Tuple[str, Dict[str, Any]]:
    """Ollama /api/generate を使ってテキストを返す（format指定なし）。"""
    url = cfg.host.rstrip("/") + "/api/generate"
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
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

    response_text = data.get("response", "")
    if not isinstance(response_text, str) or not response_text.strip():
        raise OllamaError(f"Ollama responseが空です: keys={list(data.keys())}")

    data["_elapsed_sec"] = round(time.time() - t0, 3)
    return response_text.strip(), data


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

_DEFAULT_PROMPTS = {
    "soap_v1": {
        "label": "SOAP(推測禁止)",
        "template": (
            "以下は医者と患者の診察室での会話（主に医者の発言）です。"
            "音声認識で一部同音異義語の書き違いがあります。これを考慮してSOAP形式にまとめてください。\n"
            "制約:\n"
            "- 勝手に内容を追加しない\n"
            "- 喋った内容に沿って記述\n"
            "- わからないことは記載しない（推測しない）\n\n"
            "会話テキスト:\n{asr_text}\n"
        ),
    },
    "soap_v1_short": {
        "label": "SOAP(短め)",
        "template": (
            "以下の診察会話を、推測せずSOAP形式で簡潔にまとめてください。\n\n"
            "会話テキスト:\n{asr_text}\n"
        ),
    },
}

def _load_llm_json(path: str = "llm.json") -> dict:
    if not path:  # ← None / "" 対策
        path = "llm.json"
    p = Path(path)
    if not p.exists():
        return {"prompts": _DEFAULT_PROMPTS}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("llm.json is not an object")
        prompts = data.get("prompts")
        if not isinstance(prompts, dict) or not prompts:
            return {"prompts": _DEFAULT_PROMPTS}
        return data
    except Exception:
        # 壊れてても落とさない
        return {"prompts": _DEFAULT_PROMPTS}

def list_prompt_items(path: str = "llm.json"):
    """
    UI用：プロンプト一覧
    return: [{"id":"soap_v1","label":"..."} ...]
    """
    if not path:
        path = "llm.json"
    catalog = _load_llm_json(path)
    prompts = catalog.get("prompts", _DEFAULT_PROMPTS)
    items = []
    for pid, p in prompts.items():
        if not isinstance(p, dict):
            continue
        label = p.get("label") or pid
        items.append({"id": pid, "label": label})
    return items

def build_prompt(prompt_id: str, asr_text: str, path: str = "llm.json") -> str:
    if not path:
        path = "llm.json"
    catalog = _load_llm_json(path)
    prompts = catalog.get("prompts", _DEFAULT_PROMPTS)
    p = prompts.get(prompt_id) or prompts.get("soap_v1") or _DEFAULT_PROMPTS["soap_v1"]
    template = p.get("template") or _DEFAULT_PROMPTS["soap_v1"]["template"]
    # {asr_text} が無いテンプレでも破綻しないように
    if "{asr_text}" in template:
        return template.replace("{asr_text}", asr_text)
    return template + "\n\n" + asr_text