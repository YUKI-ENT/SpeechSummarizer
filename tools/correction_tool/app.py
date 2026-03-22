from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
EXCLUDED_JSONL_NAMES = {"patient_data.jsonl"}
CORRECTION_RESPONSE_SUFFIX = (
    "\n\n出力ルール:\n"
    "- 返答は JSON のみ。説明文、Markdown、コードフェンスは禁止。\n"
    "- `<think>` や思考過程は出力しないこと。\n"
    "- 候補が1件でも、必ず JSON配列で返すこと。\n"
    '- 1件の例: [{"wrong":"セキ","correct":"咳","reason":"文脈上「咳」が適切"}]\n'
    "- 候補が0件なら [] を返すこと。\n"
)

app = FastAPI(title="Correction Dictionary Builder")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class FileListRequest(BaseModel):
    data_dir: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SessionLoadRequest(BaseModel):
    file_paths: list[str]


class SuggestRequest(BaseModel):
    source_file: str
    turns: list[dict[str, Any]]
    model: str = ""
    prompt_id: str = ""


class RuleUpsertRequest(BaseModel):
    model_name: str = ""
    wrong: str
    correct: str


class RuleDeleteRequest(BaseModel):
    model_name: str = ""
    wrong: str


SUGGEST_CHUNK_SIZE = 4


def _debug_log(message: str) -> None:
    print(message, flush=True)


def _debug_dump(label: str, payload: Any) -> None:
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        text = str(payload)
    _debug_log(f"{label}\n{text}")


def _summarize_ollama_response(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": data.get("model"),
        "created_at": data.get("created_at"),
        "done": data.get("done"),
        "done_reason": data.get("done_reason"),
        "response": data.get("response"),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_count": data.get("eval_count"),
        "eval_duration": data.get("eval_duration"),
    }


def _build_correction_prompt(template: str, transcript: str) -> str:
    prompt = template.format(transcript=transcript)
    if CORRECTION_RESPONSE_SUFFIX not in prompt:
        prompt += CORRECTION_RESPONSE_SUFFIX
    return prompt


def _parse_date_text(value: Optional[str]):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def _load_rules(path: Path) -> dict:
    if not path.exists():
        return {"version": None, "replacements": {}, "model_rules": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("corrections.json must be an object")
    data.setdefault("replacements", {})
    data.setdefault("model_rules", {})
    return data


def _save_rules(path: Path, rules: dict) -> None:
    rules["version"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


def _effective_replacements(rules: dict, model_name: str = "") -> dict[str, str]:
    base = rules.get("replacements") or {}
    scoped = ((rules.get("model_rules") or {}).get(model_name) or {}).get("replacements") or {}
    out = {}
    if isinstance(base, dict):
        out.update({str(k): str(v) for k, v in base.items()})
    if isinstance(scoped, dict):
        out.update({str(k): str(v) for k, v in scoped.items()})
    return out


def _apply_preview(text: str, rules: dict, model_name: str = "") -> str:
    out = text
    for wrong, correct in _effective_replacements(rules, model_name).items():
        out = out.replace(wrong, correct)
    return out


def _list_jsonl_files(data_dir: str, start_date: Optional[str], end_date: Optional[str]) -> list[dict]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    start_dt = _parse_date_text(start_date)
    end_dt = _parse_date_text(end_date)
    out = []
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in EXCLUDED_JSONL_NAMES:
            continue
        session_ts = None
        patient_id = None
        asr_count = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    row = json.loads(line)
                    if i == 0 and row.get("type") == "session_start":
                        session_ts = row.get("ts")
                        patient_id = row.get("patient_id")
                    if row.get("type") == "asr" and (row.get("text") or "").strip():
                        asr_count += 1
        except Exception:
            pass
        file_dt = None
        if session_ts:
            try:
                file_dt = datetime.fromisoformat(session_ts.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                file_dt = None
        if start_dt and file_dt and file_dt.date() < start_dt.date():
            continue
        if end_dt and file_dt and file_dt.date() > end_dt.date():
            continue
        out.append({
            "path": str(path),
            "name": path.name,
            "session_ts": session_ts,
            "patient_id": patient_id,
            "asr_count": asr_count,
        })
    return out


def _load_session(file_path: str) -> dict[str, Any]:
    path = Path(file_path)
    turns = []
    model_counter: Counter[str] = Counter()
    patient_id = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") != "asr":
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            asr_cfg = row.get("asr_cfg") or {}
            model_name = str(asr_cfg.get("model_name") or "").strip()
            if model_name:
                model_counter.update([model_name])
            patient_id = patient_id or row.get("patient_id")
            turns.append({
                "index": len(turns),
                "ts": row.get("ts", ""),
                "seg_id": row.get("seg_id"),
                "text": text,
                "model_name": model_name,
            })
    detected_model = model_counter.most_common(1)[0][0] if model_counter else ""
    return {
        "session_id": path.name,
        "source_file": path.name,
        "patient_id": patient_id,
        "turns": turns,
        "detected_model": detected_model,
        "detected_models": [{"name": name, "count": count} for name, count in model_counter.most_common()],
    }


def _prompt_items(request: Request) -> tuple[list[dict], dict[str, str], str]:
    items = getattr(request.app.state, "prompt_items", None) or [{"id": "correction_v1", "label": "誤変換補正候補 v1"}]
    templates = getattr(request.app.state, "prompt_templates", None) or {
        "correction_v1": (
            "以下は耳鼻科診察会話のASRテキストです。明らかな誤変換だけを高精度で抽出してください。"
            "返答はJSON配列のみで、各要素は"
            '{{"wrong":"誤変換","correct":"正しい語","reason":"短い理由"}}'
            "の形式。最大10件。自信が低いものは出さないでください。\n\n"
            "{transcript}\n"
        )
    }
    default_prompt_id = getattr(request.app.state, "default_prompt_id", None) or items[0]["id"]
    return items, templates, default_prompt_id


def _strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*think\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_first_json_value(text: str) -> str:
    match_positions = [pos for pos in (text.find("["), text.find("{")) if pos >= 0]
    start = min(match_positions) if match_positions else -1
    if start < 0:
        raise ValueError("no JSON value found in LLM response")
    depth = 0
    in_string = False
    escaped = False
    opening = text[start]
    closing = "]" if opening == "[" else "}"
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    raise ValueError("unterminated JSON value in LLM response")


def _normalize_suggestion_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        if "wrong" in payload and "correct" in payload:
            return [payload]
    return []


def _chunk_turns(turns: list[dict[str, Any]], chunk_size: int = SUGGEST_CHUNK_SIZE) -> list[list[dict[str, Any]]]:
    cleaned = [turn for turn in turns if str(turn.get("text") or "").strip()]
    return [cleaned[idx:idx + chunk_size] for idx in range(0, len(cleaned), chunk_size)]


def _transcript_from_turns(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{turn['index']}] {turn['text']}" for turn in turns)


def _merge_suggestion_items(chunks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for items in chunks:
        for item in items:
            wrong = str(item.get("wrong") or "").strip()
            correct = str(item.get("correct") or "").strip()
            if not wrong or not correct:
                continue
            key = (wrong, correct)
            row = merged.setdefault(
                key,
                {
                    "wrong": wrong,
                    "correct": correct,
                    "reason": str(item.get("reason") or "").strip(),
                    "sources": [],
                },
            )
            reason = str(item.get("reason") or "").strip()
            if reason and not row["reason"]:
                row["reason"] = reason
            for source in item.get("sources") or []:
                if source not in row["sources"]:
                    row["sources"].append(source)
    return sorted(
        merged.values(),
        key=lambda item: (-len(item.get("sources") or []), item["wrong"], item["correct"]),
    )


def _suggest_with_ollama(request: Request, transcript: str, model: str, prompt_id: str) -> list[dict]:
    llm_cfg = getattr(request.app.state, "llm_config", None) or {}
    base_url = str(llm_cfg.get("base_url") or "http://127.0.0.1:11434")
    timeout_sec = float(llm_cfg.get("timeout_sec") or 120)
    temperature = float(llm_cfg.get("temperature") or 0.0)
    top_p = float(llm_cfg.get("top_p") or 0.9)
    _, templates, default_prompt_id = _prompt_items(request)
    template = templates.get(prompt_id or default_prompt_id) or next(iter(templates.values()))
    prompt = _build_correction_prompt(template, transcript)
    body = {
        "model": model or str(llm_cfg.get("model") or "qwen3.5:9b"),
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": temperature, "top_p": top_p},
    }
    _debug_log(
        f"[correction_tool][ollama_start] model={body['model']} prompt_id={prompt_id or default_prompt_id} "
        f"base_url={base_url.rstrip('/')} prompt_chars={len(prompt)} transcript_chars={len(transcript)}"
    )
    _debug_dump("[correction_tool][ollama_request]", body)
    resp = requests.post(base_url.rstrip("/") + "/api/generate", json=body, timeout=timeout_sec)
    resp.raise_for_status()
    _debug_log(f"[correction_tool][ollama_http] status={resp.status_code}")
    _debug_log(f"[correction_tool][ollama_raw_response]\n{resp.text}")
    data = resp.json()
    _debug_dump("[correction_tool][ollama_response_summary]", _summarize_ollama_response(data))
    text = _strip_think_blocks(str(data.get("response") or ""))
    _debug_log(f"[correction_tool][ollama_done] response_chars={len(text)}")
    if not text.strip():
        _debug_log("[correction_tool][ollama_done] empty response text")
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = json.loads(_extract_first_json_value(text))
    _debug_dump("[correction_tool][ollama_parsed_payload]", payload)
    arr = _normalize_suggestion_payload(payload)
    out = []
    for item in arr:
        wrong = str(item.get("wrong") or "").strip()
        correct = str(item.get("correct") or "").strip()
        if not wrong or not correct or wrong == correct:
            continue
        out.append({
            "wrong": wrong,
            "correct": correct,
            "reason": str(item.get("reason") or "").strip(),
        })
    _debug_dump("[correction_tool][ollama_filtered_items]", out)
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def api_config(request: Request):
    llm_cfg = getattr(request.app.state, "llm_config", None) or {}
    items, _, default_prompt_id = _prompt_items(request)
    return {
        "default_data_dir": getattr(request.app.state, "default_data_dir", str(BASE_DIR.parent.parent / "data")),
        "llm_base_url": str(llm_cfg.get("base_url") or "http://127.0.0.1:11434"),
        "llm_model": str(llm_cfg.get("model") or "qwen3.5:9b"),
        "prompt_items": items,
        "default_prompt_id": default_prompt_id,
        "correction_rules_path": str(getattr(request.app.state, "correction_rules_path", BASE_DIR.parent.parent / "corrections.json")),
        "asr_models": getattr(request.app.state, "asr_models", []),
    }


@app.get("/api/ollama_models")
def api_ollama_models(request: Request):
    llm_cfg = getattr(request.app.state, "llm_config", None) or {}
    try:
        resp = requests.get(str(llm_cfg.get("base_url") or "http://127.0.0.1:11434").rstrip("/") + "/api/tags", timeout=float(llm_cfg.get("timeout_sec") or 120))
        resp.raise_for_status()
        data = resp.json()
        models = sorted([str(item.get("name") or "").strip() for item in data.get("models") or [] if item.get("name")])
        return {"ok": True, "models": models, "default_model": str(llm_cfg.get("model") or "")}
    except Exception as e:
        return {"ok": False, "models": [], "default_model": str(llm_cfg.get("model") or ""), "error": str(e)}


@app.post("/api/files")
def api_files(req: FileListRequest):
    return {"items": _list_jsonl_files(req.data_dir, req.start_date, req.end_date)}


@app.post("/api/sessions")
def api_sessions(req: SessionLoadRequest):
    return {"items": [_load_session(path) for path in req.file_paths]}


@app.get("/api/rules")
def api_rules(request: Request, model_name: str = ""):
    rules_path = Path(getattr(request.app.state, "correction_rules_path", BASE_DIR.parent.parent / "corrections.json"))
    rules = _load_rules(rules_path)
    return {
        "ok": True,
        "rules": rules,
        "effective_replacements": _effective_replacements(rules, model_name),
    }


@app.post("/api/preview")
def api_preview(request: Request, payload: dict[str, Any]):
    rules_path = Path(getattr(request.app.state, "correction_rules_path", BASE_DIR.parent.parent / "corrections.json"))
    rules = _load_rules(rules_path)
    text = str(payload.get("text") or "")
    model_name = str(payload.get("model_name") or "")
    return {"ok": True, "text": _apply_preview(text, rules, model_name)}


@app.post("/api/suggest")
def api_suggest(request: Request, req: SuggestRequest):
    chunks = _chunk_turns(req.turns)
    transcript = _transcript_from_turns(req.turns)
    _debug_dump(
        "[correction_tool][suggest_request]",
        {
            "source_file": req.source_file,
            "model": req.model,
            "prompt_id": req.prompt_id,
            "turn_count": len(req.turns),
            "chunk_size": SUGGEST_CHUNK_SIZE,
            "chunk_count": len(chunks),
            "transcript_chars": len(transcript),
            "transcript": transcript,
        },
    )
    try:
        chunk_results: list[list[dict[str, Any]]] = []
        for chunk_index, chunk_turns in enumerate(chunks):
            chunk_transcript = _transcript_from_turns(chunk_turns)
            _debug_dump(
                "[correction_tool][suggest_chunk_request]",
                {
                    "chunk_index": chunk_index,
                    "turn_indexes": [turn.get("index") for turn in chunk_turns],
                    "transcript_chars": len(chunk_transcript),
                    "transcript": chunk_transcript,
                },
            )
            items = _suggest_with_ollama(request, chunk_transcript, req.model, req.prompt_id)
            chunk_items = []
            for item in items:
                chunk_items.append({
                    **item,
                    "sources": [f"#{turn.get('index')}" for turn in chunk_turns],
                })
            _debug_dump(
                "[correction_tool][suggest_chunk_result]",
                {"chunk_index": chunk_index, "item_count": len(chunk_items), "items": chunk_items},
            )
            chunk_results.append(chunk_items)
        merged_items = _merge_suggestion_items(chunk_results)
        _debug_dump("[correction_tool][suggest_result]", {"item_count": len(merged_items), "items": merged_items})
        return {"ok": True, "items": merged_items}
    except Exception as e:
        _debug_log(f"[correction_tool][suggest_error] {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail={"message": str(e)}) from e


@app.post("/api/rules/upsert")
def api_rules_upsert(request: Request, req: RuleUpsertRequest):
    rules_path = Path(getattr(request.app.state, "correction_rules_path", BASE_DIR.parent.parent / "corrections.json"))
    rules = _load_rules(rules_path)
    wrong = req.wrong.strip()
    correct = req.correct.strip()
    if not wrong or not correct:
        raise HTTPException(status_code=400, detail="wrong and correct are required")
    if req.model_name.strip():
        model_rules = rules.setdefault("model_rules", {})
        scoped = model_rules.setdefault(req.model_name.strip(), {})
        replacements = scoped.setdefault("replacements", {})
        replacements[wrong] = correct
    else:
        replacements = rules.setdefault("replacements", {})
        replacements[wrong] = correct
    _save_rules(rules_path, rules)
    return {"ok": True, "rules": rules}


@app.post("/api/rules/delete")
def api_rules_delete(request: Request, req: RuleDeleteRequest):
    rules_path = Path(getattr(request.app.state, "correction_rules_path", BASE_DIR.parent.parent / "corrections.json"))
    rules = _load_rules(rules_path)
    wrong = req.wrong.strip()
    if req.model_name.strip():
        replacements = (((rules.get("model_rules") or {}).get(req.model_name.strip()) or {}).get("replacements") or {})
        replacements.pop(wrong, None)
    else:
        (rules.get("replacements") or {}).pop(wrong, None)
    _save_rules(rules_path, rules)
    return {"ok": True, "rules": rules}
