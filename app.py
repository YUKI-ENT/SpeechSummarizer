import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import wave
import json
import datetime
import asyncio
import sys
from pathlib import Path
from urllib.parse import unquote
import re
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from faster_whisper import WhisperModel
import httpx
import hashlib
# from datetime import datetime, timedelta

"""
=========================
この版での修正点（重要）
=========================
[1] llm.json を廃止し、config.json の llm.prompts をそのまま使う
    - items変換/正規化/旧形式互換などは全部削除
    - /api/llm/prompts は config の prompts を列挙するだけ

[2] models.json を廃止し、config.json の asr.models をそのまま使う
    - load_models_registry() は config から読む
    - model_id は config の asr.model_id のまま（default_model は作らない）

[3] 既存の ASR→jsonl、LLM→jsonl good以上抽出の流れは維持
    - LLM問い合わせは jsonl から min_quality 以上のみ送信（現状通り）
"""
# -------------------------
# App version (server software version)
# -------------------------
# ここを書き換えるだけでUI表示が変わる（config.json には置かない）
APP_VERSION = "20260224"

# -------------------------
# Config
# -------------------------
def get_app_dir() -> Path:
    # PyInstaller等で frozen のときは exe の場所
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # 通常実行時はこの .py の場所
    return Path(__file__).resolve().parent

APP_DIR = get_app_dir()
CONFIG_PATH = APP_DIR / "config.json"
CONFIG_SAMPLE_PATH = APP_DIR / "config.json.sample"

_CONFIG_CACHE: Optional[dict] = None

def load_config(force_reload: bool = False) -> dict:
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    if not CONFIG_PATH.exists():
        # 初回起動: config.json が無ければ sample から生成（既存を上書きしない）
        if CONFIG_SAMPLE_PATH.exists():
            try:
                shutil.copyfile(CONFIG_SAMPLE_PATH, CONFIG_PATH)
            except Exception as e:
                raise RuntimeError(f"failed to copy config.json.sample -> config.json: {e}")
        else:
            raise FileNotFoundError(
                f"config.json not found: {CONFIG_PATH} (and sample not found: {CONFIG_SAMPLE_PATH})"
            )

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 必須キー検証
    if "asr" not in cfg:
        raise ValueError("config.json missing 'asr' section")
    if "llm" not in cfg:
        raise ValueError("config.json missing 'llm' section")

    # llm.prompts 必須（空でもよいが、dictであること）
    llm = cfg.get("llm") or {}
    prompts = llm.get("prompts")
    if prompts is None:
        raise ValueError("config.json missing 'llm.prompts'")
    if not isinstance(prompts, dict):
        raise ValueError("config.json 'llm.prompts' must be an object/dict")

    # asr.models 必須（空はダメ：モデルが選べない）
    asr = cfg.get("asr") or {}
    models = asr.get("models")
    if models is None:
        raise ValueError("config.json missing 'asr.models'")
    if not isinstance(models, dict):
        raise ValueError("config.json 'asr.models' must be an object/dict")

    _CONFIG_CACHE = cfg
    return cfg

CFG = load_config()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
def resolve_relpath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (APP_DIR / p)

OUTPUTS_DIR = resolve_relpath(CFG.get("outputs_dir", "data/sessions"))
WAV_DIR = resolve_relpath(CFG.get("wav_dir", "data/wav"))
ensure_dir(OUTPUTS_DIR)
ensure_dir(WAV_DIR)

CORRECTION_RULES_PATH = resolve_relpath(CFG.get("correction_rules_path", "corrections.json"))
CORRECTIONS_DIR = resolve_relpath(CFG.get("corrections_dir", "data/corrections"))
ensure_dir(CORRECTIONS_DIR)
_CORRECTION_CACHE: Optional[dict] = None

def load_correction_rules(force_reload: bool = False) -> dict:
    global _CORRECTION_CACHE
    if _CORRECTION_CACHE is not None and not force_reload:
        return _CORRECTION_CACHE

    if not CORRECTION_RULES_PATH.exists():
        # ルールが無いなら「何もしない」扱いにする
        _CORRECTION_CACHE = {"version": None}
        return _CORRECTION_CACHE

    with CORRECTION_RULES_PATH.open("r", encoding="utf-8") as f:
        rules = json.load(f)

    if not isinstance(rules, dict):
        raise ValueError("corrections.json must be an object")

    _CORRECTION_CACHE = rules
    return rules

def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def apply_corrections(text: str, rules: dict) -> tuple[str, dict]:
    """
    行単位で補正:
      - replacements（単純置換）
      - regex_replacements（正規表現置換）
      - blacklist_words（単語除去）
      - drop_regex / min_chars_per_line（行の破棄）
    """
    if not text:
        return text, {"changed": False, "lines_in": 0, "lines_out": 0}

    repl = rules.get("replacements") or {}
    if not isinstance(repl, dict):
        repl = {}

    regex_repls = rules.get("regex_replacements") or []
    if not isinstance(regex_repls, list):
        regex_repls = []

    blacklist = rules.get("blacklist_words") or []
    if not isinstance(blacklist, list):
        blacklist = []

    drop_regex = rules.get("drop_regex") or []
    if not isinstance(drop_regex, list):
        drop_regex = []

    min_chars = rules.get("min_chars_per_line")
    try:
        min_chars = int(min_chars) if min_chars is not None else 0
    except Exception:
        min_chars = 0

    # 事前コンパイル
    compiled_drop = []
    for pat in drop_regex:
        try:
            compiled_drop.append(re.compile(pat))
        except Exception:
            pass

    compiled_regex_repls = []
    for it in regex_repls:
        if not isinstance(it, dict):
            continue
        pat = it.get("pattern")
        rep = it.get("repl", "")
        if not isinstance(pat, str):
            continue
        try:
            compiled_regex_repls.append((re.compile(pat), str(rep)))
        except Exception:
            pass

    lines = text.splitlines()
    out_lines: list[str] = []

    changed_count = 0
    dropped_count = 0

    for line in lines:
        s = line.strip()
        if not s:
            continue

        before = s

        # 単純置換
        for a, b in repl.items():
            if not isinstance(a, str) or not isinstance(b, str) or not a:
                continue
            s = s.replace(a, b)

        # 正規表現置換
        for cre, rep in compiled_regex_repls:
            s = cre.sub(rep, s)

        # blacklist除去（部分一致で消す簡易版）
        for w in blacklist:
            if isinstance(w, str) and w:
                s = s.replace(w, "")

        s = s.strip()
        if not s:
            dropped_count += 1
            continue

        if min_chars > 0 and len(s) < min_chars:
            dropped_count += 1
            continue

        drop_hit = False
        for cre in compiled_drop:
            if cre.match(s):
                drop_hit = True
                break
        if drop_hit:
            dropped_count += 1
            continue

        if s != before:
            changed_count += 1

        out_lines.append(s)

    out_text = "\n".join(out_lines).strip() + ("\n" if out_lines else "")
    stats = {
        "changed": (out_text != (text.strip() + ("\n" if text.strip() else ""))),
        "lines_in": len(lines),
        "lines_out": len(out_lines),
        "changed_lines": changed_count,
        "dropped_lines": dropped_count,
        "rules_version": rules.get("version"),
    }
    return out_text, stats

def append_jsonl_line(jsonl_path: Path, obj: dict):
    s = json.dumps(obj, ensure_ascii=False)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


# -------------------------
# Logging / time
# -------------------------
def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------
# Session manager (single global session)
# -------------------------
CURRENT = {
    "patient_id": None,
    "session_stamp": None,
    "text_path": None,
    "jsonl_path": None,
    "full_text": "",

    "asr": {
        "model_name": None,
        "model_path": None,
        "language": "ja",
        "temperature": 0.0,
        "prompt": "",
    },

    "dict": {
        "version": None,
        "rules_path": None,
        "applied_count": 0,
    },
    "edits": {
        "enabled": True,
        "count": 0,
    }
}

def _normalize_jsonl_text(v: str) -> str:
    """
    JSONL安全化：改行・タブ等をスペースに潰し、連続空白を整理。
    1行=1JSON を壊さないための責務を append_jsonl に閉じ込める。
    """
    if not isinstance(v, str):
        return v
    v = v.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    v = re.sub(r"\s{2,}", " ", v)
    return v.strip()


def append_jsonl(obj: dict, jsonl_path: str):
    # text / transcript / summary など「改行が入りうるフィールド」を安全化
    for k in ("text", "transcript", "summary"):
        if k in obj and isinstance(obj[k], str):
            obj[k] = _normalize_jsonl_text(obj[k])

    line = json.dumps(obj, ensure_ascii=False)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_text_line(text_path: str, s: str):
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def ensure_unknown_session():
    if CURRENT["patient_id"] is None:
        new_session("unknown")

def new_session(patient_id: str):
    stamp = now_stamp()
    txt = OUTPUTS_DIR / f"{patient_id}_{stamp}.txt"
    js  = OUTPUTS_DIR / f"{patient_id}_{stamp}.jsonl"
    CURRENT["patient_id"] = patient_id
    CURRENT["session_stamp"] = stamp
    CURRENT["text_path"] = str(txt)
    CURRENT["jsonl_path"] = str(js)
    CURRENT["full_text"] = ""

    session_meta = {
        "asr": dict(CURRENT.get("asr", {})),
        "dict": dict(CURRENT.get("dict", {})),
        "edits": dict(CURRENT.get("edits", {})),
    }

    append_jsonl(
        {
            "type": "session_start",
            "ts": now_iso(),
            "patient_id": patient_id,
            "stamp": stamp,
            "meta": session_meta,
        },
        CURRENT["jsonl_path"]
    )

def append_asr_config_event(reason: str):
    try:
        if CURRENT.get("jsonl_path"):
            append_jsonl({
                "type": "asr_config",
                "ts": now_iso(),
                "patient_id": CURRENT.get("patient_id"),
                "stamp": CURRENT.get("session_stamp"),
                "reason": reason,
                "asr": dict(CURRENT.get("asr", {})),
            }, CURRENT["jsonl_path"])
    except Exception as e:
        log(f"[ASR] append_asr_config_event failed: {e}")

# -------------------------
# LLM (Ollama) - config.json の llm から読む（llm.json廃止）
# -------------------------
LLM_CFG = CFG.get("llm", {}) or {}

OLLAMA_HOST = (LLM_CFG.get("host") or "http://127.0.0.1:11434").strip()
OLLAMA_MODEL_DEFAULT = (LLM_CFG.get("model_default") or "gemma3:12b").strip()
OLLAMA_TIMEOUT = float(LLM_CFG.get("timeout") or 120.0)
OLLAMA_TEMPERATURE = float(LLM_CFG.get("temperature") or 0.0)
OLLAMA_TOP_P = float(LLM_CFG.get("top_p") or 0.9)

LLM_PROMPTS: Dict[str, Dict[str, Any]] = LLM_CFG.get("prompts", {})  # ★ dictのまま
LLM_DEFAULT_PROMPT_ID = (LLM_CFG.get("default_prompt_id") or "soap_v1").strip()

LLM_DIR = resolve_relpath(CFG.get("llm_outputs_dir", "data/llm"))
ensure_dir(LLM_DIR)

def _safe_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s[:120] if len(s) > 120 else s

def _session_from_jsonl_name(path: Path) -> Dict[str, str]:
    m = re.match(r"^(?P<pid>.+)_(?P<stamp>\d{8}_\d{6})\.jsonl$", path.name)
    if not m:
        return {"patient_id": CURRENT.get("patient_id") or "unknown", "stamp": CURRENT.get("session_stamp") or ""}
    return {"patient_id": m.group("pid"), "stamp": m.group("stamp")}

def ollama_generate_text(*, host: str, model: str, prompt: str, timeout_sec: float, temperature: float, top_p: float) -> Dict[str, Any]:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    with httpx.Client(timeout=timeout_sec) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

def list_ollama_models(host: str, *, timeout_sec: float) -> List[str]:
    url = host.rstrip("/") + "/api/tags"
    with httpx.Client(timeout=timeout_sec) as client:
        r = client.get(url)
        r.raise_for_status()
        j = r.json()
    out = []
    for m in j.get("models", []) or []:
        name = m.get("name") or m.get("model")
        if name:
            out.append(str(name))
    return out

def list_prompt_items_from_cfg() -> Dict[str, Any]:
    """
    ★ items変換なし。configの prompts(dict) をそのまま列挙する。
    """
    items = []
    for pid, v in (LLM_PROMPTS or {}).items():
        if not isinstance(pid, str) or not pid.strip():
            continue
        label = pid
        if isinstance(v, dict):
            label = (v.get("label") or pid)
        items.append({"id": pid, "label": label})
    items.sort(key=lambda x: x["id"])

    default_id = LLM_DEFAULT_PROMPT_ID
    if default_id not in (LLM_PROMPTS or {}):
        # defaultが存在しないときは先頭or soap_v1
        default_id = items[0]["id"] if items else "soap_v1"

    return {"default": default_id, "items": items}

def build_prompt_from_cfg(prompt_id: str, asr_text: str) -> str:
    """
    ★ configの llm.prompts からテンプレを取得し、{asr_text} を埋める。
    """
    tpl = None
    if isinstance(LLM_PROMPTS, dict):
        v = LLM_PROMPTS.get(prompt_id)
        if isinstance(v, dict):
            tpl = v.get("template")

    if not tpl:
        tpl = '''以下は医者と患者の診察室での会話で主に医者の発言部分ですが、音声認識で一部同音異義語の書き違いがあります。
これを考慮して、SOAP形式にまとめてください。勝手に内容を追加せず喋った内容に沿って記述し、わからないことは記載しないでください。

=== 会話テキスト ===
{asr_text}
'''
    return str(tpl).replace("{asr_text}", asr_text)

def collect_asr_text_from_jsonl(jsonl_path: Path, *, min_quality: str = "good", max_lines: Optional[int] = None) -> str:
    order = {"bad": 0, "maybe": 1, "good": 2}
    def ok(q: str) -> bool:
        return order.get(q, 1) >= order.get(min_quality, 1)

    lines: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "asr":
                continue

            meta = rec.get("meta") or {}
            q = str(meta.get("quality", "unknown"))
            if not ok(q):
                continue

            t = (rec.get("text") or "").strip()
            if not t:
                continue

            lines.append(t)
            if max_lines and len(lines) >= max_lines:
                break
    return "\n".join(lines)

def save_llm_summary(*, jsonl_path: Path, model: str, prompt_id: str, summary: str) -> str:
    info = _session_from_jsonl_name(jsonl_path)
    pid = info.get("patient_id", "unknown")
    stamp = info.get("stamp", "")
    fname = f"{pid}_{stamp}__{_safe_name(model)}__{_safe_name(prompt_id)}__{now_stamp()}.txt"
    out_path = LLM_DIR / fname
    out_path.write_text(summary or "", encoding="utf-8")
    return fname

# ==========================================
# Auto LLM (sequential queue)
# ==========================================

AUTO_LLM_ENABLED = bool(CFG.get("auto_llm", False))
AUTO_LLM_PROMPTS = CFG.get("auto_llm_prompts") or []
if not isinstance(AUTO_LLM_PROMPTS, list):
    AUTO_LLM_PROMPTS = []

_auto_llm_q = None
# 追加：同じjsonlを複数回 enqueue しない（WS切断が複数回起きても防ぐ）
_auto_llm_enqueued: set[str] = set()

def _normalize_auto_llm_item(it: dict) -> dict | None:
    if not isinstance(it, dict):
        return None
    model_id = it.get("model_id")
    prompt_id = it.get("prompt_id")
    asr_correct = bool(it.get("asr_correct", False))
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        return None
    return {"model_id": model_id.strip(), "prompt_id": prompt_id.strip(), "asr_correct": asr_correct}

def _has_auto_llm_done(jsonl_path: Path, *, model_id: str, prompt_id: str, asr_correct: bool) -> bool:
    """
    重複送信防止：同一jsonlに type=auto_llm_done が既にあればスキップ
    """
    try:
        if not jsonl_path.exists():
            return False
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") != "auto_llm_done":
                    continue
                meta = rec.get("meta") or {}
                if (meta.get("model_id") == model_id and
                    meta.get("prompt_id") == prompt_id and
                    bool(meta.get("asr_correct")) == bool(asr_correct)):
                    return True
        return False
    except Exception:
        return False

def _has_llm_history_for_prompt(jsonl_path: Path, *, prompt_id: str) -> bool:
    """
    同一セッション（patient_id + stamp）で、同一 prompt_id のLLM出力ファイルが既に存在すれば True。
    ※model は問わない（*扱い）＝「同一プロンプトが過去に走っていればスキップ」
    """
    try:
        info = _session_from_jsonl_name(jsonl_path)
        pid = info.get("patient_id", "")
        stamp = info.get("stamp", "")
        if not pid or not stamp:
            return False

        # save_llm_summary の命名に合わせる（modelはワイルドカード）
        # {pid}_{stamp}__*__{prompt_id}__*.txt
        pat = f"{pid}_{stamp}__*__{_safe_name(prompt_id)}__*.txt"
        return any(LLM_DIR.glob(pat))
    except Exception:
        return False
    
def enqueue_auto_llm_for_session(jsonl_path: Path):
    """
    セッション終了（患者切替など）タイミングで、当該セッションのjsonlを自動でLLMに投げる。
    """
    global _auto_llm_q, _auto_llm_enqueued

    if not AUTO_LLM_ENABLED:
        return
    if _auto_llm_q is None:
        # startup前に呼ばれても落ちないように
        _auto_llm_q = asyncio.Queue()

    if not isinstance(jsonl_path, Path):
        jsonl_path = Path(str(jsonl_path))

    # ★追加：存在しないなら何もしない
    if not jsonl_path.exists():
        return
    # ★追加：メモリ内ガード（同じjsonlを多重enqueueしない）
    key = str(jsonl_path.resolve())
    if key in _auto_llm_enqueued:
        return
    _auto_llm_enqueued.add(key)

    # promptsを正規化して順番にqueueへ
    for raw in AUTO_LLM_PROMPTS:
        it = _normalize_auto_llm_item(raw)
        if not it:
            continue

        # ★追加：同一promptの過去LLM履歴があればスキップ（患者移動/マイクOnOff連打対策）
        if _has_llm_history_for_prompt(jsonl_path, prompt_id=it["prompt_id"]):
            continue
        # 既存：JSONLに auto_llm_done があるならスキップ（再起動後対策）
        if _has_auto_llm_done(jsonl_path, **it):
            continue

        _auto_llm_q.put_nowait({
            "jsonl_path": str(jsonl_path),
            **it,  # model_id, prompt_id, asr_correct
        })

async def _auto_llm_worker():
    """
    1ワーカーで逐次処理（Ollama詰まり防止）
    """
    global _auto_llm_q
    if _auto_llm_q is None:
        _auto_llm_q = asyncio.Queue()

    log(f"[AUTO_LLM] worker started enabled={AUTO_LLM_ENABLED} items={len(AUTO_LLM_PROMPTS)}")

    while True:
        job = await _auto_llm_q.get()
        try:
            await _run_auto_llm_job(job)
        except Exception as e:
            log(f"[AUTO_LLM] job failed: {e}")
        finally:
            _auto_llm_q.task_done()

async def _run_auto_llm_job(job: dict):
    jsonl_path = Path(job["jsonl_path"])
    model_id = str(job["model_id"])
    prompt_id = str(job["prompt_id"])
    asr_correct = bool(job.get("asr_correct", False))

    if not jsonl_path.exists():
        return

    # JSONLからASRテキストを復元（qualityフィルタも効くやつ）
    # 既存関数：collect_asr_text_from_jsonl :contentReference[oaicite:2]{index=2}
    asr_text = collect_asr_text_from_jsonl(jsonl_path, min_quality="good")
    asr_text = (asr_text or "").strip()
    if not asr_text:
        return

    # ASR補正（必要なら）
    # 既存：load_correction_rules / apply_corrections :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
    if asr_correct:
        try:
            rules = load_correction_rules()
            asr_text, _stats = apply_corrections(asr_text, rules)
        except Exception as e:
            log(f"[AUTO_LLM] asr_correct failed: {e}")

    # prompt生成（既存：build_prompt_from_cfg）:contentReference[oaicite:5]{index=5}
    prompt = build_prompt_from_cfg(prompt_id, asr_text)

    # Ollama呼び出し（既存：ollama_generate_text）:contentReference[oaicite:6]{index=6}
    # httpx.Client()の同期処理なので event loop を塞がないよう to_thread に逃がす
    def _call():
        return ollama_generate_text(
            host=OLLAMA_HOST,
            model=model_id,
            prompt=prompt,
            timeout_sec=OLLAMA_TIMEOUT,
            temperature=OLLAMA_TEMPERATURE,
            top_p=OLLAMA_TOP_P,
        )

    t0 = time.time()
    log(f"[AUTO_LLM] start jsonl={jsonl_path.name} model={model_id} prompt={prompt_id} asr_correct={asr_correct}")

    resp = await asyncio.to_thread(_call)
    summary = (resp.get("response") or "").strip()

    dt = time.time() - t0
    if not summary:
        log(f"[AUTO_LLM] empty response jsonl={jsonl_path.name} model={model_id} prompt={prompt_id} dt={dt:.2f}s")
        return

    # 出力保存（既存：save_llm_summary）:contentReference[oaicite:7]{index=7}
    out_name = save_llm_summary(jsonl_path=jsonl_path, model=model_id, prompt_id=prompt_id, summary=summary)

    # JSONLに「完了」マーカーを追記して、再起動後の重複送信を抑止
    append_jsonl({
        "type": "auto_llm_done",
        "ts": now_iso(),
        "patient_id": _session_from_jsonl_name(jsonl_path).get("patient_id", CURRENT.get("patient_id")),
        "meta": {
            "model_id": model_id,
            "prompt_id": prompt_id,
            "asr_correct": asr_correct,
            "dt_sec": round(dt, 2),
            "llm_output": out_name,
        }
    }, jsonl_path)

    log(f"[AUTO_LLM] done jsonl={jsonl_path.name} -> {out_name} dt={dt:.2f}s")

# ==========================================
# Auto LLM (on patient change / recording stop)
# ==========================================

AUTO_LLM_ENABLED = bool(CFG.get("auto_llm", False))
AUTO_LLM_PROMPTS = CFG.get("auto_llm_prompts", []) or []
AUTO_LLM_MIN_QUALITY = (CFG.get("auto_llm_min_quality") or "good").strip()

# 既に投げた session(jsonl) を二重送信しないためのガード
_AUTO_LLM_ENQUEUED: set[str] = set()

# 自動LLM用キュー（逐次実行）
AUTO_LLM_Q: asyncio.Queue[dict] = asyncio.Queue(maxsize=int(CFG.get("auto_llm_queue", 20)))

def enqueue_auto_llm_for_jsonl(jsonl_path: Path, *, reason: str):
    """
    指定 jsonl を auto_llm_prompts に従ってキューに積む。
    二重送信は _AUTO_LLM_ENQUEUED で防止。
    """
    if not AUTO_LLM_ENABLED:
        return
    if not jsonl_path or not jsonl_path.exists():
        return
    if not AUTO_LLM_PROMPTS:
        return

    key = str(jsonl_path.resolve())
    if key in _AUTO_LLM_ENQUEUED:
        return
    _AUTO_LLM_ENQUEUED.add(key)

    for p in AUTO_LLM_PROMPTS:
        try:
            model_id = (p.get("model_id") or "").strip() or OLLAMA_MODEL_DEFAULT
            prompt_id = (p.get("prompt_id") or "").strip() or LLM_DEFAULT_PROMPT_ID
            asr_correct = bool(p.get("asr_correct", False))

            job = {
                "type": "auto_llm",
                "ts": now_iso(),
                "reason": reason,
                "jsonl_path": str(jsonl_path),
                "model": model_id,
                "prompt_id": prompt_id,
                "asr_correct": asr_correct,
                "min_quality": AUTO_LLM_MIN_QUALITY,
            }
            try:
                AUTO_LLM_Q.put_nowait(job)
            except asyncio.QueueFull:
                log(f"[AUTO_LLM_Q] DROP (queue full) jsonl={jsonl_path.name} prompt={prompt_id}")
        except Exception as e:
            log(f"[AUTO_LLM_Q] enqueue error: {e}")
# -------------------------
# ASR config (defaults from config.json)
# -------------------------
ASR_CFG = CFG.get("asr", {}) or {}
ASR_ENABLED = bool(ASR_CFG.get("enabled", True))

ASR_LANG = ASR_CFG.get("language", "ja")
ASR_DEVICE = ASR_CFG.get("device", "cpu")
ASR_COMPUTE = ASR_CFG.get("compute_type", "int8")
ASR_BEAM = int(ASR_CFG.get("beam_size", 10))
ASR_TEMP = float(ASR_CFG.get("temperature", 0.0))
ASR_COPT = bool(ASR_CFG.get("condition_on_previous_text", False))
ASR_PROMPT_DEFAULT = ASR_CFG.get(
    "initial_prompt",
    "日本の医療現場の会話。聞こえたとおりに書き起こす。推測で補完しない。"
)

# ★ model_id と models は config.json のみ
ASR_MODEL_ID = (ASR_CFG.get("model_id") or "").strip()

def load_models_registry() -> dict[str, str]:
    """
    ★ config.json の asr.models をそのまま読む（models.json廃止）
    """
    cfg = load_config(force_reload=False)
    asr = cfg.get("asr", {}) or {}
    d = asr.get("models", {})
    out: dict[str, str] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                out[k.strip()] = v.strip()
    return out

MODELS_REGISTRY = load_models_registry()

def resolve_model_path(model_id: str) -> str:
    """
    model_id から model_path を引く。なければ model_id 自体を返す（許容）。
    """
    if model_id and model_id in MODELS_REGISTRY:
        return MODELS_REGISTRY[model_id]
    return model_id or ""

# 起動時のデフォルトモデル
ASR_MODEL = resolve_model_path(ASR_MODEL_ID) or (ASR_CFG.get("model_path") or "small")

# ★ 起動時に CURRENT.asr を defaults で初期化
CURRENT["asr"]["model_name"] = ASR_MODEL_ID if ASR_MODEL_ID else None
CURRENT["asr"]["model_path"] = ASR_MODEL
CURRENT["asr"]["language"] = ASR_LANG
CURRENT["asr"]["temperature"] = ASR_TEMP
CURRENT["asr"]["prompt"] = ASR_PROMPT_DEFAULT

def get_asr_runtime_config() -> dict:
    a = CURRENT.get("asr") or {}
    model_path = a.get("model_path") or ASR_MODEL
    model_name = a.get("model_name") or ASR_MODEL_ID
    lang = a.get("language") or ASR_LANG
    temp = float(a.get("temperature") if a.get("temperature") is not None else ASR_TEMP)
    prompt = a.get("prompt") if a.get("prompt") is not None else ASR_PROMPT_DEFAULT

    return {
        "model_name": model_name,
        "model_path": model_path,
        "language": lang,
        "temperature": temp,
        "prompt": prompt,
        "beam_size": ASR_BEAM,
        "condition_on_previous_text": ASR_COPT,
        "device": ASR_DEVICE,
        "compute_type": ASR_COMPUTE,
    }

# -------------------------
# DSP / metrics
# -------------------------
SR = int(CFG.get("sr", 48000))
FRAME_MS = int(CFG.get("frame_ms", 50))
FRAME = int(SR * FRAME_MS / 1000)

THRESHOLD_DB = float(CFG.get("threshold_db", -42.0))
START_VOICE_FRAMES = int(CFG.get("start_voice_frames", 3))
END_SILENCE_FRAMES = int(CFG.get("end_silence_frames", 12))
PRE_ROLL_MS = int(CFG.get("pre_roll_ms", 300))
PRE_ROLL_FRAMES = max(1, int(PRE_ROLL_MS / FRAME_MS))
MIN_SEC = float(CFG.get("min_sec", 0.4))
MAX_SEC = float(CFG.get("max_sec", 20.0))

def rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)

def peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x)) + 1e-12)
    return 20.0 * np.log10(p + 1e-12)

def clip_ratio(x: np.ndarray, thr: float = 0.98) -> float:
    return float(np.mean(np.abs(x) >= thr))

def silence_ratio(x: np.ndarray, thr: float = 0.005) -> float:
    return float(np.mean(np.abs(x) < thr))

def zcr(x: np.ndarray) -> float:
    s = np.sign(x)
    return float(np.mean(s[1:] != s[:-1]))

def safe_avg(vals) -> float | None:
    v = [x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)]
    if not v:
        return None
    return float(sum(v) / len(v))

def round_or_none(x: float | None, nd: int = 3):
    return None if x is None else round(x, nd)

def judge_quality(audio_meta: dict, asr_meta: dict, text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []

    rms = audio_meta.get("rms_dbfs")
    sil = audio_meta.get("silence_ratio")
    clip = audio_meta.get("clip_ratio")

    no_speech = asr_meta.get("no_speech_prob")
    avg_lp = asr_meta.get("avg_logprob")
    comp = asr_meta.get("compression_ratio")

    if rms is not None and rms < -45:
        reasons.append("low_rms")
    if sil is not None and sil > 0.60:
        reasons.append("high_silence_ratio")
    if clip is not None and clip > 0.005:
        reasons.append("clipping")

    if no_speech is not None and no_speech > 0.60:
        reasons.append("high_no_speech_prob")
    if avg_lp is not None and avg_lp < -1.0:
        reasons.append("low_avg_logprob")
    if comp is not None and comp > 2.4:
        reasons.append("high_compression_ratio")

    if text and len(text) >= 40:
        run = 1
        best = 1
        prev = text[0]
        for ch in text[1:]:
            if ch == prev:
                run += 1
                best = max(best, run)
            else:
                run = 1
                prev = ch
        if best >= 20:
            reasons.append("long_char_run")

    bad_triggers = {"high_no_speech_prob", "high_compression_ratio", "long_char_run"}
    maybe_triggers = {"low_rms", "high_silence_ratio", "low_avg_logprob", "clipping"}

    if any(r in bad_triggers for r in reasons):
        return "bad", reasons
    if any(r in maybe_triggers for r in reasons):
        return "maybe", reasons
    return "good", reasons

def save_wav_mono16(path: Path, audio_f32: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

# -------------------------
# Whisper model cache
# -------------------------
_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}

def get_model_for(model_path: str) -> WhisperModel:
    key = (model_path, ASR_DEVICE, ASR_COMPUTE)
    m = _MODEL_CACHE.get(key)
    if m is None:
        log(f"[ASR] loading model: {model_path} device={ASR_DEVICE} compute={ASR_COMPUTE}")
        m = WhisperModel(model_path, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
        _MODEL_CACHE[key] = m
        log("[ASR] model loaded")
    return m

def clear_model_cache_for(model_path: str | None):
    global _MODEL_CACHE
    if model_path is None:
        _MODEL_CACHE = {}
        return
    key = (model_path, ASR_DEVICE, ASR_COMPUTE)
    _MODEL_CACHE.pop(key, None)

def set_asr_model_by_id(model_id: str) -> tuple[bool, str]:
    """
    ★ config.json の asr.models のキーを受け取り、モデルを切り替える
    """
    global MODELS_REGISTRY, ASR_MODEL_ID, ASR_MODEL

    MODELS_REGISTRY = load_models_registry()
    if model_id not in MODELS_REGISTRY:
        return (False, f"unknown model_id: {model_id}")

    old_path = ASR_MODEL

    ASR_MODEL_ID = model_id
    ASR_MODEL = MODELS_REGISTRY[model_id]

    CURRENT["asr"]["model_name"] = model_id
    CURRENT["asr"]["model_path"] = ASR_MODEL

    clear_model_cache_for(old_path)

    log(f"[ASR] model switched -> id={ASR_MODEL_ID} path={ASR_MODEL}")
    append_asr_config_event("model_switched")
    return (True, "ok")

# -------------------------
# dyna watcher
# -------------------------
DYNA_WATCH_DIR = resolve_relpath(CFG.get("dyna_watch_dir", "data/SpeechID"))
DYNA_GLOB = CFG.get("dyna_glob", "dyna*.txt")
DYNA_DELETE_AFTER_READ = bool(CFG.get("delete_after_read", True))

def list_dyna_files() -> list[Path]:
    if not DYNA_WATCH_DIR.exists():
        return []
    return list(DYNA_WATCH_DIR.glob(DYNA_GLOB))

def pick_latest_dyna_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]

def delete_all_dyna_files(paths: list[Path]):
    if not DYNA_DELETE_AFTER_READ:
        return
    for p in paths:
        try:
            p.unlink()
        except Exception as e:
            log(f"[DYNA] delete failed {p}: {e}")

def read_patient_id_from_dyna(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            raw = f.readline()

        line = raw.decode("cp932", errors="replace").strip()
        if not line:
            return None

        pid = line.split(",")[0].strip()
        return pid if pid.isdigit() else None

    except Exception as e:
        log(f"[DYNA] read error {path}: {e}")
        return None

# WSごとのStateを持つ
CLIENTS: dict[WebSocket, "State"] = {}

async def broadcast(obj: dict, *, reset_states: bool = False):
    dead: list[WebSocket] = []
    for ws, st in list(CLIENTS.items()):
        if reset_states:
            st.reset_pending = True
        try:
            await ws.send_json(obj)
        except Exception:
            dead.append(ws)

    for ws in dead:
        CLIENTS.pop(ws, None)

async def dyna_watch_task():
    log(f"[DYNA] watcher started dir={DYNA_WATCH_DIR} glob={DYNA_GLOB}")
    last_batch_sig = None

    while True:
        try:
            paths = list_dyna_files()
            if not paths:
                await asyncio.sleep(0.5)
                continue

            latest = pick_latest_dyna_file(paths)
            if latest is None:
                await asyncio.sleep(0.5)
                continue

            st = latest.stat()
            sig = (str(latest), st.st_mtime, len(paths))
            if sig == last_batch_sig:
                await asyncio.sleep(0.5)
                continue

            await asyncio.sleep(0.25)

            pid = read_patient_id_from_dyna(latest)

            delete_all_dyna_files(paths)
            last_batch_sig = sig

            if pid:
                if CURRENT["patient_id"] != pid:
                    # 切替前セッションを自動LLMに投げる
                    try:
                        prev_jsonl = CURRENT.get("jsonl_path")
                        if prev_jsonl:
                            enqueue_auto_llm_for_session(Path(prev_jsonl))
                    except Exception as e:
                        log(f"[AUTO_LLM] enqueue failed: {e}")

                    new_session(pid)
                    log(f"[SESSION] switched patient_id={pid} txt={CURRENT['text_path']}")

                    await broadcast({
                        "type": "patient_changed",
                        "patient_id": pid,
                        "text_path": CURRENT["text_path"],
                        "jsonl_path": CURRENT["jsonl_path"],
                        "session_txt": (Path(CURRENT["text_path"]).name if CURRENT.get("text_path") else "")
                    }, reset_states=True)

            await asyncio.sleep(0.5)

        except Exception as e:
            log(f"[DYNA] watcher error: {e}")
            await asyncio.sleep(1.0)

# -------------------------
# データ表示
# -------------------------
SESSION_TXT_RE = re.compile(r"^(?P<patient_id>[^_]+)_(?P<stamp>\d{8}_\d{6})\.txt$")

def _list_session_txt_files() -> list[Path]:
    return sorted(OUTPUTS_DIR.glob("*.txt"))

def _parse_session_name(name: str):
    m = SESSION_TXT_RE.match(name)
    if not m:
        return None
    return m.group("patient_id"), m.group("stamp")

# -------------------------
# Web
# -------------------------
STATIC_DIR = APP_DIR / "static"
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/version")
async def api_version():
    return {"ok": True, "version": APP_VERSION}

@app.on_event("startup")
async def _startup():
    asyncio.create_task(dyna_watch_task())

    # Auto LLM worker（逐次キュー処理）
    asyncio.create_task(_auto_llm_worker())

    cleanup_expired_wavs(CFG)
    log("[APP] startup complete")

# -------------------------
# LLM APIs
# -------------------------
@app.get("/api/llm/models")
async def api_llm_models():
    try:
        models = list_ollama_models(OLLAMA_HOST, timeout_sec=OLLAMA_TIMEOUT)
        return {"ok": True, "models": models, "default_model": OLLAMA_MODEL_DEFAULT}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "models": [], "default_model": OLLAMA_MODEL_DEFAULT}, status_code=500)

@app.get("/api/llm/prompts")
async def api_llm_prompts():
    items = list_prompt_items_from_cfg()
    return {"ok": True, "items": items.get("items", []), "default_prompt_id": items.get("default")}

@app.get("/api/llm/history")
async def api_llm_history(session: str = "", patient_id: str = ""):
    session = (session or "").strip()
    patient_id = (patient_id or "").strip()

    pid = ""
    stamp = ""
    if session:
        m = re.match(r"^(?P<pid>.+)_(?P<stamp>\d{8}_\d{6})\.txt$", session)
        if m:
            pid = m.group("pid")
            stamp = m.group("stamp")
    if not pid and patient_id:
        pid = patient_id

    if pid and stamp:
        pat = f"{pid}_{stamp}__*.txt"
        files = sorted(LLM_DIR.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True)
    elif pid:
        pat = f"{pid}_*__*.txt"
        files = sorted(LLM_DIR.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        files = []

    items = []
    for f in files[:200]:
        name = f.name
        parts = name.split("__")
        model = parts[1] if len(parts) > 1 else ""
        prompt = parts[2] if len(parts) > 2 else ""
        items.append({"id": name, "label": f"{model} / {prompt} / {name[-19:-4]}"})

    return {"ok": True, "items": items}

@app.get("/api/llm/history/item/{item_id}")
async def api_llm_history_item(item_id: str):
    item_id = unquote(item_id)
    if "/" in item_id or "\\" in item_id or ".." in item_id:
        return JSONResponse({"ok": False, "error": "invalid id"}, status_code=400)

    p = (LLM_DIR / item_id).resolve()
    # Python>=3.9
    if not p.is_file():
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    if LLM_DIR.resolve() not in p.parents:
        return JSONResponse({"ok": False, "error": "invalid path"}, status_code=400)

    txt = p.read_text(encoding="utf-8", errors="replace")
    return {"ok": True, "summary": txt}

@app.post("/api/llm/soap")
async def api_llm_soap(payload: Dict[str, Any] = Body(...)):
    MAX_ASR_TEXT_CHARS = 50000

    session_txt = (payload.get("session") or "").strip()
    min_quality = (payload.get("min_quality") or "good").strip()
    prompt_id = (payload.get("prompt_id") or LLM_DEFAULT_PROMPT_ID or "soap_v1").strip()

    model = (payload.get("model") or "").strip() or OLLAMA_MODEL_DEFAULT
    temperature = float(payload.get("temperature") or OLLAMA_TEMPERATURE)
    top_p = float(payload.get("top_p") or OLLAMA_TOP_P)

    max_lines = payload.get("max_lines")
    try:
        max_lines = int(max_lines) if max_lines not in (None, "", 0, "0") else None
    except Exception:
        max_lines = None

    # decide jsonl path
    if session_txt:
        if not session_txt.endswith(".txt"):
            return JSONResponse({"ok": False, "error": "session must be .txt name"}, status_code=400)
        jsonl_path = (OUTPUTS_DIR / session_txt.replace(".txt", ".jsonl")).resolve()
    else:
        if not CURRENT.get("jsonl_path"):
            return JSONResponse({"ok": False, "error": "no current session"}, status_code=400)
        jsonl_path = Path(CURRENT["jsonl_path"]).resolve()

    if OUTPUTS_DIR.resolve() not in jsonl_path.parents:
        return JSONResponse({"ok": False, "error": "invalid path"}, status_code=400)
    if not jsonl_path.exists():
        return JSONResponse({"ok": False, "error": f"not found: {jsonl_path.name}"}, status_code=404)

    # payload優先：クライアントで手修正した transcript を使えるようにする
    asr_text_payload = payload.get("asr_text")
    if len(asr_text_payload) > MAX_ASR_TEXT_CHARS:
        return JSONResponse({"ok": False, "error": "ASR text too large"}, status_code=413)

    if isinstance(asr_text_payload, str) and asr_text_payload.strip():
        asr_text = asr_text_payload.strip()
    else:
        asr_text = collect_asr_text_from_jsonl(jsonl_path, min_quality=min_quality, max_lines=max_lines)

    if not asr_text.strip():
        return JSONResponse({"ok": False, "error": "ASR text empty"}, status_code=400)

    prompt = build_prompt_from_cfg(prompt_id, asr_text)

    t0 = time.time()
    try:
        raw = ollama_generate_text(
            host=OLLAMA_HOST,
            model=model,
            prompt=prompt,
            timeout_sec=OLLAMA_TIMEOUT,
            temperature=temperature,
            top_p=top_p,
        )
        summary = (raw.get("response") or "").strip()
        elapsed = round(time.time() - t0, 3)

        saved_id = save_llm_summary(jsonl_path=jsonl_path, model=model, prompt_id=prompt_id, summary=summary)
        return {
            "ok": True,
            "summary": summary,
            "model": model,
            "prompt_id": prompt_id,
            "elapsed_sec": elapsed,
            "saved_id": saved_id,
            "session_jsonl": jsonl_path.name,
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# -------------------------
# ASR APIs
# -------------------------
@app.get("/api/asr/models")
async def api_asr_models():
    reg = load_models_registry()
    cur = CURRENT.get("asr", {}).get("model_name") or ASR_MODEL_ID

    if not cur and reg:
        cur_path = CURRENT.get("asr", {}).get("model_path") or ASR_MODEL
        for k, v in reg.items():
            if v == cur_path:
                cur = k
                break

    return {
        "current": cur,
        "models": [{"id": k, "label": k} for k in sorted(reg.keys())],
    }

@app.post("/api/asr/model")
async def api_asr_set_model(payload: dict = Body(...)):
    model_id = (payload.get("id") or "").strip()
    ok, msg = set_asr_model_by_id(model_id)
    if not ok:
        return {"ok": False, "error": msg}
    return {"ok": True, "current": CURRENT["asr"]["model_name"]}

@app.post("/api/asr/config")
async def api_asr_set_config(payload: dict = Body(...)):
    a = CURRENT.setdefault("asr", {})
    changed = False

    if "language" in payload and isinstance(payload["language"], str):
        a["language"] = payload["language"].strip() or "ja"
        changed = True

    if "temperature" in payload:
        try:
            a["temperature"] = float(payload["temperature"])
            changed = True
        except Exception:
            pass

    if "prompt" in payload and isinstance(payload["prompt"], str):
        a["prompt"] = payload["prompt"]
        changed = True

    if changed:
        append_asr_config_event("config_updated")

    return {"ok": True, "asr": dict(a)}

# -------------------------
# Session APIs
# -------------------------
@app.get("/api/sessions")
def api_sessions():
    items = []
    for p in _list_session_txt_files():
        info = _parse_session_name(p.name)
        if not info:
            continue
        patient_id, stamp = info
        items.append({
            "name": p.name,
            "patient_id": patient_id,
            "stamp": stamp,
            "label": f"{patient_id} / {stamp[:8]} {stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}",
        })

    items.sort(key=lambda x: x["stamp"], reverse=True)
    return {"items": items}

@app.get("/api/session/{name}")
def api_session_get(name: str):
    name = unquote(name)

    info = _parse_session_name(name)
    if not info:
        return JSONResponse({"error": "invalid session name"}, status_code=400)

    p = (OUTPUTS_DIR / name).resolve()
    if OUTPUTS_DIR.resolve() not in p.parents:
        return JSONResponse({"error": "invalid path"}, status_code=400)

    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    text = p.read_text(encoding="utf-8", errors="replace")
    patient_id, stamp = info

    meta = None
    j = (OUTPUTS_DIR / name.replace(".txt", ".jsonl")).resolve()
    if OUTPUTS_DIR.resolve() in j.parents and j.exists():
        try:
            with open(j, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("type") == "session_start":
                        meta = obj.get("meta")
                        break
        except Exception as e:
            log(f"[SESSION] jsonl read failed: {j}: {e}")

    return {"name": name, "patient_id": patient_id, "stamp": stamp, "text": text, "meta": meta}

@app.post("/api/correct/apply")
async def api_correct_apply(payload: Dict[str, Any] = Body(...)):
    """
    方針C:
      - JSONL(ASR原本)は変更しない
      - 表示用 .txt を補正後で置換
      - 補正履歴(type=correction)だけ同じJSONLに追記
    payload:
      {"session":"<pid>_<stamp>.txt"}  # 省略時は現在セッション
    """
    session_txt = (payload.get("session") or "").strip()

    # 対象txt / jsonl を決定
    if session_txt:
        info = _parse_session_name(session_txt)
        if not info:
            return JSONResponse({"ok": False, "error": "invalid session name"}, status_code=400)
        txt_path = (OUTPUTS_DIR / session_txt).resolve()
        jsonl_path = (OUTPUTS_DIR / session_txt.replace(".txt", ".jsonl")).resolve()
    else:
        if not CURRENT.get("text_path") or not CURRENT.get("jsonl_path"):
            return JSONResponse({"ok": False, "error": "no current session"}, status_code=400)
        txt_path = Path(CURRENT["text_path"]).resolve()
        jsonl_path = Path(CURRENT["jsonl_path"]).resolve()
        session_txt = txt_path.name

    # path traversal 防止
    if OUTPUTS_DIR.resolve() not in txt_path.parents:
        return JSONResponse({"ok": False, "error": "invalid path"}, status_code=400)
    if OUTPUTS_DIR.resolve() not in jsonl_path.parents:
        return JSONResponse({"ok": False, "error": "invalid path"}, status_code=400)

    if not txt_path.exists():
        return JSONResponse({"ok": False, "error": "txt not found"}, status_code=404)
    if not jsonl_path.exists():
        return JSONResponse({"ok": False, "error": "jsonl not found"}, status_code=404)

    rules = load_correction_rules()
    before_text = txt_path.read_text(encoding="utf-8", errors="replace")

    after_text, stats = apply_corrections(before_text, rules)

    # 変更が無ければそのまま返す
    if not stats.get("changed"):
        return {"ok": True, "changed": False, "text": before_text, "stats": stats}

    # 念のためバックアップ
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_name = f"{session_txt}.{ts}.bak"
    bak_path = (CORRECTIONS_DIR / bak_name).resolve()
    bak_path.write_text(before_text, encoding="utf-8")

    # .txt を置換（表示はこちらが参照される）
    txt_path.write_text(after_text, encoding="utf-8")

    # JSONLへ補正履歴だけ追記（ASR原本レコードは維持）
    ev = {
        "type": "correction",
        "ts": datetime.datetime.now().isoformat(),
        "session_txt": session_txt,
        "rules_version": rules.get("version"),
        "before_sha1": _sha1_text(before_text),
        "after_sha1": _sha1_text(after_text),
        "stats": stats,
        "backup_txt": bak_name,
    }
    append_jsonl_line(jsonl_path, ev)

    # 複数クライアントが見ている時に同期したいならブロードキャスト
    try:
        await broadcast({"type": "corrected", "session_txt": session_txt, "text": after_text, "stats": stats})
    except Exception:
        pass

    return {"ok": True, "changed": True, "text": after_text, "stats": stats, "backup": bak_name}

@app.post("/api/session/rebuild")
async def api_session_rebuild(payload: Dict[str, Any] = Body(...)):
    """
    JSONL原本(type=asr)から .txt を作り直して上書きする（原本に戻す）
    payload:
      {"session":"<pid>_<stamp>.txt"}  # 省略時は現在セッション
    """
    session_txt = (payload.get("session") or "").strip()

    # 対象txt / jsonl を決定（applyと同じ流儀）
    if session_txt:
        info = _parse_session_name(session_txt)
        if not info:
            return JSONResponse({"ok": False, "error": "invalid session name"}, status_code=400)
        txt_path = (OUTPUTS_DIR / session_txt).resolve()
        jsonl_path = (OUTPUTS_DIR / session_txt.replace(".txt", ".jsonl")).resolve()
    else:
        if not CURRENT.get("text_path") or not CURRENT.get("jsonl_path"):
            return JSONResponse({"ok": False, "error": "no current session"}, status_code=400)
        txt_path = Path(CURRENT["text_path"]).resolve()
        jsonl_path = Path(CURRENT["jsonl_path"]).resolve()
        session_txt = txt_path.name

    # path traversal 防止（applyと同じ）
    if OUTPUTS_DIR.resolve() not in txt_path.parents:
        return JSONResponse({"ok": False, "error": "invalid path"}, status_code=400)
    if OUTPUTS_DIR.resolve() not in jsonl_path.parents:
        return JSONResponse({"ok": False, "error": "invalid path"}, status_code=400)

    if not jsonl_path.exists():
        return JSONResponse({"ok": False, "error": "jsonl not found"}, status_code=404)

    # JSONLから type=asr の text を復元
    parts = []
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") == "asr":
                    t = rec.get("text")
                    if isinstance(t, str):
                        t = t.strip()
                        if t:
                            parts.append(t)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"read jsonl failed: {e}"}, status_code=500)

    rebuilt = "\n".join(parts).strip()
    if rebuilt:
        rebuilt += "\n"

    # 念のため現txtをバックアップ（applyと同じ流儀）
    backup_name = None
    try:
        if txt_path.exists():
            before_text = txt_path.read_text(encoding="utf-8", errors="replace")
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{session_txt}.{ts}.rebuild.bak"
            bak_path = (CORRECTIONS_DIR / backup_name).resolve()
            bak_path.write_text(before_text, encoding="utf-8")
    except Exception:
        # バックアップ失敗は致命ではないので続行（必要ならreturnに変えてもOK）
        backup_name = None

    # .txt を上書き
    try:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(rebuilt, encoding="utf-8")
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"write txt failed: {e}"}, status_code=500)

    # JSONLへ履歴追記（原本は不変）
    try:
        ev = {
            "type": "rebuild",
            "ts": datetime.datetime.now().isoformat(),
            "session_txt": session_txt,
            "source": "jsonl_asr",
            "backup_txt": backup_name,
            "rebuilt_chars": len(rebuilt),
            "rebuilt_segments": len(parts),
        }
        append_jsonl_line(jsonl_path, ev)
    except Exception:
        pass

    # クライアント同期（任意）
    try:
        await broadcast({"type": "rebuilt", "session_txt": session_txt, "text": rebuilt})
    except Exception:
        pass

    return {
        "ok": True,
        "text": rebuilt,
        "stats": {"segments": len(parts), "chars": len(rebuilt)},
        "backup": backup_name,
    }
    
# -------------------------
# Per-connection state
# -------------------------
class State:
    def __init__(self):
        self.buf = np.zeros((0,), dtype=np.float32)
        self.ring: list[np.ndarray] = []
        self.in_speech = False
        self.voice_run = 0
        self.silence_run = 0
        self.utter_frames: list[np.ndarray] = []
        self.seg_index = 0
        self.last_level_sent = 0.0

        self.asr_q: asyncio.Queue[dict] = asyncio.Queue(maxsize=int(CFG.get("asr_queue", 20)))
        self.asr_task: asyncio.Task | None = None
        self.reset_pending = False

    def reset_audio(self):
        self.buf = np.zeros((0,), dtype=np.float32)
        self.ring = []
        self.in_speech = False
        self.voice_run = 0
        self.silence_run = 0
        self.utter_frames = []

        try:
            while True:
                _ = self.asr_q.get_nowait()
                self.asr_q.task_done()
        except asyncio.QueueEmpty:
            pass

    def push_ring(self, f: np.ndarray):
        self.ring.append(f)
        if len(self.ring) > PRE_ROLL_FRAMES:
            self.ring.pop(0)

    def finalize_segment(self):
        if not self.utter_frames:
            self.in_speech = False
            self.voice_run = 0
            self.silence_run = 0
            self.ring = []
            return None

        audio = np.concatenate(self.utter_frames).astype(np.float32, copy=False)

        self.utter_frames = []
        self.in_speech = False
        self.voice_run = 0
        self.silence_run = 0
        self.ring = []

        dur = audio.shape[0] / SR
        if dur < MIN_SEC:
            return None

        self.seg_index += 1
        wavpath = WAV_DIR / f"seg_{now_stamp()}_{self.seg_index:03d}.wav"
        save_wav_mono16(wavpath, audio, SR)

        audio_meta = {
            "rms_dbfs": round(rms_dbfs(audio), 2),
            "peak_dbfs": round(peak_dbfs(audio), 2),
            "clip_ratio": round(clip_ratio(audio), 4),
            "silence_ratio": round(silence_ratio(audio), 4),
            "zcr": round(zcr(audio), 4),
        }

        seg = {
            "seg_id": self.seg_index,
            "dur": round(dur, 2),
            "wav": str(wavpath),
            "audio_meta": audio_meta,
        }

        ensure_unknown_session()
        append_jsonl(
            {"type": "saved", "ts": now_iso(), "patient_id": CURRENT["patient_id"], **seg},
            CURRENT["jsonl_path"]
        )

        cleanup_expired_wavs(CFG)
        return seg

# ==========================================
# WAV Auto Cleanup
# ==========================================

_last_wav_cleanup = None

def cleanup_expired_wavs(cfg: dict):
    """
    cfg["wav_expire_days"]:
        0 -> 削除しない
        N -> N日より古いwavを削除
    """
    global _last_wav_cleanup

    expire_days = cfg.get("wav_expire_days",0)  

    if not expire_days or int(expire_days) <= 0:
        return

    now = datetime.datetime.now()

    # 1時間に1回だけ実行（負荷軽減）
    if _last_wav_cleanup and (now - _last_wav_cleanup).seconds < 3600:
        return

    wav_dir = cfg.get("wav_dir", "data/wav")
    cutoff = now - datetime.timedelta(days=int(expire_days))

    if not os.path.isdir(wav_dir):
        return

    deleted = 0
    for fname in os.listdir(wav_dir):
        if not fname.lower().endswith(".wav"):
            continue

        fpath = os.path.join(wav_dir, fname)

        try:
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
                deleted += 1
        except Exception:
            pass

    if deleted > 0:
        print(f"[WAV CLEANUP] deleted {deleted} expired files")

    _last_wav_cleanup = now

# -------------------------
# ASR worker
# -------------------------
async def asr_worker(ws: WebSocket, st: State):
    if not ASR_ENABLED:
        log("[ASR] disabled -> worker exit")
        return

    log("[ASR] worker started")

    while True:
        job = await st.asr_q.get()
        try:
            seg_id = job.get("seg_id")
            wav = job.get("wav")
            dur = job.get("dur")
            audio_meta = job.get("audio_meta", {})

            cfg_rt = get_asr_runtime_config()
            model_path = cfg_rt["model_path"]

            log(f"[ASR] dequeued seg#{seg_id} wav={wav} model={cfg_rt.get('model_name')}")

            try:
                model = get_model_for(model_path)
            except Exception as e:
                log(f"[ASR] model load FAILED: {e}")
                await ws.send_json({"type": "error", "where": "asr_model_load", "error": str(e), "model_path": model_path})
                continue

            t0 = time.time()
            log(f"[ASR] transcribe start seg#{seg_id:03d} dur={dur}s")

            segments, _info = model.transcribe(
                wav,
                language=cfg_rt["language"],
                vad_filter=False,
                beam_size=cfg_rt["beam_size"],
                temperature=cfg_rt["temperature"],
                condition_on_previous_text=cfg_rt["condition_on_previous_text"],
                initial_prompt=cfg_rt["prompt"],
            )

            texts = []
            avg_logprob_list = []
            no_speech_list = []
            comp_ratio_list = []

            for s in segments:
                t = (getattr(s, "text", "") or "").strip()
                if t:
                    texts.append(t)
                avg_logprob_list.append(getattr(s, "avg_logprob", None))
                no_speech_list.append(getattr(s, "no_speech_prob", None))
                comp_ratio_list.append(getattr(s, "compression_ratio", None))

            # UI / .txt 用：セグメントごとに改行
            text_ui = "\n".join(texts).strip()

            # 品質判定は「改行無し」のほうが安定するなら、ここで潰した版を使う
            text_for_judge = text_ui.replace("\r", " ").replace("\n", " ").strip()

            dt = time.time() - t0

            asr_meta = {
                "asr_sec": round(dt, 2),
                "avg_logprob": round_or_none(safe_avg(avg_logprob_list), 3),
                "no_speech_prob": round_or_none(safe_avg(no_speech_list), 3),
                "compression_ratio": round_or_none(safe_avg(comp_ratio_list), 3),
            }

            quality, reasons = judge_quality(audio_meta, asr_meta, text_for_judge)

            ensure_unknown_session()
            pid = CURRENT["patient_id"]
            txt_path = CURRENT["text_path"]
            jsonl_path = CURRENT["jsonl_path"]

            # .txt には読みやすさ優先で改行入りを保存
            if text_ui:
                append_text_line(txt_path, text_ui)

            # JSONL には改行入りを渡してOK（append_jsonl 側で改行を潰して1行保証）
            append_jsonl({
                "type": "asr",
                "ts": now_iso(),
                "patient_id": pid,
                "seg_id": seg_id,
                "dur": dur,
                "wav": wav,
                "text": text_ui,
                "asr_cfg": {
                    "model_name": cfg_rt.get("model_name"),
                    "model_path": cfg_rt.get("model_path"),
                    "language": cfg_rt.get("language"),
                    "temperature": cfg_rt.get("temperature"),
                    "prompt": cfg_rt.get("prompt"),
                    "beam_size": cfg_rt.get("beam_size"),
                    "condition_on_previous_text": cfg_rt.get("condition_on_previous_text"),
                },
                "meta": {"audio": audio_meta, "asr": asr_meta, "quality": quality, "reasons": reasons}
            }, jsonl_path)

            log(
                f"[ASR] done seg#{seg_id:03d} sec={dt:.2f} "
                f"q={quality} reasons={reasons} "
                f"rms={audio_meta.get('rms_dbfs')} no_speech={asr_meta.get('no_speech_prob')} "
                f"comp={asr_meta.get('compression_ratio')} text_head={text_for_judge[:60]!r}"
            )

            # WebSocket へは改行入り表示を送る
            await ws.send_json({
                "type": "asr",
                "patient_id": pid,
                "seg_id": seg_id,
                "dur": dur,
                "wav": wav,
                "text": text_ui,
                "asr_cfg": {
                    "model_name": cfg_rt.get("model_name"),
                    "model_path": cfg_rt.get("model_path"),
                    "language": cfg_rt.get("language"),
                    "temperature": cfg_rt.get("temperature"),
                    "prompt": cfg_rt.get("prompt"),
                },
                "meta": {"audio": audio_meta, "asr": asr_meta, "quality": quality, "reasons": reasons}
            })

        except Exception as e:
            log(f"[ASR] ERROR: {e}")
            await ws.send_json({"type": "error", "where": "asr", "error": str(e), "job": job})
        finally:
            st.asr_q.task_done()

# -------------------------
# WebSocket endpoint
# -------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    st = State()
    CLIENTS[ws] = st

    log(f"[WS] open enabled_asr={ASR_ENABLED}")
    st.asr_task = asyncio.create_task(asr_worker(ws, st))

    try:
        await ws.send_json({
            "type": "status",
            "msg": "connected",
            "patient_id": CURRENT["patient_id"],
            "session_txt": (Path(CURRENT["text_path"]).name if CURRENT.get("text_path") else "")
        })

        while True:
            if st.reset_pending:
                st.reset_pending = False
                st.reset_audio()
                log("[WS] state reset (patient_changed)")

            data = await ws.receive_bytes()
            x = np.frombuffer(data, dtype=np.float32)
            if x.size == 0:
                continue

            st.buf = np.concatenate([st.buf, x])

            while st.buf.shape[0] >= FRAME:
                f = st.buf[:FRAME]
                st.buf = st.buf[FRAME:]

                level = rms_dbfs(f)
                now = time.time()
                if now - st.last_level_sent >= 0.5:
                    st.last_level_sent = now
                    await ws.send_json({"type": "level", "dbfs": round(level, 2)})

                st.push_ring(f)
                is_voice = level > THRESHOLD_DB

                if not st.in_speech:
                    if is_voice:
                        st.voice_run += 1
                        if st.voice_run >= START_VOICE_FRAMES:
                            st.in_speech = True
                            st.utter_frames = list(st.ring)
                            st.silence_run = 0
                    else:
                        st.voice_run = 0
                else:
                    st.utter_frames.append(f)
                    if is_voice:
                        st.silence_run = 0
                    else:
                        st.silence_run += 1
                        if st.silence_run >= END_SILENCE_FRAMES:
                            seg = st.finalize_segment()
                            if seg:
                                await ws.send_json({"type": "saved", "patient_id": CURRENT["patient_id"], **seg})
                                try:
                                    st.asr_q.put_nowait(seg)
                                    log(f"[ASR_Q] enqueued seg#{seg['seg_id']:03d} dur={seg['dur']}s wav={seg['wav']}")
                                except asyncio.QueueFull:
                                    log(f"[ASR_Q] DROP seg#{seg['seg_id']:03d} (queue full)")
                                    await ws.send_json({"type": "asr_drop", "seg_id": seg["seg_id"], "reason": "asr_q full"})
                            continue

                    dur = sum(a.shape[0] for a in st.utter_frames) / SR
                    if dur >= MAX_SEC:
                        seg = st.finalize_segment()
                        if seg:
                            await ws.send_json({"type": "saved", "patient_id": CURRENT["patient_id"], **seg})
                            try:
                                st.asr_q.put_nowait(seg)
                            except asyncio.QueueFull:
                                await ws.send_json({"type": "asr_drop", "seg_id": seg["seg_id"], "reason": "asr_q full"})

    except WebSocketDisconnect:
        log("[WS] close")
    finally:
        # ★追加：最後の患者は patient_changed が起きないので、録音停止で auto_llm
        try:
            cur_jsonl = CURRENT.get("jsonl_path")
            if cur_jsonl:
                enqueue_auto_llm_for_session(Path(cur_jsonl))
        except Exception as e:
            log(f"[AUTO_LLM] enqueue on stop failed: {e}")

        CLIENTS.pop(ws, None)
        if st.asr_task:
            st.asr_task.cancel()

# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    log(f"[APP] start SR={SR} frame={FRAME_MS}ms threshold={THRESHOLD_DB} dyna_dir={DYNA_WATCH_DIR}")
    SSL_CFG = CFG.get("ssl", {}) or {}

    if SSL_CFG.get("enabled"):
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(CFG.get("port", 8000)),
            ssl_certfile=SSL_CFG.get("certfile"),
            ssl_keyfile=SSL_CFG.get("keyfile")
        )
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(CFG.get("port", 8000))
        )
