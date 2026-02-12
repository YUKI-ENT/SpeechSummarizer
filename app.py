import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import wave
import json
import datetime
import asyncio
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
from faster_whisper import WhisperModel

from urllib.parse import unquote
import re

from typing import Any, Dict, List, Optional

import httpx

# -------------------------
# Config
# -------------------------
CONFIG_PATH = Path("config.json")

def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

CFG = load_config()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

OUTPUTS_DIR = Path(CFG.get("outputs_dir", "data/sessions"))
WAV_DIR = Path(CFG.get("wav_dir", "data/wav"))
ensure_dir(OUTPUTS_DIR)
ensure_dir(WAV_DIR)


# -------------------------
# LLM (Ollama)
# -------------------------
OLLAMA_HOST = (CFG.get("ollama_host") or "http://127.0.0.1:11434").strip()
OLLAMA_MODEL_DEFAULT = (CFG.get("ollama_model_default") or "gemma3:12b").strip()
OLLAMA_TIMEOUT = float(CFG.get("ollama_timeout") or 120.0)
OLLAMA_TEMPERATURE = float(CFG.get("ollama_temperature") or 0.0)
OLLAMA_TOP_P = float(CFG.get("ollama_top_p") or 0.9)

LLM_PROMPTS_PATH = (CFG.get("llm_prompts_path") or "").strip() or None

LLM_DIR = Path(CFG.get("llm_outputs_dir", "data/llm"))
ensure_dir(LLM_DIR)

def _safe_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    return s[:120] if len(s) > 120 else s

def _session_from_jsonl_name(path: Path) -> Dict[str, str]:
    # <patient>_<stamp>.jsonl
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

def load_prompt_catalog(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {"default": "soap_v1", "items": []}
    pth = Path(path)
    if not pth.exists():
        return {"default": "soap_v1", "items": []}
    try:
        return json.loads(pth.read_text(encoding="utf-8"))
    except Exception:
        return {"default": "soap_v1", "items": []}

def list_prompt_items(path: Optional[str]) -> Dict[str, Any]:
    cat = load_prompt_catalog(path)
    items = cat.get("items") or []
    out_items = []
    for it in items:
        pid = (it.get("id") or "").strip()
        if not pid:
            continue
        out_items.append({"id": pid, "label": (it.get("label") or pid)})
    default_id = (cat.get("default") or (out_items[0]["id"] if out_items else "soap_v1"))
    return {"default": default_id, "items": out_items}

def build_prompt(prompt_id: str, asr_text: str, *, path: Optional[str]) -> str:
    cat = load_prompt_catalog(path)
    items = cat.get("items") or []
    tpl = None
    for it in items:
        if (it.get("id") or "").strip() == prompt_id:
            tpl = it.get("prompt")
            break
    if not tpl:
        # fallback
        tpl = '''以下は医者と患者の診察室での会話で主に医者の発言部分ですが、音声認識で一部同音異義語の書き違いがあります。
これを考慮して、SOAP形式にまとめてください。勝手に内容を追加せず喋った内容に沿って記述し、わからないことは記載しないでください。

=== 会話テキスト ===
{asr_text}
'''
    return tpl.replace("{asr_text}", asr_text)

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
            q = str((rec.get("meta") or {}).get("quality", "unknown"))
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

# dyna watcher
DYNA_WATCH_DIR = Path(CFG.get("dyna_watch_dir", "/home/yuki/SpeechID"))
DYNA_GLOB = CFG.get("dyna_glob", "dyna*.txt")
DYNA_DELETE_AFTER_READ = bool(CFG.get("delete_after_read", True))

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

    # 追加：ASR設定（UIで選んだものをここに入れる）
    "asr": {
        "model_name": None,     # 例: "large-v3" (models.jsonのキー)
        "model_path": None,     # 例: "/home/yuki/.cache/.../snapshots/..."
        "language": "ja",
        "temperature": 0.0,
        "prompt": "",
        # 必要なら beam_size, vad なども追加
    },

    # 追加：辞書・編集関連
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

def append_jsonl(obj: dict, jsonl_path: str):
    line = json.dumps(obj, ensure_ascii=False)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def append_text_line(text_path: str, s: str):
    with open(text_path, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def ensure_unknown_session():
    """患者IDが来る前に録音を開始した場合の保険"""
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

    # セッション開始時点のASR/辞書/編集設定をスナップショット
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
    """
    ★ ASR設定が変わったタイミングをJSONLに残す
    （セッションが開始済みのときだけ）
    """
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

    # 同一文字の連続（うううう…）を検知
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
# ASR config (defaults from config.json)
# -------------------------
ASR_CFG = CFG.get("asr", {})
ASR_ENABLED = bool(ASR_CFG.get("enabled", True))
ASR_MODEL = ASR_CFG.get("model_path", "small")  # path推奨
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

# -------------------------
# ASR model registry (models.json)
# -------------------------
MODELS_PATH = Path("models.json")
ASR_MODEL_ID = ASR_CFG.get("model_id")  # config.json側にあれば使う（なくてもOK）

def load_models_registry() -> dict[str, str]:
    if not MODELS_PATH.exists():
        return {}
    try:
        with open(MODELS_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        out = {}
        for k, v in (d or {}).items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                out[k.strip()] = v.strip()
        return out
    except Exception as e:
        log(f"[ASR] models.json load failed: {e}")
        return {}

MODELS_REGISTRY = load_models_registry()

# 起動時に model_id が指定されていて、models.jsonにあるなら優先
if ASR_MODEL_ID and ASR_MODEL_ID in MODELS_REGISTRY:
    ASR_MODEL = MODELS_REGISTRY[ASR_MODEL_ID]

# ★ 起動時に CURRENT.asr を defaults で初期化（ここ重要）
CURRENT["asr"]["model_name"] = ASR_MODEL_ID
CURRENT["asr"]["model_path"] = ASR_MODEL
CURRENT["asr"]["language"] = ASR_LANG
CURRENT["asr"]["temperature"] = ASR_TEMP
CURRENT["asr"]["prompt"] = ASR_PROMPT_DEFAULT

def get_asr_runtime_config() -> dict:
    """
    ★ ASR実行時に使う設定を CURRENT から引く（未設定ならdefaultへフォールバック）
    """
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
        "beam_size": ASR_BEAM,                  # これは現状グローバルのまま
        "condition_on_previous_text": ASR_COPT, # これも現状グローバルのまま
        "device": ASR_DEVICE,
        "compute_type": ASR_COMPUTE,
    }

# -------------------------
# Whisper model cache
# -------------------------
# ★ モデル切替に対応するため、path/device/compute でキャッシュ
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
    """
    ★ 特定モデルだけ捨てる（model_pathがNoneなら全部捨てる）
    """
    global _MODEL_CACHE
    if model_path is None:
        _MODEL_CACHE = {}
        return
    key_prefix = (model_path, ASR_DEVICE, ASR_COMPUTE)
    _MODEL_CACHE.pop(key_prefix, None)

def set_asr_model_by_id(model_id: str) -> tuple[bool, str]:
    """
    models.json のキーを受け取り、ASR_MODEL を差し替える。
    ★ 同時に CURRENT["asr"] も更新し、jsonlにも asr_config を残す。
    """
    global ASR_MODEL, ASR_MODEL_ID, MODELS_REGISTRY

    MODELS_REGISTRY = load_models_registry()  # models.json編集しても反映
    if model_id not in MODELS_REGISTRY:
        return (False, f"unknown model_id: {model_id}")

    old_path = ASR_MODEL

    ASR_MODEL_ID = model_id
    ASR_MODEL = MODELS_REGISTRY[model_id]

    # ★ CURRENTへ反映
    CURRENT["asr"]["model_name"] = model_id
    CURRENT["asr"]["model_path"] = ASR_MODEL

    # ★ 古いモデルキャッシュを捨てる（全部捨ててもOK）
    clear_model_cache_for(old_path)

    log(f"[ASR] model switched -> id={ASR_MODEL_ID} path={ASR_MODEL}")

    # ★ 既にセッションがあるなら、その時点の設定変更をjsonlに残す
    append_asr_config_event("model_switched")

    return (True, "ok")

# -------------------------
# dyna watcher
# -------------------------
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
    """
    dyna???????.txt の1行目:  ID,漢字氏名,ﾌﾘｶﾞﾅ,...
    をSJIS(cp932)で読み、先頭カラム(ID)だけ返す。
    """
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
    last_batch_sig = None  # (latest_path, latest_mtime, count)

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
    files = sorted(OUTPUTS_DIR.glob("*.txt"))
    return files

def _parse_session_name(name: str):
    m = SESSION_TXT_RE.match(name)
    if not m:
        return None
    return m.group("patient_id"), m.group("stamp")

# -------------------------
# Web
# -------------------------
app = FastAPI()


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
    items = list_prompt_items(LLM_PROMPTS_PATH)
    return {"ok": True, "items": items.get("items", []), "default_prompt_id": items.get("default")}

@app.get("/api/llm/history")
async def api_llm_history(session: str = "", patient_id: str = ""):
    """指定セッション（<pid>_<stamp>.txt）か patient_id でLLM要約履歴を返す。"""
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

    items = []
    if pid and stamp:
        pat = f"{pid}_{stamp}__*.txt"
        files = sorted(LLM_DIR.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True)
    elif pid:
        pat = f"{pid}_*__*.txt"
        files = sorted(LLM_DIR.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        files = []

    for f in files[:200]:
        # label: model/prompt/time
        name = f.name
        parts = name.split("__")
        model = parts[1] if len(parts) > 1 else ""
        prompt = parts[2] if len(parts) > 2 else ""
        items.append({"id": name, "label": f"{model} / {prompt} / {name[-19:-4]}"})

    return {"ok": True, "items": items}

@app.get("/api/llm/history/{item_id}")
async def api_llm_history_item(item_id: str):
    item_id = unquote(item_id)
    # no path traversal
    if "/" in item_id or "\\" in item_id or ".." in item_id:
        return JSONResponse({"ok": False, "error": "invalid id"}, status_code=400)
    p = (LLM_DIR / item_id).resolve()
    if LLM_DIR.resolve() not in p.parents:
        return JSONResponse({"ok": False, "error": "invalid path"}, status_code=400)
    if not p.exists():
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    txt = p.read_text(encoding="utf-8")
    return {"ok": True, "summary": txt}

@app.post("/api/llm/soap")
async def api_llm_soap(payload: Dict[str, Any] = Body(...)):
    """現在/過去セッションのJSONLからSOAP要約を作って返す（ついでに履歴保存）。"""
    session_txt = (payload.get("session") or "").strip()
    min_quality = (payload.get("min_quality") or "good").strip()
    include_meta = bool(payload.get("include_meta", False))
    prompt_id = (payload.get("prompt_id") or "soap_v1").strip()

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

    asr_text = collect_asr_text_from_jsonl(jsonl_path, min_quality=min_quality, max_lines=max_lines)
    if not asr_text.strip():
        return JSONResponse({"ok": False, "error": "ASR text empty"}, status_code=400)

    prompt = build_prompt(prompt_id, asr_text, path=LLM_PROMPTS_PATH)

    import time as _time
    t0 = _time.time()
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
        elapsed = round(_time.time() - t0, 3)

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
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.on_event("startup")
async def _startup():
    asyncio.create_task(dyna_watch_task())
    log("[APP] startup complete")

@app.get("/api/asr/models")
async def api_asr_models():
    reg = load_models_registry()

    # ★ current は “CURRENT優先” で返す（表示が一貫する）
    cur = CURRENT.get("asr", {}).get("model_name") or ASR_MODEL_ID

    if not cur and reg:
        # CURRENTにもASR_MODEL_IDにも無い場合だけ、path逆引き
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
    """
    ★ UIから prompt / temperature / language を反映する
    payload例:
      {"language":"ja", "temperature":0.0, "prompt":"..."}
    """
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

    # ★ 対応する jsonl があれば session_start/meta を読む（モデル表示に使える）
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

        return seg

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

            cfg_rt = get_asr_runtime_config()  # ★ 実行時点の設定を取得
            model_path = cfg_rt["model_path"]

            log(f"[ASR] dequeued seg#{seg_id} wav={wav} model={cfg_rt.get('model_name')}")

            try:
                model = get_model_for(model_path)  # ★ パスに応じたモデル
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

            text = "".join(texts).strip()
            dt = time.time() - t0

            asr_meta = {
                "asr_sec": round(dt, 2),
                "avg_logprob": round_or_none(safe_avg(avg_logprob_list), 3),
                "no_speech_prob": round_or_none(safe_avg(no_speech_list), 3),
                "compression_ratio": round_or_none(safe_avg(comp_ratio_list), 3),
            }

            quality, reasons = judge_quality(audio_meta, asr_meta, text)

            ensure_unknown_session()
            pid = CURRENT["patient_id"]
            txt_path = CURRENT["text_path"]
            jsonl_path = CURRENT["jsonl_path"]

            if text:
                append_text_line(txt_path, text)

            # ★ jsonl に「この asr がどの設定で走ったか」を一緒に保存
            append_jsonl({
                "type": "asr",
                "ts": now_iso(),
                "patient_id": pid,
                "seg_id": seg_id,
                "dur": dur,
                "wav": wav,
                "text": text,
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
                f"comp={asr_meta.get('compression_ratio')} text_head={text[:60]!r}"
            )

            await ws.send_json({
                "type": "asr",
                "patient_id": pid,
                "seg_id": seg_id,
                "dur": dur,
                "wav": wav,
                "text": text,
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
        await ws.send_json({"type": "status", "msg": "connected", "patient_id": CURRENT["patient_id"], "session_txt": (Path(CURRENT["text_path"]).name if CURRENT.get("text_path") else "")})

        while True:
            if st.reset_pending:
                st.reset_pending = False
                st.reset_audio()
                log("[WS] state reset (patient_changed)")

            data = await ws.receive_bytes()  # float32 PCM
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
        CLIENTS.pop(ws, None)
        if st.asr_task:
            st.asr_task.cancel()

# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    log(f"[APP] start SR={SR} frame={FRAME_MS}ms threshold={THRESHOLD_DB} dyna_dir={DYNA_WATCH_DIR}")
    SSL_CFG = CFG.get("ssl", {})

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
