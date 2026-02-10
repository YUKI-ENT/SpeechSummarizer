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
    "full_text": ""
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

    append_jsonl(
        {"type": "session_start", "ts": now_iso(), "patient_id": patient_id, "stamp": stamp},
        CURRENT["jsonl_path"]
    )

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
# ASR config
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
        # 念のため文字列だけにする
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


# -------------------------
# Whisper model cache
# -------------------------
_MODEL = None

def get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        log(f"[ASR] loading model: {ASR_MODEL} device={ASR_DEVICE} compute={ASR_COMPUTE}")
        _MODEL = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
        log("[ASR] model loaded")
    return _MODEL

def set_asr_model_by_id(model_id: str) -> tuple[bool, str]:
    """
    models.json のキーを受け取り、ASR_MODEL を差し替え、WhisperModelキャッシュを破棄する。
    """
    global ASR_MODEL, ASR_MODEL_ID, _MODEL, MODELS_REGISTRY

    MODELS_REGISTRY = load_models_registry()  # ついでにリロード（models.json編集しても反映）
    if model_id not in MODELS_REGISTRY:
        return (False, f"unknown model_id: {model_id}")

    ASR_MODEL_ID = model_id
    ASR_MODEL = MODELS_REGISTRY[model_id]

    # ここが重要：モデルキャッシュ破棄（次のASRで再ロード）
    _MODEL = None
    log(f"[ASR] model switched -> id={ASR_MODEL_ID} path={ASR_MODEL}")
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
        # SMBで書き込み中でも読めるように、まず1行だけバイナリで読む
        with open(path, "rb") as f:
            raw = f.readline()

        line = raw.decode("cp932", errors="replace").strip()
        if not line:
            return None

        pid = line.split(",")[0].strip()
        # 念のため数字以外を弾く（必要なら緩める）
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
            st.reset_pending = True  # ★次の受信ループでリセットさせる
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

            # SMB書き込み完了待ち
            await asyncio.sleep(0.25)

            pid = read_patient_id_from_dyna(latest)

            # ★ ここで「最新を読んだら、そのファイル含め全部削除」
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
                        "jsonl_path": CURRENT["jsonl_path"]
                    }, reset_states=True)

            # 次の作成を待つ
            await asyncio.sleep(0.5)

        except Exception as e:
            log(f"[DYNA] watcher error: {e}")
            await asyncio.sleep(1.0)

# -------------------------
# データ表示
# -------------------------
SESSION_TXT_RE = re.compile(r"^(?P<patient_id>[^_]+)_(?P<stamp>\d{8}_\d{6})\.txt$")

def _list_session_txt_files() -> list[Path]:
    # OUTPUTS_DIR は既に app.py 内で定義されている前提（data/sessions）
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
    # 毎回リロード（models.json編集に追従）
    reg = load_models_registry()
    current_id = ASR_MODEL_ID if ASR_MODEL_ID else None

    # current_id が未設定なら、ASR_MODELの値から逆引きできるものがあればセット
    if not current_id and reg:
        for k, v in reg.items():
            if v == ASR_MODEL:
                current_id = k
                break

    return {
        "current": current_id,
        "models": [{"id": k, "label": k} for k in sorted(reg.keys())],
    }

@app.post("/api/asr/model")
async def api_asr_set_model(payload: dict = Body(...)):
    model_id = (payload.get("id") or "").strip()
    ok, msg = set_asr_model_by_id(model_id)
    if not ok:
        return {"ok": False, "error": msg}

    return {"ok": True, "current": ASR_MODEL_ID}

@app.get("/api/sessions")
def api_sessions():
    items = []
    for p in _list_session_txt_files():
        info = _parse_session_name(p.name)
        if not info:
            continue
        patient_id, stamp = info
        # stamp: YYYYMMDD_HHMMSS
        items.append({
            "name": p.name,               # 例: 16231_20260209_121559.txt
            "patient_id": patient_id,     # 例: 16231
            "stamp": stamp,               # 例: 20260209_121559
            "label": f"{patient_id} / {stamp[:8]} {stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}",
        })

    # 新しい順（stamp降順）
    items.sort(key=lambda x: x["stamp"], reverse=True)
    return {"items": items}

@app.get("/api/session/{name}")
def api_session_get(name: str):
    # URLエンコード対策
    name = unquote(name)

    # 安全のため、ファイル名だけ許可
    info = _parse_session_name(name)
    if not info:
        return JSONResponse({"error": "invalid session name"}, status_code=400)

    p = (OUTPUTS_DIR / name).resolve()
    # ディレクトリ外参照ブロック
    if OUTPUTS_DIR.resolve() not in p.parents:
        return JSONResponse({"error": "invalid path"}, status_code=400)

    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    text = p.read_text(encoding="utf-8", errors="replace")
    patient_id, stamp = info
    return {"name": name, "patient_id": patient_id, "stamp": stamp, "text": text}

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
        # ★追加：カルテ切替などで音声状態をリセットしたいときに立てるフラグ
        self.reset_pending = False

    def reset_audio(self):
        # """次の患者に混ざらないよう、音声バッファ/状態をクリア"""
        self.buf = np.zeros((0,), dtype=np.float32)
        self.ring = []
        self.in_speech = False
        self.voice_run = 0
        self.silence_run = 0
        self.utter_frames = []

        # ASR待ちも捨てる（前患者のwavを回さない）
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

        # reset
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

        # JSONL: saved
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

    try:
        model = get_model()
    except Exception as e:
        log(f"[ASR] model load FAILED: {e}")
        await ws.send_json({"type": "error", "where": "asr_model_load", "error": str(e)})
        return

    log("[ASR] worker started")

    while True:
        job = await st.asr_q.get()
        try:
            seg_id = job.get("seg_id")
            wav = job.get("wav")
            dur = job.get("dur")
            audio_meta = job.get("audio_meta", {})

            log(f"[ASR] dequeued seg#{seg_id} wav={wav}")

            t0 = time.time()
            log(f"[ASR] transcribe start seg#{seg_id:03d} dur={dur}s")

            segments, _info = model.transcribe(
                wav,
                language=ASR_LANG,
                vad_filter=False,
                beam_size=ASR_BEAM,
                temperature=ASR_TEMP,
                condition_on_previous_text=ASR_COPT,
                initial_prompt="日本の医療現場の会話。聞こえたとおりに書き起こす。推測で補完しない。"
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

            # 保存（患者セッションへ）
            ensure_unknown_session()
            pid = CURRENT["patient_id"]
            txt_path = CURRENT["text_path"]
            jsonl_path = CURRENT["jsonl_path"]

            if text:
                append_text_line(txt_path, text)

            append_jsonl({
                "type": "asr",
                "ts": now_iso(),
                "patient_id": pid,
                "seg_id": seg_id,
                "dur": dur,
                "wav": wav,
                "text": text,
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
    CLIENTS[ws] = st  # ★setではなくdictへ登録

    log(f"[WS] open enabled_asr={ASR_ENABLED}")
    st.asr_task = asyncio.create_task(asr_worker(ws, st))

    try:
        await ws.send_json({"type": "status", "msg": "connected", "patient_id": CURRENT["patient_id"]})

        while True:
            # ★追加：カルテ切替などのリセット指示が来ていたら、ここで安全にリセット
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
                                # saved（wsへ）
                                await ws.send_json({"type": "saved", "patient_id": CURRENT["patient_id"], **seg})

                                # asr投入（詰まったら落とす）
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