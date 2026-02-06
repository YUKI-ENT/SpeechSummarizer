import time, wave, datetime, asyncio
from pathlib import Path
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from faster_whisper import WhisperModel

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# -------------------------
# Settings
# -------------------------
OUT_WAV_DIR = Path("data/wav")
OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

SR = 48000
FRAME_MS = 50
FRAME = int(SR * FRAME_MS / 1000)

THRESHOLD_DB = -42.0
START_VOICE_FRAMES = 3
END_SILENCE_FRAMES = 12
PRE_ROLL_MS = 300
PRE_ROLL_FRAMES = max(1, int(PRE_ROLL_MS / FRAME_MS))
MIN_SEC = 0.4
MAX_SEC = 20.0

# --- ASR settings ---
ASR_ENABLED = True
#ASR_MODEL = "large-v3-turbo"     # tiny / base / small / medium / large-v3 など
ASR_MODEL = "/home/yuki/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
ASR_LANG = "ja"
ASR_DEVICE = "cuda"      # "cpu" or "cuda"
ASR_COMPUTE = "float16"  # cuda: float16推奨 / cpu: int8推奨

def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x*x) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)

def save_wav_mono16(path: Path, audio_f32: np.ndarray, sr: int):
    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

def peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x)) + 1e-12)
    return 20.0 * np.log10(p + 1e-12)

def clip_ratio(x: np.ndarray, thr: float = 0.98) -> float:
    return float(np.mean(np.abs(x) >= thr))

def silence_ratio(x: np.ndarray, thr: float = 0.005) -> float:
    # thr=0.005 はだいたい -46dBFS 相当の雑な無音判定。必要なら調整。
    return float(np.mean(np.abs(x) < thr))

def zcr(x: np.ndarray) -> float:
    # zero-crossing rate（雑音/摩擦音の目安）
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
    """
    quality: good/maybe/bad
    reasons: ['low_rms', 'high_no_speech', ...]
    ルールは仮。あとで調整しやすいように理由を返す。
    """
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

    # 反復っぽい（同一文字が長く続く）を雑に検知： "ううううう..." 等
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

    # 判定（badがあればbad、maybeがあればmaybe）
    bad_triggers = {
        "high_no_speech_prob", "high_compression_ratio", "long_char_run"
    }
    maybe_triggers = {
        "low_rms", "high_silence_ratio", "low_avg_logprob", "clipping"
    }

    if any(r in bad_triggers for r in reasons):
        return "bad", reasons
    if any(r in maybe_triggers for r in reasons):
        return "maybe", reasons
    return "good", reasons

# -------------------------
# Whisper model cache (global: 再ロード禁止)
# -------------------------
_MODEL = None
def get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        log(f"[ASR] loading model: {ASR_MODEL} device={ASR_DEVICE} compute={ASR_COMPUTE}")
        _MODEL = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
        log("[ASR] model loaded")
    return _MODEL

# -------------------------
# Web
# -------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

# -------------------------
# Per-connection state
# -------------------------
class State:
    def __init__(self):
        self.buf = np.zeros((0,), dtype=np.float32)
        self.ring = []
        self.in_speech = False
        self.voice_run = 0
        self.silence_run = 0
        self.utter_frames = []
        self.seg_index = 0
        self.last_level_sent = 0.0

        self.asr_q: asyncio.Queue[dict] = asyncio.Queue(maxsize=20)
        self.asr_task: asyncio.Task | None = None

    def push_ring(self, f):
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
        wavpath = OUT_WAV_DIR / f"seg_{now_stamp()}_{self.seg_index:03d}.wav"
        save_wav_mono16(wavpath, audio, SR)

        # --- audio meta ---
        a_rms = rms_dbfs(audio)
        a_peak = peak_dbfs(audio)
        a_clip = clip_ratio(audio)
        a_sil = silence_ratio(audio)
        a_zcr = zcr(audio)

        audio_meta = {
            "rms_dbfs": round(a_rms, 2),
            "peak_dbfs": round(a_peak, 2),
            "clip_ratio": round(a_clip, 4),
            "silence_ratio": round(a_sil, 4),
            "zcr": round(a_zcr, 4),
        }

        return {
            "seg_id": self.seg_index,
            "dur": round(dur, 2),
            "wav": str(wavpath),
            "audio_meta": audio_meta
        }

async def asr_worker(ws: WebSocket, st: State):
    """
    st.asr_q から wav job を取り出してASRし、wsへ asr イベント送信
    job は finalize_segment() の戻り（audio_meta含む）を想定
    """
    if not ASR_ENABLED:
        log("[ASR] disabled -> worker exit")
        return

    try:
        model = get_model()  # warm-up
    except Exception as e:
        log(f"[ASR] model load FAILED: {e}")
        await ws.send_json({"type": "error", "where": "asr_model_load", "error": str(e)})
        return

    log("[ASR] worker started")

    while True:
        job = await st.asr_q.get()
        seg_id = job.get("seg_id")
        wav = job.get("wav")
        log(f"[ASR] dequeued seg#{seg_id} wav={wav}")

        try:
            wav = job["wav"]
            seg_id = job["seg_id"]
            dur = job["dur"]
            audio_meta = job.get("audio_meta", {})  # 無い場合も落ちない

            t0 = time.time()
            log(f"[ASR] transcribe start seg#{seg_id:03d} dur={dur}s")

            segments, info = model.transcribe(
                wav,
                language=ASR_LANG,
                vad_filter=False,
                beam_size=10,
                temperature=0.0,
                condition_on_previous_text=True,
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

                # 版によって無いことがあるので getattr
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

            log(
                f"[ASR] done seg#{seg_id:03d} sec={dt:.2f} "
                f"q={quality} reasons={reasons} "
                f"rms={audio_meta.get('rms_dbfs')} no_speech={asr_meta.get('no_speech_prob')} "
                f"comp={asr_meta.get('compression_ratio')} text_head={text[:60]!r}"
            )

            await ws.send_json({
                "type": "asr",
                "seg_id": seg_id,
                "dur": dur,
                "wav": wav,
                "text": text,
                "meta": {
                    "audio": audio_meta,
                    "asr": asr_meta,
                    "quality": quality,
                    "reasons": reasons
                }
            })

        except Exception as e:
            log(f"[ASR] ERROR seg#{job.get('seg_id')}: {e}")
            await ws.send_json({"type": "error", "where": "asr", "error": str(e), "job": job})
        finally:
            st.asr_q.task_done()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    st = State()

    # ASRワーカー起動
    log(f"[WS] connected, starting ASR worker enabled={ASR_ENABLED}")
    st.asr_task = asyncio.create_task(asr_worker(ws, st))

    try:
        await ws.send_json({"type": "status", "msg": "connected"})
        while True:
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
                                await ws.send_json({"type": "saved", **seg})
                                # ASRへ投入（詰まったら落とす）
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
                            await ws.send_json({"type": "saved", **seg})
                            try:
                                st.asr_q.put_nowait(seg)
                            except asyncio.QueueFull:
                                await ws.send_json({"type": "asr_drop", "seg_id": seg["seg_id"], "reason": "asr_q full"})

    except WebSocketDisconnect:
        pass
    finally:
        if st.asr_task:
            st.asr_task.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
