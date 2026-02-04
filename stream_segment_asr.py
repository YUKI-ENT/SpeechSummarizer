import argparse
import queue
import sys
import time
import wave
from pathlib import Path
from threading import Thread, Event

import requests, json, datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


# -----------------------------
# Utilities
# -----------------------------
def rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """軽量リサンプル（ASR前だけ）。品質が必要なら別ライブラリに差し替え可。"""
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    n_src = x.shape[0]
    if n_src < 2:
        return x.astype(np.float32, copy=False)
    n_dst = int(round(n_src * dst_sr / src_sr))
    if n_dst < 2:
        return x[:1].astype(np.float32, copy=False)

    xp = np.arange(n_src, dtype=np.float32)
    x_new = np.linspace(0, n_src - 1, n_dst, dtype=np.float32)
    y = np.interp(x_new, xp, x.astype(np.float32, copy=False)).astype(np.float32)
    return y


def save_wav_mono16(path: Path, audio_f32: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


# -----------------------------
# Whisper model cache
# -----------------------------
_MODEL_CACHE = {}


def get_model(model_size: str, asr_device: str, compute_type: str) -> WhisperModel:
    key = (model_size, asr_device, compute_type)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(model_size, device=asr_device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def transcribe_float32_audio(audio_f32_16k: np.ndarray,
                             model_size: str,
                             language: str,
                             asr_device: str,
                             compute_type: str) -> str:
    model = get_model(model_size, asr_device, compute_type)
    segments, _info = model.transcribe(
        audio_f32_16k,
        language=language,
        vad_filter=False,  # ここは既に分割済みなので不要
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt="これは日本語の医療現場の会話です。聞こえたとおりに書き起こしてください。定型句で補完しないでください。",
        hotwords="咳 せき 咳嗽 痰 たん 鼻水 はなみず 喉 のど 発熱 ねつ 下痢 嘔吐 腹痛 めまい 耳なり 出血"
    )
    texts = []
    for seg in segments:
        t = seg.text.strip()
        if t:
            texts.append(t)
    return "".join(texts).strip()


# -----------------------------
# Ollama + JSONL helpers
# -----------------------------
def ollama_generate(base_url: str, model: str, prompt: str, timeout_sec: int = 180) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return r.json().get("response", "")


def parse_json_loose(s: str) -> tuple[bool, dict | None, str]:
    t = s.strip()
    # ```json ... ``` 対策
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    # 最初の{から最後の}だけ拾う
    if "{" in t and "}" in t:
        t = t[t.find("{"):t.rfind("}") + 1]
    try:
        return True, json.loads(t), ""
    except Exception as e:
        return False, None, str(e)


def now_iso_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def emit_jsonl(obj: dict, path: str):
    line = json.dumps(obj, ensure_ascii=False)
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def append_text(path: str, s: str):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def build_classify_prompt(texts: list[str]) -> str:
    joined = "\n".join([f"- {t}" for t in texts])
    return (
        "あなたは医療会話の文字起こしから『事実』だけを抽出し、S/O/A/Pに分類するツールです。\n"
        "【絶対ルール】\n"
        "- 推測・補完・診断・病名・治療提案は禁止。\n"
        "- 因果関係（〜すると〜）を作らない。発話に明確に含まれる場合のみ。\n"
        "- 言葉遊び、言い間違い訂正、相槌、笑いは医学情報にしない（ignoreへ）。\n"
        "- 不確実なものはuncertainへ。事実と混ぜない。\n"
        "- 出力はJSONのみ。コードフェンス ``` は禁止。日本語で。\n\n"
        "【入力（文字起こし断片）】\n"
        f"{joined}\n\n"
        "【出力JSONスキーマ】\n"
        "{"
        "\"items\": ["
        "{\"slot\":\"S|O|A|P\",\"facts\":[string],\"evidence\":[string]}"
        "],"
        "\"ignore\":[string],"
        "\"uncertain\":[string]"
        "}\n\n"
        "【補足】factsは短い箇条書き。evidenceは入力からの引用（そのまま）。\n"
    )


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--sr", type=int, default=0, help="0ならデバイス既定SR")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--blocksize", type=int, default=0)

    # segmentation
    ap.add_argument("--frame-ms", type=int, default=50)
    ap.add_argument("--auto-threshold", action="store_true")
    ap.add_argument("--calib-sec", type=float, default=2.0)
    ap.add_argument("--margin-db", type=float, default=18.0)
    ap.add_argument("--threshold-db", type=float, default=-46.0)

    ap.add_argument("--start-voice-frames", type=int, default=3)
    ap.add_argument("--end-silence-frames", type=int, default=12)
    ap.add_argument("--pre-roll-ms", type=int, default=300)
    ap.add_argument("--min-sec", type=float, default=0.4)
    ap.add_argument("--max-sec", type=float, default=20.0)

    # output
    ap.add_argument("--outdir", default="recordings/segments")
    ap.add_argument("--save-wav", action="store_true", help="各セグメントをwav保存する")
    ap.add_argument("--show-level", action="store_true")
    ap.add_argument("--show-status", action="store_true")

    # ASR
    ap.add_argument("--asr", action="store_true", help="ASRを有効化")
    ap.add_argument("--asr-sr", type=int, default=16000)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--asr-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--compute", default=None, help="cudaならfloat16推奨 / cpuならint8推奨")
    ap.add_argument("--asr-queue", type=int, default=10, help="ASR待ちキューの深さ。溢れると新しい発話を捨てる")

    # LLM (Ollama classify)
    ap.add_argument("--classify", action="store_true", help="ASR後にOllamaでS/O/A/P分類JSONを出力")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="gemma3:4b")
    ap.add_argument("--classify-batch", type=int, default=1, help="何セグメントまとめて分類するか(1-3推奨)")
    ap.add_argument("--jsonl-out", default="", help="イベントをJSONL追記保存（空なら保存しない）")
    ap.add_argument("--classify-timeout", type=int, default=120)

    # live transcript (full text)
    ap.add_argument("--fulltext-out", default="session_full.txt",
                    help="ASR全文を追記保存するテキストファイル（空なら保存しない）")
    ap.add_argument("--emit-delta", action="store_true",
                    help="ASR確定ごとに full_text_delta イベントをJSONLに出す（Web向け）")
    ap.add_argument("--emit-fulltext-every", type=float, default=1.0,
                    help="full_text（全文スナップショット）をJSONLに出す間隔（秒）。0なら出さない")
    ap.add_argument("--max-fulltext-chars", type=int, default=20000,
                    help="全文が長くなりすぎた時の上限（古い先頭を捨てる）")

    args = ap.parse_args()

    dev = sd.query_devices(args.device, "input")
    sr = int(dev["default_samplerate"]) if args.sr == 0 else args.sr
    frame = int(sr * args.frame_ms / 1000)
    pre_roll_frames = max(1, int(args.pre_roll_ms / args.frame_ms))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    compute_type = args.compute
    if compute_type is None:
        compute_type = "float16" if args.asr_device == "cuda" else "int8"

    print(f"device={args.device} name={dev['name']}")
    print(f"sr={sr} ch={args.channels} block={args.blocksize} frame={args.frame_ms}ms ({frame} samples)")
    print(f"seg: start={args.start_voice_frames} end={args.end_silence_frames} pre_roll={args.pre_roll_ms}ms min={args.min_sec}s max={args.max_sec}s")
    if args.asr:
        print(f"ASR: model={args.model} lang={args.lang} asr_device={args.asr_device} compute={compute_type} asr_sr={args.asr_sr}")

    # 自動しきい値（短時間だけ rec はOK）
    if args.auto_threshold:
        print(f"Calibrating noise for {args.calib_sec:.1f}s... (黙ってて)")
        x = sd.rec(int(args.calib_sec * sr), samplerate=sr, channels=1, dtype="float32", device=args.device)
        sd.wait()
        noise_db = rms_dbfs(x[:, 0])
        args.threshold_db = noise_db + args.margin_db
        print(f"noise={noise_db:.2f} dBFS -> threshold={args.threshold_db:.2f} dBFS (margin {args.margin_db}dB)")

    # 入力chunk queue（callbackから受ける）
    q_in: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2000)

    def cb(indata, frames, time_info, status):
        if status and args.show_status:
            print("PortAudio status:", status, file=sys.stderr)
        try:
            q_in.put_nowait(indata.copy())
        except queue.Full:
            print("WARNING: input queue full (dropped chunk)", file=sys.stderr)

    # ASR用 queue（セグメント確定ごと）
    q_asr: "queue.Queue[tuple[int, float, np.ndarray, Path|None]]" = queue.Queue(maxsize=args.asr_queue)
    # classify用 queue（ASRのtextを投げる）
    q_cls: "queue.Queue[tuple[int, str]]" = queue.Queue(maxsize=200)

    stop_ev = Event()

    # セッションログ（後でFinalizeに渡す素材）
    session_texts: list[str] = []

    full_text = ""
    last_full_emit = 0.0

    # --- ASR worker ---
    def asr_worker():
        nonlocal session_texts
        nonlocal full_text
        nonlocal last_full_emit
        if not args.asr:
            return
        _ = get_model(args.model, args.asr_device, compute_type)  # warm up

        while not stop_ev.is_set():
            try:
                seg_id, dur, audio_sr, wavpath = q_asr.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                t0 = time.time()
                audio_16k = resample_linear(audio_sr, sr, args.asr_sr)
                text = transcribe_float32_audio(audio_16k, args.model, args.lang, args.asr_device, compute_type)
                dt = time.time() - t0

                tag = f"seg#{seg_id:03d}"
                if wavpath is not None:
                    tag += f" {wavpath.name}"
                print(f"[ASR] {tag} dur={dur:.2f}s asr={dt:.2f}s  -> {text if text else '(empty)'}")

                # JSONLでASRイベントを吐く（Web/DBで拾える）
                emit_jsonl(
                    {"type": "asr", "ts": now_iso_local(), "seg_id": seg_id, "dur": round(dur, 2), "text": text},
                    args.jsonl_out
                )

                if text:
                    session_texts.append(text)

                    # 1) ファイルに追記（保険＆後で校正）
                    append_text(args.fulltext_out, text)

                    # 2) メモリ上の全文（Webのtextbox用の“現在の字幕”）
                    if full_text:
                        full_text += " " + text
                    else:
                        full_text = text

                    # 長くなりすぎるのを防ぐ（textboxも重くなるので）
                    if args.max_fulltext_chars > 0 and len(full_text) > args.max_fulltext_chars:
                        full_text = full_text[-args.max_fulltext_chars:]

                    # 3) Web向け：増分だけ出す（軽い）
                    if args.emit_delta:
                        emit_jsonl(
                            {"type": "full_text_delta", "ts": now_iso_local(), "seg_id": seg_id, "append": text},
                            args.jsonl_out
                        )

                    # 4) Web向け：全文スナップショット（再接続時に便利）
                    if args.emit_fulltext_every and args.emit_fulltext_every > 0:
                        now = time.time()
                        if now - last_full_emit >= args.emit_fulltext_every:
                            emit_jsonl(
                                {"type": "full_text", "ts": now_iso_local(), "text": full_text},
                                args.jsonl_out
                            )
                            last_full_emit = now

                # classifyへ（ここでは「投げるだけ」）
                if args.classify and text:
                    try:
                        q_cls.put_nowait((seg_id, text))
                    except queue.Full:
                        emit_jsonl(
                            {"type": "classify_drop", "ts": now_iso_local(), "seg_id": seg_id, "reason": "q_cls full"},
                            args.jsonl_out
                        )
            finally:
                q_asr.task_done()

    # --- classify worker ---
    def classify_worker():
        if not args.classify:
            return

        batch: list[tuple[int, str]] = []
        last_flush = time.time()

        def flush_batch():
            nonlocal batch, last_flush
            if not batch:
                return

            seg_ids = [b[0] for b in batch]
            texts = [b[1] for b in batch]

            prompt = build_classify_prompt(texts)
            t0 = time.time()
            try:
                raw = ollama_generate(args.ollama_url, args.ollama_model, prompt, timeout_sec=args.classify_timeout)
                dt = time.time() - t0
                ok, obj, err = parse_json_loose(raw)

                if ok and isinstance(obj, dict):
                    out = {
                        "type": "classified",
                        "ts": now_iso_local(),
                        "seg_ids": seg_ids,
                        "items": obj.get("items", []),
                        "ignore": obj.get("ignore", []),
                        "uncertain": obj.get("uncertain", []),
                        "llm_sec": round(dt, 2),
                    }
                else:
                    out = {"type": "classify_error", "ts": now_iso_local(), "seg_ids": seg_ids, "error": err, "raw_head": raw[:300]}
            except Exception as e:
                out = {"type": "classify_error", "ts": now_iso_local(), "seg_ids": seg_ids, "error": str(e)}

            emit_jsonl(out, args.jsonl_out)
            batch = []
            last_flush = time.time()

        while not stop_ev.is_set():
            try:
                seg_id, text = q_cls.get(timeout=0.2)
            except queue.Empty:
                # ちょい溜め（batch運用時）
                if batch and (time.time() - last_flush) > 0.8:
                    flush_batch()
                continue

            try:
                batch.append((seg_id, text))
                if len(batch) >= max(1, args.classify_batch):
                    flush_batch()
            finally:
                q_cls.task_done()

        # 終了時に残りを流す
        flush_batch()

    # スレッド起動（あなたのコードに無かった classify を追加）
    threads: list[Thread] = []
    t_asr = Thread(target=asr_worker, daemon=True)
    t_asr.start()
    threads.append(t_asr)

    t_cls = Thread(target=classify_worker, daemon=True)
    t_cls.start()
    threads.append(t_cls)

    # --------------- segmentation state ---------------
    buf = np.zeros((0, args.channels), dtype=np.float32)
    ring = []  # pre-roll 用：frame単位（最大 pre_roll_frames）
    in_speech = False
    voice_run = 0
    silence_run = 0
    utter_frames = []  # frame単位で保持（mono）
    seg_index = 0
    last_level_print = time.time()

    def push_ring(f: np.ndarray):
        ring.append(f)
        if len(ring) > pre_roll_frames:
            ring.pop(0)

    def finalize_segment():
        nonlocal utter_frames, in_speech, voice_run, silence_run, seg_index, ring
        if not utter_frames:
            in_speech = False
            voice_run = 0
            silence_run = 0
            ring = []
            return

        audio = np.concatenate(utter_frames, axis=0)
        utter_frames = []
        in_speech = False
        voice_run = 0
        silence_run = 0
        ring = []

        dur = audio.shape[0] / sr
        if dur < args.min_sec:
            return

        seg_index += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        wavpath = None
        if args.save_wav:
            wavpath = outdir / f"seg_{ts}_{seg_index:03d}.wav"
            save_wav_mono16(wavpath, audio, sr)
            print(f"[saved] {wavpath.name}  dur={dur:.2f}s")
        else:
            print(f"[seg] seg#{seg_index:03d} dur={dur:.2f}s")

        if args.asr:
            try:
                q_asr.put_nowait((seg_index, dur, audio.astype(np.float32, copy=False), wavpath))
            except queue.Full:
                print("WARNING: ASR queue full (dropped segment)", file=sys.stderr)

    print("Listening... Ctrl+C to stop")

    try:
        with sd.InputStream(device=args.device, samplerate=sr, channels=args.channels,
                            dtype="float32", blocksize=args.blocksize, callback=cb):
            while True:
                x = q_in.get()  # (n, ch)
                buf = np.concatenate([buf, x], axis=0)

                while buf.shape[0] >= frame:
                    f = buf[:frame, :]
                    buf = buf[frame:, :]

                    mono = f[:, 0].astype(np.float32, copy=False)
                    level = rms_dbfs(mono)

                    if args.show_level and (time.time() - last_level_print >= 1.0):
                        print(f"level {level:7.2f} dBFS")
                        last_level_print = time.time()

                    push_ring(mono)
                    is_voice = level > args.threshold_db

                    if not in_speech:
                        if is_voice:
                            voice_run += 1
                            if voice_run >= args.start_voice_frames:
                                in_speech = True
                                utter_frames = list(ring)  # pre-roll を含めて開始
                                silence_run = 0
                        else:
                            voice_run = 0
                    else:
                        utter_frames.append(mono)

                        if is_voice:
                            silence_run = 0
                        else:
                            silence_run += 1
                            if silence_run >= args.end_silence_frames:
                                finalize_segment()
                                continue

                        dur = sum(a.shape[0] for a in utter_frames) / sr
                        if dur >= args.max_sec:
                            finalize_segment()
                            continue

    except KeyboardInterrupt:
        print("\nStopping...")
        finalize_segment()
    finally:
        stop_ev.set()
        # ここで待ちたいなら join を使う（任意）
        # q_asr.join(); q_cls.join()
        # daemonなので基本はこのまま終了でOK


if __name__ == "__main__":
    main()
