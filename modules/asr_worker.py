# modules/asr_worker.py
from __future__ import annotations
import queue
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import wave
from faster_whisper import WhisperModel

from .audio_utils import resample_linear
from .quality import merge_asr_quality
from .events import now_iso_local

_MODEL_CACHE: Dict[Tuple[str, str, str], WhisperModel] = {}

def get_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    key = (model_size, device, compute_type)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]

def load_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        samp = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if samp != 2:
        raise ValueError(f"Only 16-bit PCM wav supported (sampwidth={samp})")

    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch >= 2:
        pcm = pcm.reshape(-1, ch)[:, 0]
    return pcm.astype(np.float32, copy=False), sr

@dataclass
class AsrConfig:
    model: str = "medium"
    lang: str = "ja"
    asr_device: str = "cuda"   # "cpu" or "cuda"
    compute_type: Optional[str] = None
    asr_sr: int = 16000

    beam_size: int = 10
    temperature: float = 0.0
    condition_on_previous_text: bool = True

class AsrWorker:
    def __init__(self, cfg: AsrConfig, stop_ev, q_asr: "queue.Queue[dict]", on_asr_event):
        self.cfg = cfg
        self.stop_ev = stop_ev
        self.q_asr = q_asr
        self.on_asr_event = on_asr_event  # dict を渡す
        if self.cfg.compute_type is None:
            self.cfg.compute_type = "float16" if self.cfg.asr_device == "cuda" else "int8"

        # warm
        _ = get_model(self.cfg.model, self.cfg.asr_device, self.cfg.compute_type)

    def _transcribe(self, audio16k: np.ndarray) -> tuple[str, dict]:
        model = get_model(self.cfg.model, self.cfg.asr_device, self.cfg.compute_type)

        hot = "咳, 痰, 鼻水, 喉, 発熱, 下痢, 嘔吐, 腹痛, めまい, 耳鳴り, 出血"
        segments, info = model.transcribe(
            audio16k,
            language=self.cfg.lang,
            vad_filter=False,
            beam_size=self.cfg.beam_size,
            temperature=self.cfg.temperature,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            initial_prompt=(
                "日本の医療現場の会話。聞こえたとおりに書き起こす。推測で補完しない。"
                f" 症状語: {hot}。"
            )
        )

        texts: List[str] = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                texts.append(t)

        text = "".join(texts).strip()

        # faster-whisper の info から取れる範囲で（環境差あるので get で安全に）
        asr_meta = {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
        }
        # セグメント側にも avg_logprob 等が入ることが多いので、平均を計算
        # seg.avg_logprob / seg.no_speech_prob / seg.compression_ratio
        alps, nsps, crs = [], [], []
        for seg in segments:
            if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                alps.append(float(seg.avg_logprob))
            if hasattr(seg, "no_speech_prob") and seg.no_speech_prob is not None:
                nsps.append(float(seg.no_speech_prob))
            if hasattr(seg, "compression_ratio") and seg.compression_ratio is not None:
                crs.append(float(seg.compression_ratio))

        if alps: asr_meta["avg_logprob"] = float(np.mean(alps))
        if nsps: asr_meta["no_speech_prob"] = float(np.mean(nsps))
        if crs:  asr_meta["compression_ratio"] = float(np.mean(crs))

        return text, asr_meta

    def run_forever(self):
        while not self.stop_ev.is_set():
            try:
                job = self.q_asr.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                seg_id = job["seg_id"]
                wav = job["wav"]
                base_q = job.get("quality", "good")
                base_r = job.get("drop_reason", [])
                t0 = time.time()

                audio, sr = load_wav_mono_f32(wav)
                audio16k = resample_linear(audio, sr, self.cfg.asr_sr)

                text, asr_meta = self._transcribe(audio16k)
                dt = time.time() - t0

                q, reasons = merge_asr_quality(base_q, base_r, asr_meta)

                ev = {
                    "type": "asr",
                    "ts": now_iso_local(),
                    "seg_id": seg_id,
                    "wav": wav,
                    "dur": round(float(job.get("dur", 0.0)), 2),
                    "text": text,
                    "asr_sec": round(dt, 2),
                    "asr_quality": {
                        "avg_logprob": asr_meta.get("avg_logprob", None),
                        "no_speech_prob": asr_meta.get("no_speech_prob", None),
                        "compression_ratio": asr_meta.get("compression_ratio", None),
                    },
                    "quality": q,
                    "drop_reason": reasons,
                }
                self.on_asr_event(ev)

            except Exception as e:
                self.on_asr_event({
                    "type": "asr_error",
                    "ts": now_iso_local(),
                    "seg_id": job.get("seg_id", None),
                    "error": str(e),
                    "wav": job.get("wav", ""),
                })
            finally:
                self.q_asr.task_done()
