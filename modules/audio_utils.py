# modules/audio_utils.py
from __future__ import annotations
import wave
from pathlib import Path
import numpy as np

def rms_dbfs(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)

def peak_dbfs(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    p = float(np.max(np.abs(x)) + 1e-12)
    return 20.0 * np.log10(p + 1e-12)

def clip_ratio(x: np.ndarray, thr: float = 0.98) -> float:
    x = x.astype(np.float32, copy=False)
    return float(np.mean(np.abs(x) >= thr))

def silence_ratio(x: np.ndarray, thr_abs: float = 0.002) -> float:
    x = x.astype(np.float32, copy=False)
    return float(np.mean(np.abs(x) < thr_abs))

def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
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

def save_wav_mono16(path: Path, audio_f32: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
