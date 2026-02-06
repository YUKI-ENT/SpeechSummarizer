# modules/quality.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
from .audio_utils import rms_dbfs, peak_dbfs, clip_ratio, silence_ratio

@dataclass
class AcousticMeta:
    rms_dbfs: float
    peak_dbfs: float
    clip_ratio: float
    silence_ratio: float
    dur_sec: float
    noise_dbfs: Optional[float] = None
    snr_db: Optional[float] = None

def calc_acoustic_meta(audio_mono: np.ndarray, sr: int, noise_dbfs: Optional[float] = None) -> AcousticMeta:
    audio_mono = audio_mono.astype(np.float32, copy=False)
    dur = float(audio_mono.shape[0] / sr)
    r = float(rms_dbfs(audio_mono))
    p = float(peak_dbfs(audio_mono))
    cr = float(clip_ratio(audio_mono))
    srati = float(silence_ratio(audio_mono))
    snr = (r - noise_dbfs) if (noise_dbfs is not None) else None
    return AcousticMeta(
        rms_dbfs=r, peak_dbfs=p, clip_ratio=cr, silence_ratio=srati, dur_sec=dur,
        noise_dbfs=noise_dbfs, snr_db=snr
    )

def judge_quality(ac: AcousticMeta) -> tuple[str, List[str]]:
    """
    まずは音響だけで暫定判定（ASRメタが来たら合算して上書きする想定）
    """
    reasons: List[str] = []
    if ac.dur_sec < 0.4:
        reasons.append("too_short")
    if ac.clip_ratio > 0.01:
        reasons.append("clipping")
    if ac.silence_ratio > 0.85:
        reasons.append("mostly_silence")
    if ac.snr_db is not None and ac.snr_db < 8:
        reasons.append("low_snr")

    # ランク決め（雑でOK、あとで調整）
    if "too_short" in reasons or "mostly_silence" in reasons:
        return "bad", reasons
    if "low_snr" in reasons or "clipping" in reasons:
        return "maybe", reasons
    return "good", reasons

def merge_asr_quality(
    base_quality: str,
    base_reasons: List[str],
    asr_meta: Dict[str, Any]
) -> tuple[str, List[str]]:
    """
    Whisper系メタで上書き/補強する。
    asr_meta例: {"avg_logprob":..., "no_speech_prob":..., "compression_ratio":...}
    """
    q = base_quality
    reasons = list(base_reasons)

    nsp = asr_meta.get("no_speech_prob", None)
    alp = asr_meta.get("avg_logprob", None)
    cr = asr_meta.get("compression_ratio", None)

    if nsp is not None and nsp > 0.6:
        reasons.append("no_speech_prob_high")
        q = "bad"
    if alp is not None and alp < -1.0:
        reasons.append("avg_logprob_low")
        q = "bad" if q == "maybe" else q  # いったん弱め
    if cr is not None and cr > 2.4:
        reasons.append("repetition_suspect")
        q = "maybe" if q == "good" else q

    # 重複除去
    reasons = sorted(set(reasons))
    return q, reasons
