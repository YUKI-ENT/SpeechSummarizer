import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import ctranslate2

_MODEL_CACHE = {}

def list_input_devices() -> None:
    devices = sd.query_devices()
    print("=== Input Devices ===")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            hostapi = sd.query_hostapis(d["hostapi"])["name"]
            print(f"[{i:2d}] ch={d['max_input_channels']}  {hostapi} | {d['name']}")
    print("=====================")


def record_wav(out_path: Path, seconds: float, samplerate: int, channels: int, device: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording {seconds:.1f}s ...")

    audio = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype="int16",
        device=device,
    )
    sd.wait()

    audio_np = np.asarray(audio)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(samplerate)
        wf.writeframes(audio_np.tobytes())

    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    print(f"Saved: {out_path}  peak={peak:.0f}")

def get_model(model_size: str, asr_device: str, compute_type: str) -> WhisperModel:
    key = (model_size, asr_device, compute_type)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(model_size, device=asr_device, compute_type=compute_type)
    return _MODEL_CACHE[key]

def transcribe_wav(wav_path: Path, model_size: str, language: str, asr_device: str, compute_type: str, vad: bool) -> str:
    model = get_model(model_size, asr_device, compute_type)
    print("CUDA supported:", ctranslate2.get_supported_compute_types("cuda"))
    print("ASR device:", asr_device, "compute:", compute_type)

    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 600, "speech_pad_ms": 200} if vad else None,
        beam_size=5,
    )

    lines = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            lines.append(f"[{seg.start:6.2f}-{seg.end:6.2f}] {text}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="入力デバイス一覧")
    ap.add_argument("--device", type=int, default=None, help="入力デバイス index")
    ap.add_argument("--seconds", type=float, default=8.0, help="録音秒数")
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--out", default="recordings/last.wav")
    ap.add_argument("--model", default="small")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--compute", default="int8", help="CPU向け: int8 / int8_float16 / float32")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--asr-device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.list:
        list_input_devices()
        return

    wav_path = Path(args.out)

    record_wav(wav_path, args.seconds, args.samplerate, args.channels, args.device)

    print("Transcribing...")
    t0 = time.time()
    text = transcribe_wav(wav_path, args.model, args.lang, args.asr_device, args.compute, vad=not args.no_vad)
    dt = time.time() - t0

    print("\n----- TRANSCRIPT -----")
    print(text if text else "(empty)")
    print("----------------------")
    print(f"Time: {dt:.2f} sec")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
