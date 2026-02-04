import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from faster_whisper import WhisperModel

_MODEL_CACHE = {}

def get_model(model_size: str, device: str, compute: str) -> WhisperModel:
    key = (model_size, device, compute)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute)
    return _MODEL_CACHE[key]

def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio[:, 0]
    return audio.astype(np.float32, copy=False), int(sr)

def asr_transcribe(wav_path: Path, model_size: str, lang: str, device: str, compute: str) -> str:
    model = get_model(model_size, device, compute)
    t0 = time.time()
    segments, _info = model.transcribe(
        str(wav_path),             # ★ numpyじゃなくパス
        language=lang,
        vad_filter=False,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt="これは日本語の診療会話です。聞こえたとおりに書き起こしてください。",
        hotwords="咳, せき, 咳嗽, 痰, 喘鳴, 鼻水, 喉, 発熱"
    )
    text = "".join([s.text for s in segments]).strip()
    dt = time.time() - t0
    print(f"[ASR] {wav_path.name}  asr={dt:.2f}s  -> {text}")
    return text

def ollama_generate(base_url: str, model: str, prompt: str, timeout_sec: int = 180) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def parse_json_loose(s: str) -> tuple[bool, dict | None, str]:
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    # 末尾にゴミが付くケース対策（最初の { から最後の } までを拾う）
    if "{" in t and "}" in t:
        t2 = t[t.find("{"):t.rfind("}") + 1]
    else:
        t2 = t
    try:
        return True, json.loads(t2), ""
    except Exception as e:
        return False, None, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="input wav (segment wav)")
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--asr-model", default="medium")
    ap.add_argument("--asr-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--compute", default=None)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="gemma3:12b")
    args = ap.parse_args()

    compute = args.compute or ("float16" if args.asr_device == "cuda" else "int8")

    text = asr_transcribe(Path(args.wav), args.asr_model, args.lang, args.asr_device, compute)

    prompt = (
        "あなたは医療事務作業を支援するアシスタントです。"
        "入力は診療会話の文字起こしです。"
        "SOAP形式をJSONで返してください。JSON以外は出力しないでください。\n\n"
        "次の会話テキストをSOAPに変換してください。\n\n"
        f"{text}\n\n"
        "出力JSONスキーマ:\n"
        "{"
        "\"S\": string, \"O\": string, \"A\": string, \"P\": string, "
        "\"warnings\": string[]"
        "}\n"
    )

    t0 = time.time()
    raw = ollama_generate(args.ollama_url, args.ollama_model, prompt)
    dt = time.time() - t0

    ok, obj, err = parse_json_loose(raw)
    if ok:
        print(f"[LLM] {dt:.2f}s  SOAP JSON ok")
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(f"[LLM] {dt:.2f}s  SOAP JSON parse FAILED: {err}")
        print("---- RAW ----")
        print(raw)

if __name__ == "__main__":
    main()
