import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


# -----------------------------
# Audio utils
# -----------------------------
def read_wav_as_float32_mono(path: Path) -> tuple[np.ndarray, int]:
    """WAVを読み込み、float32 mono(-1..1)とサンプルレートを返す"""
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        pcm = wf.readframes(nframes)

    if sampwidth == 2:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        # 32bit PCM想定（環境によってはfloat32の場合もあるが、ここはPCM前提）
        x = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]  # 左chだけ（必要なら平均に変更してもOK）

    return x.astype(np.float32, copy=False), sr


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """軽量リサンプル（ASR前用）。"""
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


# -----------------------------
# ASR
# -----------------------------
_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}


def get_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    key = (model_name, device, compute_type)
    m = _MODEL_CACHE.get(key)
    if m is None:
        m = WhisperModel(model_name, device=device, compute_type=compute_type)
        _MODEL_CACHE[key] = m
    return m


def transcribe_audio(
    audio_f32: np.ndarray,
    model_name: str,
    lang: str,
    device: str,
    compute_type: str,
    beam_size: int,
    temperature: float,
    no_context: bool,
    initial_prompt: str,
) -> str:
    model = get_model(model_name, device, compute_type)

    segments, _info = model.transcribe(
        audio_f32,
        language=lang if lang else None,
        vad_filter=False,  # 既に区切られたwav前提（必要ならTrueでも可）
        beam_size=beam_size,
        temperature=temperature,
        condition_on_previous_text=not no_context,
        initial_prompt=initial_prompt if initial_prompt else None,
    )

    texts = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            texts.append(t)
    return "".join(texts).strip()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="入力WAVファイル、またはフォルダ")
    ap.add_argument("--glob", default="*.wav", help="フォルダ指定時のglob (default: *.wav)")
    ap.add_argument("--out", default="", help="出力テキストファイル（空ならstdout）")
    ap.add_argument("--append", action="store_true", help="--out を追記モードにする")

    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--lang", default="ja", help="空文字で自動判定")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--compute", default=None, help="cudaならfloat16推奨 / cpuならint8推奨")

    ap.add_argument("--asr-sr", type=int, default=16000, help="ASR前にこのSRへリサンプル")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--no-context", action="store_true", help="condition_on_previous_text=False")
    ap.add_argument("--initial-prompt", default="日本語の会話音声です。聞こえたとおりに書き起こしてください。推測で補完しないでください。")

    args = ap.parse_args()

    compute_type = args.compute
    if compute_type is None:
        compute_type = "float16" if args.device == "cuda" else "int8"

    in_path = Path(args.wav)

    # 入力一覧
    if in_path.is_dir():
        wav_list = sorted(in_path.glob(args.glob))
        if not wav_list:
            print(f"No wav files found: {in_path} / {args.glob}", file=sys.stderr)
            sys.exit(1)
    else:
        wav_list = [in_path]

    # 出力先
    out_f = None
    if args.out:
        out_f = open(args.out, "a" if args.append else "w", encoding="utf-8")

    # 先にモデルをロードしてウォームアップ（初回の遅延を前に）
    _ = get_model(args.model, args.device, compute_type)

    for wavp in wav_list:
        audio, sr = read_wav_as_float32_mono(wavp)
        sec = audio.shape[0] / sr if sr else 0.0

        # ASR用に 16k に変換（whisper系は基本16k）
        audio_16k = resample_linear(audio, sr, args.asr_sr)

        t0 = time.time()
        text = transcribe_audio(
            audio_16k,
            model_name=args.model,
            lang=args.lang,
            device=args.device,
            compute_type=compute_type,
            beam_size=args.beam,
            temperature=args.temp,
            no_context=args.no_context,
            initial_prompt=args.initial_prompt,
        )
        dt = time.time() - t0

        line = f"{wavp.name}\tsec={sec:.2f}\tasr={dt:.2f}s\t{text}\n"

        if out_f:
            out_f.write(line)
            out_f.flush()
        else:
            print(line, end="", flush=True)

    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
