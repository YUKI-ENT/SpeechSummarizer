import argparse
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(
    wav_path: Path,
    model_size: str = "small",
    language: str = "ja",
    device: str = "cpu",
    compute_type: str = "int8",
    vad: bool = True,
) -> str:
    """
    faster-whisperでWAVを文字起こしする（CPU向けデフォルト）。
    Ubuntu+GPUでは device='cuda', compute_type='float16' に切り替える。
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=vad,
        vad_parameters={
            # 無音区間の扱い。環境で調整可能
            "min_silence_duration_ms": 600,
            "speech_pad_ms": 200,
        } if vad else None,
        beam_size=5,
    )

    lines = []
    for seg in segments:
        t0 = seg.start
        t1 = seg.end
        text = seg.text.strip()
        if text:
            lines.append(f"[{t0:6.2f}-{t1:6.2f}] {text}")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a WAV file with faster-whisper")
    ap.add_argument("wav", help="input wav path (16kHz mono recommended)")
    ap.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3 ...")
    ap.add_argument("--lang", default="ja", help="language code (ja/en/...)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu or cuda")
    ap.add_argument("--compute", default=None, help="compute_type: cpu=int8/int8_float16/float32, cuda=float16/int8_float16")
    ap.add_argument("--no-vad", action="store_true", help="disable VAD")
    args = ap.parse_args()

    wav_path = Path(args.wav)

    # CPUデフォルトは軽くする
    if args.compute is None:
        compute_type = "int8" if args.device == "cpu" else "float16"
    else:
        compute_type = args.compute

    t0 = time.time()
    text = transcribe(
        wav_path=wav_path,
        model_size=args.model,
        language=args.lang,
        device=args.device,
        compute_type=compute_type,
        vad=not args.no_vad,
    )
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
