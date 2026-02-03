import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


def list_input_devices() -> None:
    """入力デバイス一覧を表示（indexが後で --device に使える）"""
    devices = sd.query_devices()
    print("=== Input Devices ===")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            name = d["name"]
            hostapi = sd.query_hostapis(d["hostapi"])["name"]
            ch = d["max_input_channels"]
            default_sr = int(d.get("default_samplerate", 0) or 0)
            print(f"[{i:2d}] ch={ch} sr={default_sr:5d}  {hostapi} | {name}")
    print("=====================")


def record_to_wav(
    out_path: Path,
    seconds: float,
    samplerate: int,
    channels: int,
    device: int | None,
    dtype: str = "int16",
) -> None:
    """指定秒数だけ録音してWAV保存"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording {seconds:.1f}s ...")
    try:
        audio = sd.rec(
            int(seconds * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            device=device,
        )
        sd.wait()
    except Exception as e:
        raise RuntimeError(
            f"録音に失敗しました: {e}\n"
            f"ヒント: --list でデバイスを確認し、--device を指定してください。"
        ) from e

    # sounddeviceは形が (samples, channels)
    audio_np: np.ndarray = np.asarray(audio)

    # WAV書き込み
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(np.dtype(dtype).itemsize)  # int16なら2
        wf.setframerate(samplerate)
        wf.writeframes(audio_np.tobytes())

    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    print(f"Saved: {out_path}  peak={peak:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record from microphone and save to WAV (Windows)")
    ap.add_argument("--list", action="store_true", help="入力デバイス一覧を表示して終了")
    ap.add_argument("--device", type=int, default=None, help="入力デバイス index（--listで確認）")
    ap.add_argument("--seconds", type=float, default=5.0, help="録音秒数")
    ap.add_argument("--samplerate", type=int, default=16000, help="サンプルレート（ASRなら16000推奨）")
    ap.add_argument("--channels", type=int, default=1, help="チャンネル数（1推奨）")
    ap.add_argument("--out", default="recordings/test.wav", help="出力WAVパス")
    args = ap.parse_args()

    if args.list:
        list_input_devices()
        return

    out_path = Path(args.out)

    # デバイス未指定なら既定入力を使う（うまくいかない場合は --device 指定）
    record_to_wav(
        out_path=out_path,
        seconds=args.seconds,
        samplerate=args.samplerate,
        channels=args.channels,
        device=args.device,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCanceled.")
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
