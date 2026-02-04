import argparse
import numpy as np
import sounddevice as sd
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--channels", type=int, default=1)
    args = ap.parse_args()

    print("device:", args.device)
    print("samplerate:", args.samplerate, "channels:", args.channels)

    x = sd.rec(int(args.seconds * args.samplerate),
               samplerate=args.samplerate,
               channels=args.channels,
               dtype="float32",
               device=args.device)
    sd.wait()

    if x.ndim == 2:
        x0 = x[:, 0]
    else:
        x0 = x

    peak = float(np.max(np.abs(x0))) if x0.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(x0)) + 1e-12)) if x0.size else 0.0
    print("peak:", peak, "rms:", rms)

if __name__ == "__main__":
    main()
