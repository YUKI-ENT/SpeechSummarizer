# modules/recorder.py
from __future__ import annotations
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

from .audio_utils import rms_dbfs, save_wav_mono16
from .quality import calc_acoustic_meta, judge_quality
from .events import now_iso_local

@dataclass
class RecorderConfig:
    device: int
    sr: int                 # 0ならdefault_samplerateを使う（app側で確定）
    channels: int = 1
    blocksize: int = 0

    frame_ms: int = 50
    threshold_db: float = -46.0
    auto_threshold: bool = False
    calib_sec: float = 2.0
    margin_db: float = 18.0

    start_voice_frames: int = 3
    end_silence_frames: int = 12
    pre_roll_ms: int = 300
    min_sec: float = 0.4
    max_sec: float = 20.0

    outdir: str = "recordings/segments"
    save_wav: bool = True

    show_level: bool = False
    show_status: bool = False

class Recorder:
    def __init__(
        self,
        cfg: RecorderConfig,
        stop_ev: Event,
        on_segment: Callable[[dict], None],
        asr_queue: "queue.Queue[dict]"
    ):
        self.cfg = cfg
        self.stop_ev = stop_ev
        self.on_segment = on_segment  # segイベント（dict）を渡す
        self.q_asr = asr_queue

        self.outdir = Path(cfg.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        dev = sd.query_devices(cfg.device, "input")
        self.device_name = dev["name"]
        self.sr = int(dev["default_samplerate"]) if cfg.sr == 0 else int(cfg.sr)

        self.frame = int(self.sr * cfg.frame_ms / 1000)
        self.pre_roll_frames = max(1, int(cfg.pre_roll_ms / cfg.frame_ms))

        self.noise_dbfs: Optional[float] = None
        if cfg.auto_threshold:
            self._calibrate_noise()

        self.q_in: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2000)

        # segmentation state
        self.buf = np.zeros((0, cfg.channels), dtype=np.float32)
        self.ring: list[np.ndarray] = []
        self.in_speech = False
        self.voice_run = 0
        self.silence_run = 0
        self.utter_frames: list[np.ndarray] = []
        self.seg_index = 0
        self.last_level_print = time.time()

    def _calibrate_noise(self) -> None:
        cfg = self.cfg
        print(f"Calibrating noise for {cfg.calib_sec:.1f}s... (黙ってて)")
        x = sd.rec(int(cfg.calib_sec * self.sr), samplerate=self.sr, channels=1,
                   dtype="float32", device=cfg.device)
        sd.wait()
        mono = x[:, 0].astype(np.float32, copy=False)
        self.noise_dbfs = rms_dbfs(mono)
        cfg.threshold_db = self.noise_dbfs + cfg.margin_db
        print(f"noise={self.noise_dbfs:.2f} dBFS -> threshold={cfg.threshold_db:.2f} dBFS (margin {cfg.margin_db}dB)")

    def _cb(self, indata, frames, time_info, status):
        if status and self.cfg.show_status:
            print("PortAudio status:", status, file=sys.stderr)
        try:
            self.q_in.put_nowait(indata.copy())
        except queue.Full:
            print("WARNING: input queue full (dropped chunk)", file=sys.stderr)

    def _push_ring(self, f_mono: np.ndarray):
        self.ring.append(f_mono)
        if len(self.ring) > self.pre_roll_frames:
            self.ring.pop(0)

    def _finalize_segment(self):
        if not self.utter_frames:
            self.in_speech = False
            self.voice_run = 0
            self.silence_run = 0
            self.ring = []
            return

        audio = np.concatenate(self.utter_frames, axis=0).astype(np.float32, copy=False)

        # reset state
        self.utter_frames = []
        self.in_speech = False
        self.voice_run = 0
        self.silence_run = 0
        self.ring = []

        dur = float(audio.shape[0] / self.sr)
        if dur < self.cfg.min_sec:
            return

        self.seg_index += 1
        ts = time.strftime("%Y%m%d_%H%M%S")
        wavpath = None
        if self.cfg.save_wav:
            wavpath = self.outdir / f"seg_{ts}_{self.seg_index:03d}.wav"
            save_wav_mono16(wavpath, audio, self.sr)

        # 音響メタ
        ac = calc_acoustic_meta(audio, self.sr, noise_dbfs=self.noise_dbfs)
        quality, reasons = judge_quality(ac)

        seg_event = {
            "type": "seg",
            "ts": now_iso_local(),
            "seg_id": self.seg_index,
            "dur": round(dur, 2),
            "sr": self.sr,
            "wav": str(wavpath) if wavpath else "",
            "acoustic": {
                "rms_dbfs": round(ac.rms_dbfs, 2),
                "peak_dbfs": round(ac.peak_dbfs, 2),
                "clip_ratio": round(ac.clip_ratio, 4),
                "silence_ratio": round(ac.silence_ratio, 4),
                "noise_dbfs": round(ac.noise_dbfs, 2) if ac.noise_dbfs is not None else None,
                "snr_db": round(ac.snr_db, 2) if ac.snr_db is not None else None,
            },
            "quality": quality,
            "drop_reason": reasons,
        }
        self.on_segment(seg_event)

        # ASRへ渡す（まずはWAVパス運用で安定させる）
        if wavpath:
            try:
                self.q_asr.put_nowait({
                    "seg_id": self.seg_index,
                    "dur": dur,
                    "sr": self.sr,
                    "wav": str(wavpath),
                    "quality": quality,
                    "drop_reason": reasons,
                    "acoustic": seg_event["acoustic"],
                })
            except queue.Full:
                # q_asrが溢れたら drop イベントを上げた方が後で追いやすい
                self.on_segment({
                    "type": "asr_drop",
                    "ts": now_iso_local(),
                    "seg_id": self.seg_index,
                    "reason": "q_asr_full"
                })

    def run_forever(self):
        cfg = self.cfg
        print(f"device={cfg.device} name={self.device_name}")
        print(f"sr={self.sr} ch={cfg.channels} block={cfg.blocksize} frame={cfg.frame_ms}ms ({self.frame} samples)")
        print(f"seg: start={cfg.start_voice_frames} end={cfg.end_silence_frames} pre_roll={cfg.pre_roll_ms}ms min={cfg.min_sec}s max={cfg.max_sec}s")
        print("Listening... Ctrl+C to stop")

        try:
            with sd.InputStream(device=cfg.device, samplerate=self.sr, channels=cfg.channels,
                                dtype="float32", blocksize=cfg.blocksize, callback=self._cb):
                while not self.stop_ev.is_set():
                    x = self.q_in.get()  # (n, ch)
                    self.buf = np.concatenate([self.buf, x], axis=0)

                    while self.buf.shape[0] >= self.frame:
                        f = self.buf[:self.frame, :]
                        self.buf = self.buf[self.frame:, :]

                        mono = f[:, 0].astype(np.float32, copy=False)
                        level = rms_dbfs(mono)

                        if cfg.show_level and (time.time() - self.last_level_print >= 1.0):
                            print(f"level {level:7.2f} dBFS")
                            self.last_level_print = time.time()

                        self._push_ring(mono)
                        is_voice = level > cfg.threshold_db

                        if not self.in_speech:
                            if is_voice:
                                self.voice_run += 1
                                if self.voice_run >= cfg.start_voice_frames:
                                    self.in_speech = True
                                    self.utter_frames = list(self.ring)
                                    self.silence_run = 0
                            else:
                                self.voice_run = 0
                        else:
                            self.utter_frames.append(mono)

                            if is_voice:
                                self.silence_run = 0
                            else:
                                self.silence_run += 1
                                if self.silence_run >= cfg.end_silence_frames:
                                    self._finalize_segment()
                                    continue

                            dur = sum(a.shape[0] for a in self.utter_frames) / self.sr
                            if dur >= cfg.max_sec:
                                self._finalize_segment()
                                continue

        except KeyboardInterrupt:
            print("\nStopping...")
            self._finalize_segment()
            self.stop_ev.set()
