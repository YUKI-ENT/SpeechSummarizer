"""Shared schedule validation and recording deadlines (no browser dependencies)."""
from dataclasses import dataclass
from datetime import datetime
import math
import re

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_LABELS = ("月", "火", "水", "木", "金", "土", "日")


def default_auto_recording():
    return {
        "enabled": False,
        "stop_after_minutes": 20,
        "sound_enabled": False,
        "notifications_enabled": False,
        "schedule": {day: (["09:00-19:00"] if i < 5 else []) for i, day in enumerate(DAYS)},
    }


def parse_period(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2}", value):
        raise ValueError("時間帯は HH:MM-HH:MM 形式で入力してください。")
    start, end = value.split("-")
    def minute(text):
        hour, minute = map(int, text.split(":"))
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("時刻は00:00〜23:59で入力してください。")
        return hour * 60 + minute
    start, end = minute(start), minute(end)
    if start >= end:
        raise ValueError("終了時刻は開始時刻より後にしてください（日をまたぐ設定は曜日ごとに分けてください）。")
    return start, end


def normalize_auto_recording(raw=None):
    cfg = default_auto_recording()
    if raw is None:
        return cfg
    if not isinstance(raw, dict):
        raise ValueError("auto_recording はオブジェクトで指定してください。")
    for key in ("enabled", "sound_enabled", "notifications_enabled"):
        if key in raw:
            if not isinstance(raw[key], bool):
                raise ValueError(f"auto_recording.{key} は true / false で指定してください。")
            cfg[key] = raw[key]
    minutes = raw.get("stop_after_minutes", cfg["stop_after_minutes"])
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not math.isfinite(minutes) or not 1 <= minutes <= 1440:
        raise ValueError("自動停止時間は1〜1440分で指定してください。")
    cfg["stop_after_minutes"] = minutes
    schedule = raw.get("schedule", cfg["schedule"])
    if not isinstance(schedule, dict) or set(schedule) - set(DAYS):
        raise ValueError("曜日は mon, tue, wed, thu, fri, sat, sun で指定してください。")
    cfg["schedule"] = {}
    for day, label in zip(DAYS, DAY_LABELS):
        periods = schedule.get(day, [])
        if not isinstance(periods, list):
            raise ValueError(f"{label}曜日の時間帯は配列で指定してください。")
        cleaned = []
        for period in periods:
            period = period.strip() if isinstance(period, str) else period
            try:
                start, end = parse_period(period)
            except ValueError as e:
                raise ValueError(f"{label}曜日: {e}") from e
            cleaned.append((start, end, period))
        cleaned.sort()
        if any(left[1] > right[0] for left, right in zip(cleaned, cleaned[1:])):
            raise ValueError(f"{label}曜日の時間帯が重複しています。")
        cfg["schedule"][day] = [item[2] for item in cleaned]
    return cfg


def within_schedule(cfg, now=None):
    # The schedule uses the server's local time (also on Windows).
    now = now or datetime.now().astimezone()
    minute = now.hour * 60 + now.minute
    return cfg["enabled"] and any(
        start <= minute < end
        for start, end in map(parse_period, cfg["schedule"][DAYS[now.weekday()]])
    )


@dataclass
class RecordingSession:
    auto_enabled: bool = False
    active: bool = False
    blocked: bool = True
    deadline: float | None = None
    patient_id: str = ""
    request_id: str = ""
    request_deadline: float = 0

    def begin(self, now, minutes, patient_id):
        self.active = True
        self.blocked = False
        self.deadline = now + minutes * 60
        self.patient_id = patient_id
        self.request_id = ""

    def switch_patient(self, now, minutes, patient_id):
        if self.active and self.patient_id != patient_id:
            self.patient_id = patient_id
            self.deadline = now + minutes * 60

    def stop(self):
        self.active = False
        self.blocked = True
        self.deadline = None
        self.request_id = ""

    def expired(self, now):
        return self.active and self.deadline is not None and now >= self.deadline
