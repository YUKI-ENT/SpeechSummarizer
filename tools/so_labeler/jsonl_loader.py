from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    from .models import AsrEvent
except ImportError:
    from models import AsrEvent

_JSONL_PATTERN = re.compile(r'^(?P<pid>\d+)_?(?P<stamp>\d{8}_\d{6}).*\.jsonl$', re.IGNORECASE)
EXCLUDED_JSONL_NAMES = {'patient_data.jsonl'}


@dataclass
class JsonlFileInfo:
    path: str
    name: str
    session_ts: Optional[str]
    patient_id: Optional[str]
    asr_count: int


def _parse_date_text(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, '%Y-%m-%d')


def _extract_session_stamp_from_name(path: Path) -> tuple[Optional[str], Optional[str]]:
    m = _JSONL_PATTERN.match(path.name)
    if not m:
        return None, None
    pid = m.group('pid')
    stamp = m.group('stamp')
    try:
        dt = datetime.strptime(stamp, '%Y%m%d_%H%M%S')
        return dt.isoformat(), pid
    except ValueError:
        return None, pid


def _read_session_header(path: Path) -> tuple[Optional[str], Optional[str], int]:
    session_ts, patient_id = _extract_session_stamp_from_name(path)
    asr_count = 0
    try:
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                if i == 0 and row.get('type') == 'session_start':
                    session_ts = row.get('ts') or session_ts
                    patient_id = row.get('patient_id') or patient_id
                if row.get('type') == 'asr' and (row.get('text') or '').strip():
                    asr_count += 1
    except Exception:
        pass
    return session_ts, patient_id, asr_count


def list_jsonl_files(data_dir: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> list[JsonlFileInfo]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f'data_dir not found: {data_dir}')

    start_dt = _parse_date_text(start_date)
    end_dt = _parse_date_text(end_date)

    result: list[JsonlFileInfo] = []
    for path in sorted(root.rglob('*.jsonl')):
        if path.name in EXCLUDED_JSONL_NAMES:
            continue
        session_ts, patient_id, asr_count = _read_session_header(path)
        if session_ts:
            try:
                file_dt = datetime.fromisoformat(session_ts.replace('Z', '+00:00')).replace(tzinfo=None)
            except ValueError:
                file_dt = None
        else:
            file_dt = None

        if start_dt and file_dt and file_dt.date() < start_dt.date():
            continue
        if end_dt and file_dt and file_dt.date() > end_dt.date():
            continue

        result.append(JsonlFileInfo(
            path=str(path),
            name=path.name,
            session_ts=session_ts,
            patient_id=patient_id,
            asr_count=asr_count,
        ))
    return result


def load_asr_events(file_path: str) -> list[AsrEvent]:
    path = Path(file_path)
    source_file = path.name
    events: list[AsrEvent] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('type') != 'asr':
                continue
            text = (row.get('text') or '').strip()
            if not text:
                continue
            events.append(
                AsrEvent(
                    source_file=source_file,
                    ts=row.get('ts', ''),
                    patient_id=row.get('patient_id'),
                    seg_id=row.get('seg_id'),
                    dur=row.get('dur'),
                    wav=row.get('wav'),
                    text=text,
                    meta=row.get('meta', {}),
                )
            )
    return events
