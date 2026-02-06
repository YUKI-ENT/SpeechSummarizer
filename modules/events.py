# modules/events.py
from __future__ import annotations
import json
import datetime
from typing import Optional, Dict, Any

def now_iso_local() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

class EventSink:
    """
    - JSONL保存（任意）
    - Socket.IOでWebへemit（任意）
    """
    def __init__(self, jsonl_path: str = "", sio_server=None):
        self.jsonl_path = jsonl_path
        self.sio = sio_server  # socketio.AsyncServer

    def emit(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        print(line, flush=True)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        # Socket.IOは async なので、app側で create_task して呼ぶのが安全
        # ここでは「sioがあれば返す」だけにして、呼び元で送る
