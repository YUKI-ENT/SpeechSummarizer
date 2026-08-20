"""Non-GUI helpers for the Windows launcher."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path


def resolve_launcher_path(value: str, app_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (app_dir / path).resolve()


def qwen_api_is_ready(base_url: str, timeout_sec: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/ready", timeout=timeout_sec) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("schema_version") == 1 and payload.get("status") == "ready"
    except (OSError, ValueError, urllib.error.URLError):
        return False
