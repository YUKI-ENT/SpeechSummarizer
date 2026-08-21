"""Non-GUI helpers for the Windows launcher."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


def resolve_launcher_path(value: str, app_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (app_dir / path).resolve()


@dataclass(frozen=True)
class QwenReadyStatus:
    reachable: bool
    ready: bool
    http_status: int | None
    payload: dict[str, Any]
    error: str | None = None


def fetch_qwen_ready_status(base_url: str, timeout_sec: float = 1.5) -> QwenReadyStatus:
    """Fetch Qwen3-ASR /ready while preserving useful non-200 response details."""
    url = f"{base_url.rstrip('/')}/ready"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            http_status = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (OSError, UnicodeError, ValueError):
            payload = {}
    except (OSError, UnicodeError, ValueError, urllib.error.URLError) as exc:
        return QwenReadyStatus(False, False, None, {}, str(exc))

    if not isinstance(payload, dict):
        return QwenReadyStatus(True, False, http_status, {}, "JSON response is not an object")

    schema_ok = payload.get("schema_version") == 1
    ready = http_status == 200 and schema_ok and payload.get("status") == "ready"
    error = None
    if not ready:
        error_info = payload.get("error")
        if isinstance(error_info, dict):
            error = str(error_info.get("message") or error_info.get("code") or "not ready")
        elif not schema_ok:
            error = f"unsupported schema_version: {payload.get('schema_version')}"
        else:
            error = str(payload.get("status") or f"HTTP {http_status}")
    return QwenReadyStatus(True, ready, http_status, payload, error)


def qwen_api_is_ready(base_url: str, timeout_sec: float = 1.5) -> bool:
    return fetch_qwen_ready_status(base_url, timeout_sec).ready
