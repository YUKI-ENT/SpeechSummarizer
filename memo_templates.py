import json
import os
import tempfile
from pathlib import Path
from typing import Any


def validate_memo_templates(data: Any) -> dict:
    if not isinstance(data, dict):
        raise ValueError("memo templates must be a JSON object")

    version = data.get("version", 1)
    if version != 1:
        raise ValueError(f"unsupported memo templates version: {version}")

    raw_templates = data.get("templates")
    if not isinstance(raw_templates, list):
        raise ValueError("memo templates 'templates' must be an array")

    templates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_templates, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"memo template #{index} must be an object")

        template_id = item.get("id")
        label = item.get("label")
        text = item.get("text")
        enabled = item.get("enabled", True)
        if not isinstance(template_id, str) or not template_id.strip():
            raise ValueError(f"memo template #{index} id is required")
        template_id = template_id.strip()
        if template_id in seen_ids:
            raise ValueError(f"duplicate memo template id: {template_id}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"memo template '{template_id}' label is required")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"memo template '{template_id}' text is required")
        if not isinstance(enabled, bool):
            raise ValueError(f"memo template '{template_id}' enabled must be true or false")

        seen_ids.add(template_id)
        templates.append({
            "id": template_id,
            "label": label.strip(),
            "text": text.strip(),
            "enabled": enabled,
        })

    return {"version": 1, "templates": templates}


def load_memo_templates(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return validate_memo_templates(json.load(f))


def save_memo_templates(path: Path, data: Any) -> None:
    normalized = validate_memo_templates(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            serialized = json.dumps(normalized, ensure_ascii=False, indent=2).replace("\n", "\r\n")
            f.write(f"{serialized}\r\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
