from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from tools.correction_tool.app import app as correction_tool_app

BASE_DIR = Path(__file__).resolve().parent


def _resolve_resource_dir(name: str) -> Path:
    candidates = [BASE_DIR / name]
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
        candidates.append(app_dir / "tools" / BASE_DIR.name / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


STATIC_DIR = _resolve_resource_dir("static")
TEMPLATE_DIR = _resolve_resource_dir("templates")

app = FastAPI(title="Analysis Tools")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/correction-tool", correction_tool_app)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
