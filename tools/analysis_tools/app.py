from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from tools.correction_tool.app import app as correction_tool_app

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

app = FastAPI(title="Analysis Tools")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/correction-tool", correction_tool_app)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
