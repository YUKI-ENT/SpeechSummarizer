import json
import locale
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import urlparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from app_version import APP_VERSION
from launcher_helpers import (
    QwenReadyStatus,
    build_qwen_server_command,
    build_vibevoice_server_command,
    fetch_asr_ready_status,
    resolve_launcher_path,
)
from memo_ai_prompts import normalize_memo_ai_settings
from memo_templates import load_memo_templates, save_memo_templates


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = get_app_dir()
CONFIG_PATH = APP_DIR / "config.json"
CONFIG_SAMPLE_PATH = APP_DIR / "config.json.sample"
PATH_KEYS = {
    ("dyna_watch_dir",),
    ("outputs_dir",),
    ("wav_dir",),
    ("llm_outputs_dir",),
    ("memo_templates_path",),
    ("ssl", "certfile"),
    ("ssl", "keyfile"),
}
MODEL_PATH_PREFIX = ("asr", "models")
DEFAULT_HEARING_TRANSLATION_LANGUAGES = [
    {"id": "en", "label": "英語", "name": "English"},
    {"id": "zh", "label": "中国語", "name": "Simplified Chinese"},
    {"id": "ko", "label": "韓国語", "name": "Korean"},
]
FIELD_DEFAULTS = {
    "memo_templates_path": "./memo_templates.json",
    "hearing_translation_enabled": True,
    "hearing_translation_timeout": 30,
    "hearing_translation_default_language": "en",
    "asr_provider": "whisper",
    "qwen_base_url": "http://127.0.0.1:8010",
    "qwen_timeout": 35,
    "qwen_language": "Japanese",
    "qwen_context": "",
    "qwen_managed": True,
    "qwen_model_alias": "1.7b",
    "qwen_executable": "../QwenASR/dist/QwenASR-Server/QwenASR-Server.exe",
    "qwen_config": "../QwenASR/dist/QwenASR-Server/config.json",
    "qwen_startup_timeout": 90,
    "vibevoice_base_url": "http://127.0.0.1:8020",
    "vibevoice_timeout": 135,
    "vibevoice_language": "Japanese",
    "vibevoice_context": "",
    "vibevoice_hotwords": "",
    "vibevoice_include_segments": True,
    "vibevoice_managed": False,
    "vibevoice_model_alias": "7b",
    "vibevoice_python": "../VibeVoiceASR/.venv/Scripts/python.exe",
    "vibevoice_server_script": "../VibeVoiceASR/server.py",
    "vibevoice_config": "../VibeVoiceASR/config.json",
    "vibevoice_startup_timeout": 180,
}


def ensure_config_file() -> None:
    if CONFIG_PATH.exists():
        return
    if CONFIG_SAMPLE_PATH.exists():
        shutil.copyfile(CONFIG_SAMPLE_PATH, CONFIG_PATH)
        return
    raise FileNotFoundError(f"config file not found: {CONFIG_PATH}")


def load_config() -> dict:
    ensure_config_file()
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_config_sample() -> dict:
    if not CONFIG_SAMPLE_PATH.exists():
        return {}
    with CONFIG_SAMPLE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(cfg: dict, path: tuple[str, ...], default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_nested(cfg: dict, path: tuple[str, ...], value) -> None:
    cur = cfg
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def normalize_path_text(value: str) -> str:
    return value.replace("\\", "/").strip()


def validate_hearing_translation_languages(
    rows: list[tuple[str, str, str]], default_language: str
) -> list[dict[str, str]]:
    languages: list[dict[str, str]] = []
    seen: set[str] = set()
    for language_id, label, name in rows:
        language_id = language_id.strip()
        label = label.strip()
        name = name.strip()
        if not language_id and not label and not name:
            continue
        if not language_id or not label or not name:
            raise ValueError("翻訳言語のID、表示名、LLM向け言語名をすべて入力してください。")
        if len(language_id) > 24:
            raise ValueError(f"翻訳言語IDは24文字以内で入力してください: {language_id}")
        if len(name) > 80:
            raise ValueError(f"LLM向け言語名は80文字以内で入力してください: {name}")
        if language_id in seen:
            raise ValueError(f"翻訳言語IDが重複しています: {language_id}")
        seen.add(language_id)
        languages.append({"id": language_id, "label": label, "name": name})

    if not languages:
        raise ValueError("翻訳言語を1件以上追加してください。")
    if default_language.strip() not in seen:
        raise ValueError("既定の翻訳言語を言語一覧から選択してください。")
    return languages


def get_hearing_translation_languages(cfg: dict) -> list[dict[str, str]]:
    languages = get_nested(cfg, ("hearing_translation", "languages"), None)
    if not isinstance(languages, list) or not languages:
        return [dict(item) for item in DEFAULT_HEARING_TRANSLATION_LANGUAGES]
    return languages


def get_memo_templates_path(cfg: dict) -> Path:
    configured = Path(str(cfg.get("memo_templates_path") or "memo_templates.json"))
    return configured if configured.is_absolute() else APP_DIR / configured


def ensure_memo_templates_file(cfg: dict) -> Path:
    path = get_memo_templates_path(cfg)
    if path.exists():
        return path
    sample_path = path.with_name(f"{path.name}.sample")
    if not sample_path.exists() and path.name == "memo_templates.json":
        sample_path = APP_DIR / "memo_templates.json.sample"
    if not sample_path.exists():
        raise FileNotFoundError(f"memo templates file not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample_path, path)
    return path


class ScrollableTab(ttk.Frame):
    def __init__(self, parent, padding=0):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas, padding=padding)
        self._content_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        for widget in (self, self.canvas, self.content):
            widget.bind("<Enter>", self._bind_mousewheel, add="+")
            widget.bind("<Leave>", self._unbind_mousewheel, add="+")

    def _on_content_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._content_window, width=event.width)

    def _bind_mousewheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _can_scroll(self) -> bool:
        top, bottom = self.canvas.yview()
        return (bottom - top) < 0.999

    def _on_mousewheel(self, event) -> None:
        if not self._can_scroll() or event.delta == 0:
            return
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_mousewheel_linux(self, event) -> None:
        if not self._can_scroll():
            return
        delta = -1 if event.num == 4 else 1
        self.canvas.yview_scroll(delta, "units")


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SpeechSummarizer Launcher")
        self.root.geometry("1120x900")
        self.root.minsize(980, 780)

        self.proc: subprocess.Popen | None = None
        self.qwen_proc: subprocess.Popen | None = None
        self._starting = False
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.action_queue: queue.Queue[tuple[str, object | None]] = queue.Queue()
        self.vars: dict[str, tk.Variable] = {}
        self.field_meta: dict[str, dict] = {}
        self.model_rows: list[tuple[tk.StringVar, tk.StringVar]] = []
        self.prompt_rows: list[tuple[tk.StringVar, tk.StringVar, ScrolledText]] = []
        self.translation_language_rows: list[dict[str, object]] = []
        self.memo_ai_prompts: list[dict[str, str]] = []
        self.memo_ai_selected_index: int | None = None
        self._memo_ai_selection_changing = False
        self.memo_templates: list[dict] = []
        self.memo_template_selected_index: int | None = None
        self._memo_template_selection_changing = False
        self._auto_start_attempted = False
        self._cancel_start = threading.Event()
        self._qwen_status_checking = False
        self._qwen_restarting = False

        self.cfg = load_config()
        self._build_ui()
        self._load_form_from_config()
        self._update_asr_provider_ui()
        self._update_status()
        self._poll_log_queue()
        self.root.after(250, self.refresh_qwen_status)
        self.root.after(5000, self._periodic_qwen_status_refresh)
        self.root.after(150, self._maybe_auto_start_server)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        title_box = ttk.Frame(top)
        title_box.pack(side="left")
        ttk.Label(title_box, text="SpeechSummarizer", font=("", 16, "bold")).pack(side="left")
        ttk.Label(title_box, text=f"ver {APP_VERSION}").pack(side="left", padx=(10, 0))

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(10, 10))

        self.btn_save = ttk.Button(btns, text="設定保存", command=self.save_form)
        self.btn_save.pack(side="left")

        self.btn_reload = ttk.Button(btns, text="再読込", command=self.reload_form)
        self.btn_reload.pack(side="left", padx=(8, 0))

        self.btn_start = ttk.Button(btns, text="サーバー起動", command=self.start_server)
        self.btn_start.pack(side="left", padx=(8, 0))

        self.btn_stop = ttk.Button(btns, text="サーバー停止", command=self.stop_server)
        self.btn_stop.pack(side="left", padx=(8, 0))

        self.btn_open = ttk.Button(btns, text="ブラウザで開く", command=self.open_browser)
        self.btn_open.pack(side="left", padx=(8, 0))

        status_box = ttk.Frame(btns)
        status_box.pack(side="left", padx=(18, 0))
        self.status_indicator = tk.Canvas(status_box, width=20, height=20, highlightthickness=0, borderwidth=0)
        self.status_indicator.pack(side="left", padx=(0, 8))
        self.status_indicator_oval = self.status_indicator.create_oval(2, 2, 18, 18, outline="", fill="#9ca3af")
        self.status_var = tk.StringVar(value="停止中")
        ttk.Label(status_box, textvariable=self.status_var, font=("", 11, "bold")).pack(side="left")

        main_pane = ttk.Panedwindow(outer, orient="vertical")
        main_pane.pack(fill="both", expand=True)

        upper = ttk.Frame(main_pane)
        lower = ttk.Frame(main_pane, padding=(0, 10, 0, 0))
        main_pane.add(upper, weight=4)
        main_pane.add(lower, weight=2)

        notebook = ttk.Notebook(upper)
        notebook.pack(fill="both", expand=True)

        general_tab = ScrollableTab(notebook, padding=12)
        asr_tab = ScrollableTab(notebook, padding=12)
        llm_tab = ScrollableTab(notebook, padding=12)
        hearing_translation_tab = ScrollableTab(notebook, padding=12)
        memo_ai_tab = ScrollableTab(notebook, padding=12)
        memo_templates_tab = ScrollableTab(notebook, padding=12)

        notebook.add(general_tab, text="一般")
        notebook.add(asr_tab, text="ASR")
        notebook.add(llm_tab, text="LLM")
        notebook.add(hearing_translation_tab, text="難聴翻訳")
        notebook.add(memo_ai_tab, text="メモAI")
        notebook.add(memo_templates_tab, text="メモ定型文")

        self._build_general_tab(general_tab.content)
        self._build_asr_tab(asr_tab.content)
        self._build_llm_tab(llm_tab.content)
        self._build_hearing_translation_tab(hearing_translation_tab.content)
        self._build_memo_ai_tab(memo_ai_tab.content)
        self._build_memo_templates_tab(memo_templates_tab.content)
        self._build_log_panel(lower)

    def _build_general_tab(self, parent: ttk.Frame) -> None:
        for i in range(3):
            parent.columnconfigure(i, weight=1)

        row = 0
        self._add_entry(parent, "port", "Port", ("port",), kind="int", row=row, width=12)
        row += 1
        self._add_path_entry(parent, "dyna_watch_dir", "ID監視フォルダ", ("dyna_watch_dir",), row=row, select="dir")
        row += 1
        self._add_path_entry(parent, "outputs_dir", "セッション保存先", ("outputs_dir",), row=row, select="dir")
        row += 1
        self._add_path_entry(parent, "wav_dir", "WAV保存先", ("wav_dir",), row=row, select="dir")
        row += 1
        self._add_path_entry(parent, "llm_outputs_dir", "LLM保存先", ("llm_outputs_dir",), row=row, select="dir")
        row += 1
        self._add_entry(parent, "wav_expire_days", "WAV保持日数", ("wav_expire_days",), kind="int", row=row, width=12)
        row += 1
        self._add_bool(parent, "windows_auto_start_server", "Windows GUI起動時にサーバーも自動起動", ("windows_launcher", "auto_start_server"), row=row)
        row += 1
        self._add_bool(parent, "ssl_enabled", "SSL有効", ("ssl", "enabled"), row=row)
        row += 1
        self._add_path_entry(parent, "certfile", "SSL certfile", ("ssl", "certfile"), row=row, select="file")
        row += 1
        self._add_path_entry(parent, "keyfile", "SSL keyfile", ("ssl", "keyfile"), row=row, select="file")

    def _build_asr_tab(self, parent: ttk.Frame) -> None:
        for i in range(4):
            parent.columnconfigure(i, weight=1)

        row = 0
        provider_box = self._add_choice(
            parent, "asr_provider", "ASR Provider", ("asr", "provider"),
            ["whisper", "qwen3-asr", "vibevoice-asr"], row=row,
        )
        row += 1

        whisper_box = ttk.LabelFrame(parent, text="Whisper設定", padding=10)
        whisper_box.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        self.whisper_settings_box = whisper_box
        for i in range(4):
            whisper_box.columnconfigure(i, weight=1)

        whisper_row = 0
        self._add_entry(whisper_box, "asr_model_id", "選択モデルID", ("asr", "model_id"), kind="str", row=whisper_row, width=20)
        self._add_entry(whisper_box, "asr_language", "言語", ("asr", "language"), kind="str", row=whisper_row, col=2, width=12)
        whisper_row += 1
        self._add_choice(whisper_box, "asr_device", "Device", ("asr", "device"), ["cpu", "cuda"], row=whisper_row)
        self._add_choice(whisper_box, "asr_compute_type", "Compute Type", ("asr", "compute_type"), ["int8", "float16"], row=whisper_row, col=2)
        whisper_row += 1
        self._add_entry(whisper_box, "asr_beam_size", "Beam Size", ("asr", "beam_size"), kind="int", row=whisper_row, width=12)
        self._add_entry(whisper_box, "asr_temperature", "Temperature", ("asr", "temperature"), kind="float", row=whisper_row, col=2, width=12)
        whisper_row += 1
        self._add_bool(whisper_box, "asr_condition_prev", "前文脈を利用", ("asr", "condition_on_previous_text"), row=whisper_row)
        whisper_row += 1
        self._add_text(whisper_box, "asr_initial_prompt", "Initial Prompt", ("asr", "initial_prompt"), row=whisper_row, height=5)
        whisper_row += 1
        row += 1

        qwen_box = ttk.LabelFrame(parent, text="Qwen3-ASR API", padding=10)
        qwen_box.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        self.qwen_settings_box = qwen_box
        for i in range(4):
            qwen_box.columnconfigure(i, weight=1)
        qwen_row = 0
        self._add_entry(qwen_box, "qwen_base_url", "API URL", ("asr", "qwen", "base_url"), kind="str", row=qwen_row, width=36)
        qwen_row += 1
        self._add_entry(
            qwen_box, "qwen_timeout", "認識Timeout秒", ("asr", "qwen", "timeout_sec"),
            kind="float", row=qwen_row, width=12,
        )
        self._add_entry(
            qwen_box, "qwen_language", "言語", ("asr", "qwen", "language"),
            kind="str", row=qwen_row, col=2, width=12,
        )
        qwen_row += 1
        self._add_text(
            qwen_box, "qwen_context", "Context", ("asr", "qwen", "context"),
            row=qwen_row, height=4,
        )
        qwen_row += 1

        status_box = ttk.LabelFrame(qwen_box, text="API稼働状況", padding=8)
        status_box.grid(row=qwen_row, column=0, columnspan=4, sticky="nsew", pady=(6, 10))
        status_box.columnconfigure(1, weight=1)
        self.qwen_status_indicator = tk.Canvas(
            status_box, width=18, height=18, highlightthickness=0, borderwidth=0
        )
        self.qwen_status_indicator.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.qwen_status_indicator_oval = self.qwen_status_indicator.create_oval(
            2, 2, 16, 16, outline="", fill="#9ca3af"
        )
        self.qwen_status_var = tk.StringVar(value="未確認")
        ttk.Label(status_box, textvariable=self.qwen_status_var, font=("", 10, "bold")).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(status_box, text="更新", command=self.refresh_qwen_status, width=8).grid(
            row=0, column=2, sticky="e", padx=(8, 0)
        )
        self.btn_qwen_restart = ttk.Button(
            status_box, text="Qwen再起動", command=self.restart_managed_qwen, width=12
        )
        self.btn_qwen_restart.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self.qwen_status_details_var = tk.StringVar(value="/ready の応答を確認します。")
        ttk.Label(
            status_box, textvariable=self.qwen_status_details_var, wraplength=780,
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        qwen_row += 1

        self._add_bool(
            qwen_box, "qwen_managed", "Windows GUIランチャーでQwenASRを起動・停止",
            ("asr", "qwen", "managed_by_launcher"), row=qwen_row,
        )
        qwen_row += 1
        self._add_choice(
            qwen_box, "qwen_model_alias", "起動モデル", ("asr", "qwen", "model_alias"),
            ["0.6b", "1.7b"], row=qwen_row,
        )
        qwen_row += 1
        self._add_path_entry(
            qwen_box, "qwen_executable", "QwenASR-Server.exe", ("asr", "qwen", "executable_path"),
            row=qwen_row, select="file",
        )
        qwen_row += 1
        self._add_path_entry(
            qwen_box, "qwen_config", "Qwen config.json", ("asr", "qwen", "config_path"),
            row=qwen_row, select="file",
        )
        qwen_row += 1
        self._add_entry(
            qwen_box, "qwen_startup_timeout", "起動タイムアウト秒", ("asr", "qwen", "startup_timeout_sec"),
            kind="float", row=qwen_row, width=12,
        )
        ttk.Label(
            qwen_box,
            text="モデル読込後に /ready がReadyになるまで待つ上限時間です。固定時間待機する設定ではありません。",
            wraplength=780,
        ).grid(row=qwen_row + 1, column=0, columnspan=4, sticky="w", pady=(0, 6))
        row += 1

        vibevoice_box = ttk.LabelFrame(parent, text="VibeVoice-ASR API", padding=10)
        vibevoice_box.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        self.vibevoice_settings_box = vibevoice_box
        for i in range(4):
            vibevoice_box.columnconfigure(i, weight=1)
        vibe_row = 0
        self._add_entry(vibevoice_box, "vibevoice_base_url", "API URL", ("asr", "vibevoice", "base_url"), kind="str", row=vibe_row, width=36)
        vibe_row += 1
        self._add_entry(vibevoice_box, "vibevoice_timeout", "認識Timeout秒", ("asr", "vibevoice", "timeout_sec"), kind="float", row=vibe_row, width=12)
        self._add_entry(vibevoice_box, "vibevoice_language", "言語", ("asr", "vibevoice", "language"), kind="str", row=vibe_row, col=2, width=12)
        vibe_row += 1
        self._add_text(vibevoice_box, "vibevoice_context", "Context", ("asr", "vibevoice", "context"), row=vibe_row, height=4)
        vibe_row += 1
        self._add_text(vibevoice_box, "vibevoice_hotwords", "Hotwords（1行1語）", ("asr", "vibevoice", "hotwords"), row=vibe_row, height=5)
        vibe_row += 1
        self._add_bool(vibevoice_box, "vibevoice_include_segments", "話者・timestamp情報を保存", ("asr", "vibevoice", "include_segments"), row=vibe_row)
        vibe_row += 1

        vibe_status_box = ttk.LabelFrame(vibevoice_box, text="API稼働状況", padding=8)
        vibe_status_box.grid(row=vibe_row, column=0, columnspan=4, sticky="nsew", pady=(6, 10))
        vibe_status_box.columnconfigure(1, weight=1)
        self.vibevoice_status_indicator = tk.Canvas(vibe_status_box, width=18, height=18, highlightthickness=0, borderwidth=0)
        self.vibevoice_status_indicator.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.vibevoice_status_indicator_oval = self.vibevoice_status_indicator.create_oval(2, 2, 16, 16, outline="", fill="#9ca3af")
        self.vibevoice_status_var = tk.StringVar(value="未確認")
        ttk.Label(vibe_status_box, textvariable=self.vibevoice_status_var, font=("", 10, "bold")).grid(row=0, column=1, sticky="w")
        ttk.Button(vibe_status_box, text="更新", command=self.refresh_qwen_status, width=8).grid(row=0, column=2, sticky="e", padx=(8, 0))
        self.btn_vibevoice_restart = ttk.Button(vibe_status_box, text="VibeVoice再起動", command=self.restart_managed_qwen, width=16)
        self.btn_vibevoice_restart.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self.vibevoice_status_details_var = tk.StringVar(value="/ready の応答を確認します。")
        ttk.Label(vibe_status_box, textvariable=self.vibevoice_status_details_var, wraplength=780, justify="left").grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        vibe_row += 1

        self._add_bool(vibevoice_box, "vibevoice_managed", "Windows GUIランチャーでVibeVoiceASRを起動・停止", ("asr", "vibevoice", "managed_by_launcher"), row=vibe_row)
        vibe_row += 1
        self._add_entry(vibevoice_box, "vibevoice_model_alias", "起動モデルalias", ("asr", "vibevoice", "model_alias"), kind="str", row=vibe_row, width=16)
        vibe_row += 1
        self._add_path_entry(vibevoice_box, "vibevoice_python", "VibeVoice Python", ("asr", "vibevoice", "python_executable"), row=vibe_row, select="file")
        vibe_row += 1
        self._add_path_entry(vibevoice_box, "vibevoice_server_script", "VibeVoice server.py", ("asr", "vibevoice", "server_script"), row=vibe_row, select="file")
        vibe_row += 1
        self._add_path_entry(vibevoice_box, "vibevoice_config", "VibeVoice config.json", ("asr", "vibevoice", "config_path"), row=vibe_row, select="file")
        vibe_row += 1
        self._add_entry(vibevoice_box, "vibevoice_startup_timeout", "起動タイムアウト秒", ("asr", "vibevoice", "startup_timeout_sec"), kind="float", row=vibe_row, width=12)
        ttk.Label(vibevoice_box, text="VibeVoice用の仮想環境を指定し、モデル読込後に /ready になるまで待ちます。", wraplength=780).grid(row=vibe_row + 1, column=0, columnspan=4, sticky="w", pady=(0, 6))
        row += 1

        vad_box = ttk.LabelFrame(parent, text="VAD（全ASR provider共通）", padding=10)
        vad_box.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        for i in range(4):
            vad_box.columnconfigure(i, weight=1)

        vad_row = 0
        self._add_choice(vad_box, "vad_mode", "Mode", ("vad", "mode"), ["auto", "manual"], row=vad_row)
        self._add_entry(vad_box, "vad_manual_threshold_db", "Manual Threshold dB", ("vad", "manual_threshold_db"), kind="float", row=vad_row, col=2, width=12)
        vad_row += 1
        self._add_entry(vad_box, "vad_calibration_sec", "Calibration sec", ("vad", "calibration_sec"), kind="float", row=vad_row, width=12)
        self._add_entry(vad_box, "vad_margin_db", "Margin dB", ("vad", "margin_db"), kind="float", row=vad_row, col=2, width=12)
        vad_row += 1
        self._add_entry(vad_box, "vad_min_threshold_db", "Min Threshold dB", ("vad", "min_threshold_db"), kind="float", row=vad_row, width=12)
        self._add_entry(vad_box, "vad_max_threshold_db", "Max Threshold dB", ("vad", "max_threshold_db"), kind="float", row=vad_row, col=2, width=12)
        vad_row += 1
        self._add_entry(vad_box, "vad_noise_window_sec", "Noise Window sec", ("vad", "noise_window_sec"), kind="float", row=vad_row, width=12)
        self._add_entry(vad_box, "vad_update_margin_db", "Update Margin dB", ("vad", "update_margin_db"), kind="float", row=vad_row, col=2, width=12)
        vad_row += 1
        self._add_entry(vad_box, "vad_start_voice_frames", "Start Voice Frames", ("vad", "start_voice_frames"), kind="int", row=vad_row, width=12)
        self._add_entry(vad_box, "vad_end_silence_frames", "End Silence Frames", ("vad", "end_silence_frames"), kind="int", row=vad_row, col=2, width=12)
        vad_row += 1
        self._add_entry(vad_box, "vad_pre_roll_ms", "Pre-roll ms", ("vad", "pre_roll_ms"), kind="int", row=vad_row, width=12)
        self._add_entry(vad_box, "vad_quiet_percentile", "Quiet Percentile", ("vad", "quiet_percentile"), kind="float", row=vad_row, col=2, width=12)
        vad_row += 1
        self._add_entry(vad_box, "vad_min_sec", "Min sec", ("vad", "min_sec"), kind="float", row=vad_row, width=12)
        self._add_entry(vad_box, "vad_max_sec", "Max sec", ("vad", "max_sec"), kind="float", row=vad_row, col=2, width=12)
        row += 1

        model_box = ttk.LabelFrame(whisper_box, text="WhisperモデルIDとパス", padding=10)
        model_box.grid(row=whisper_row, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        model_box.columnconfigure(1, weight=1)

        ttk.Label(model_box, text="ID").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Label(model_box, text="Path").grid(row=0, column=1, sticky="w", padx=(0, 8), pady=(0, 6))

        model_cfg = get_nested(self.cfg, ("asr", "models"), {}) or {}
        row_idx = 1
        for model_id, model_path in model_cfg.items():
            id_var = tk.StringVar(value=str(model_id))
            path_var = tk.StringVar(value=str(model_path))
            self.model_rows.append((id_var, path_var))

            ttk.Entry(model_box, textvariable=id_var, width=18).grid(row=row_idx, column=0, sticky="we", padx=(0, 8), pady=4)
            ttk.Entry(model_box, textvariable=path_var).grid(row=row_idx, column=1, sticky="we", padx=(0, 8), pady=4)
            ttk.Button(
                model_box,
                text="参照",
                command=lambda v=path_var: self._browse_into_var(v, select="dir"),
                width=8,
            ).grid(row=row_idx, column=2, sticky="w", pady=4)
            row_idx += 1

        self.model_box = model_box

        btn_row = ttk.Frame(model_box)
        btn_row.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(btn_row, text="行追加", command=self.add_model_row).pack(side="left")
        provider_box.bind("<<ComboboxSelected>>", self._on_asr_provider_changed)

    def _build_llm_tab(self, parent: ttk.Frame) -> None:
        for i in range(4):
            parent.columnconfigure(i, weight=1)
        parent.rowconfigure(7, weight=1)

        row = 0
        self._add_entry(parent, "llm_server", "LLMサーバー (IP/ホスト名)", ("llm", "server"), kind="str", row=row, width=32)
        self._add_entry(parent, "llm_port", "Port", ("llm", "port"), kind="int", row=row, col=2, width=12)
        row += 1
        self._add_bool(parent, "llm_use_https", "HTTPS", ("llm", "use_https"), row=row)
        self._add_entry(parent, "llm_model_default", "既定モデル", ("llm", "model_default"), kind="str", row=row, col=2, width=20)
        row += 1
        self._add_entry(parent, "llm_api_key", "API Key", ("llm", "api_key"), kind="str", row=row, width=32, show="*")
        self._add_entry(parent, "llm_api_key_env", "API Key環境変数", ("llm", "api_key_env"), kind="str", row=row, col=2, width=20)
        row += 1
        self._add_entry(parent, "llm_timeout", "Timeout", ("llm", "timeout"), kind="float", row=row, width=12)
        self._add_entry(parent, "llm_temperature", "Temperature", ("llm", "temperature"), kind="float", row=row, col=2, width=12)
        row += 1
        self._add_entry(parent, "llm_top_p", "Top P", ("llm", "top_p"), kind="float", row=row, width=12)
        self._add_entry(parent, "llm_default_prompt_id", "既定Prompt ID", ("llm", "default_prompt_id"), kind="str", row=row, col=2, width=20)
        row += 1
        self._add_bool(
            parent,
            "llm_reasoning_enabled",
            "Reasoningを有効にする（OFF推奨）",
            ("llm", "reasoning_enabled"),
            row=row,
        )
        row += 1
        self._add_bool(parent, "auto_llm", "自動LLM", ("auto_llm",), row=row)
        row += 1

        auto_box = ttk.LabelFrame(parent, text="自動LLM 先頭設定", padding=10)
        auto_box.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        auto_box.columnconfigure(1, weight=1)

        self._add_entry(auto_box, "auto_llm_model_id", "Model ID", ("auto_llm_prompts", "0", "model_id"), kind="str", row=0, width=20)
        self._add_entry(auto_box, "auto_llm_prompt_id", "Prompt ID", ("auto_llm_prompts", "0", "prompt_id"), kind="str", row=1, width=20)
        self._add_bool(auto_box, "auto_llm_asr_correct", "ASR補正後に送る", ("auto_llm_prompts", "0", "asr_correct"), row=2)
        row += 1

        prompt_box = ttk.LabelFrame(parent, text="LLM Prompt一覧", padding=10)
        prompt_box.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        prompt_box.columnconfigure(2, weight=1)

        ttk.Label(prompt_box, text="Prompt ID").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Label(prompt_box, text="Label").grid(row=0, column=1, sticky="w", padx=(0, 8), pady=(0, 6))
        ttk.Label(prompt_box, text="Template").grid(row=0, column=2, sticky="w", pady=(0, 6))

        prompt_cfg = get_nested(self.cfg, ("llm", "prompts"), {}) or {}
        row_idx = 1
        for prompt_id, prompt_meta in prompt_cfg.items():
            label = ""
            template = ""
            if isinstance(prompt_meta, dict):
                label = str(prompt_meta.get("label", ""))
                template = str(prompt_meta.get("template", ""))
            self._add_prompt_row_widgets(prompt_box, row_idx, str(prompt_id), label, template)
            row_idx += 1

        self.prompt_box = prompt_box

        btn_row = ttk.Frame(prompt_box)
        btn_row.grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(btn_row, text="追加", command=self.add_prompt_row).pack(side="left")

    def _build_hearing_translation_tab(self, parent: ttk.Frame) -> None:
        for i in range(4):
            parent.columnconfigure(i, weight=1)

        row = 0
        self._add_bool(
            parent, "hearing_translation_enabled", "難聴モードの翻訳を有効にする",
            ("hearing_translation", "enabled"), row=row,
        )
        row += 1

        ttk.Label(parent, text="翻訳モデル").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=6
        )
        translation_model_var = tk.StringVar()
        self.vars["hearing_translation_model"] = translation_model_var
        self.field_meta["hearing_translation_model"] = {
            "path": ("hearing_translation", "model"), "kind": "str"
        }
        self.hearing_translation_model_box = ttk.Combobox(
            parent, textvariable=translation_model_var, state="normal", width=32
        )
        self.hearing_translation_model_box.grid(
            row=row, column=1, sticky="we", padx=(0, 16), pady=6
        )
        self.vars["llm_model_default"].trace_add(
            "write", lambda *_args: self._refresh_translation_model_choices()
        )
        self.vars["auto_llm_model_id"].trace_add(
            "write", lambda *_args: self._refresh_translation_model_choices()
        )
        self._add_entry(
            parent, "hearing_translation_timeout", "Timeout",
            ("hearing_translation", "timeout"), kind="float", row=row, col=2, width=12,
        )
        row += 1

        ttk.Label(parent, text="既定の翻訳言語").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=6
        )
        default_language_var = tk.StringVar()
        self.vars["hearing_translation_default_language"] = default_language_var
        self.field_meta["hearing_translation_default_language"] = {
            "path": ("hearing_translation", "default_language"), "kind": "choice"
        }
        self.hearing_translation_default_box = ttk.Combobox(
            parent, textvariable=default_language_var, state="readonly", width=18
        )
        self.hearing_translation_default_box.grid(
            row=row, column=1, sticky="w", padx=(0, 16), pady=6
        )
        row += 1

        ttk.Label(
            parent,
            text=(
                "翻訳モデルは通常の要約モデルとは別に指定できます。"
                "接続先を個別指定していない場合は、LLMタブの設定を使用します。"
                "変更はサーバー再起動後に反映されます。"
            ),
            wraplength=900,
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 10))
        row += 1

        language_box = ttk.LabelFrame(parent, text="翻訳言語一覧", padding=10)
        language_box.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(4, 0))
        language_box.columnconfigure(0, weight=1)
        language_box.columnconfigure(1, weight=2)
        language_box.columnconfigure(2, weight=3)
        ttk.Label(language_box, text="ID").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(language_box, text="表示名").grid(row=0, column=1, sticky="w", padx=(0, 8))
        ttk.Label(language_box, text="LLM向け言語名").grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.translation_language_box = language_box
        self.translation_language_button_row = ttk.Frame(language_box)
        ttk.Button(
            self.translation_language_button_row,
            text="言語を追加",
            command=self.add_translation_language_row,
        ).pack(side="left")

        language_cfg = get_hearing_translation_languages(self.cfg)
        self._set_translation_language_rows(language_cfg)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="サーバーログ", padding=10)
        panel.pack(fill="both", expand=True)

        self.log_text = ScrolledText(panel, height=14, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _build_memo_ai_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        settings = ttk.Frame(parent)
        settings.grid(row=0, column=0, sticky="we", pady=(0, 8))
        ttk.Label(settings, text="既定Prompt ID").pack(side="left", padx=(0, 8))
        self.memo_ai_default_var = tk.StringVar()
        self.memo_ai_default_box = ttk.Combobox(
            settings, textvariable=self.memo_ai_default_var, state="readonly", width=28
        )
        self.memo_ai_default_box.pack(side="left")
        self.memo_ai_default_box.bind("<<ComboboxSelected>>", self._on_memo_ai_default_changed)
        ttk.Label(
            parent,
            text="{text} がASR本文の差し込み位置です。変更は設定保存後、サーバーを再起動すると反映されます。",
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        body = ttk.Frame(parent)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        list_box = ttk.LabelFrame(body, text="メモAIプロンプト一覧", padding=10)
        list_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        list_box.columnconfigure(0, weight=1)
        list_box.rowconfigure(0, weight=1)
        self.memo_ai_list = tk.Listbox(list_box, exportselection=False, height=18)
        self.memo_ai_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_box, orient="vertical", command=self.memo_ai_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.memo_ai_list.configure(yscrollcommand=scrollbar.set)
        self.memo_ai_list.bind("<<ListboxSelect>>", self._on_memo_ai_selected)

        buttons = ttk.Frame(list_box)
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="追加", command=self.add_memo_ai_prompt).pack(side="left")
        ttk.Button(buttons, text="削除", command=self.delete_memo_ai_prompt).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="↑", width=4, command=lambda: self.move_memo_ai_prompt(-1)).pack(side="left", padx=(12, 0))
        ttk.Button(buttons, text="↓", width=4, command=lambda: self.move_memo_ai_prompt(1)).pack(side="left", padx=(4, 0))

        editor = ttk.LabelFrame(body, text="選択中のプロンプト", padding=10)
        editor.grid(row=0, column=1, sticky="nsew")
        editor.columnconfigure(1, weight=1)
        editor.rowconfigure(2, weight=1)
        ttk.Label(editor, text="Prompt ID").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        self.memo_ai_id_var = tk.StringVar()
        self.memo_ai_id_entry = ttk.Entry(editor, textvariable=self.memo_ai_id_var)
        self.memo_ai_id_entry.grid(row=0, column=1, sticky="we", pady=(0, 8))
        ttk.Label(editor, text="表示名").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        self.memo_ai_label_var = tk.StringVar()
        self.memo_ai_label_entry = ttk.Entry(editor, textvariable=self.memo_ai_label_var)
        self.memo_ai_label_entry.grid(row=1, column=1, sticky="we", pady=(0, 8))
        ttk.Label(editor, text="プロンプト本文").grid(row=2, column=0, sticky="nw", padx=(0, 8))
        self.memo_ai_template_text = ScrolledText(editor, height=14, wrap="word")
        self.memo_ai_template_text.grid(row=2, column=1, sticky="nsew")
        self._set_memo_ai_editor_enabled(False)

    def _set_memo_ai_editor_enabled(self, enabled: bool) -> None:
        state = ["!disabled"] if enabled else ["disabled"]
        self.memo_ai_id_entry.state(state)
        self.memo_ai_label_entry.state(state)
        self.memo_ai_template_text.configure(state="normal" if enabled else "disabled")

    def _commit_memo_ai_editor(self) -> None:
        index = self.memo_ai_selected_index
        if index is None or not 0 <= index < len(self.memo_ai_prompts):
            return
        item = self.memo_ai_prompts[index]
        old_id = item["id"]
        new_id = self.memo_ai_id_var.get().strip()
        item["id"] = new_id
        item["label"] = self.memo_ai_label_var.get().strip()
        item["template"] = self.memo_ai_template_text.get("1.0", "end").strip()
        if self.memo_ai_default_var.get() == old_id:
            self.memo_ai_default_var.set(new_id)

    def _show_memo_ai_prompt(self, index: int | None) -> None:
        self.memo_ai_selected_index = index
        enabled = index is not None and 0 <= index < len(self.memo_ai_prompts)
        self._set_memo_ai_editor_enabled(enabled)
        self.memo_ai_id_var.set("")
        self.memo_ai_label_var.set("")
        if not enabled:
            self.memo_ai_template_text.configure(state="normal")
            self.memo_ai_template_text.delete("1.0", "end")
            self.memo_ai_template_text.configure(state="disabled")
            return
        item = self.memo_ai_prompts[index]
        self.memo_ai_id_var.set(item["id"])
        self.memo_ai_label_var.set(item["label"])
        self.memo_ai_template_text.delete("1.0", "end")
        self.memo_ai_template_text.insert("1.0", item["template"])

    def _refresh_memo_ai_list(self, selected_index: int | None = None) -> None:
        self._memo_ai_selection_changing = True
        try:
            self.memo_ai_list.delete(0, "end")
            default_id = self.memo_ai_default_var.get()
            prompt_ids = []
            for item in self.memo_ai_prompts:
                prompt_id = item.get("id", "")
                prompt_ids.append(prompt_id)
                prefix = "[既定] " if prompt_id == default_id else ""
                self.memo_ai_list.insert("end", f"{prefix}{item.get('label') or prompt_id}")
            self.memo_ai_default_box.configure(values=prompt_ids)
            if selected_index is not None and 0 <= selected_index < len(self.memo_ai_prompts):
                self.memo_ai_list.selection_set(selected_index)
                self.memo_ai_list.see(selected_index)
            else:
                selected_index = None
        finally:
            self._memo_ai_selection_changing = False
        self._show_memo_ai_prompt(selected_index)

    def _on_memo_ai_selected(self, _event=None) -> None:
        if self._memo_ai_selection_changing:
            return
        selection = self.memo_ai_list.curselection()
        new_index = int(selection[0]) if selection else None
        self._commit_memo_ai_editor()
        self._refresh_memo_ai_list(new_index)

    def _on_memo_ai_default_changed(self, _event=None) -> None:
        if self._memo_ai_selection_changing:
            return
        self._commit_memo_ai_editor()
        self._refresh_memo_ai_list(self.memo_ai_selected_index)

    def add_memo_ai_prompt(self) -> None:
        self._commit_memo_ai_editor()
        prompt_id = f"memo_prompt_{uuid.uuid4().hex[:12]}"
        self.memo_ai_prompts.append({
            "id": prompt_id,
            "label": "新しいAI処理",
            "template": "以下のメモを処理してください。\n\n{text}",
        })
        if not self.memo_ai_default_var.get():
            self.memo_ai_default_var.set(prompt_id)
        self._refresh_memo_ai_list(len(self.memo_ai_prompts) - 1)
        self.memo_ai_label_entry.focus_set()
        self.memo_ai_label_entry.selection_range(0, "end")

    def delete_memo_ai_prompt(self) -> None:
        index = self.memo_ai_selected_index
        if index is None:
            return
        deleted_id = self.memo_ai_prompts[index]["id"]
        del self.memo_ai_prompts[index]
        if self.memo_ai_default_var.get() == deleted_id:
            self.memo_ai_default_var.set(self.memo_ai_prompts[0]["id"] if self.memo_ai_prompts else "")
        next_index = min(index, len(self.memo_ai_prompts) - 1) if self.memo_ai_prompts else None
        self._refresh_memo_ai_list(next_index)

    def move_memo_ai_prompt(self, offset: int) -> None:
        index = self.memo_ai_selected_index
        if index is None:
            return
        target = index + offset
        if not 0 <= target < len(self.memo_ai_prompts):
            return
        self._commit_memo_ai_editor()
        self.memo_ai_prompts[index], self.memo_ai_prompts[target] = self.memo_ai_prompts[target], self.memo_ai_prompts[index]
        self._refresh_memo_ai_list(target)

    def _build_memo_templates_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        self._add_path_entry(
            parent,
            "memo_templates_path",
            "定型文ファイル",
            ("memo_templates_path",),
            row=0,
            select="file",
        )
        ttk.Label(
            parent,
            text="一覧の順番がメモ画面の表示順になります。設定保存後、メモ画面を再読み込みすると反映されます。",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))

        body = ttk.Frame(parent)
        body.grid(row=2, column=0, columnspan=3, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        list_box = ttk.LabelFrame(body, text="定型文一覧", padding=10)
        list_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        list_box.columnconfigure(0, weight=1)
        list_box.rowconfigure(0, weight=1)
        self.memo_template_list = tk.Listbox(list_box, exportselection=False, height=18)
        self.memo_template_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_box, orient="vertical", command=self.memo_template_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.memo_template_list.configure(yscrollcommand=scrollbar.set)
        self.memo_template_list.bind("<<ListboxSelect>>", self._on_memo_template_selected)

        list_buttons = ttk.Frame(list_box)
        list_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(list_buttons, text="追加", command=self.add_memo_template).pack(side="left")
        ttk.Button(list_buttons, text="削除", command=self.delete_memo_template).pack(side="left", padx=(6, 0))
        ttk.Button(list_buttons, text="↑", width=4, command=lambda: self.move_memo_template(-1)).pack(side="left", padx=(12, 0))
        ttk.Button(list_buttons, text="↓", width=4, command=lambda: self.move_memo_template(1)).pack(side="left", padx=(4, 0))

        editor = ttk.LabelFrame(body, text="選択中の定型文", padding=10)
        editor.grid(row=0, column=1, sticky="nsew")
        editor.columnconfigure(1, weight=1)
        editor.rowconfigure(2, weight=1)
        self.memo_template_enabled_var = tk.BooleanVar(value=True)
        self.memo_template_enabled_check = ttk.Checkbutton(
            editor, text="メモ画面に表示する", variable=self.memo_template_enabled_var
        )
        self.memo_template_enabled_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(editor, text="表示名").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
        self.memo_template_label_var = tk.StringVar()
        self.memo_template_label_entry = ttk.Entry(editor, textvariable=self.memo_template_label_var)
        self.memo_template_label_entry.grid(row=1, column=1, sticky="we", pady=(0, 8))
        ttk.Label(editor, text="本文").grid(row=2, column=0, sticky="nw", padx=(0, 8))
        self.memo_template_text = ScrolledText(editor, height=14, wrap="word")
        self.memo_template_text.grid(row=2, column=1, sticky="nsew")
        self._set_memo_template_editor_enabled(False)

    def _set_memo_template_editor_enabled(self, enabled: bool) -> None:
        self.memo_template_enabled_check.state(["!disabled"] if enabled else ["disabled"])
        self.memo_template_label_entry.state(["!disabled"] if enabled else ["disabled"])
        self.memo_template_text.configure(state="normal" if enabled else "disabled")

    def _commit_memo_template_editor(self) -> None:
        index = self.memo_template_selected_index
        if index is None or not 0 <= index < len(self.memo_templates):
            return
        self.memo_templates[index]["label"] = self.memo_template_label_var.get().strip()
        self.memo_templates[index]["text"] = self.memo_template_text.get("1.0", "end").strip()
        self.memo_templates[index]["enabled"] = bool(self.memo_template_enabled_var.get())

    def _show_memo_template(self, index: int | None) -> None:
        self.memo_template_selected_index = index
        enabled = index is not None and 0 <= index < len(self.memo_templates)
        self._set_memo_template_editor_enabled(enabled)
        self.memo_template_label_var.set("")
        if not enabled:
            self.memo_template_text.configure(state="normal")
            self.memo_template_text.delete("1.0", "end")
            self.memo_template_text.configure(state="disabled")
            return
        item = self.memo_templates[index]
        self.memo_template_enabled_var.set(bool(item.get("enabled", True)))
        self.memo_template_label_var.set(str(item.get("label", "")))
        self.memo_template_text.delete("1.0", "end")
        self.memo_template_text.insert("1.0", str(item.get("text", "")))

    def _refresh_memo_template_list(self, selected_index: int | None = None) -> None:
        self._memo_template_selection_changing = True
        try:
            self.memo_template_list.delete(0, "end")
            for item in self.memo_templates:
                prefix = "" if item.get("enabled", True) else "[非表示] "
                self.memo_template_list.insert("end", f"{prefix}{item.get('label') or item.get('id')}")
            if selected_index is not None and 0 <= selected_index < len(self.memo_templates):
                self.memo_template_list.selection_set(selected_index)
                self.memo_template_list.see(selected_index)
            else:
                selected_index = None
        finally:
            self._memo_template_selection_changing = False
        self._show_memo_template(selected_index)

    def _on_memo_template_selected(self, _event=None) -> None:
        if self._memo_template_selection_changing:
            return
        selection = self.memo_template_list.curselection()
        new_index = int(selection[0]) if selection else None
        self._commit_memo_template_editor()
        self._refresh_memo_template_list(new_index)

    def add_memo_template(self) -> None:
        self._commit_memo_template_editor()
        self.memo_templates.append({
            "id": f"template_{uuid.uuid4().hex[:12]}",
            "label": "新しい定型文",
            "text": "",
            "enabled": True,
        })
        self._refresh_memo_template_list(len(self.memo_templates) - 1)
        self.memo_template_label_entry.focus_set()
        self.memo_template_label_entry.selection_range(0, "end")

    def delete_memo_template(self) -> None:
        index = self.memo_template_selected_index
        if index is None:
            return
        del self.memo_templates[index]
        next_index = min(index, len(self.memo_templates) - 1) if self.memo_templates else None
        self._refresh_memo_template_list(next_index)

    def move_memo_template(self, offset: int) -> None:
        index = self.memo_template_selected_index
        if index is None:
            return
        target = index + offset
        if not 0 <= target < len(self.memo_templates):
            return
        self._commit_memo_template_editor()
        self.memo_templates[index], self.memo_templates[target] = self.memo_templates[target], self.memo_templates[index]
        self._refresh_memo_template_list(target)

    def _add_entry(self, parent, name: str, label: str, path: tuple[str, ...], kind: str, row: int, col: int = 0, width: int = 24, show: str = "") -> None:
        var = tk.StringVar()
        self.vars[name] = var
        self.field_meta[name] = {"path": path, "kind": kind}

        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(parent, textvariable=var, width=width, show=show).grid(row=row, column=col + 1, sticky="we", padx=(0, 16), pady=6)

    def _add_path_entry(self, parent, name: str, label: str, path: tuple[str, ...], row: int, select: str) -> None:
        var = tk.StringVar()
        self.vars[name] = var
        self.field_meta[name] = {"path": path, "kind": "path", "select": select}

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="we", padx=(0, 8), pady=6)
        ttk.Button(parent, text="参照", command=lambda: self._browse_named_path(name), width=8).grid(row=row, column=2, sticky="w", pady=6)

    def _add_choice(self, parent, name: str, label: str, path: tuple[str, ...], choices: list[str], row: int, col: int = 0) -> ttk.Combobox:
        var = tk.StringVar()
        self.vars[name] = var
        self.field_meta[name] = {"path": path, "kind": "choice"}

        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=6)
        box = ttk.Combobox(parent, textvariable=var, values=choices, state="readonly", width=18)
        box.grid(row=row, column=col + 1, sticky="w", padx=(0, 16), pady=6)
        return box

    def _add_bool(self, parent, name: str, label: str, path: tuple[str, ...], row: int, col: int = 0) -> None:
        var = tk.BooleanVar()
        self.vars[name] = var
        self.field_meta[name] = {"path": path, "kind": "bool"}
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=col, columnspan=2, sticky="w", padx=(0, 16), pady=6)

    def _add_text(self, parent, name: str, label: str, path: tuple[str, ...], row: int, height: int = 4) -> None:
        text = ScrolledText(parent, height=height, wrap="word")
        self.vars[name] = text
        self.field_meta[name] = {"path": path, "kind": "text"}

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=6)
        text.grid(row=row, column=1, columnspan=3, sticky="nsew", pady=6)

    def _browse_named_path(self, name: str) -> None:
        meta = self.field_meta[name]
        self._browse_into_var(self.vars[name], select=meta["select"])

    def _browse_into_var(self, var, select: str) -> None:
        if select == "dir":
            path = filedialog.askdirectory(parent=self.root)
        else:
            path = filedialog.askopenfilename(parent=self.root)
        if path:
            var.set(normalize_path_text(path))

    def add_model_row(self) -> None:
        row_idx = len(self.model_rows) + 1
        id_var = tk.StringVar()
        path_var = tk.StringVar()
        self.model_rows.append((id_var, path_var))

        ttk.Entry(self.model_box, textvariable=id_var, width=18).grid(row=row_idx, column=0, sticky="we", padx=(0, 8), pady=4)
        ttk.Entry(self.model_box, textvariable=path_var).grid(row=row_idx, column=1, sticky="we", padx=(0, 8), pady=4)
        ttk.Button(
            self.model_box,
            text="参照",
            command=lambda v=path_var: self._browse_into_var(v, select="dir"),
            width=8,
        ).grid(row=row_idx, column=2, sticky="w", pady=4)

    def _add_prompt_row_widgets(self, parent, row_idx: int, prompt_id: str = "", label: str = "", template: str = "") -> None:
        id_var = tk.StringVar(value=prompt_id)
        label_var = tk.StringVar(value=label)
        text = ScrolledText(parent, height=6, wrap="word")
        text.insert("1.0", template)
        self.prompt_rows.append((id_var, label_var, text))

        ttk.Entry(parent, textvariable=id_var, width=18).grid(row=row_idx, column=0, sticky="we", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=label_var, width=20).grid(row=row_idx, column=1, sticky="we", padx=(0, 8), pady=4)
        text.grid(row=row_idx, column=2, sticky="nsew", pady=4)

    def add_prompt_row(self) -> None:
        row_idx = len(self.prompt_rows) + 1
        self._add_prompt_row_widgets(self.prompt_box, row_idx)

    def _add_translation_language_row_widgets(
        self, language_id: str = "", label: str = "", name: str = ""
    ) -> None:
        id_var = tk.StringVar(value=language_id)
        label_var = tk.StringVar(value=label)
        name_var = tk.StringVar(value=name)
        id_entry = ttk.Entry(self.translation_language_box, textvariable=id_var, width=16)
        label_entry = ttk.Entry(self.translation_language_box, textvariable=label_var, width=22)
        name_entry = ttk.Entry(self.translation_language_box, textvariable=name_var, width=32)
        delete_button = ttk.Button(self.translation_language_box, text="削除", width=8)
        item: dict[str, object] = {
            "id_var": id_var,
            "label_var": label_var,
            "name_var": name_var,
            "widgets": (id_entry, label_entry, name_entry, delete_button),
        }
        delete_button.configure(command=lambda current=item: self.delete_translation_language_row(current))
        id_entry.bind("<FocusOut>", self._on_translation_language_id_changed)
        id_entry.bind("<Return>", self._on_translation_language_id_changed)
        self.translation_language_rows.append(item)
        self._layout_translation_language_rows()

    def _layout_translation_language_rows(self) -> None:
        for index, item in enumerate(self.translation_language_rows, start=1):
            widgets = item["widgets"]
            widgets[0].grid(row=index, column=0, sticky="we", padx=(0, 8), pady=4)
            widgets[1].grid(row=index, column=1, sticky="we", padx=(0, 8), pady=4)
            widgets[2].grid(row=index, column=2, sticky="we", padx=(0, 8), pady=4)
            widgets[3].grid(row=index, column=3, sticky="w", pady=4)
        self.translation_language_button_row.grid(
            row=len(self.translation_language_rows) + 1,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(8, 0),
        )

    def _set_translation_language_rows(self, languages) -> None:
        for item in self.translation_language_rows:
            for widget in item["widgets"]:
                widget.destroy()
        self.translation_language_rows.clear()
        for language in languages:
            if not isinstance(language, dict):
                continue
            self._add_translation_language_row_widgets(
                str(language.get("id") or ""),
                str(language.get("label") or ""),
                str(language.get("name") or ""),
            )
        self._layout_translation_language_rows()
        self._refresh_translation_default_choices()

    def add_translation_language_row(self) -> None:
        self._add_translation_language_row_widgets()
        self.translation_language_rows[-1]["widgets"][0].focus_set()

    def delete_translation_language_row(self, item: dict[str, object]) -> None:
        if item not in self.translation_language_rows:
            return
        for widget in item["widgets"]:
            widget.destroy()
        self.translation_language_rows.remove(item)
        self._layout_translation_language_rows()
        self._refresh_translation_default_choices()

    def _on_translation_language_id_changed(self, _event=None) -> None:
        self._refresh_translation_default_choices()

    def _refresh_translation_default_choices(self) -> None:
        language_ids = [
            item["id_var"].get().strip()
            for item in self.translation_language_rows
            if item["id_var"].get().strip()
        ]
        self.hearing_translation_default_box.configure(values=language_ids)
        current = self.vars["hearing_translation_default_language"].get().strip()
        if language_ids and current not in language_ids:
            self.vars["hearing_translation_default_language"].set(language_ids[0])
        elif not language_ids:
            self.vars["hearing_translation_default_language"].set("")

    def _refresh_translation_model_choices(self) -> None:
        candidates = [
            self.vars["hearing_translation_model"].get(),
            self.vars["llm_model_default"].get(),
            self.vars["auto_llm_model_id"].get(),
            get_nested(self.cfg, ("hearing_translation", "model"), ""),
            get_nested(self.cfg, ("llm", "model_default"), ""),
        ]
        candidates.extend(
            item.get("model_id", "")
            for item in (self.cfg.get("auto_llm_prompts") or [])
            if isinstance(item, dict)
        )
        models = list(dict.fromkeys(str(value).strip() for value in candidates if str(value).strip()))
        self.hearing_translation_model_box.configure(values=models)

    def _load_form_from_config(self) -> None:
        if hasattr(self, "whisper_settings_box"):
            self._set_widget_tree_enabled(self.whisper_settings_box, True)
            self._set_widget_tree_enabled(self.qwen_settings_box, True)
            self._set_widget_tree_enabled(self.vibevoice_settings_box, True)
        self.cfg = load_config()
        for name, field in self.field_meta.items():
            path = field["path"]
            kind = field["kind"]

            if path[:2] == ("auto_llm_prompts", "0"):
                prompt_items = self.cfg.get("auto_llm_prompts") or []
                base = prompt_items[0] if prompt_items else {}
                key = path[2]
                value = base.get(key, False if kind == "bool" else "")
            else:
                fallback = FIELD_DEFAULTS.get(name, False if kind == "bool" else "")
                if name in {"qwen_context", "vibevoice_context"}:
                    fallback = get_nested(self.cfg, ("asr", "initial_prompt"), fallback)
                if name == "hearing_translation_model":
                    fallback = get_nested(self.cfg, ("llm", "model_default"), fallback)
                value = get_nested(self.cfg, path, fallback)
                if name == "vibevoice_hotwords" and isinstance(value, list):
                    value = "\n".join(str(word) for word in value)
                if name in {"llm_server", "llm_port", "llm_use_https"} and value in (None, "", False):
                    old_url = str(
                        get_nested(self.cfg, ("llm", "base_url"), "")
                        or get_nested(self.cfg, ("llm", "host"), "")
                        or ""
                    ).strip()
                    if old_url:
                        parsed = urlparse(old_url if "://" in old_url else f"http://{old_url}")
                        if name == "llm_server":
                            value = parsed.hostname or "127.0.0.1"
                        elif name == "llm_port":
                            value = parsed.port or (443 if parsed.scheme == "https" else 11434)
                        else:
                            value = parsed.scheme == "https"
                    elif name == "llm_server":
                        value = "127.0.0.1"
                    elif name == "llm_port":
                        value = 11434

            widget = self.vars[name]
            if kind == "bool":
                widget.set(bool(value))
            elif kind == "text":
                widget.delete("1.0", "end")
                widget.insert("1.0", "" if value is None else str(value))
            else:
                widget.set("" if value is None else str(value))

        model_cfg = get_nested(self.cfg, ("asr", "models"), {}) or {}
        for idx, (model_id, model_path) in enumerate(model_cfg.items()):
            if idx >= len(self.model_rows):
                self.add_model_row()
            self.model_rows[idx][0].set(str(model_id))
            self.model_rows[idx][1].set(str(model_path))
        for idx in range(len(model_cfg), len(self.model_rows)):
            self.model_rows[idx][0].set("")
            self.model_rows[idx][1].set("")

        prompt_cfg = get_nested(self.cfg, ("llm", "prompts"), {}) or {}
        prompt_items = list(prompt_cfg.items())
        for idx, (prompt_id, prompt_meta) in enumerate(prompt_items):
            if idx >= len(self.prompt_rows):
                self.add_prompt_row()
            label = ""
            template = ""
            if isinstance(prompt_meta, dict):
                label = str(prompt_meta.get("label", ""))
                template = str(prompt_meta.get("template", ""))
            self.prompt_rows[idx][0].set(str(prompt_id))
            self.prompt_rows[idx][1].set(label)
            self.prompt_rows[idx][2].delete("1.0", "end")
            self.prompt_rows[idx][2].insert("1.0", template)
        for idx in range(len(prompt_items), len(self.prompt_rows)):
            self.prompt_rows[idx][0].set("")
            self.prompt_rows[idx][1].set("")
            self.prompt_rows[idx][2].delete("1.0", "end")

        language_cfg = get_hearing_translation_languages(self.cfg)
        self._set_translation_language_rows(language_cfg)
        self._refresh_translation_model_choices()
        self._refresh_translation_default_choices()

        try:
            memo_ai_prompts, memo_ai_default_id = normalize_memo_ai_settings(
                self.cfg, load_config_sample()
            )
            self.memo_ai_prompts = [
                {"id": prompt_id, "label": meta["label"], "template": meta["template"]}
                for prompt_id, meta in memo_ai_prompts.items()
            ]
            self.memo_ai_default_var.set(memo_ai_default_id)
            self._refresh_memo_ai_list(0 if self.memo_ai_prompts else None)
        except Exception as e:
            self.memo_ai_prompts = []
            self.memo_ai_default_var.set("")
            self._refresh_memo_ai_list()
            messagebox.showerror("メモAI設定読込エラー", str(e), parent=self.root)

        try:
            templates_path = ensure_memo_templates_file(self.cfg)
            templates_data = load_memo_templates(templates_path)
            self.memo_templates = [dict(item) for item in templates_data["templates"]]
            selected_index = 0 if self.memo_templates else None
            self._refresh_memo_template_list(selected_index)
        except Exception as e:
            self.memo_templates = []
            self._refresh_memo_template_list()
            messagebox.showerror("定型文読込エラー", str(e), parent=self.root)

    def reload_form(self) -> None:
        self._load_form_from_config()
        self._update_asr_provider_ui()
        self._append_log(f"[launcher] config reloaded: {CONFIG_PATH.name}")
        self.refresh_qwen_status()

    @staticmethod
    def _set_widget_tree_enabled(widget: tk.Misc, enabled: bool) -> None:
        for child in widget.winfo_children():
            if isinstance(child, tk.Text):
                child.configure(state="normal" if enabled else "disabled")
            elif isinstance(child, ttk.Widget):
                child.state(["!disabled"] if enabled else ["disabled"])
            LauncherApp._set_widget_tree_enabled(child, enabled)

    def _update_asr_provider_ui(self) -> None:
        provider = self.vars["asr_provider"].get().strip().lower()
        qwen_selected = provider in {"qwen", "qwen3-asr"}
        vibevoice_selected = provider in {"vibevoice", "vibevoice-asr"}
        self._set_widget_tree_enabled(self.whisper_settings_box, not qwen_selected and not vibevoice_selected)
        self._set_widget_tree_enabled(self.qwen_settings_box, qwen_selected)
        self._set_widget_tree_enabled(self.vibevoice_settings_box, vibevoice_selected)

    def _on_asr_provider_changed(self, _event=None) -> None:
        self._update_asr_provider_ui()
        provider = self.vars["asr_provider"].get().strip().lower()
        if provider != "whisper":
            self.refresh_qwen_status()

    def refresh_qwen_status(self) -> None:
        if self._qwen_status_checking:
            return
        provider = self.vars["asr_provider"].get().strip().lower()
        vibevoice = provider in {"vibevoice", "vibevoice-asr"}
        base_url = self.vars["vibevoice_base_url" if vibevoice else "qwen_base_url"].get().strip()
        if not base_url:
            self._apply_qwen_status(QwenReadyStatus(False, False, None, {}, "API URLが空です。"), vibevoice)
            return

        self._qwen_status_checking = True
        status_var = self.vibevoice_status_var if vibevoice else self.qwen_status_var
        indicator = self.vibevoice_status_indicator if vibevoice else self.qwen_status_indicator
        indicator_oval = self.vibevoice_status_indicator_oval if vibevoice else self.qwen_status_indicator_oval
        status_var.set("確認中...")
        indicator.itemconfigure(indicator_oval, fill="#eab308")
        threading.Thread(
            target=self._fetch_qwen_status, args=(base_url, vibevoice), daemon=True
        ).start()

    def _fetch_qwen_status(self, base_url: str, vibevoice: bool = False) -> None:
        status = fetch_asr_ready_status(base_url)
        self.action_queue.put(("qwen_status", (status, vibevoice)))

    def _periodic_qwen_status_refresh(self) -> None:
        provider = self.vars["asr_provider"].get().strip().lower()
        if provider != "whisper":
            self.refresh_qwen_status()
        self.root.after(5000, self._periodic_qwen_status_refresh)

    def _apply_qwen_status(self, status: QwenReadyStatus, vibevoice: bool = False) -> None:
        self._qwen_status_checking = False
        status_var = self.vibevoice_status_var if vibevoice else self.qwen_status_var
        details_var = self.vibevoice_status_details_var if vibevoice else self.qwen_status_details_var
        indicator = self.vibevoice_status_indicator if vibevoice else self.qwen_status_indicator
        indicator_oval = self.vibevoice_status_indicator_oval if vibevoice else self.qwen_status_indicator_oval
        payload = status.payload
        if status.ready:
            model = str(payload.get("model") or "-")
            device = str(payload.get("device") or "-")
            queue_depth = payload.get("queue_depth", "-")
            queue_capacity = payload.get("queue_capacity", "-")
            status_var.set(
                f"Ready / model {model} / {device} / queue {queue_depth}/{queue_capacity}"
            )
            indicator.itemconfigure(indicator_oval, fill="#16a34a")
            details = [
                f"model_id: {payload.get('model_id') or '-'}",
                f"engine: {payload.get('engine') or '-'} / {payload.get('backend') or '-'}",
                f"app: {payload.get('app_version') or '-'} / schema: {payload.get('schema_version') or '-'}",
            ]
            language = self.vars["vibevoice_language"].get().strip().lower() if vibevoice else ""
            if vibevoice and str(payload.get("backend") or "").lower() == "vibeasr-cpp" and language in {"ja", "japanese", "日本語"}:
                details.append("警告: BitNetの明示対応言語に日本語は含まれません。Transformers版を使用してください。")
            details_var.set("   ".join(details))
            return

        if status.reachable:
            http_text = f"HTTP {status.http_status}" if status.http_status is not None else "応答あり"
            status_var.set(f"未準備 / {http_text}")
            color = "#eab308"
        else:
            status_var.set("接続不可")
            color = "#dc2626"
        indicator.itemconfigure(indicator_oval, fill=color)
        details_var.set(status.error or "Ready応答を取得できませんでした。")

    def save_form(self) -> bool:
        cfg = load_config()
        try:
            self._commit_memo_ai_editor()
            self._commit_memo_template_editor()
            self._refresh_translation_default_choices()
            for name, field in self.field_meta.items():
                path = field["path"]
                kind = field["kind"]
                widget = self.vars[name]

                if kind == "bool":
                    value = bool(widget.get())
                elif kind == "int":
                    value = int(widget.get().strip())
                elif kind == "float":
                    value = float(widget.get().strip())
                elif kind == "text":
                    value = widget.get("1.0", "end").strip()
                else:
                    value = widget.get().strip()

                if kind == "path" or path in PATH_KEYS:
                    value = normalize_path_text(str(value))

                if path[:2] == ("auto_llm_prompts", "0"):
                    items = cfg.get("auto_llm_prompts")
                    if not isinstance(items, list) or not items:
                        items = [{}]
                        cfg["auto_llm_prompts"] = items
                    items[0][path[2]] = value
                else:
                    set_nested(cfg, path, value)

            models: dict[str, str] = {}
            for id_var, path_var in self.model_rows:
                model_id = id_var.get().strip()
                model_path = normalize_path_text(path_var.get())
                if not model_id and not model_path:
                    continue
                if not model_id or not model_path:
                    raise ValueError("ASRモデルのIDとパスは両方入力してください。")
                models[model_id] = model_path
            set_nested(cfg, MODEL_PATH_PREFIX, models)

            qwen_cfg = get_nested(cfg, ("asr", "qwen"), {})
            if isinstance(qwen_cfg, dict):
                qwen_cfg.pop("python_executable", None)
                qwen_cfg.pop("server_script", None)

            llm_cfg = get_nested(cfg, ("llm",), {})
            if isinstance(llm_cfg, dict):
                llm_cfg.pop("host", None)
                llm_cfg.pop("base_url", None)
                llm_server = str(llm_cfg.get("server") or "").strip()
                llm_port = int(llm_cfg.get("port") or 0)
                if not llm_server:
                    raise ValueError("llm.server は必須です。")
                if "://" in llm_server or "/" in llm_server:
                    raise ValueError("llm.server にはURLではなくIPアドレスまたはホスト名だけを入力してください。")
                if not 1 <= llm_port <= 65535:
                    raise ValueError("llm.port は1～65535で指定してください。")

            translation_enabled = bool(
                get_nested(cfg, ("hearing_translation", "enabled"), True)
            )
            translation_model = str(
                get_nested(cfg, ("hearing_translation", "model"), "")
            ).strip()
            translation_timeout = float(
                get_nested(cfg, ("hearing_translation", "timeout"), 0)
            )
            if translation_enabled and not translation_model:
                raise ValueError("難聴翻訳を有効にする場合は翻訳モデルを選択してください。")
            if translation_timeout <= 0:
                raise ValueError("難聴翻訳のTimeoutは0より大きい値を指定してください。")
            translation_rows = [
                (
                    item["id_var"].get(),
                    item["label_var"].get(),
                    item["name_var"].get(),
                )
                for item in self.translation_language_rows
            ]
            translation_languages = validate_hearing_translation_languages(
                translation_rows,
                str(get_nested(cfg, ("hearing_translation", "default_language"), "")),
            )
            set_nested(cfg, ("hearing_translation", "languages"), translation_languages)

            if not get_nested(cfg, ("asr", "model_id"), "").strip():
                raise ValueError("asr.model_id が空です。")
            if get_nested(cfg, ("asr", "model_id")) not in models:
                raise ValueError("asr.model_id が asr.models に存在しません。")

            prompts: dict[str, dict[str, str]] = {}
            for prompt_id_var, label_var, template_widget in self.prompt_rows:
                prompt_id = prompt_id_var.get().strip()
                label = label_var.get().strip()
                template = template_widget.get("1.0", "end").strip()
                if not prompt_id and not label and not template:
                    continue
                if not prompt_id:
                    raise ValueError("LLM prompt の Prompt ID は必須です。")
                if not label:
                    raise ValueError(f"LLM prompt '{prompt_id}' の label は必須です。")
                if not template:
                    raise ValueError(f"LLM prompt '{prompt_id}' の template は必須です。")
                prompts[prompt_id] = {"label": label, "template": template}
            set_nested(cfg, ("llm", "prompts"), prompts)

            default_prompt_id = get_nested(cfg, ("llm", "default_prompt_id"), "").strip()
            if not default_prompt_id:
                raise ValueError("llm.default_prompt_id は必須です。")
            if default_prompt_id not in prompts:
                raise ValueError("llm.default_prompt_id が llm.prompts に存在しません。")

            memo_ai_prompts: dict[str, dict[str, str]] = {}
            for item in self.memo_ai_prompts:
                prompt_id = str(item.get("id") or "").strip()
                if not prompt_id:
                    raise ValueError("メモAI prompt の Prompt ID は必須です。")
                if prompt_id in memo_ai_prompts:
                    raise ValueError(f"メモAI prompt の Prompt ID が重複しています: {prompt_id}")
                memo_ai_prompts[prompt_id] = {
                    "label": str(item.get("label") or "").strip(),
                    "template": str(item.get("template") or "").strip(),
                }
            set_nested(cfg, ("llm", "memo_prompts"), memo_ai_prompts)
            set_nested(cfg, ("llm", "memo_default_prompt_id"), self.memo_ai_default_var.get().strip())
            normalize_memo_ai_settings(cfg)

            templates_path = get_memo_templates_path(cfg)
            save_memo_templates(templates_path, {
                "version": 1,
                "templates": self.memo_templates,
            })
            save_config(cfg)
            self.cfg = cfg
            self._refresh_memo_ai_list(self.memo_ai_selected_index)
            self._refresh_memo_template_list(self.memo_template_selected_index)
            self._append_log(f"[launcher] memo templates saved: {templates_path.name}")
            self._append_log(f"[launcher] config saved: {CONFIG_PATH.name}")
            return True
        except Exception as e:
            messagebox.showerror("設定保存エラー", str(e), parent=self.root)
            return False

    def _start_speechsummarizer_process(self) -> None:
        if self._cancel_start.is_set():
            self._starting = False
            self._update_status()
            return
        cmd = [sys.executable, "--server"] if getattr(sys, "frozen", False) else [sys.executable, str(Path(__file__).resolve()), "--server"]
        try:
            popen_encoding = locale.getpreferredencoding(False) or "utf-8"
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=popen_encoding,
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            messagebox.showerror("起動エラー", str(e), parent=self.root)
            self.proc = None
            self._stop_owned_qwen()
            self._starting = False
            self._update_status()
            return

        threading.Thread(
            target=self._read_process_output, args=(self.proc, "server"), daemon=True
        ).start()
        self._append_log(f"[launcher] server start: {' '.join(cmd)}")
        self._starting = False
        self._update_status()

    def start_server(self) -> None:
        if self._starting or (self.proc and self.proc.poll() is None):
            return
        if not self.save_form():
            return

        self._cancel_start.clear()
        provider = str(get_nested(self.cfg, ("asr", "provider"), "whisper")).strip().lower()
        provider_key = "vibevoice" if provider in {"vibevoice", "vibevoice-asr"} else "qwen"
        managed = bool(get_nested(self.cfg, ("asr", provider_key, "managed_by_launcher"), False))
        if provider == "whisper" or not managed:
            self._start_speechsummarizer_process()
            return

        self._starting = True
        self._update_status()
        threading.Thread(target=self._start_managed_qwen_then_server, daemon=True).start()

    def _start_managed_qwen_then_server(self, start_speechsummarizer: bool = True) -> None:
        provider = str(get_nested(self.cfg, ("asr", "provider"), "qwen3-asr")).strip().lower()
        is_vibevoice = provider in {"vibevoice", "vibevoice-asr"}
        provider_key = "vibevoice" if is_vibevoice else "qwen"
        provider_label = "VibeVoiceASR" if is_vibevoice else "QwenASR"
        remote_cfg = get_nested(self.cfg, ("asr", provider_key), {}) or {}
        default_url = "http://127.0.0.1:8020" if is_vibevoice else "http://127.0.0.1:8010"
        base_url = str(remote_cfg.get("base_url", default_url)).rstrip("/")
        try:
            model_alias = str(remote_cfg.get("model_alias", "7b" if is_vibevoice else "1.7b")).strip().lower()
            if not is_vibevoice and model_alias not in {"0.6b", "1.7b"}:
                raise ValueError("Qwen model_alias must be 0.6b or 1.7b")
            if not model_alias:
                raise ValueError("VibeVoice model_alias must not be empty")

            ready_status = fetch_asr_ready_status(base_url)
            if ready_status.ready:
                running_model = str(ready_status.payload.get("model") or "").strip().lower()
                if running_model != model_alias:
                    raise RuntimeError(
                        f"{provider_label} is already running with model {running_model or 'unknown'}; "
                        f"selected model is {model_alias}. Stop that process before starting."
                    )
                self.log_queue.put(f"[launcher] {provider_label} is already ready; using external process")
                action = "start_speechsummarizer" if start_speechsummarizer else "qwen_restart_complete"
                self.action_queue.put((action, None))
                return

            config_path = resolve_launcher_path(str(remote_cfg.get("config_path", "")), APP_DIR)
            timeout_sec = float(remote_cfg.get("startup_timeout_sec", 180.0 if is_vibevoice else 90.0))
            if timeout_sec <= 0:
                raise ValueError(f"{provider_label} startup_timeout_sec must be greater than zero")
            required_paths = [(f"{provider_label} config.json", config_path)]
            if is_vibevoice:
                python_path = resolve_launcher_path(str(remote_cfg.get("python_executable", "")), APP_DIR)
                server_script = resolve_launcher_path(str(remote_cfg.get("server_script", "")), APP_DIR)
                required_paths.extend([("VibeVoice Python", python_path), ("VibeVoice server.py", server_script)])
                cmd = build_vibevoice_server_command(python_path, server_script, config_path, model_alias)
                working_dir = server_script.parent
            else:
                executable_path = resolve_launcher_path(str(remote_cfg.get("executable_path", "")), APP_DIR)
                required_paths.append(("QwenASR-Server.exe", executable_path))
                cmd = build_qwen_server_command(executable_path, config_path, model_alias)
                working_dir = executable_path.parent
            for label, path in required_paths:
                if not path.is_file():
                    raise FileNotFoundError(f"{label} not found: {path}")

            popen_encoding = locale.getpreferredencoding(False) or "utf-8"
            self.qwen_proc = subprocess.Popen(
                cmd,
                cwd=str(working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=popen_encoding,
                errors="replace",
                bufsize=1,
            )
            threading.Thread(
                target=self._read_process_output, args=(self.qwen_proc, provider_key), daemon=True
            ).start()
            self.log_queue.put(f"[launcher] {provider_label} start: {' '.join(cmd)}")

            deadline = time.monotonic() + timeout_sec
            while time.monotonic() < deadline and not self._cancel_start.is_set():
                if self.qwen_proc.poll() is not None:
                    raise RuntimeError(f"{provider_label} exited before ready (rc={self.qwen_proc.returncode})")
                ready_status = fetch_asr_ready_status(base_url)
                if self._cancel_start.is_set():
                    return
                if ready_status.ready:
                    running_model = str(ready_status.payload.get("model") or "").strip().lower()
                    if running_model != model_alias:
                        raise RuntimeError(
                            f"{provider_label} ready model is {running_model or 'unknown'}, expected {model_alias}"
                        )
                    self.log_queue.put(f"[launcher] {provider_label} ready: model={model_alias}")
                    action = "start_speechsummarizer" if start_speechsummarizer else "qwen_restart_complete"
                    self.action_queue.put((action, None))
                    return
                self._cancel_start.wait(0.25)
            if self._cancel_start.is_set():
                return
            raise TimeoutError(f"{provider_label} did not become ready within {timeout_sec:g} seconds")
        except Exception as exc:
            action = "managed_start_failed" if start_speechsummarizer else "qwen_restart_failed"
            self.action_queue.put((action, str(exc)))

    def restart_managed_qwen(self) -> None:
        if self._qwen_restarting or self._starting:
            return
        if not self.save_form():
            return
        provider = str(get_nested(self.cfg, ("asr", "provider"), "whisper")).strip().lower()
        is_vibevoice = provider in {"vibevoice", "vibevoice-asr"}
        provider_key = "vibevoice" if is_vibevoice else "qwen"
        provider_label = "VibeVoiceASR" if is_vibevoice else "QwenASR"
        managed = bool(get_nested(self.cfg, ("asr", provider_key, "managed_by_launcher"), False))
        if provider == "whisper" or not managed:
            messagebox.showinfo(f"{provider_label}再起動", f"{provider_label}のランチャー管理を有効にしてください。", parent=self.root)
            return
        if self.qwen_proc is None or self.qwen_proc.poll() is not None:
            messagebox.showinfo(
                f"{provider_label}再起動",
                f"ランチャーが起動した{provider_label}だけを再起動できます。\n外部起動したプロセスは停止しません。",
                parent=self.root,
            )
            return
        self._cancel_start.clear()
        self._qwen_restarting = True
        self._update_status()
        threading.Thread(target=self._restart_owned_qwen, daemon=True).start()

    def _restart_owned_qwen(self) -> None:
        proc = self.qwen_proc
        self._terminate_process(proc)
        if self.qwen_proc is proc:
            self.qwen_proc = None
        self.log_queue.put("[launcher] managed ASR API stopped for restart")
        if self._cancel_start.is_set():
            return
        self._start_managed_qwen_then_server(start_speechsummarizer=False)

    def _qwen_restart_finished(self, error: str | None = None) -> None:
        self._qwen_restarting = False
        if error:
            self._stop_owned_qwen()
            self._append_log(f"[launcher] ASR API restart failed: {error}")
            messagebox.showerror("ASR API再起動エラー", error, parent=self.root)
        else:
            self._append_log("[launcher] ASR API restart complete")
            self.refresh_qwen_status()
        self._update_status()

    def _managed_start_failed(self, error: str) -> None:
        self._stop_owned_qwen()
        self._starting = False
        self._append_log(f"[launcher] ASR API start failed: {error}")
        messagebox.showerror("ASR API起動エラー", error, parent=self.root)
        self._update_status()

    def _maybe_auto_start_server(self) -> None:
        if self._auto_start_attempted:
            return
        self._auto_start_attempted = True

        auto_start = bool(get_nested(self.cfg, ("windows_launcher", "auto_start_server"), False))
        if not auto_start:
            return
        if self.proc and self.proc.poll() is None:
            return

        self._append_log("[launcher] auto start enabled: starting server")
        self.start_server()

    @staticmethod
    def _terminate_process(proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _stop_owned_qwen(self) -> None:
        if self.qwen_proc is None:
            return
        self._terminate_process(self.qwen_proc)
        self.qwen_proc = None
        self._append_log("[launcher] managed ASR API stopped")

    def stop_server(self) -> None:
        self._cancel_start.set()
        self._terminate_process(self.proc)
        self.proc = None
        self._stop_owned_qwen()
        self._starting = False
        self._qwen_restarting = False
        self._append_log("[launcher] server stopped")
        self._update_status()

    def open_browser(self) -> None:
        cfg = load_config()
        port = int(cfg.get("port", 8000))
        ssl_enabled = bool(get_nested(cfg, ("ssl", "enabled"), False))
        scheme = "https" if ssl_enabled else "http"
        webbrowser.open(f"{scheme}://127.0.0.1:{port}")

    def _read_process_output(self, proc: subprocess.Popen, label: str) -> None:
        if not proc.stdout:
            return
        for line in proc.stdout:
            self.log_queue.put(f"[{label}] {line.rstrip()}")
        rc = proc.poll()
        self.log_queue.put(f"[launcher] {label} exited rc={rc}")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                action, value = self.action_queue.get_nowait()
                if action == "start_speechsummarizer":
                    self.refresh_qwen_status()
                    self._start_speechsummarizer_process()
                elif action == "managed_start_failed":
                    self._managed_start_failed(str(value or "QwenASR start failed"))
                elif action == "qwen_restart_complete":
                    self._qwen_restart_finished()
                elif action == "qwen_restart_failed":
                    self._qwen_restart_finished(str(value or "QwenASR restart failed"))
                elif action == "qwen_status" and isinstance(value, tuple) and isinstance(value[0], QwenReadyStatus):
                    self._apply_qwen_status(value[0], bool(value[1]))
        except queue.Empty:
            pass
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self._update_status()
        self.root.after(200, self._poll_log_queue)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_status(self) -> None:
        running = self.proc is not None and self.proc.poll() is None
        busy = self._starting or self._qwen_restarting
        self.status_var.set("ASR再起動中" if self._qwen_restarting else ("起動準備中" if self._starting else ("起動中" if running else "停止中")))
        color = "#eab308" if busy else ("#16a34a" if running else "#9ca3af")
        self.status_indicator.itemconfigure(self.status_indicator_oval, fill=color)
        self.btn_start.configure(state="disabled" if running or busy else "normal")
        managed_running = self.qwen_proc is not None and self.qwen_proc.poll() is None
        provider = self.vars["asr_provider"].get().strip().lower()
        is_vibevoice = provider in {"vibevoice", "vibevoice-asr"}
        self.btn_stop.configure(state="normal" if running or managed_running or busy else "disabled")
        self.btn_qwen_restart.configure(
            state="normal" if managed_running and not busy and not is_vibevoice else "disabled"
        )
        self.btn_vibevoice_restart.configure(
            state="normal" if managed_running and not busy and is_vibevoice else "disabled"
        )

    def _on_close(self) -> None:
        any_running = (
            self._starting
            or self._qwen_restarting
            or (self.proc is not None and self.proc.poll() is None)
            or (self.qwen_proc is not None and self.qwen_proc.poll() is None)
        )
        if any_running:
            if not messagebox.askyesno("終了確認", "サーバーを停止して終了しますか？", parent=self.root):
                return
            self.stop_server()
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


def main() -> None:
    if "--check-server-imports" in sys.argv[1:]:
        import app as _server_import_check  # noqa: F401
        return
    if "--server" in sys.argv[1:]:
        import app as server_app

        server_app.run_server()
        return
    run_gui()


if __name__ == "__main__":
    main()
