import json
import locale
import queue
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


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
    ("ssl", "certfile"),
    ("ssl", "keyfile"),
}
MODEL_PATH_PREFIX = ("asr", "models")


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


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SpeechSummarizer Launcher")
        self.root.geometry("1040x820")
        self.root.minsize(900, 720)

        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.vars: dict[str, tk.Variable] = {}
        self.field_meta: dict[str, dict] = {}
        self.model_rows: list[tuple[tk.StringVar, tk.StringVar]] = []
        self.prompt_rows: list[tuple[tk.StringVar, tk.StringVar, ScrolledText]] = []

        self.cfg = load_config()
        self._build_ui()
        self._load_form_from_config()
        self._update_status()
        self._poll_log_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        ttk.Label(top, text="SpeechSummarizer", font=("", 16, "bold")).pack(side="left")

        self.status_var = tk.StringVar(value="停止中")
        ttk.Label(top, textvariable=self.status_var).pack(side="right")

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

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        general_tab = ttk.Frame(notebook, padding=12)
        asr_tab = ttk.Frame(notebook, padding=12)
        llm_tab = ttk.Frame(notebook, padding=12)
        log_tab = ttk.Frame(notebook, padding=12)

        notebook.add(general_tab, text="一般")
        notebook.add(asr_tab, text="ASR")
        notebook.add(llm_tab, text="LLM")
        notebook.add(log_tab, text="ログ")

        self._build_general_tab(general_tab)
        self._build_asr_tab(asr_tab)
        self._build_llm_tab(llm_tab)
        self._build_log_tab(log_tab)

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
        self._add_bool(parent, "ssl_enabled", "SSL有効", ("ssl", "enabled"), row=row)
        row += 1
        self._add_path_entry(parent, "certfile", "SSL certfile", ("ssl", "certfile"), row=row, select="file")
        row += 1
        self._add_path_entry(parent, "keyfile", "SSL keyfile", ("ssl", "keyfile"), row=row, select="file")

    def _build_asr_tab(self, parent: ttk.Frame) -> None:
        for i in range(4):
            parent.columnconfigure(i, weight=1)

        row = 0
        self._add_entry(parent, "asr_model_id", "選択モデルID", ("asr", "model_id"), kind="str", row=row, width=20)
        self._add_entry(parent, "asr_language", "言語", ("asr", "language"), kind="str", row=row, col=2, width=12)
        row += 1
        self._add_choice(parent, "asr_device", "Device", ("asr", "device"), ["cpu", "cuda"], row=row)
        self._add_choice(parent, "asr_compute_type", "Compute Type", ("asr", "compute_type"), ["int8", "float16"], row=row, col=2)
        row += 1
        self._add_entry(parent, "asr_beam_size", "Beam Size", ("asr", "beam_size"), kind="int", row=row, width=12)
        self._add_entry(parent, "asr_temperature", "Temperature", ("asr", "temperature"), kind="float", row=row, col=2, width=12)
        row += 1
        self._add_bool(parent, "asr_condition_prev", "前文脈を利用", ("asr", "condition_on_previous_text"), row=row)
        row += 1
        self._add_text(parent, "asr_initial_prompt", "Initial Prompt", ("asr", "initial_prompt"), row=row, height=5)
        row += 1

        vad_box = ttk.LabelFrame(parent, text="VAD", padding=10)
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

        model_box = ttk.LabelFrame(parent, text="モデルIDとパス", padding=10)
        model_box.grid(row=row, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
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

    def _build_llm_tab(self, parent: ttk.Frame) -> None:
        for i in range(4):
            parent.columnconfigure(i, weight=1)
        parent.rowconfigure(4, weight=1)

        row = 0
        self._add_entry(parent, "llm_host", "Host", ("llm", "host"), kind="str", row=row, width=32)
        self._add_entry(parent, "llm_model_default", "既定モデル", ("llm", "model_default"), kind="str", row=row, col=2, width=20)
        row += 1
        self._add_entry(parent, "llm_timeout", "Timeout", ("llm", "timeout"), kind="float", row=row, width=12)
        self._add_entry(parent, "llm_temperature", "Temperature", ("llm", "temperature"), kind="float", row=row, col=2, width=12)
        row += 1
        self._add_entry(parent, "llm_top_p", "Top P", ("llm", "top_p"), kind="float", row=row, width=12)
        self._add_entry(parent, "llm_default_prompt_id", "既定Prompt ID", ("llm", "default_prompt_id"), kind="str", row=row, col=2, width=20)
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

    def _build_log_tab(self, parent: ttk.Frame) -> None:
        self.log_text = ScrolledText(parent, height=30, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _add_entry(self, parent, name: str, label: str, path: tuple[str, ...], kind: str, row: int, col: int = 0, width: int = 24) -> None:
        var = tk.StringVar()
        self.vars[name] = var
        self.field_meta[name] = {"path": path, "kind": kind}

        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="we", padx=(0, 16), pady=6)

    def _add_path_entry(self, parent, name: str, label: str, path: tuple[str, ...], row: int, select: str) -> None:
        var = tk.StringVar()
        self.vars[name] = var
        self.field_meta[name] = {"path": path, "kind": "path", "select": select}

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="we", padx=(0, 8), pady=6)
        ttk.Button(parent, text="参照", command=lambda: self._browse_named_path(name), width=8).grid(row=row, column=2, sticky="w", pady=6)

    def _add_choice(self, parent, name: str, label: str, path: tuple[str, ...], choices: list[str], row: int, col: int = 0) -> None:
        var = tk.StringVar()
        self.vars[name] = var
        self.field_meta[name] = {"path": path, "kind": "choice"}

        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=6)
        box = ttk.Combobox(parent, textvariable=var, values=choices, state="readonly", width=18)
        box.grid(row=row, column=col + 1, sticky="w", padx=(0, 16), pady=6)

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

    def _load_form_from_config(self) -> None:
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
                value = get_nested(self.cfg, path, False if kind == "bool" else "")

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

    def reload_form(self) -> None:
        self._load_form_from_config()
        self._append_log(f"[launcher] config reloaded: {CONFIG_PATH.name}")

    def save_form(self) -> bool:
        cfg = load_config()
        try:
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

            save_config(cfg)
            self.cfg = cfg
            self._append_log(f"[launcher] config saved: {CONFIG_PATH.name}")
            return True
        except Exception as e:
            messagebox.showerror("設定保存エラー", str(e), parent=self.root)
            return False

    def start_server(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        if not self.save_form():
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
            return

        threading.Thread(target=self._read_process_output, daemon=True).start()
        self._append_log(f"[launcher] server start: {' '.join(cmd)}")
        self._update_status()

    def stop_server(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self._append_log("[launcher] server stopped")
        self._update_status()

    def open_browser(self) -> None:
        cfg = load_config()
        port = int(cfg.get("port", 8000))
        ssl_enabled = bool(get_nested(cfg, ("ssl", "enabled"), False))
        scheme = "https" if ssl_enabled else "http"
        webbrowser.open(f"{scheme}://127.0.0.1:{port}")

    def _read_process_output(self) -> None:
        assert self.proc is not None
        if not self.proc.stdout:
            return
        for line in self.proc.stdout:
            self.log_queue.put(line.rstrip())
        rc = self.proc.poll()
        self.log_queue.put(f"[launcher] server exited rc={rc}")

    def _poll_log_queue(self) -> None:
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
        self.status_var.set("起動中" if running else "停止中")
        self.btn_start.configure(state="disabled" if running else "normal")
        self.btn_stop.configure(state="normal" if running else "disabled")

    def _on_close(self) -> None:
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("終了確認", "サーバーを停止して終了しますか？", parent=self.root):
                return
            self.stop_server()
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


def main() -> None:
    if "--server" in sys.argv[1:]:
        import app as server_app

        server_app.run_server()
        return
    run_gui()


if __name__ == "__main__":
    main()
