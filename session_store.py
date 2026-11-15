"""Rebuildable session/LLM metadata index. Transcript and summary bodies stay in files."""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
import threading


SESSION_RE = re.compile(
    r"^(?P<patient_id>[^_/\\]+?)(?:_b(?P<branch>[0-9]))?_"
    r"(?P<stamp>\d{8}_\d{6})(?:_c(?P<collision>[0-9a-f]{32}))?\.(?:txt|jsonl)$"
)


def parse_session_name(name: str) -> dict | None:
    match = SESSION_RE.fullmatch(name)
    if not match:
        return None
    return {
        "patient_id": match["patient_id"],
        "chart_branch": int(match["branch"]) if match["branch"] is not None else None,
        "stamp": match["stamp"],
        "session_id": name.rsplit(".", 1)[0],
    }


def decode_chart_id(value: str, includes_branch: bool) -> dict:
    value = value.strip()
    if not value.isascii() or not value.isdigit():
        raise ValueError("カルテNoは半角数字で入力してください。")
    if includes_branch:
        if len(value) < 2:
            raise ValueError("枝番ありのカルテNoは2桁以上で入力してください。")
        return {"patient_id": str(int(value[:-1])), "chart_branch": int(value[-1]),
                "source_chart_id": value}
    return {"patient_id": value, "chart_branch": None, "source_chart_id": value}


class SessionIndex:
    def __init__(self, outputs_dir: Path, llm_dir: Path):
        self.outputs_dir = Path(outputs_dir)
        self.llm_dir = Path(llm_dir)
        self.path = self.outputs_dir / "session_index.sqlite3"
        self._lock = threading.RLock()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    name TEXT PRIMARY KEY, patient_id TEXT NOT NULL,
                    chart_branch INTEGER, stamp TEXT NOT NULL,
                    approved_file TEXT, jsonl_mtime INTEGER, jsonl_size INTEGER
                );
                CREATE INDEX IF NOT EXISTS sessions_patient_time
                    ON sessions(patient_id, stamp DESC, name DESC);
                CREATE INDEX IF NOT EXISTS sessions_resume
                    ON sessions(patient_id, chart_branch, stamp DESC);
                CREATE TABLE IF NOT EXISTS llm_files (
                    name TEXT PRIMARY KEY, session_name TEXT NOT NULL, mtime INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS llm_session ON llm_files(session_name, mtime DESC);
            """)

    @contextmanager
    def connection(self):
        # Short lived connections also work for FastAPI and ASR worker threads on Windows.
        with self._lock:
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()

    def reconcile(self):
        """Scan directories once at startup; read only new/changed JSONLs."""
        with self.connection() as db:
            previous = {row["name"]: dict(row) for row in db.execute("SELECT * FROM sessions")}
            names = set()
            for path in self.outputs_dir.glob("*.txt"):
                info = parse_session_name(path.name)
                if info is None:
                    continue
                names.add(path.name)
                jsonl = path.with_suffix(".jsonl")
                try:
                    stat = jsonl.stat()
                    signature = (stat.st_mtime_ns, stat.st_size)
                except FileNotFoundError:
                    signature = (None, None)
                old = previous.get(path.name)
                if old and (old["jsonl_mtime"], old["jsonl_size"]) == signature:
                    continue
                approved = None
                if signature[0] is not None:
                    with jsonl.open(encoding="utf-8", errors="replace") as stream:
                        for line in stream:
                            try:
                                event = json.loads(line)
                            except (ValueError, TypeError):
                                continue
                            if isinstance(event, dict) and event.get("type") == "approved":
                                approved = event.get("llm_file")
                # Filename is durable identity, independent of current import settings.
                db.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (path.name, info["patient_id"], info["chart_branch"], info["stamp"],
                            approved, *signature))
            db.executemany("DELETE FROM sessions WHERE name = ?",
                           [(name,) for name in previous.keys() - names])
            db.execute("DELETE FROM llm_files")
            db.executemany("INSERT INTO llm_files VALUES (?, ?, ?)",
                           [row for path in self.llm_dir.glob("*.txt")
                            if (row := self._llm_row(path)) is not None])

    def record_event(self, path: Path, event: dict):
        if event.get("type") not in {"session_start", "approved"}:
            return
        info = parse_session_name(path.name)
        if info is None or path.parent.resolve() != self.outputs_dir.resolve():
            return
        name = path.with_suffix(".txt").name
        stat = path.stat()
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO sessions VALUES (?, ?, ?, ?, NULL, NULL, NULL)",
                       (name, info["patient_id"], info["chart_branch"], info["stamp"]))
            if event["type"] == "approved":
                db.execute("UPDATE sessions SET approved_file = ? WHERE name = ?",
                           (event.get("llm_file"), name))
            db.execute("UPDATE sessions SET jsonl_mtime = ?, jsonl_size = ? WHERE name = ?",
                       (stat.st_mtime_ns, stat.st_size, name))

    @staticmethod
    def _llm_row(path):
        stem = path.name.split("__", 1)[0]
        if "__" not in path.name or parse_session_name(stem + ".txt") is None:
            return None
        return path.name, stem + ".txt", path.stat().st_mtime_ns

    def record_llm(self, path: Path):
        row = self._llm_row(path)
        if row:
            with self.connection() as db:
                db.execute("INSERT OR REPLACE INTO llm_files VALUES (?, ?, ?)", row)

    def sessions(self, patient_id=None):
        with self.connection() as db:
            query = "SELECT * FROM sessions"
            args = ()
            if patient_id is not None:
                query += " WHERE patient_id = ?"
                args = (patient_id,)
            rows = [dict(row) for row in db.execute(query + " ORDER BY stamp DESC, name DESC", args)]
            llm = {}
            query = "SELECT l.* FROM llm_files l"
            if patient_id is not None:
                query += " JOIN sessions s ON s.name = l.session_name WHERE s.patient_id = ?"
            for row in db.execute(query + " ORDER BY l.mtime DESC, l.name DESC", args):
                llm.setdefault(row["session_name"], []).append(row["name"])
            for row in rows:
                row["llm_files"] = llm.get(row["name"], [])
            return rows

    def recent_patients(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute("""
                SELECT name, patient_id, stamp FROM (
                    SELECT name, patient_id, stamp,
                        ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY stamp DESC, name DESC) AS n
                    FROM sessions
                ) WHERE n = 1 ORDER BY stamp DESC, name DESC
            """)]

    def latest_today(self, patient_id, chart_branch, ymd):
        with self.connection() as db:
            row = db.execute("""SELECT name FROM sessions
                WHERE patient_id = ? AND chart_branch IS ? AND stamp >= ? AND stamp <= ?
                ORDER BY stamp DESC, name DESC LIMIT 1""",
                (patient_id, chart_branch, ymd + "_000000", ymd + "_235959")).fetchone()
            return row["name"] if row else None

    def llm_history(self, *, session=None, patient_id=None, limit=200):
        with self.connection() as db:
            if session:
                query = "SELECT l.name FROM llm_files l WHERE l.session_name = ?"
                args = (session,)
            else:
                query = ("SELECT l.name FROM llm_files l JOIN sessions s ON s.name = l.session_name "
                         "WHERE s.patient_id = ?")
                args = (patient_id,)
            query += " ORDER BY l.mtime DESC, l.name DESC"
            if limit is not None:
                query += " LIMIT ?"
                args += (limit,)
            return [row["name"] for row in db.execute(query, args)]
