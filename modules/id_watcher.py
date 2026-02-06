# modules/id_watcher.py
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

@dataclass
class IdWatchConfig:
    dir_path: str
    glob: str = "dyna*.txt"        # dyna384300.txt 等
    poll_sec: float = 0.5
    read_delay_ms: int = 300
    encoding: str = "cp932"        # SJIS互換（Windows由来ならcp932が無難）
    delete_after_read: bool = True

class IdWatcher:
    def __init__(self, cfg: IdWatchConfig, on_id: Callable[[int, str, dict], None]):
        """
        on_id(karte_id:int, source_path:str, meta:dict)
        metaには氏名なども入れられる（必要なら）
        """
        self.cfg = cfg
        self.on_id = on_id
        self._seen: set[str] = set()

    def _read_first_line(self, path: Path) -> Optional[str]:
        # 書き込み直後対策
        time.sleep(self.cfg.read_delay_ms / 1000.0)

        last_err = None
        for _ in range(5):
            try:
                with open(path, "r", encoding=self.cfg.encoding, errors="replace") as f:
                    line = f.readline().strip()
                return line
            except Exception as e:
                last_err = e
                time.sleep(0.1)
        return None

    def run_forever(self, stop_ev):
        base = Path(self.cfg.dir_path)
        base.mkdir(parents=True, exist_ok=True)

        while not stop_ev.is_set():
            try:
                for p in base.glob(self.cfg.glob):
                    if not p.is_file():
                        continue

                    sp = str(p.resolve())
                    if sp in self._seen:
                        continue
                    self._seen.add(sp)

                    line = self._read_first_line(p)
                    if not line:
                        continue

                    cols = [c.strip() for c in line.split(",")]
                    if not cols or not cols[0]:
                        continue

                    id_str = cols[0]  # 枝番なしのIDがここに入る前提
                    try:
                        karte_id = int(id_str)
                    except ValueError:
                        # 数値じゃないなら捨てる（ログは呼び元で）
                        continue

                    meta = {}
                    if len(cols) >= 2:
                        meta["name_kanji"] = cols[1]
                    if len(cols) >= 3:
                        meta["name_kana"] = cols[2]

                    self.on_id(karte_id, sp, meta)

                    if self.cfg.delete_after_read:
                        try:
                            p.unlink()
                        except Exception:
                            pass

                time.sleep(self.cfg.poll_sec)
            except Exception:
                time.sleep(self.cfg.poll_sec)
