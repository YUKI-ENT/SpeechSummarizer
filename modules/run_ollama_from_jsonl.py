# run_ollama_from_jsonl.py
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .llm_ollama import (
    OllamaConfig,
    build_s_extraction_prompt,
    ollama_generate_json,
    OllamaError,
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_quality(rec: Dict[str, Any]) -> str:
    try:
        return str(rec.get("meta", {}).get("quality", "unknown"))
    except Exception:
        return "unknown"


def quality_ok(q: str, min_quality: str) -> bool:
    order = {"bad": 0, "maybe": 1, "good": 2}
    return order.get(q, 1) >= order.get(min_quality, 1)


def collect_asr_lines(
    jsonl_path: Path,
    *,
    min_quality: str = "good",
    max_lines: Optional[int] = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for rec in iter_jsonl(jsonl_path):
        if rec.get("type") != "asr":
            continue
        q = get_quality(rec)
        if not quality_ok(q, min_quality):
            continue
        txt = (rec.get("text") or "").strip()
        if not txt:
            continue
        items.append(rec)
        if max_lines and len(items) >= max_lines:
            break
    return items


def build_asr_input_text(asr_items: List[Dict[str, Any]], *, include_meta: bool = True) -> str:
    lines = []
    for rec in asr_items:
        seg_id = rec.get("seg_id")
        text = (rec.get("text") or "").strip()
        q = get_quality(rec)
        if include_meta:
            lines.append(f"[seg:{seg_id} q:{q}] {text}")
        else:
            lines.append(f"{text}")
    return "\n".join(lines)


def default_out_path(
    in_path: Path,
    *,
    model: str,
    min_quality: str,
    mode: str,
) -> Path:
    """
    mode:
      - fixed: <input>.llm.jsonl
      - auto : <input>.llm_s_extract.<model>.<minq>.<timestamp>.jsonl
    """
    if mode == "fixed":
        return in_path.with_suffix(in_path.suffix + ".llm.jsonl")

    # auto
    safe_model = model.replace("/", "_").replace(":", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{in_path.name}.llm_s_extract.{safe_model}.minq_{min_quality}.{ts}.jsonl"
    return in_path.parent / name


def write_jsonl(path: Path, obj: Dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"Output exists: {path}\n"
            f"上書きするなら --overwrite を付けるか、--out を変えるか、--out-mode auto を使ってください。"
        )
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=str, help="入力JSONL（録音/ASRログ）")
    ap.add_argument("--out", type=str, default="", help="出力JSONL（指定があればそれを使う）")
    ap.add_argument("--out-mode", type=str, default="fixed", choices=["fixed", "auto"],
                    help="fixed: 常に同名 / auto: 条件+時刻で別ファイル")
    ap.add_argument("--overwrite", action="store_true", help="出力を上書きする（fixed時に便利）")

    ap.add_argument("--host", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--model", type=str, default="gemma3:12b")
    ap.add_argument("--timeout", type=float, default=120.0)

    ap.add_argument("--min-quality", type=str, default="good", choices=["bad", "maybe", "good"])
    ap.add_argument("--max-lines", type=int, default=0, help="0なら全部")
    ap.add_argument("--include-meta", action="store_true", help="入力にseg_id/qualityを含める")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)

    args = ap.parse_args()

    in_path = Path(args.jsonl)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    out_path = Path(args.out) if args.out else default_out_path(
        in_path, model=args.model, min_quality=args.min_quality, mode=args.out_mode
    )

    asr_items = collect_asr_lines(
        in_path,
        min_quality=args.min_quality,
        max_lines=(args.max_lines if args.max_lines > 0 else None),
    )
    if not asr_items:
        raise SystemExit("ASR items not found (check --min-quality or file content)")

    asr_text = build_asr_input_text(asr_items, include_meta=args.include_meta)
    prompt = build_s_extraction_prompt(asr_text)

    cfg = OllamaConfig(host=args.host, model=args.model, timeout_sec=args.timeout)

    print(f"[info] input_asr_lines={len(asr_items)} model={cfg.model} host={cfg.host}")
    try:
        parsed, raw = ollama_generate_json(cfg, prompt, temperature=args.temperature, top_p=args.top_p)

    except OllamaError as e:
        raise SystemExit(f"OllamaError: {e}")

    out_rec: Dict[str, Any] = {
        "type": "llm_s_extract",
        "source_jsonl": str(in_path),
        "min_quality": args.min_quality,
        "model": cfg.model,
        "host": cfg.host,
        "ollama_elapsed_sec": raw.get("_elapsed_sec"),
        "input_lines": len(asr_items),
        "input_preview": asr_text[:1000],
        "result": parsed,
    }

    write_jsonl(out_path, out_rec, overwrite=args.overwrite)

    print(f"[ok] wrote: {out_path}")
    print(json.dumps(parsed, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
