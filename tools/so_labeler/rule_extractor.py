from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

try:
    from .reviewer_store import load_review_records
except ImportError:
    from reviewer_store import load_review_records


def _normalize_phrase(text: str) -> str:
    return ' '.join((text or '').strip().split())


def extract_rules(annotation_path: str, output_path: str, min_count: int = 1) -> dict:
    records = [
        record
        for record in load_review_records(annotation_path)
        if record.human_checked and record.human_is_boundary
    ]

    trigger_counter: Counter[str] = Counter()
    source_examples: dict[str, list[dict[str, str]]] = {}

    for record in records:
        phrases = record.human_trigger_phrases or record.llm_trigger_phrases
        for phrase in phrases:
            normalized = _normalize_phrase(phrase)
            if not normalized:
                continue
            trigger_counter.update([normalized])
            source_examples.setdefault(normalized, [])
            if len(source_examples[normalized]) < 3:
                source_examples[normalized].append({
                    'source_file': record.source_file,
                    'prev_text': record.prev_text,
                    'next_text': record.next_text,
                })

    triggers = {
        phrase: {
            'score': float(count),
            'examples': source_examples.get(phrase, []),
        }
        for phrase, count in trigger_counter.items()
        if count >= min_count
    }

    payload = {
        'rule_type': 's_to_o_boundary_triggers',
        'count': len(records),
        'triggers': triggers,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload
