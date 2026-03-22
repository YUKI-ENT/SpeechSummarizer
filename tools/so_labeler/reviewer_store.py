from __future__ import annotations

import json
from pathlib import Path

try:
    from .models import SessionBoundaryLabelResult, SessionCandidate, SessionReviewRecord
except ImportError:
    from models import SessionBoundaryLabelResult, SessionCandidate, SessionReviewRecord


def make_review_record(
    session: SessionCandidate,
    llm: SessionBoundaryLabelResult,
    llm_mode: str,
    llm_model: str,
    llm_prompt_id: str,
) -> SessionReviewRecord:
    return SessionReviewRecord(
        session_id=session.session_id,
        source_file=session.source_file,
        patient_id=session.patient_id,
        turns=session.turns,
        llm_mode=llm_mode,
        llm_model=llm_model,
        llm_prompt_id=llm_prompt_id,
        llm_has_boundary=llm.has_boundary,
        llm_boundary_index=llm.boundary_index,
        llm_confidence=llm.confidence,
        llm_trigger_text=llm.trigger_text,
        llm_trigger_phrases=llm.trigger_phrases,
        llm_reason=llm.reason,
        human_has_boundary=llm.has_boundary,
        human_boundary_index=llm.boundary_index,
        human_trigger_phrases=list(llm.trigger_phrases),
        human_checked=False,
    )


def append_review_record(output_path: str, record: SessionReviewRecord) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(record.model_dump_json(ensure_ascii=False) + '\n')


def load_review_records(annotation_path: str) -> list[SessionReviewRecord]:
    path = Path(annotation_path)
    if not path.exists():
        return []
    rows: list[SessionReviewRecord] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(SessionReviewRecord(**json.loads(line)))
    return rows
