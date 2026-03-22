from __future__ import annotations

import json
from pathlib import Path

try:
    from .models import BoundaryCandidate, BoundaryLabelResult, BoundaryReviewRecord
except ImportError:
    from models import BoundaryCandidate, BoundaryLabelResult, BoundaryReviewRecord


def make_review_record(
    candidate: BoundaryCandidate,
    llm: BoundaryLabelResult,
    llm_mode: str,
) -> BoundaryReviewRecord:
    return BoundaryReviewRecord(
        boundary_id=candidate.boundary_id,
        source_file=candidate.source_file,
        patient_id=candidate.patient_id,
        event_index=candidate.event_index,
        prev_ts=candidate.prev_ts,
        next_ts=candidate.next_ts,
        prev_text=candidate.prev_text,
        next_text=candidate.next_text,
        context_before=candidate.context_before,
        context_after=candidate.context_after,
        prev_seg_id=candidate.prev_seg_id,
        next_seg_id=candidate.next_seg_id,
        llm_mode=llm_mode,
        llm_phase_before=llm.phase_before,
        llm_phase_after=llm.phase_after,
        llm_is_boundary=llm.is_boundary,
        llm_confidence=llm.confidence,
        llm_trigger_text=llm.trigger_text,
        llm_trigger_phrases=llm.trigger_phrases,
        llm_reason=llm.reason,
        human_is_boundary=llm.is_boundary,
        human_trigger_phrases=list(llm.trigger_phrases),
        human_checked=False,
    )


def append_review_record(output_path: str, record: BoundaryReviewRecord) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(record.model_dump_json(ensure_ascii=False) + '\n')


def load_review_records(annotation_path: str) -> list[BoundaryReviewRecord]:
    path = Path(annotation_path)
    if not path.exists():
        return []
    rows: list[BoundaryReviewRecord] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(BoundaryReviewRecord(**json.loads(line)))
    return rows
