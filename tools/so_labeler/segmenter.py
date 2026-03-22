from __future__ import annotations

try:
    from .models import AsrEvent, BoundaryCandidate
except ImportError:
    from models import AsrEvent, BoundaryCandidate


def build_boundary_candidates(
    events: list[AsrEvent],
    context_size: int = 3,
    max_candidates: int = 0,
) -> list[BoundaryCandidate]:
    if context_size < 0:
        raise ValueError('context_size must be >= 0')
    if max_candidates < 0:
        raise ValueError('max_candidates must be >= 0')

    candidates: list[BoundaryCandidate] = []
    for idx in range(len(events) - 1):
        prev_event = events[idx]
        next_event = events[idx + 1]

        before_events = events[max(0, idx - context_size):idx]
        after_events = events[idx + 2:idx + 2 + context_size]

        candidate = BoundaryCandidate(
            boundary_id=f'{prev_event.source_file}__boundary_{idx:05d}',
            source_file=prev_event.source_file,
            patient_id=prev_event.patient_id or next_event.patient_id,
            event_index=idx,
            prev_ts=prev_event.ts,
            next_ts=next_event.ts,
            prev_text=prev_event.text,
            next_text=next_event.text,
            context_before=[event.text for event in before_events],
            context_after=[event.text for event in after_events],
            prev_seg_id=prev_event.seg_id,
            next_seg_id=next_event.seg_id,
        )
        candidates.append(candidate)
        if max_candidates and len(candidates) >= max_candidates:
            break

    return candidates
