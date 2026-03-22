from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

PhaseLabel = Literal['S', 'O', 'UNKNOWN']
BoundaryMode = Literal['stub', 'ollama']


class AsrEvent(BaseModel):
    source_file: str
    ts: str
    patient_id: Optional[str] = None
    seg_id: Optional[int] = None
    dur: Optional[float] = None
    wav: Optional[str] = None
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class BoundaryCandidate(BaseModel):
    boundary_id: str
    source_file: str
    patient_id: Optional[str] = None
    event_index: int
    prev_ts: str = ''
    next_ts: str = ''
    prev_text: str
    next_text: str
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)
    prev_seg_id: Optional[int] = None
    next_seg_id: Optional[int] = None


class BoundaryLabelResult(BaseModel):
    phase_before: PhaseLabel = 'UNKNOWN'
    phase_after: PhaseLabel = 'UNKNOWN'
    is_boundary: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    trigger_text: str = ''
    trigger_phrases: list[str] = Field(default_factory=list)
    reason: str = ''


class BoundaryReviewRecord(BaseModel):
    boundary_id: str
    source_file: str
    patient_id: Optional[str] = None
    event_index: int
    prev_ts: str = ''
    next_ts: str = ''
    prev_text: str
    next_text: str
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)
    prev_seg_id: Optional[int] = None
    next_seg_id: Optional[int] = None
    llm_mode: BoundaryMode = 'stub'
    llm_phase_before: PhaseLabel = 'UNKNOWN'
    llm_phase_after: PhaseLabel = 'UNKNOWN'
    llm_is_boundary: bool = False
    llm_confidence: Optional[float] = None
    llm_trigger_text: str = ''
    llm_trigger_phrases: list[str] = Field(default_factory=list)
    llm_reason: str = ''
    human_is_boundary: Optional[bool] = None
    human_trigger_phrases: list[str] = Field(default_factory=list)
    human_note: str = ''
    human_checked: bool = False


class FileListRequest(BaseModel):
    data_dir: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class BoundaryCandidateRequest(BaseModel):
    file_paths: list[str]
    context_size: int = 3
    max_candidates_per_file: int = 0


class BatchBoundaryLabelRequest(BaseModel):
    candidates: list[BoundaryCandidate]
    mode: BoundaryMode = 'stub'


class SaveReviewRequest(BaseModel):
    output_path: str
    record: BoundaryReviewRecord


class ExtractRuleRequest(BaseModel):
    annotation_path: str
    output_path: str
    min_count: int = 1
