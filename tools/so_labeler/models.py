from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

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


class TranscriptTurn(BaseModel):
    index: int
    ts: str = ''
    seg_id: Optional[int] = None
    text: str


class SessionCandidate(BaseModel):
    session_id: str
    source_file: str
    patient_id: Optional[str] = None
    turns: list[TranscriptTurn] = Field(default_factory=list)


class SessionBoundaryLabelResult(BaseModel):
    has_boundary: bool = False
    boundary_index: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0)
    trigger_text: str = ''
    trigger_phrases: list[str] = Field(default_factory=list)
    reason: str = ''


class SessionReviewRecord(BaseModel):
    session_id: str
    source_file: str
    patient_id: Optional[str] = None
    turns: list[TranscriptTurn] = Field(default_factory=list)
    llm_mode: BoundaryMode = 'stub'
    llm_model: str = ''
    llm_prompt_id: str = ''
    llm_has_boundary: bool = False
    llm_boundary_index: Optional[int] = None
    llm_confidence: Optional[float] = None
    llm_trigger_text: str = ''
    llm_trigger_phrases: list[str] = Field(default_factory=list)
    llm_reason: str = ''
    human_has_boundary: bool = False
    human_boundary_index: Optional[int] = None
    human_trigger_phrases: list[str] = Field(default_factory=list)
    human_note: str = ''
    human_checked: bool = False


class FileListRequest(BaseModel):
    data_dir: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class SessionCandidateRequest(BaseModel):
    file_paths: list[str]


class BatchSessionLabelRequest(BaseModel):
    sessions: list[SessionCandidate]
    mode: BoundaryMode = 'stub'
    model: str = ''
    prompt_id: str = ''


class SaveReviewRequest(BaseModel):
    output_path: str
    record: SessionReviewRecord


class ExtractRuleRequest(BaseModel):
    annotation_path: str
    output_path: str
    min_count: int = 1
