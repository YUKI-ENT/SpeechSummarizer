from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from .jsonl_loader import list_jsonl_files, load_asr_events
    from .llm_client import LlmConfig, label_boundary_stub, label_boundary_with_ollama
    from .models import BatchBoundaryLabelRequest, BoundaryCandidateRequest, ExtractRuleRequest, FileListRequest, SaveReviewRequest
    from .reviewer_store import append_review_record, make_review_record
    from .rule_extractor import extract_rules
    from .segmenter import build_boundary_candidates
except ImportError:
    from jsonl_loader import list_jsonl_files, load_asr_events
    from llm_client import LlmConfig, label_boundary_stub, label_boundary_with_ollama
    from models import BatchBoundaryLabelRequest, BoundaryCandidateRequest, ExtractRuleRequest, FileListRequest, SaveReviewRequest
    from reviewer_store import append_review_record, make_review_record
    from rule_extractor import extract_rules
    from segmenter import build_boundary_candidates

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
TEMPLATE_DIR = BASE_DIR / 'templates'

app = FastAPI(title='S/O Boundary Reviewer')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


def _llm_config_from_state(request: Request) -> LlmConfig:
    data = getattr(request.app.state, 'llm_config', None) or {}
    return LlmConfig(
        base_url=str(data.get('base_url') or 'http://127.0.0.1:11434'),
        model=str(data.get('model') or 'qwen3.5:9b'),
        timeout_sec=int(data.get('timeout_sec') or 120),
    )


@app.get('/', response_class=HTMLResponse)
def index() -> str:
    return (TEMPLATE_DIR / 'index.html').read_text(encoding='utf-8')


@app.get('/api/config')
def api_config(request: Request):
    llm_cfg = _llm_config_from_state(request)
    return {
        'default_data_dir': getattr(request.app.state, 'default_data_dir', str(BASE_DIR.parent.parent / 'data')),
        'default_llm_mode': getattr(request.app.state, 'default_llm_mode', 'stub'),
        'llm_base_url': llm_cfg.base_url,
        'llm_model': llm_cfg.model,
    }


@app.post('/api/files')
def api_files(req: FileListRequest):
    items = list_jsonl_files(req.data_dir, req.start_date, req.end_date)
    return {'items': [item.__dict__ for item in items]}


@app.post('/api/boundaries')
def api_boundaries(req: BoundaryCandidateRequest):
    all_candidates: list[dict] = []
    for fp in req.file_paths:
        events = load_asr_events(fp)
        candidates = build_boundary_candidates(
            events,
            context_size=req.context_size,
            max_candidates=req.max_candidates_per_file,
        )
        all_candidates.extend([candidate.model_dump() for candidate in candidates])
    return {'items': all_candidates}


@app.post('/api/label_boundaries')
def api_label_boundaries(req: BatchBoundaryLabelRequest, request: Request):
    llm_cfg = _llm_config_from_state(request)
    items = []
    for candidate in req.candidates:
        if req.mode == 'ollama':
            llm = label_boundary_with_ollama(candidate, llm_cfg)
        else:
            llm = label_boundary_stub(candidate)
        record = make_review_record(candidate, llm, req.mode)
        items.append(record.model_dump())
    return {'items': items}


@app.post('/api/save_review')
def api_save_review(req: SaveReviewRequest):
    append_review_record(req.output_path, req.record)
    return {'ok': True}


@app.post('/api/extract_rules')
def api_extract_rules(req: ExtractRuleRequest):
    payload = extract_rules(req.annotation_path, req.output_path, req.min_count)
    return {'ok': True, 'rules': payload}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app:app', host='127.0.0.1', port=8765, reload=True)
