from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from .jsonl_loader import list_jsonl_files, load_asr_events
    from .llm_client import DEFAULT_BOUNDARY_PROMPT_TEMPLATE, LlmConfig, label_session_stub, label_session_with_ollama, list_ollama_models
    from .models import BatchSessionLabelRequest, ExtractRuleRequest, FileListRequest, SaveReviewRequest, SessionCandidate, SessionCandidateRequest, TranscriptTurn
    from .reviewer_store import append_review_record, make_review_record
    from .rule_extractor import extract_rules
except ImportError:
    from jsonl_loader import list_jsonl_files, load_asr_events
    from llm_client import DEFAULT_BOUNDARY_PROMPT_TEMPLATE, LlmConfig, label_session_stub, label_session_with_ollama, list_ollama_models
    from models import BatchSessionLabelRequest, ExtractRuleRequest, FileListRequest, SaveReviewRequest, SessionCandidate, SessionCandidateRequest, TranscriptTurn
    from reviewer_store import append_review_record, make_review_record
    from rule_extractor import extract_rules

BASE_DIR = Path(__file__).resolve().parent


def _resolve_resource_dir(name: str) -> Path:
    candidates = [BASE_DIR / name]
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
        candidates.append(app_dir / 'tools' / BASE_DIR.name / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


STATIC_DIR = _resolve_resource_dir('static')
TEMPLATE_DIR = _resolve_resource_dir('templates')
logger = logging.getLogger('so_labeler.api')

app = FastAPI(title='S/O Boundary Reviewer')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


def _debug_log(message: str) -> None:
    print(message, flush=True)
    logger.info(message)


def _llm_config_from_state(request: Request) -> LlmConfig:
    data = getattr(request.app.state, 'llm_config', None) or {}
    return LlmConfig(
        base_url=str(data.get('base_url') or 'http://127.0.0.1:11434'),
        model=str(data.get('model') or 'qwen3.5:9b'),
        timeout_sec=int(data.get('timeout_sec') or 120),
        temperature=float(data.get('temperature') or 0.0),
        top_p=float(data.get('top_p') or 0.9),
    )


def _prompt_config_from_state(request: Request) -> tuple[list[dict], dict[str, str], str]:
    items = getattr(request.app.state, 'so_labeler_prompt_items', None) or [
        {'id': 'boundary_v1', 'label': 'S/O Boundary v1'},
    ]
    templates = getattr(request.app.state, 'so_labeler_prompt_templates', None) or {
        'boundary_v1': DEFAULT_BOUNDARY_PROMPT_TEMPLATE,
    }
    default_prompt_id = getattr(request.app.state, 'default_prompt_id', None) or items[0]['id']
    if default_prompt_id not in templates and items:
        default_prompt_id = items[0]['id']
    return items, templates, default_prompt_id


def _build_session_candidate(file_path: str) -> SessionCandidate:
    events = load_asr_events(file_path)
    source_file = Path(file_path).name
    turns = [
        TranscriptTurn(
            index=idx,
            ts=event.ts,
            seg_id=event.seg_id,
            text=event.text,
        )
        for idx, event in enumerate(events)
    ]
    patient_id = events[0].patient_id if events else None
    return SessionCandidate(
        session_id=source_file,
        source_file=source_file,
        patient_id=patient_id,
        turns=turns,
    )


@app.get('/', response_class=HTMLResponse)
def index() -> str:
    return (TEMPLATE_DIR / 'index.html').read_text(encoding='utf-8')


@app.get('/api/config')
def api_config(request: Request):
    llm_cfg = _llm_config_from_state(request)
    prompt_items, _, default_prompt_id = _prompt_config_from_state(request)
    return {
        'default_data_dir': getattr(request.app.state, 'default_data_dir', str(BASE_DIR.parent.parent / 'data')),
        'default_llm_mode': getattr(request.app.state, 'default_llm_mode', 'stub'),
        'llm_base_url': llm_cfg.base_url,
        'llm_model': llm_cfg.model,
        'prompt_items': prompt_items,
        'default_prompt_id': default_prompt_id,
    }


@app.get('/api/ollama_models')
def api_ollama_models(request: Request):
    llm_cfg = _llm_config_from_state(request)
    try:
        models = list_ollama_models(llm_cfg)
        return {'ok': True, 'models': models, 'default_model': llm_cfg.model}
    except Exception as e:
        return {'ok': False, 'models': [], 'default_model': llm_cfg.model, 'error': str(e)}


@app.post('/api/files')
def api_files(req: FileListRequest):
    items = list_jsonl_files(req.data_dir, req.start_date, req.end_date)
    return {'items': [item.__dict__ for item in items]}


@app.post('/api/boundaries')
def api_sessions(req: SessionCandidateRequest):
    items = [_build_session_candidate(fp).model_dump() for fp in req.file_paths]
    return {'items': items}


@app.post('/api/label_boundaries')
def api_label_sessions(req: BatchSessionLabelRequest, request: Request):
    llm_cfg = _llm_config_from_state(request)
    _, prompt_templates, default_prompt_id = _prompt_config_from_state(request)
    prompt_id = req.prompt_id or default_prompt_id
    prompt_template = prompt_templates.get(prompt_id) or DEFAULT_BOUNDARY_PROMPT_TEMPLATE
    model = req.model or llm_cfg.model
    total = len(req.sessions)
    _debug_log(
        f'[so_labeler][label_sessions_start] total={total} mode={req.mode} '
        f'model={model if req.mode == "ollama" else ""} '
        f'prompt_id={prompt_id if req.mode == "ollama" else ""} timeout={llm_cfg.timeout_sec}s'
    )
    batch_started = time.perf_counter()
    items = []
    for idx, session in enumerate(req.sessions, start=1):
        started = time.perf_counter()
        preview = ' | '.join(turn.text for turn in session.turns[:6])[:240]
        _debug_log(
            f'[so_labeler][session_start] {idx}/{total} session_id={session.session_id} '
            f'turns={len(session.turns)} transcript_preview={preview!r}'
        )
        try:
            if req.mode == 'ollama':
                llm = label_session_with_ollama(session, llm_cfg, model=model, prompt_template=prompt_template)
            else:
                llm = label_session_stub(session)
            record = make_review_record(session, llm, req.mode, model if req.mode == 'ollama' else '', prompt_id if req.mode == 'ollama' else '')
            items.append(record.model_dump())
            _debug_log(
                f'[so_labeler][session_done] {idx}/{total} session_id={session.session_id} '
                f'elapsed={time.perf_counter() - started:.2f}s'
            )
        except Exception as e:
            logger.exception('[so_labeler][session_error] session_id=%s', session.session_id)
            raise HTTPException(
                status_code=502,
                detail={
                    'message': str(e),
                    'session_id': session.session_id,
                    'model': model if req.mode == 'ollama' else '',
                    'prompt_id': prompt_id if req.mode == 'ollama' else '',
                    'mode': req.mode,
                    'session_index': idx,
                    'session_total': total,
                },
            ) from e
    _debug_log(
        f'[so_labeler][label_sessions_done] total={total} elapsed={time.perf_counter() - batch_started:.2f}s'
    )
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
