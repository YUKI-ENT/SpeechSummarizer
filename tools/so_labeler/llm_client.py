from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

try:
    from .models import SessionBoundaryLabelResult, SessionCandidate
except ImportError:
    from models import SessionBoundaryLabelResult, SessionCandidate


logger = logging.getLogger('so_labeler.llm')


def _debug_log(message: str) -> None:
    print(message, flush=True)
    logger.info(message)


SYSTEM_PROMPT = '''あなたは耳鼻科診察会話の「S→O遷移」判定器です。
返答は必ずJSONのみで返してください。
目的は、問診や症状確認(S)が続いたあと、診察・観察・身体所見(O)へ移る最初の境界を1箇所だけ見つけることです。
JSON形式:
{"has_boundary":true|false,"boundary_index":0以上の整数またはnull,"confidence":0.0-1.0,"trigger_text":"短い表現","trigger_phrases":["短い句"],"reason":"短い理由"}
boundary_index は「その発話の後に切り替わる」位置です。たとえば index=5 なら、turn[5] の後で O に入る意味です。
'''


DEFAULT_BOUNDARY_PROMPT_TEMPLATE = '''以下は耳鼻科診察会話の全文です。
S(問診)からO(診察・所見)へ切り替わる最初の境界を1箇所だけ選んでください。
切り替わりが無ければ has_boundary=false にしてください。
「見ていきますね」「お口開けて」「喉を見ます」のような診察開始の合図を重視してください。

会話全文:
{transcript}
'''


STUB_TRIGGER_PHRASES = [
    '見ていきますね',
    '見ますね',
    '診ますね',
    '口を開けて',
    'お口開けて',
    '喉を見ます',
    '鼻を見ます',
    '耳を見ます',
    'ちょっと見せて',
    'じゃあ見て',
]


@dataclass
class LlmConfig:
    base_url: str = 'http://127.0.0.1:11434'
    model: str = 'qwen3.5:9b'
    timeout_sec: int = 120
    temperature: float = 0.0
    top_p: float = 0.9


def list_ollama_models(cfg: LlmConfig) -> list[str]:
    url = cfg.base_url.rstrip('/') + '/api/tags'
    _debug_log(f'[so_labeler][ollama_models] GET {url} timeout={cfg.timeout_sec}s')
    resp = requests.get(url, timeout=cfg.timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    items = data.get('models') or []
    names = [str(item.get('name') or '').strip() for item in items if item.get('name')]
    return sorted(names)


def transcript_text(session: SessionCandidate) -> str:
    return '\n'.join(
        f'[{turn.index}] {turn.text}'
        for turn in session.turns
    )


def build_user_prompt(session: SessionCandidate, prompt_template: str | None = None) -> str:
    template = prompt_template or DEFAULT_BOUNDARY_PROMPT_TEMPLATE
    turns = session.turns or []
    context_before = '\n'.join(f'[{turn.index}] {turn.text}' for turn in turns[:8]) or '(none)'
    context_after = '\n'.join(f'[{turn.index}] {turn.text}' for turn in turns[-8:]) or '(none)'
    prev_text = turns[-2].text if len(turns) >= 2 else (turns[0].text if turns else '')
    next_text = turns[-1].text if turns else ''
    try:
        return template.format(
            transcript=transcript_text(session),
            context_before=context_before,
            context_after=context_after,
            prev_text=prev_text,
            next_text=next_text,
        )
    except KeyError as e:
        raise ValueError(f'invalid so_labeler prompt template variable: {e}') from e


def _extract_first_json_object(text: str) -> str:
    start = text.find('{')
    if start < 0:
        raise ValueError('no JSON object found in LLM response')

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    raise ValueError('unterminated JSON object in LLM response')


def parse_llm_json(text: str) -> SessionBoundaryLabelResult:
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        text = text.replace('json\n', '', 1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = json.loads(_extract_first_json_object(text))
    return SessionBoundaryLabelResult(**obj)


def label_session_with_ollama(
    session: SessionCandidate,
    cfg: LlmConfig,
    *,
    model: str | None = None,
    prompt_template: str | None = None,
) -> SessionBoundaryLabelResult:
    url = cfg.base_url.rstrip('/') + '/api/generate'
    selected_model = model or cfg.model
    prompt_text = build_user_prompt(session, prompt_template)
    full_prompt = f'{SYSTEM_PROMPT}\n\n{prompt_text}'
    body: dict[str, Any] = {
        'model': selected_model,
        'stream': False,
        'format': 'json',
        'prompt': full_prompt,
        'think': False,
        'options': {
            'temperature': cfg.temperature,
            'top_p': cfg.top_p,
        },
    }
    prompt_preview = full_prompt[:240].replace('\n', ' ')
    _debug_log(
        f'[so_labeler][ollama_start] session_id={session.session_id} model={selected_model} '
        f'timeout={cfg.timeout_sec}s prompt_chars={len(full_prompt)} prompt_preview={prompt_preview!r}'
    )
    started = time.perf_counter()
    try:
        resp = requests.post(url, json=body, timeout=cfg.timeout_sec)
        elapsed = time.perf_counter() - started
        _debug_log(
            f'[so_labeler][ollama_http] session_id={session.session_id} '
            f'status={resp.status_code} elapsed={elapsed:.2f}s bytes={len(resp.content or b"")}'
        )
        resp.raise_for_status()
        data = resp.json()
        content = str(data.get('response') or '')
        _debug_log(
            f'[so_labeler][ollama_json] session_id={session.session_id} '
            f'keys={sorted(data.keys())} response_chars={len(content)} '
            f'thinking_chars={len(str(data.get("thinking") or ""))} '
            f'done={data.get("done")} done_reason={data.get("done_reason")!r}'
        )
        _debug_log(
            f'[so_labeler][ollama_content] session_id={session.session_id} '
            f'content_chars={len(content)} content_preview={content[:240]!r}'
        )
        if not content.strip():
            raise ValueError(f'empty response from Ollama: keys={sorted(data.keys())} done_reason={data.get("done_reason")!r}')
        result = parse_llm_json(content)
        _debug_log(
            f'[so_labeler][ollama_done] session_id={session.session_id} '
            f'has_boundary={result.has_boundary} boundary_index={result.boundary_index} '
            f'confidence={result.confidence} trigger={result.trigger_text!r} elapsed={elapsed:.2f}s'
        )
        return result
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception(
            '[so_labeler][ollama_error] session_id=%s model=%s elapsed=%.2fs',
            session.session_id,
            selected_model,
            elapsed,
        )
        raise


def label_session_stub(session: SessionCandidate) -> SessionBoundaryLabelResult:
    for turn in session.turns:
        for phrase in STUB_TRIGGER_PHRASES:
            if phrase in turn.text:
                return SessionBoundaryLabelResult(
                    has_boundary=True,
                    boundary_index=turn.index,
                    confidence=0.78,
                    trigger_text=phrase,
                    trigger_phrases=[phrase],
                    reason='診察開始を示す定型句を検出',
                )
    return SessionBoundaryLabelResult(
        has_boundary=False,
        boundary_index=None,
        confidence=0.35,
        trigger_text='',
        trigger_phrases=[],
        reason='切り替わりを示す明確な語が見つからない',
    )
