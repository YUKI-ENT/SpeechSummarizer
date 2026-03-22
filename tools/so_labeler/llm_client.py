from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

try:
    from .models import BoundaryCandidate, BoundaryLabelResult
except ImportError:
    from models import BoundaryCandidate, BoundaryLabelResult


SYSTEM_PROMPT = '''あなたは耳鼻科診察会話の「S→O遷移」判定器です。
目的は、問診や症状確認(S)が続いたあと、診察・観察・身体所見(O)へ移る境界を見つけることです。
注目点:
- 「じゃあ見ていきますね」「お口開けてください」「喉を見ます」など、診察開始の合図
- 単なる相槌や雑談では境界にしない
- 前後の流れを見て、Sが続いていてOへ入る瞬間だけを拾う
返答は必ずJSONのみ:
{"phase_before":"S|O|UNKNOWN","phase_after":"S|O|UNKNOWN","is_boundary":true|false,"confidence":0.0-1.0,"trigger_text":"境界のきっかけになった短い表現","trigger_phrases":["短い句"],"reason":"短い理由"}
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


def build_user_prompt(candidate: BoundaryCandidate) -> str:
    payload = {
        'task': 'SからOへの境界かどうかを判定する',
        'context_before': candidate.context_before,
        'prev_text': candidate.prev_text,
        'next_text': candidate.next_text,
        'context_after': candidate.context_after,
        'notes': [
            'prev_textとnext_textの間に境界があるか判定する',
            'trigger_textは境界のきっかけになった短い表現を返す',
            'trigger_phrasesは辞書に入れやすい短い句へ分割する',
            '境界がなければis_boundary=falseでよい',
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_llm_json(text: str) -> BoundaryLabelResult:
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        text = text.replace('json\n', '', 1).strip()
    obj = json.loads(text)
    return BoundaryLabelResult(**obj)


def label_boundary_with_ollama(candidate: BoundaryCandidate, cfg: LlmConfig) -> BoundaryLabelResult:
    url = cfg.base_url.rstrip('/') + '/api/chat'
    body: dict[str, Any] = {
        'model': cfg.model,
        'stream': False,
        'format': 'json',
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_prompt(candidate)},
        ],
        'options': {
            'temperature': 0,
        },
    }
    resp = requests.post(url, json=body, timeout=cfg.timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    content = data.get('message', {}).get('content', '')
    return parse_llm_json(content)


def _extract_stub_trigger(text: str) -> str:
    for phrase in STUB_TRIGGER_PHRASES:
        if phrase in text:
            return phrase
    return ''


def label_boundary_stub(candidate: BoundaryCandidate) -> BoundaryLabelResult:
    prev_text = candidate.prev_text
    next_text = candidate.next_text
    combined = f'{prev_text} {next_text}'
    trigger = _extract_stub_trigger(combined)

    if trigger:
        return BoundaryLabelResult(
            phase_before='S',
            phase_after='O',
            is_boundary=True,
            confidence=0.78,
            trigger_text=trigger,
            trigger_phrases=[trigger],
            reason='診察開始を示す定型句を検出',
        )

    s_hints = ['いつから', '熱', '痛い', '鼻水', '咳', 'せき', 'どうしました', 'しんどい']
    if any(hint in combined for hint in s_hints):
        return BoundaryLabelResult(
            phase_before='S',
            phase_after='S',
            is_boundary=False,
            confidence=0.62,
            trigger_text='',
            trigger_phrases=[],
            reason='症状確認が継続している可能性が高い',
        )

    return BoundaryLabelResult(
        phase_before='UNKNOWN',
        phase_after='UNKNOWN',
        is_boundary=False,
        confidence=0.35,
        trigger_text='',
        trigger_phrases=[],
        reason='境界の判断材料が少ない',
    )
