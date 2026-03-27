const state = {
  sessions: [],
  sessionIndex: 0,
  busy: false,
  config: null,
  currentRules: null,
  suggestions: [],
  currentRuleModelName: '',
  selectionRawText: '',
  selectionCorrectedText: '',
  annotatedTurns: [],
  rawDisplay: false,
};

const API_BASE = '/analysis-tools/correction-tool/api';

function qs(id) { return document.getElementById(id); }
function replaceAllText(value, from, to) {
  return String(value || '').split(from).join(to);
}
function escapeHtml(value) {
  let text = String(value || '');
  text = replaceAllText(text, '&', '&amp;');
  text = replaceAllText(text, '<', '&lt;');
  text = replaceAllText(text, '>', '&gt;');
  return text;
}
function patientLabel(session) {
  const patientId = String(session?.patient_id || '').trim();
  const patientName = String(session?.patient_info?.name || '').trim();
  if (patientId && patientName) return `${patientId} | ${patientName}`;
  return patientId || '-';
}
function formatDateForInput(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}
function setDefaultDate() {
  qs('targetDate').value = formatDateForInput(new Date());
}
async function postJson(url, body) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
async function getJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}
function setBusy(isBusy, message = '') {
  state.busy = isBusy;
  document.body.classList.toggle('is-busy', isBusy);
  qs('busyStatus').textContent = message;
  ['btnPrev','btnNext','btnSuggest','btnSaveRule','btnPreview','btnRestoreRaw','llmModel','promptId','ruleScope','targetDate'].forEach((id) => {
    const el = qs(id);
    if (el) el.disabled = isBusy;
  });
}
function currentSession() {
  return state.sessions[state.sessionIndex];
}
function activeRuleModelName() {
  if (qs('ruleScope').value === 'global') return '';
  const session = currentSession();
  return session ? (session.detected_model || '') : '';
}
function activeRuleScopeLabel() {
  const modelName = activeRuleModelName();
  if (qs('ruleScope').value === 'global') return 'global';
  return modelName || 'model unavailable';
}
function applyCurrentRules(text) {
  let out = String(text || '');
  const replacements = (state.currentRules && state.currentRules.effective_replacements) || {};
  Object.entries(replacements).forEach(([wrong, correct]) => {
    out = replaceAllText(out, wrong, correct);
  });
  return out;
}
function setWrongText(value) {
  qs('wrongText').value = (value || '').trim();
}
function syncSelectionInput() {
  if (!state.selectionRawText) return;
  setWrongText(state.rawDisplay ? state.selectionRawText : state.selectionCorrectedText);
}
function setSelectionSource(rawText, correctedText) {
  state.selectionRawText = (rawText || '').trim();
  state.selectionCorrectedText = (correctedText || applyCurrentRules(state.selectionRawText)).trim();
  syncSelectionInput();
  qs('selectionMeta').textContent = state.selectionRawText && state.selectionRawText !== state.selectionCorrectedText
    ? `原文: ${state.selectionRawText}`
    : '';
}
function clearSelectionSource() {
  state.selectionRawText = '';
  state.selectionCorrectedText = '';
  qs('selectionMeta').textContent = '';
}
function clearTextSelection() {
  const selection = window.getSelection ? window.getSelection() : null;
  if (selection && selection.removeAllRanges) selection.removeAllRanges();
}
function selectedTurnText(turnBody) {
  const selection = window.getSelection ? window.getSelection() : null;
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return '';
  const text = selection.toString().replace(/\s+/g, ' ').trim();
  if (!text) return '';
  const anchorNode = selection.anchorNode;
  const focusNode = selection.focusNode;
  if (!anchorNode || !focusNode) return '';
  if (!turnBody.contains(anchorNode) || !turnBody.contains(focusNode)) return '';
  return text;
}
function currentAnnotatedTurn(turnIndex) {
  return state.annotatedTurns[turnIndex] || null;
}
function visibleTurnEntries(session = currentSession()) {
  if (!session) return [];
  if (state.rawDisplay) {
    return session.turns
      .filter((turn) => Boolean((turn.text || '').trim()))
      .map((turn) => ({ turn, annotatedTurn: null }));
  }
  return session.turns
    .map((turn) => ({ turn, annotatedTurn: currentAnnotatedTurn(turn.index) }))
    .filter(({ turn, annotatedTurn }) => {
      if (!annotatedTurn) return Boolean((turn.text || '').trim());
      return annotatedTurn.visible !== false && Boolean((annotatedTurn.text || '').trim());
    });
}
function renderContinuousSegments(root, entries, clickable = false) {
  root.innerHTML = '';
  const flow = document.createElement('div');
  flow.className = clickable ? 'transcript-flow' : 'preview-flow';
  let selectedText = '';
  if (clickable) {
    flow.addEventListener('mousedown', () => { selectedText = ''; });
    flow.addEventListener('mouseup', () => {
      selectedText = selectedTurnText(flow);
      if (selectedText) setSelectionSource(selectedText, selectedText);
    });
  }
  entries.forEach(({ turn, annotatedTurn }, index) => {
    const chunk = document.createElement('span');
    chunk.className = clickable ? 'transcript-chunk' : 'preview-chunk';
    chunk.dataset.turnIndex = String(turn.index);
    chunk.innerHTML = renderAnnotatedTurnBody(annotatedTurn, turn.text);
    if (clickable) {
      chunk.addEventListener('click', () => {
        if (selectedText) {
          clearTextSelection();
          selectedText = '';
          return;
        }
        setSelectionSource(turn.text, annotatedTurn ? annotatedTurn.text : '');
      });
    }
    flow.appendChild(chunk);
    if (index < entries.length - 1) flow.appendChild(document.createTextNode(' '));
  });
  root.appendChild(flow);
}
function populatePatientSelect() {
  const select = qs('sessionPatientSelect');
  if (!select) return;
  const patients = [];
  const seen = new Set();
  state.sessions.forEach((session) => {
    const patientId = String(session.patient_id || '').trim();
    if (!patientId || seen.has(patientId)) return;
    seen.add(patientId);
    patients.push({
      patientId,
      name: String(session?.patient_info?.name || '').trim(),
    });
  });
  const options = ['<option value="">患者選択</option>'];
  patients.forEach(({ patientId, name }) => {
    const label = name ? `${patientId} | ${name}` : patientId;
    options.push(`<option value="${escapeHtml(patientId)}">${escapeHtml(label)}</option>`);
  });
  select.innerHTML = options.join('');
  syncPatientSelect();
}
function syncPatientSelect() {
  const select = qs('sessionPatientSelect');
  const session = currentSession();
  if (!select) return;
  select.value = session ? String(session.patient_id || '').trim() : '';
}
function renderTranscript() {
  const session = currentSession();
  const visibleEntries = visibleTurnEntries(session);
  syncPatientSelect();
  syncSelectionInput();
  qs('btnRestoreRaw').disabled = state.rawDisplay;
  qs('btnPreview').disabled = !state.rawDisplay;
  qs('sessionCount').textContent = state.sessions.length ? `${state.sessionIndex + 1} / ${state.sessions.length}` : '';
  qs('sessionMeta').textContent = session
    ? `${session.source_file} / patient: ${patientLabel(session)} / detected_model: ${session.detected_model || '-'} / active_scope: ${activeRuleScopeLabel()} / display: ${state.rawDisplay ? 'raw' : 'corrected'} / turns: ${visibleEntries.length}/${session.turns.length}`
    : '';
  const root = qs('transcriptPane');
  root.innerHTML = '';
  if (!session) {
    root.innerHTML = '<div class="muted">JSONL を開くとここに全文が出ます。</div>';
    return;
  }
  if (!visibleEntries.length) {
    root.innerHTML = '<div class="muted">表示対象の ASR テキストがありません。</div>';
    return;
  }
  renderContinuousSegments(root, visibleEntries, true);
}
function renderAnnotatedTurnBody(annotatedTurn, fallbackText) {
  if (!annotatedTurn || !annotatedTurn.segments || !annotatedTurn.segments.length) {
    return escapeHtml(fallbackText || '');
  }
  return annotatedTurn.segments.map((seg) => {
    const text = escapeHtml(seg.text || '');
    return seg.changed ? `<span class="changed-text">${text}</span>` : text;
  }).join('');
}
function renderSuggestions() {
  const root = qs('suggestions');
  root.innerHTML = '';
  if (!state.suggestions.length) {
    root.innerHTML = '<div class="muted">LLM 候補はまだありません。</div>';
    return;
  }
  state.suggestions.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'suggestion-row';
    const sources = (item.sources || []).join(', ');
    row.innerHTML = `<div><strong>${escapeHtml(item.wrong)}</strong><div class="muted">${escapeHtml(item.reason || '')}</div>${sources ? `<div class="muted">出現: ${escapeHtml(sources)}</div>` : ''}</div><div>${escapeHtml(item.correct)}</div><button class="button-secondary">反映</button>`;
    row.querySelector('button').addEventListener('click', () => {
      setWrongText(item.wrong);
      qs('correctText').value = item.correct;
    });
    root.appendChild(row);
  });
}
function renderRules() {
  const root = qs('rulesList');
  root.innerHTML = '';
  if (!state.currentRules) {
    root.innerHTML = '<div class="muted">辞書を読み込むとここに表示します。</div>';
    return;
  }
  const modelName = state.currentRuleModelName;
  const effective = state.currentRules.effective_replacements || {};
  if (qs('ruleScope').value === 'model' && !modelName) {
    root.innerHTML = '<div class="muted">このセッションではモデルが検出できないため、モデル別辞書は表示できません。</div>';
    return;
  }
  const entries = Object.entries(effective).sort((a,b) => a[0].localeCompare(b[0], 'ja'));
  if (!entries.length) {
    root.innerHTML = '<div class="muted">このモデルの辞書はまだ空です。</div>';
    return;
  }
  entries.forEach(([wrong, correct]) => {
    const row = document.createElement('div');
    row.className = 'rule-row';
    row.innerHTML = `<div>${escapeHtml(modelName || 'global')}</div><div>${escapeHtml(wrong)}</div><div>${escapeHtml(correct)}</div><button class="button-secondary">削除</button>`;
    row.querySelector('button').addEventListener('click', async () => {
      setBusy(true, '辞書を削除中...');
      try {
        await postJson(`${API_BASE}/rules/delete`, { model_name: modelName, wrong });
        await loadRules();
      } catch (err) {
        alert(String(err.message || err));
      } finally {
        setBusy(false, '');
      }
    });
    root.appendChild(row);
  });
}
async function loadRules() {
  state.currentRuleModelName = activeRuleModelName();
  if (qs('ruleScope').value === 'model' && !state.currentRuleModelName) {
    state.currentRules = { ok: true, effective_replacements: {} };
  } else {
    state.currentRules = await getJson(`${API_BASE}/rules?model_name=${encodeURIComponent(state.currentRuleModelName)}`);
  }
  await loadAnnotatedTurns();
  renderRules();
  renderTranscript();
  if (state.selectionRawText) setSelectionSource(state.selectionRawText);
}
async function loadAnnotatedTurns() {
  const session = currentSession();
  if (!session) {
    state.annotatedTurns = [];
    return;
  }
  const data = await postJson(`${API_BASE}/annotate`, {
    turns: session.turns,
    model_name: activeRuleModelName(),
  });
  state.annotatedTurns = data.items || [];
}
function populateRuleScopeOptions() {
  const select = qs('ruleScope');
  select.innerHTML = '';
  [
    { value: 'model', label: 'セッション検出モデル' },
    { value: 'global', label: 'Global' },
  ].forEach((item) => {
    const option = document.createElement('option');
    option.value = item.value;
    option.textContent = item.label;
    select.appendChild(option);
  });
  select.value = 'model';
}
function populateLlmModels(models) {
  const select = qs('llmModel');
  select.innerHTML = '';
  models.forEach((name) => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name || 'global';
    select.appendChild(option);
  });
}
async function loadConfig() {
  state.config = await getJson(`${API_BASE}/config`);
  setDefaultDate();
  populateRuleScopeOptions();
  const promptSelect = qs('promptId');
  promptSelect.innerHTML = '';
  (state.config.prompt_items || []).forEach((item) => {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = item.label || item.id;
    if (item.id === state.config.default_prompt_id) option.selected = true;
    promptSelect.appendChild(option);
  });
  populateLlmModels([state.config.llm_model || '']);
  clearSelectionSource();
  await loadRules();

  getJson(`${API_BASE}/ollama_models`)
    .then((data) => {
      if (data.models && data.models.length) {
        populateLlmModels(data.models);
        if (data.default_model) qs('llmModel').value = data.default_model;
      }
    })
    .catch((err) => {
      console.error(err);
    });
}

async function loadSessionsForDate(targetDate) {
  let statusMessage = '';
  if (!state.config) {
    alert('初期設定の読込中です。数秒おいてもう一度お試しください。');
    return;
  }
  setBusy(true, '指定日のセッションを読み込み中...');
  try {
    const files = await postJson(`${API_BASE}/files`, {
      data_dir: state.config.default_data_dir || '',
      start_date: targetDate,
      end_date: targetDate,
    });
    const filePaths = (files.items || []).map((item) => item.path);
    const data = await postJson(`${API_BASE}/sessions`, { file_paths: filePaths });
    state.sessions = data.items || [];
    state.sessionIndex = 0;
    state.suggestions = [];
    state.rawDisplay = false;
    populatePatientSelect();
    clearSelectionSource();
    await loadRules();
    renderSuggestions();
    statusMessage = state.sessions.length
      ? `${targetDate || ''} の ${state.sessions.length} 件を読み込みました`
      : `${targetDate || ''} の jsonl は見つかりませんでした`;
  } catch (err) {
    statusMessage = '';
    alert(String(err.message || err));
  } finally {
    setBusy(false, statusMessage);
  }
}

qs('ruleScope').addEventListener('change', loadRules);

qs('targetDate').addEventListener('change', async () => {
  const targetDate = qs('targetDate').value || null;
  await loadSessionsForDate(targetDate);
});

qs('btnPrev').onclick = () => {
  if (!state.sessions.length) return;
  state.sessionIndex = Math.max(0, state.sessionIndex - 1);
  state.suggestions = [];
  state.rawDisplay = false;
  clearSelectionSource();
  loadRules().catch((err) => alert(String(err.message || err)));
  renderSuggestions();
};
qs('btnNext').onclick = () => {
  if (!state.sessions.length) return;
  state.sessionIndex = Math.min(state.sessions.length - 1, state.sessionIndex + 1);
  state.suggestions = [];
  state.rawDisplay = false;
  clearSelectionSource();
  loadRules().catch((err) => alert(String(err.message || err)));
  renderSuggestions();
};

qs('sessionPatientSelect').addEventListener('change', () => {
  const patientId = (qs('sessionPatientSelect').value || '').trim();
  if (!patientId) return;
  const nextIndex = state.sessions.findIndex((session) => String(session.patient_id || '').trim() === patientId);
  if (nextIndex < 0) return;
  state.sessionIndex = nextIndex;
  state.suggestions = [];
  state.rawDisplay = false;
  clearSelectionSource();
  loadRules().catch((err) => alert(String(err.message || err)));
  renderSuggestions();
});

qs('btnSuggest').onclick = async () => {
  const session = currentSession();
  if (!session) return;
  setBusy(true, 'LLM が補正候補を抽出中...');
  try {
    const data = await postJson(`${API_BASE}/suggest`, {
      source_file: session.source_file,
      turns: session.turns,
      model: qs('llmModel').value,
      prompt_id: qs('promptId').value,
      model_name: activeRuleModelName(),
    });
    state.suggestions = data.items || [];
    renderSuggestions();
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnSaveRule').onclick = async () => {
  setBusy(true, '辞書を保存中...');
  try {
    await postJson(`${API_BASE}/rules/upsert`, {
      model_name: activeRuleModelName(),
      wrong: qs('wrongText').value,
      correct: qs('correctText').value,
    });
    await loadRules();
    clearSelectionSource();
    qs('wrongText').value = '';
    qs('correctText').value = '';
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnPreview').onclick = async () => {
  const session = currentSession();
  if (!session) return;
  setBusy(true, '補正表示に戻しています...');
  try {
    state.rawDisplay = false;
    renderTranscript();
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnRestoreRaw').onclick = () => {
  state.rawDisplay = true;
  renderTranscript();
};

loadConfig()
  .then(() => loadSessionsForDate(qs('targetDate').value || null))
  .catch((err) => {
    console.error(err);
    alert(String(err.message || err));
  });
