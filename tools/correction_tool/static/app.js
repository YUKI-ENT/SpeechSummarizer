const state = {
  files: [],
  sessions: [],
  sessionIndex: 0,
  busy: false,
  config: null,
  currentRules: null,
  suggestions: [],
};

const API_BASE = '/analysis-tools/correction-tool/api';

function qs(id) { return document.getElementById(id); }
function escapeHtml(value) {
  return (value || '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}
function formatDateForInput(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}
function setDefaultDateRange() {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - 7);
  qs('startDate').value = formatDateForInput(start);
  qs('endDate').value = formatDateForInput(end);
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
  ['btnList','btnCheckAll','btnUncheckAll','btnLoadSessions','btnPrev','btnNext','btnSuggest','btnSaveRule','btnPreview','llmModel','promptId','ruleModel'].forEach((id) => {
    const el = qs(id);
    if (el) el.disabled = isBusy;
  });
}
function selectedFiles() {
  return [...document.querySelectorAll('#fileList input[type=checkbox]:checked')].map((el) => state.files[Number(el.dataset.index)].path);
}
function currentSession() {
  return state.sessions[state.sessionIndex];
}
function setWrongText(value) {
  qs('wrongText').value = (value || '').trim();
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
function setFileChecks(checked) {
  document.querySelectorAll('#fileList input[type=checkbox]').forEach((el) => { el.checked = checked; });
}
function renderFiles() {
  const root = qs('fileList');
  root.innerHTML = '';
  state.files.forEach((file, index) => {
    const row = document.createElement('label');
    row.className = 'file-row';
    row.innerHTML = `<input type="checkbox" data-index="${index}" checked><div><strong>${escapeHtml(file.name)}</strong><div class="muted">${escapeHtml(file.session_ts || '')}</div></div><span class="tag">ASR ${file.asr_count}</span>`;
    root.appendChild(row);
  });
}
function renderTranscript() {
  const session = currentSession();
  qs('sessionCount').textContent = state.sessions.length ? `${state.sessionIndex + 1} / ${state.sessions.length}` : '';
  qs('sessionMeta').textContent = session ? `${session.source_file} / detected_model: ${session.detected_model || '-'} / turns: ${session.turns.length}` : '';
  const root = qs('transcriptPane');
  root.innerHTML = '';
  if (!session) {
    root.innerHTML = '<div class="muted">JSONL を開くとここに全文が出ます。</div>';
    return;
  }
  session.turns.forEach((turn) => {
    const div = document.createElement('div');
    div.className = 'turn';
    div.innerHTML = `<div class="turn-meta"><span class="tag">#${turn.index}</span><span class="muted">${escapeHtml(turn.ts || '')}</span><span class="muted">${escapeHtml(turn.model_name || '')}</span></div><div class="turn-body">${escapeHtml(turn.text)}</div>`;
    const turnBody = div.querySelector('.turn-body');
    let selectedText = '';
    turnBody.addEventListener('mousedown', () => { selectedText = ''; });
    turnBody.addEventListener('mouseup', () => {
      selectedText = selectedTurnText(turnBody);
      if (selectedText) {
        setWrongText(selectedText);
      }
    });
    div.addEventListener('click', () => {
      if (selectedText) {
        clearTextSelection();
        selectedText = '';
        return;
      }
      setWrongText(turn.text);
    });
    root.appendChild(div);
  });
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
  const modelName = qs('ruleModel').value;
  const effective = state.currentRules.effective_replacements || {};
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
  state.currentRules = await getJson(`${API_BASE}/rules?model_name=${encodeURIComponent(qs('ruleModel').value || '')}`);
  renderRules();
}
function populateModelOptions(models) {
  ['ruleModel', 'llmModel'].forEach((id) => {
    const select = qs(id);
    select.innerHTML = '';
    models.forEach((name) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name || 'global';
      select.appendChild(option);
    });
  });
}
async function loadConfig() {
  state.config = await getJson(`${API_BASE}/config`);
  qs('dataDir').value = state.config.default_data_dir || '';
  qs('llmInfo').value = `${state.config.llm_model || ''} @ ${state.config.llm_base_url || ''}`.trim();
  setDefaultDateRange();
  const models = [''].concat(state.config.asr_models || []);
  populateModelOptions(models);
  const promptSelect = qs('promptId');
  promptSelect.innerHTML = '';
  (state.config.prompt_items || []).forEach((item) => {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = item.label || item.id;
    if (item.id === state.config.default_prompt_id) option.selected = true;
    promptSelect.appendChild(option);
  });
  try {
    const data = await getJson(`${API_BASE}/ollama_models`);
    if (data.models && data.models.length) {
      const llmModel = qs('llmModel');
      llmModel.innerHTML = '';
      data.models.forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        if (name === data.default_model) option.selected = true;
        llmModel.appendChild(option);
      });
    }
  } catch (err) {
    console.error(err);
  }
  await loadRules();
}

qs('btnCheckAll').onclick = () => setFileChecks(true);
qs('btnUncheckAll').onclick = () => setFileChecks(false);
qs('ruleModel').addEventListener('change', loadRules);

qs('btnList').onclick = async () => {
  setBusy(true, 'jsonl 一覧を読み込み中...');
  try {
    const data = await postJson(`${API_BASE}/files`, { data_dir: qs('dataDir').value, start_date: qs('startDate').value || null, end_date: qs('endDate').value || null });
    state.files = data.items || [];
    renderFiles();
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnLoadSessions').onclick = async () => {
  setBusy(true, 'セッションを読み込み中...');
  try {
    const data = await postJson(`${API_BASE}/sessions`, { file_paths: selectedFiles() });
    state.sessions = data.items || [];
    state.sessionIndex = 0;
    state.suggestions = [];
    renderTranscript();
    renderSuggestions();
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnPrev').onclick = () => {
  if (!state.sessions.length) return;
  state.sessionIndex = Math.max(0, state.sessionIndex - 1);
  state.suggestions = [];
  qs('previewText').textContent = '';
  renderTranscript();
  renderSuggestions();
};
qs('btnNext').onclick = () => {
  if (!state.sessions.length) return;
  state.sessionIndex = Math.min(state.sessions.length - 1, state.sessionIndex + 1);
  state.suggestions = [];
  qs('previewText').textContent = '';
  renderTranscript();
  renderSuggestions();
};

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
      model_name: qs('ruleModel').value,
      wrong: qs('wrongText').value,
      correct: qs('correctText').value,
    });
    await loadRules();
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
  setBusy(true, '補正プレビューを生成中...');
  try {
    const data = await postJson(`${API_BASE}/preview`, {
      model_name: qs('ruleModel').value || session.detected_model || '',
      text: session.turns.map((turn) => turn.text).join('\n'),
    });
    qs('previewText').textContent = data.text || '';
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

loadConfig().catch((err) => {
  console.error(err);
  alert(String(err.message || err));
});
