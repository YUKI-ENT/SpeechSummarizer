const state = {
  files: [],
  sessions: [],
  reviews: [],
  reviewIndex: 0,
  busy: false,
};

const API_BASE = '/so-labeler/api';

function qs(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return (value || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function formatDateForInput(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function setDefaultDateRange() {
  const end = new Date();
  const start = new Date(end);
  start.setDate(start.getDate() - 7);
  qs('startDate').value = formatDateForInput(start);
  qs('endDate').value = formatDateForInput(end);
}

function splitPhrases(text) {
  return (text || '')
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const raw = await res.text();
    try {
      const parsed = JSON.parse(raw);
      const detail = parsed.detail;
      if (detail && typeof detail === 'object') {
        throw new Error(JSON.stringify(detail, null, 2));
      }
    } catch (parseError) {
      if (parseError instanceof SyntaxError) {
        throw new Error(raw);
      }
      throw parseError;
    }
  }
  return await res.json();
}

function setBusy(isBusy, message = '') {
  state.busy = isBusy;
  document.body.classList.toggle('is-busy', isBusy);
  qs('busyStatus').textContent = message;
  [
    'btnList',
    'btnCheckAll',
    'btnUncheckAll',
    'btnCandidates',
    'btnAnalyze',
    'btnPrev',
    'btnNext',
    'btnApplySuggestion',
    'btnSaveReview',
    'btnExtract',
    'llmMode',
    'llmModel',
    'promptId',
  ].forEach((id) => {
    const el = qs(id);
    if (el) {
      el.disabled = isBusy;
    }
  });
}

async function loadConfig() {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  const cfg = await res.json();
  if (cfg.default_data_dir) {
    qs('dataDir').value = cfg.default_data_dir;
  }
  if (cfg.default_llm_mode) {
    qs('llmMode').value = cfg.default_llm_mode;
  }
  setPromptOptions(cfg.prompt_items || [], cfg.default_prompt_id || '');
  qs('llmInfo').value = `${cfg.llm_model || ''} @ ${cfg.llm_base_url || ''}`.trim();
  setDefaultDateRange();
  await loadOllamaModels(cfg.llm_model || '');
}

function setPromptOptions(items, defaultPromptId) {
  const select = qs('promptId');
  select.innerHTML = '';
  items.forEach((item) => {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = item.label || item.id;
    if (item.id === defaultPromptId) {
      option.selected = true;
    }
    select.appendChild(option);
  });
}

async function loadOllamaModels(defaultModel) {
  const select = qs('llmModel');
  select.innerHTML = '';
  try {
    const res = await fetch(`${API_BASE}/ollama_models`);
    if (!res.ok) {
      throw new Error(await res.text());
    }
    const data = await res.json();
    const models = data.models || [];
    if (!models.length) {
      throw new Error(data.error || 'models not found');
    }
    models.forEach((name) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      if (name === defaultModel || name === data.default_model) {
        option.selected = true;
      }
      select.appendChild(option);
    });
  } catch (err) {
    const option = document.createElement('option');
    option.value = defaultModel;
    option.textContent = defaultModel || 'model unavailable';
    option.selected = true;
    select.appendChild(option);
    console.error('failed to load ollama models', err);
  }
}

function selectedFiles() {
  return [...document.querySelectorAll('#fileList input[type=checkbox]:checked')]
    .map((el) => state.files[Number(el.dataset.index)].path);
}

function setFileChecks(checked) {
  document.querySelectorAll('#fileList input[type=checkbox]').forEach((el) => {
    el.checked = checked;
  });
}

function makeEmptyReview(session) {
  return {
    session_id: session.session_id,
    source_file: session.source_file,
    patient_id: session.patient_id || null,
    turns: session.turns || [],
    llm_mode: '',
    llm_model: '',
    llm_prompt_id: '',
    llm_has_boundary: false,
    llm_boundary_index: null,
    llm_confidence: null,
    llm_trigger_text: '',
    llm_trigger_phrases: [],
    llm_reason: '',
    human_has_boundary: false,
    human_boundary_index: null,
    human_trigger_phrases: [],
    human_note: '',
    human_checked: false,
  };
}

function reviewedItems() {
  return state.reviews.filter((item) => item.human_checked);
}

function boundaryItems() {
  return state.reviews.filter((item) => item.human_has_boundary && item.human_boundary_index !== null);
}

function updateSummary() {
  qs('candidateCount').textContent = String(state.sessions.length);
  qs('reviewedCount').textContent = String(reviewedItems().length);
  qs('boundaryCount').textContent = String(boundaryItems().length);
}

function renderFiles() {
  const root = qs('fileList');
  root.innerHTML = '';
  qs('fileSummary').textContent = `${state.files.length} 件`;

  state.files.forEach((file, index) => {
    const row = document.createElement('label');
    row.className = 'file-row';
    row.innerHTML = `
      <input type="checkbox" data-index="${index}" checked>
      <div class="file-main">
        <strong>${escapeHtml(file.name)}</strong>
        <span class="muted">${escapeHtml(file.session_ts || '')}</span>
      </div>
      <span class="tag">ASR ${file.asr_count}</span>
    `;
    root.appendChild(row);
  });
}

function renderCandidatePreview() {
  const root = qs('candidatePreview');
  root.innerHTML = '';
  const preview = state.sessions.slice(0, 6);
  if (!preview.length) {
    root.innerHTML = '<p class="muted">対象 JSONL を選ぶと、ここに先頭数件を表示します。</p>';
    updateSummary();
    return;
  }

  preview.forEach((session) => {
    const item = document.createElement('article');
    item.className = 'candidate-card';
    item.innerHTML = `
      <div class="candidate-head">
        <strong>${escapeHtml(session.source_file)}</strong>
        <span class="muted">発話数 ${session.turns.length}</span>
      </div>
      <div class="utterance emphasis">
        <span class="utterance-label">冒頭</span>
        <p>${escapeHtml((session.turns || []).slice(0, 4).map((turn) => turn.text).join(' / '))}</p>
      </div>
    `;
    root.appendChild(item);
  });
  updateSummary();
}

function currentReviewRecord() {
  return state.reviews[state.reviewIndex];
}

function syncReviewInputs(record) {
  qs('humanBoundary').value = record.human_has_boundary ? 'yes' : 'no';
  qs('humanTriggerPhrases').value = (record.human_trigger_phrases || []).join(', ');
  qs('humanNote').value = record.human_note || '';
}

function boundaryLabelText(index) {
  return `発話 ${index} の後`;
}

function renderTranscript(turns, record) {
  const humanBoundaryIndex = record.human_boundary_index;
  const llmBoundaryIndex = record.llm_boundary_index;
  return turns.map((turn, idx) => {
    const humanSelected = humanBoundaryIndex === idx;
    const llmSelected = llmBoundaryIndex === idx;
    const nextButton = idx < turns.length - 1
      ? `<button class="boundary-pick ${humanSelected ? 'is-active' : ''}" data-boundary-index="${idx}">
           ここで切替${llmSelected ? ' / LLM候補' : ''}
         </button>`
      : '';
    return `
      <div class="turn-block">
        <div class="turn-row">
          <span class="tag">#${turn.index}</span>
          <span class="muted">${escapeHtml(turn.ts || '')}</span>
        </div>
        <div class="utterance ${humanSelected ? 'emphasis' : ''}">
          <p>${escapeHtml(turn.text)}</p>
        </div>
        ${nextButton}
      </div>
    `;
  }).join('');
}

function renderReview() {
  const pane = qs('reviewPane');
  if (!state.reviews.length) {
    pane.innerHTML = '<p class="muted">JSONL を読み込むと、ここに診察会話全文を表示します。</p>';
    qs('reviewCount').textContent = '';
    updateSummary();
    return;
  }

  const record = currentReviewRecord();
  qs('reviewCount').textContent = `${state.reviewIndex + 1} / ${state.reviews.length}`;
  syncReviewInputs(record);

  const llmBoundaryText = record.llm_boundary_index === null ? '未提案' : boundaryLabelText(record.llm_boundary_index);
  const humanBoundaryText = record.human_boundary_index === null ? '未確定' : boundaryLabelText(record.human_boundary_index);
  const suggestedPhrases = (record.llm_trigger_phrases || []).join(', ');

  pane.innerHTML = `
    <div class="review-meta">
      <span class="tag">${escapeHtml(record.source_file)}</span>
      <span class="tag">発話数 ${record.turns.length}</span>
      <span class="tag">LLM候補 ${escapeHtml(llmBoundaryText)}</span>
      <span class="tag">人手確定 ${escapeHtml(humanBoundaryText)}</span>
    </div>
    <section class="llm-panel">
      <h3>LLM 仮判定</h3>
      <div class="llm-grid">
        <div><span class="muted">モード</span><strong>${escapeHtml(record.llm_mode || '-')}</strong></div>
        <div><span class="muted">モデル</span><strong>${escapeHtml(record.llm_model || '-')}</strong></div>
        <div><span class="muted">prompt</span><strong>${escapeHtml(record.llm_prompt_id || '-')}</strong></div>
        <div><span class="muted">候補位置</span><strong>${escapeHtml(llmBoundaryText)}</strong></div>
        <div><span class="muted">信頼度</span><strong>${record.llm_confidence ?? ''}</strong></div>
      </div>
      <p><span class="muted">trigger_text</span> ${escapeHtml(record.llm_trigger_text || 'なし')}</p>
      <p><span class="muted">trigger_phrases</span> ${escapeHtml(suggestedPhrases || 'なし')}</p>
      <p class="muted">${escapeHtml(record.llm_reason || '')}</p>
    </section>
    <section class="speech-column transcript-column">
      <h3>診察会話全文</h3>
      <p class="muted">各ボタンの「ここで切替」を押すと、人手の区切り位置をその場所に設定します。</p>
      <div class="transcript-list">${renderTranscript(record.turns, record)}</div>
    </section>
  `;

  pane.querySelectorAll('[data-boundary-index]').forEach((button) => {
    button.addEventListener('click', () => {
      record.human_has_boundary = true;
      record.human_boundary_index = Number(button.dataset.boundaryIndex);
      syncReviewInputs(record);
      renderReview();
    });
  });
  updateSummary();
}

function updateCurrentReviewFromInputs() {
  if (!state.reviews.length) {
    return;
  }
  const record = currentReviewRecord();
  record.human_has_boundary = qs('humanBoundary').value === 'yes';
  if (!record.human_has_boundary) {
    record.human_boundary_index = null;
  }
  record.human_trigger_phrases = splitPhrases(qs('humanTriggerPhrases').value);
  record.human_note = qs('humanNote').value.trim();
}

qs('humanBoundary').addEventListener('change', () => {
  updateCurrentReviewFromInputs();
  renderReview();
});
qs('humanTriggerPhrases').addEventListener('input', updateCurrentReviewFromInputs);
qs('humanNote').addEventListener('input', updateCurrentReviewFromInputs);
qs('btnCheckAll').onclick = () => setFileChecks(true);
qs('btnUncheckAll').onclick = () => setFileChecks(false);

qs('btnList').onclick = async () => {
  setBusy(true, 'jsonl 一覧を読み込み中...');
  try {
    const body = {
      data_dir: qs('dataDir').value,
      start_date: qs('startDate').value || null,
      end_date: qs('endDate').value || null,
    };
    const data = await postJson(`${API_BASE}/files`, body);
    state.files = data.items;
    renderFiles();
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnCandidates').onclick = async () => {
  setBusy(true, '会話全文を読み込み中...');
  try {
    const body = {
      file_paths: selectedFiles(),
    };
    const data = await postJson(`${API_BASE}/boundaries`, body);
    state.sessions = data.items;
    state.reviews = state.sessions.map((session) => makeEmptyReview(session));
    state.reviewIndex = 0;
    renderCandidatePreview();
    renderReview();
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnAnalyze').onclick = async () => {
  if (!state.reviews.length) {
    alert('先に会話全文を読み込んでください');
    return;
  }
  const current = currentReviewRecord();
  const session = {
    session_id: current.session_id,
    source_file: current.source_file,
    patient_id: current.patient_id,
    turns: current.turns,
  };
  setBusy(true, `現在の 1 診察を LLM へ送信中... (${state.reviewIndex + 1}/${state.reviews.length})`);
  try {
    const body = {
      sessions: [session],
      mode: qs('llmMode').value,
      model: qs('llmModel').value,
      prompt_id: qs('promptId').value,
    };
    const data = await postJson(`${API_BASE}/label_boundaries`, body);
    if (data.items && data.items[0]) {
      const existing = state.reviews[state.reviewIndex];
      const reviewed = data.items[0];
      state.reviews[state.reviewIndex] = {
        ...existing,
        ...reviewed,
        human_has_boundary: existing.human_checked ? existing.human_has_boundary : reviewed.human_has_boundary,
        human_boundary_index: existing.human_checked ? existing.human_boundary_index : reviewed.human_boundary_index,
        human_trigger_phrases: existing.human_checked ? existing.human_trigger_phrases : reviewed.human_trigger_phrases,
        human_note: existing.human_checked ? existing.human_note : reviewed.human_note,
        human_checked: existing.human_checked,
      };
    }
    renderReview();
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnPrev').onclick = () => {
  if (!state.reviews.length) {
    return;
  }
  updateCurrentReviewFromInputs();
  state.reviewIndex = Math.max(0, state.reviewIndex - 1);
  renderReview();
};

qs('btnNext').onclick = () => {
  if (!state.reviews.length) {
    return;
  }
  updateCurrentReviewFromInputs();
  state.reviewIndex = Math.min(state.reviews.length - 1, state.reviewIndex + 1);
  renderReview();
};

qs('btnApplySuggestion').onclick = () => {
  if (!state.reviews.length) {
    return;
  }
  const record = currentReviewRecord();
  if (record.llm_boundary_index !== null) {
    record.human_has_boundary = true;
    record.human_boundary_index = record.llm_boundary_index;
  }
  qs('humanTriggerPhrases').value = (record.llm_trigger_phrases || []).join(', ');
  updateCurrentReviewFromInputs();
  renderReview();
};

qs('btnSaveReview').onclick = async () => {
  if (!state.reviews.length) {
    return;
  }
  setBusy(true, 'レビューを保存中...');
  try {
    updateCurrentReviewFromInputs();
    const record = structuredClone(currentReviewRecord());
    record.human_checked = true;
    await postJson(`${API_BASE}/save_review`, {
      output_path: qs('outputPath').value,
      record,
    });
    state.reviews[state.reviewIndex] = record;
    renderReview();
    alert('保存しました');
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

qs('btnExtract').onclick = async () => {
  setBusy(true, 'トリガー辞書を抽出中...');
  try {
    const data = await postJson(`${API_BASE}/extract_rules`, {
      annotation_path: qs('annotationPath').value,
      output_path: qs('rulesPath').value,
      min_count: Number(qs('minCount').value),
    });
    qs('rulesPreview').textContent = JSON.stringify(data.rules, null, 2);
  } catch (err) {
    alert(String(err.message || err));
  } finally {
    setBusy(false, '');
  }
};

loadConfig()
  .then(() => {
    renderCandidatePreview();
    renderReview();
  })
  .catch((err) => {
    console.error('failed to load so_labeler config', err);
  });
