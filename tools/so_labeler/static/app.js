const state = {
  files: [],
  candidates: [],
  reviews: [],
  reviewIndex: 0,
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
    throw new Error(await res.text());
  }
  return await res.json();
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
  qs('llmInfo').value = `${cfg.llm_model || ''} @ ${cfg.llm_base_url || ''}`.trim();
  setDefaultDateRange();
}

function selectedFiles() {
  return [...document.querySelectorAll('#fileList input[type=checkbox]:checked')]
    .map((el) => state.files[Number(el.dataset.index)].path);
}

function reviewedItems() {
  return state.reviews.filter((item) => item.human_checked);
}

function boundaryItems() {
  return state.reviews.filter((item) => item.human_is_boundary);
}

function updateSummary() {
  qs('candidateCount').textContent = String(state.candidates.length);
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
  const preview = state.candidates.slice(0, 8);
  if (!preview.length) {
    root.innerHTML = '<p class="muted">候補を作ると、ここに先頭数件を表示します。</p>';
    updateSummary();
    return;
  }

  preview.forEach((candidate) => {
    const item = document.createElement('article');
    item.className = 'candidate-card';
    item.innerHTML = `
      <div class="candidate-head">
        <strong>${escapeHtml(candidate.source_file)}</strong>
        <span class="muted">${escapeHtml(candidate.prev_ts || '')} -> ${escapeHtml(candidate.next_ts || '')}</span>
      </div>
      <div class="utterance">
        <span class="utterance-label">直前</span>
        <p>${escapeHtml(candidate.prev_text)}</p>
      </div>
      <div class="utterance emphasis">
        <span class="utterance-label">直後</span>
        <p>${escapeHtml(candidate.next_text)}</p>
      </div>
    `;
    root.appendChild(item);
  });
  updateSummary();
}

function reviewRecordAt(index) {
  return state.reviews[index];
}

function syncReviewInputs(record) {
  qs('humanBoundary').value = record.human_is_boundary ? 'yes' : 'no';
  qs('humanTriggerPhrases').value = (record.human_trigger_phrases || []).join(', ');
  qs('humanNote').value = record.human_note || '';
}

function renderReview() {
  const pane = qs('reviewPane');
  if (!state.reviews.length) {
    pane.innerHTML = '<p class="muted">LLM 仮判定を作成すると、ここで 1 件ずつ確認できます。</p>';
    qs('reviewCount').textContent = '';
    updateSummary();
    return;
  }

  const record = reviewRecordAt(state.reviewIndex);
  qs('reviewCount').textContent = `${state.reviewIndex + 1} / ${state.reviews.length}`;
  syncReviewInputs(record);

  const llmBoundaryText = record.llm_is_boundary ? 'S→O 境界あり' : '境界なし';
  const suggestedPhrases = (record.llm_trigger_phrases || []).join(', ');
  const beforeContext = (record.context_before || []).map((text) => `<li>${escapeHtml(text)}</li>`).join('');
  const afterContext = (record.context_after || []).map((text) => `<li>${escapeHtml(text)}</li>`).join('');

  pane.innerHTML = `
    <div class="review-meta">
      <span class="tag">${escapeHtml(record.source_file)}</span>
      <span class="tag">index ${record.event_index}</span>
      <span class="tag">${escapeHtml(record.prev_ts || '')} -> ${escapeHtml(record.next_ts || '')}</span>
    </div>
    <div class="review-grid">
      <section class="speech-column">
        <h3>直前までの流れ</h3>
        <ul class="context-list">${beforeContext || '<li class="muted">なし</li>'}</ul>
        <div class="utterance">
          <span class="utterance-label">境界の直前</span>
          <p>${escapeHtml(record.prev_text)}</p>
        </div>
      </section>
      <section class="speech-column">
        <h3>境界の直後</h3>
        <div class="utterance emphasis">
          <span class="utterance-label">境界の直後</span>
          <p>${escapeHtml(record.next_text)}</p>
        </div>
        <h3>続きの流れ</h3>
        <ul class="context-list">${afterContext || '<li class="muted">なし</li>'}</ul>
      </section>
    </div>
    <section class="llm-panel">
      <h3>LLM 仮判定</h3>
      <div class="llm-grid">
        <div><span class="muted">モード</span><strong>${escapeHtml(record.llm_mode)}</strong></div>
        <div><span class="muted">判定</span><strong>${llmBoundaryText}</strong></div>
        <div><span class="muted">信頼度</span><strong>${record.llm_confidence ?? ''}</strong></div>
        <div><span class="muted">推定遷移</span><strong>${escapeHtml(record.llm_phase_before)} -> ${escapeHtml(record.llm_phase_after)}</strong></div>
      </div>
      <p><span class="muted">trigger_text</span> ${escapeHtml(record.llm_trigger_text || 'なし')}</p>
      <p><span class="muted">trigger_phrases</span> ${escapeHtml(suggestedPhrases || 'なし')}</p>
      <p class="muted">${escapeHtml(record.llm_reason || '')}</p>
    </section>
  `;
  updateSummary();
}

function updateCurrentReviewFromInputs() {
  if (!state.reviews.length) {
    return;
  }
  const record = reviewRecordAt(state.reviewIndex);
  record.human_is_boundary = qs('humanBoundary').value === 'yes';
  record.human_trigger_phrases = splitPhrases(qs('humanTriggerPhrases').value);
  record.human_note = qs('humanNote').value.trim();
}

qs('humanBoundary').addEventListener('change', updateCurrentReviewFromInputs);
qs('humanTriggerPhrases').addEventListener('input', updateCurrentReviewFromInputs);
qs('humanNote').addEventListener('input', updateCurrentReviewFromInputs);

qs('btnList').onclick = async () => {
  const body = {
    data_dir: qs('dataDir').value,
    start_date: qs('startDate').value || null,
    end_date: qs('endDate').value || null,
  };
  const data = await postJson(`${API_BASE}/files`, body);
  state.files = data.items;
  renderFiles();
};

qs('btnCandidates').onclick = async () => {
  const body = {
    file_paths: selectedFiles(),
    context_size: Number(qs('contextSize').value),
    max_candidates_per_file: Number(qs('maxCandidates').value),
  };
  const data = await postJson(`${API_BASE}/boundaries`, body);
  state.candidates = data.items;
  state.reviews = [];
  state.reviewIndex = 0;
  renderCandidatePreview();
  renderReview();
};

qs('btnAnalyze').onclick = async () => {
  if (!state.candidates.length) {
    alert('先に境界候補を作成してください');
    return;
  }
  const body = {
    candidates: state.candidates,
    mode: qs('llmMode').value,
  };
  const data = await postJson(`${API_BASE}/label_boundaries`, body);
  state.reviews = data.items;
  state.reviewIndex = 0;
  renderReview();
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
  const record = reviewRecordAt(state.reviewIndex);
  qs('humanTriggerPhrases').value = (record.llm_trigger_phrases || []).join(', ');
  updateCurrentReviewFromInputs();
};

qs('btnSaveReview').onclick = async () => {
  if (!state.reviews.length) {
    return;
  }
  updateCurrentReviewFromInputs();
  const record = structuredClone(reviewRecordAt(state.reviewIndex));
  record.human_checked = true;
  await postJson(`${API_BASE}/save_review`, {
    output_path: qs('outputPath').value,
    record,
  });
  state.reviews[state.reviewIndex] = record;
  renderReview();
  alert('保存しました');
};

qs('btnExtract').onclick = async () => {
  const data = await postJson(`${API_BASE}/extract_rules`, {
    annotation_path: qs('annotationPath').value,
    output_path: qs('rulesPath').value,
    min_count: Number(qs('minCount').value),
  });
  qs('rulesPreview').textContent = JSON.stringify(data.rules, null, 2);
};

loadConfig()
  .then(() => {
    renderCandidatePreview();
    renderReview();
  })
  .catch((err) => {
    console.error('failed to load so_labeler config', err);
  });
