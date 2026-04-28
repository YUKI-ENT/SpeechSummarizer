/**
 * SpeechSummarizer – common.js  (Phase 1: カルテビューUI)
 * ★ 変更点:
 *   - 左サイドバー（受診履歴）+ 右タイムラインカード表示
 *   - ASR欄はデフォルト折りたたみ (<details>)
 *   - LLM欄に draft/approved バッジ + [確定]ボタン
 *   - カルテNo手動入力 → /api/patient/switch
 *   - 難聴モードは hearingOverlay で現在ライブASRを拡大表示
 *   - 録音・ASRモデル切替など既存機能を維持
 */
(() => {
  // ===== DOM =====
  const appVersionEl = document.getElementById('appVersion');
  const logEl = document.getElementById('log');
  const levelEl = document.getElementById('level');
  const btnRec = document.getElementById('btnRec');
  const selAsrModel = document.getElementById('selAsrModel');
  // const chkHearing = document.getElementById('chkHearing');
  const hearingOverlay = document.getElementById('hearingOverlay');
  const hearingTextEl = document.getElementById('hearingText');
  const btnHearingToggle = document.getElementById('btnHearingToggle');
  const btnHearingClose = document.getElementById('btnHearingClose');
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');

  const patientSelectEl = document.getElementById('patientSelect');
  const patientInputEl = document.getElementById('patientInput');
  const btnSwitchPatient = document.getElementById('btnSwitchPatient');

  const sidebarListEl = document.getElementById('sidebarList');
  const sidebarPidEl = document.getElementById('sidebarPid');
  const karteTimeline = document.getElementById('karteTimeline');


  // 非表示でも参照のみ使う要素（既存ロジック互換）
  const selLlmModel = document.getElementById('selLlmModel');
  const selSoapPrompt = document.getElementById('selSoapPrompt');
  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');

  // ===== State =====
  let ws = null;
  let audioCtx = null, srcNode = null, procNode = null, stream = null;
  let isRecording = false;
  let recordingWatchdogTimer = null;
  let lastAudioProcessAt = 0;
  let lastAudioChunkSentAt = 0;
  let recoveringAudio = false;

  let currentPatientId = '';   // 現在選択中の患者ID
  let currentSessionTxt = '';   // 現在録音中 / 選択中のセッション(.txt)名
  let liveAsrText = '';   // ライブASRの累積テキスト
  let recentPatientIds = [];   // 新しい順・重複なし
  let hearingZoom = 1.0;
  let suppressHearingInput = false;
  let hearingSaveTimer = null;
  let hearingLastSavedSession = '';
  let hearingLastSavedText = '';
  let patientSessionRefreshTimer = null;


  // ===== ログ =====
  function log(s) {
    if (!logEl) return;
    logEl.textContent += s + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }

  // ===== ログ折りたたみ =====
  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('btnToggleLog');
    document.body.classList.add('log-collapsed');

    function setLogVisible(show) {
      document.body.classList.toggle('log-collapsed', !show);
      document.body.classList.toggle('log-shown', show);
      if (btn) btn.setAttribute('aria-pressed', show ? 'true' : 'false');
    }
    if (btn) btn.addEventListener('click', () => {
      setLogVisible(document.body.classList.contains('log-collapsed'));
    });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') setLogVisible(false); });
  });

  // ===== 難聴モード =====

  function setHearingMode(on) {

    document.body.classList.toggle('hearing', on);

    if (hearingOverlay)
      hearingOverlay.classList.toggle('visible', on);

    // ボタンの状態
    if (btnHearingToggle)
      btnHearingToggle.setAttribute(
        'aria-pressed',
        on ? 'true' : 'false'
      );

    if (on) {
      applyHearingZoom();
    }
  }

  function applyHearingZoom() {
    if (hearingTextEl) {
      hearingTextEl.style.setProperty('--hearing-zoom', hearingZoom);
    }
  }

  function syncCurrentCardAsr(text) {
    if (!currentSessionTxt || !karteTimeline) return;
    const preEl = karteTimeline.querySelector(`#asr-${CSS.escape(currentSessionTxt)}`);
    if (preEl) preEl.textContent = text;
  }

  async function saveHearingTranscript(sessionTxt, text) {
    const normalized = (text || '').trim();
    if (!sessionTxt) return;
    if (sessionTxt === hearingLastSavedSession && normalized === hearingLastSavedText) return;
    try {
      const r = await fetch('/api/session/transcript', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: sessionTxt, text: normalized })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));
      hearingLastSavedSession = sessionTxt;
      hearingLastSavedText = normalized;
    } catch (e) {
      log(`[hearing] save failed: ${e}`);
    }
  }

  function scheduleHearingTranscriptSave() {
    if (!hearingTextEl || !currentSessionTxt) return;
    const sessionTxt = currentSessionTxt;
    const text = hearingTextEl.value || '';
    clearTimeout(hearingSaveTimer);
    hearingSaveTimer = setTimeout(() => {
      hearingSaveTimer = null;
      saveHearingTranscript(sessionTxt, text);
    }, 800);
  }

  async function flushHearingTranscriptSave() {
    if (hearingSaveTimer) {
      clearTimeout(hearingSaveTimer);
      hearingSaveTimer = null;
    }
    if (!hearingTextEl || !currentSessionTxt) return;
    await saveHearingTranscript(currentSessionTxt, hearingTextEl.value || '');
  }

  if (btnZoomIn) {
    btnZoomIn.onclick = () => {
      hearingZoom = Math.min(hearingZoom + 0.1, 3.0);
      applyHearingZoom();
    };
  }
  if (btnZoomOut) {
    btnZoomOut.onclick = () => {
      hearingZoom = Math.max(hearingZoom - 0.1, 0.5);
      applyHearingZoom();
    };
  }

  if (hearingTextEl) {
    hearingTextEl.addEventListener('input', () => {
      if (suppressHearingInput) return;
      liveAsrText = hearingTextEl.value || '';
      syncCurrentCardAsr(liveAsrText);
      scheduleHearingTranscriptSave();
    });
    hearingTextEl.addEventListener('blur', () => {
      flushHearingTranscriptSave();
    });
  }

  function updateHearingText(text) {
    if (!hearingTextEl) return;
    const next = text || '';
    // 同一内容なら何もしない
    if (hearingTextEl.value === next)
      return;
    const isFocused = document.activeElement === hearingTextEl;
    const prevLength = hearingTextEl.value.length;
    const selStart = isFocused ? hearingTextEl.selectionStart : null;
    const selEnd = isFocused ? hearingTextEl.selectionEnd : null;
    // 下端にいる場合のみ自動スクロール
    const nearBottom =
      hearingTextEl.scrollHeight -
      hearingTextEl.scrollTop -
      hearingTextEl.clientHeight < 80;

    const prevScrollTop = hearingTextEl.scrollTop;
    suppressHearingInput = true;
    try {
      hearingTextEl.value = next;

      if (nearBottom) {
        hearingTextEl.scrollTop =
          hearingTextEl.scrollHeight;
      } else {
        hearingTextEl.scrollTop = prevScrollTop;
      }

      if (isFocused && selStart !== null && selEnd !== null) {
        const appended = Math.max(0, next.length - prevLength);
        const nextSelStart = Math.min(next.length, selStart + appended);
        const nextSelEnd = Math.min(next.length, selEnd + appended);
        hearingTextEl.setSelectionRange(nextSelStart, nextSelEnd);
      }
    } finally {
      suppressHearingInput = false;
    }
  }

  // トグル
  function toggleHearing() {
    const on = !document.body.classList.contains('hearing');
    setHearingMode(on);
  }


  // 耳ボタン
  if (btnHearingToggle) {
    btnHearingToggle.addEventListener('click', (e) => {
      e.preventDefault();
      toggleHearing();
    });
  }


  // ×ボタン
  if (btnHearingClose) {
    btnHearingClose.addEventListener('click', (e) => {
      e.preventDefault();
      setHearingMode(false);
    });
  }


  // 背景クリックで閉じる
  if (hearingOverlay) {
    hearingOverlay.addEventListener('click', (e) => {
      if (e.target === hearingOverlay)
        setHearingMode(false);
    });
  }


  // Escで閉じる
  document.addEventListener('keydown', (e) => {

    if (e.key !== 'Escape')
      return;

    if (document.body.classList.contains('hearing'))
      setHearingMode(false);

  });

  // ===== 録音状態UI =====
  function renderRecState() {
    if (!btnRec) return;
    btnRec.classList.toggle('recording', isRecording);
  }

  // ===== API helpers =====
  async function loadAppVersion() {
    if (!appVersionEl) return;
    try {
      const r = await fetch('/api/version');
      const j = await r.json();
      appVersionEl.textContent = (j && j.ok && j.version) ? j.version : '-';
    } catch { appVersionEl.textContent = '-'; }
  }

  async function loadAsrModels() {
    if (!selAsrModel) return;
    try {
      const r = await fetch('/api/asr/models');
      const j = await r.json();
      selAsrModel.innerHTML = '<option value="">ASR model...</option>';
      (j.models || []).forEach(m => {
        const o = document.createElement('option');
        o.value = m.id; o.textContent = m.label;
        selAsrModel.appendChild(o);
      });
      if (j.current) selAsrModel.value = j.current;
    } catch (e) { log(`[ui] loadAsrModels failed: ${e}`); }
  }

  // LLMモデル一覧（カード内ドロップダウン用）
  let llmModelList = [];
  async function loadLlmModels() {
    try {
      const r = await fetch('/api/llm/models');
      const j = await r.json();
      llmModelList = j.ok ? (j.models || []) : [];
      // hidden select も更新
      if (selLlmModel) {
        selLlmModel.innerHTML = '<option value="">LLM model...</option>' +
          llmModelList.map(m => `<option value="${m}">${m}</option>`).join('');
        if (j.default_model) selLlmModel.value = j.default_model;
      }
    } catch (e) { log(`[ui] loadLlmModels failed: ${e}`); }
  }

  // LLMプロンプト一覧（カード内ドロップダウン用）
  let llmPromptList = [];
  let llmDefaultPromptId = '';
  async function loadLlmPrompts() {
    try {
      const r = await fetch('/api/llm/prompts');
      const j = await r.json();
      if (!j || !j.ok) return;
      llmPromptList = j.items || [];
      llmDefaultPromptId = j.default_prompt_id || '';
      if (selSoapPrompt) {
        selSoapPrompt.innerHTML = '<option value="">Prompt...</option>' +
          llmPromptList.map(it => `<option value="${it.id}">${it.label || it.id}</option>`).join('');
        if (llmDefaultPromptId) selSoapPrompt.value = llmDefaultPromptId;
      }
    } catch (e) { log(`[ui] loadLlmPrompts failed: ${e}`); }
  }

  async function loadRecentPatients() {
    if (!patientSelectEl) return;

    try {
      const r = await fetch('/api/sessions');
      const j = await r.json();
      const items = Array.isArray(j?.items) ? j.items : [];

      const seen = new Set();
      recentPatientIds = [];
      const pidToName = {};

      for (const it of items) {
        const pid = String(it?.patient_id || '').trim();
        if (!pid || seen.has(pid)) continue;
        seen.add(pid);
        recentPatientIds.push(pid);
        if (it?.patient_info?.name) {
          pidToName[pid] = it.patient_info.name;
        }
      }

      const optionHtml = recentPatientIds
        .map(pid => {
          const label = pidToName[pid]
            ? `${e_(pid)} | ${e_(pidToName[pid])}`
            : `${e_(pid)}`;
          return `<option value="${e_(pid)}">${label}</option>`;
        })
        .join('');

      patientSelectEl.innerHTML = `<option value="">履歴</option>${optionHtml}`;
      if (currentPatientId) {
        patientSelectEl.value = recentPatientIds.includes(currentPatientId) ? currentPatientId : '';
      }

      log(`[patient] recent list loaded: ${recentPatientIds.length}`);
    } catch (e) {
      log(`[patient] loadRecentPatients failed: ${e}`);
    }
  }

  // ===== 患者セッション一覧の読み込み →サイドバー + タイムライン描画 =====
  async function loadPatientSessions(pid, opts = {}) {
    if (!pid) return;
    try {
      // APIからセッションと患者情報を取得
      const [rSessions, rInfo] = await Promise.all([
        fetch(`/api/patient/${encodeURIComponent(pid)}/sessions`),
        fetch(`/api/patient/${encodeURIComponent(pid)}/info`).then(res => res.ok ? res.json() : null).catch(() => null)
      ]);
      const jSessions = await rSessions.json();
      if (!jSessions || !jSessions.ok) throw new Error(jSessions?.error || 'failed');

      const pInfo = (rInfo && rInfo.ok) ? rInfo.patient_info : null;
      renderSidebar(pid, jSessions.sessions || [], pInfo);
      renderTimeline(jSessions.sessions || [], opts);
    } catch (e) {
      log(`[patient] loadPatientSessions failed: ${e}`);
    }
  }

  function calculateAge(dobString) {
    if (!dobString) return '';
    // Expected dobString format: yyyy/mm/dd or yyyy-mm-dd
    const parts = dobString.split(/[\/\-]/);
    if (parts.length !== 3) return '';
    const birthDate = new Date(parts[0], parts[1] - 1, parts[2]);
    const today = new Date();
    let age = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    return age;
  }

  function formatGender(g) {
    if (g === '1') return '男';
    if (g === '2') return '女';
    return g || '-';
  }

  // ===== サイドバー描画 =====
  function renderSidebar(pid, sessions, pInfo) {
    if (sidebarPidEl) sidebarPidEl.textContent = pid || '-';

    // 患者情報の描画
    const infoEl = document.getElementById('sidebarPatientInfo');
    if (infoEl) {
      if (pInfo) {
        const _name = e_(pInfo.name) || '-';
        const _gender = formatGender(pInfo.gender);
        const _age = calculateAge(pInfo.dob);
        const _ageStr = _age !== '' ? `${_age}歳` : '-';
        const _dob = e_(pInfo.dob) || '-';

        infoEl.innerHTML = `
          <div class="info-row"><span class="info-label">ID</span><span class="info-val">${pid}</span></div>
          <div class="info-row"><span class="info-label">氏名</span><span class="info-val">${_name}</span></div>
          <div class="info-row"><span class="info-label">生年月日</span><span class="info-val">${_dob}</span></div>
          <div class="info-row"><span class="info-label">性別</span><span class="info-val">${_gender}</span></div>
          <div class="info-row"><span class="info-label">年齢</span><span class="info-val">${_ageStr}</span></div>
        `;
        infoEl.style.display = 'block';
      } else {
        infoEl.style.display = 'none';
        infoEl.innerHTML = '';
      }
    }

    if (!sidebarListEl) return;

    if (!sessions.length) {
      sidebarListEl.innerHTML = '<div class="muted2 sidebar-empty">セッションなし</div>';
      return;
    }

    sidebarListEl.innerHTML = sessions.map(s => {
      const isCurrent = s.is_current;
      const hasApproved = !!s.approved_file;
      const hasLlm = s.llm_files && s.llm_files.length > 0;
      const icon = isCurrent ? '🎙' : (hasApproved ? '✅' : (hasLlm ? '📄' : '📝'));
      const cls = isCurrent ? 'sidebar-item sidebar-item--current' : 'sidebar-item';
      return `<div class="${cls}" data-session="${e_(s.session_txt)}" title="${e_(s.session_txt)}">
        <span class="sidebar-icon">${icon}</span>
        <span class="sidebar-date">${e_(s.date)}</span>
        <span class="sidebar-time">${e_(s.time)}</span>
      </div>`;
    }).join('');

    // サイドバーアイテムクリックでタイムラインにスクロール
    sidebarListEl.querySelectorAll('.sidebar-item').forEach(el => {
      el.addEventListener('click', () => {
        const sid = el.dataset.session;
        const card = karteTimeline?.querySelector(`[data-session="${sid}"]`);
        if (card) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });
  }

  // HTML escapeユーティリティ
  function e_(s) {
    return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function parseSessionStamp(sessionTxt) {
    const m = String(sessionTxt || '').match(/^(.+?)_(\d{8}_\d{6})\.txt$/);
    if (!m) return null;
    return { patientId: m[1], stamp: m[2] };
  }

  function buildOptimisticSession(sessionTxt) {
    const parsed = parseSessionStamp(sessionTxt);
    if (!parsed) return null;
    const { stamp } = parsed;
    return {
      session_txt: sessionTxt,
      stamp,
      date: `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}`,
      time: `${stamp.slice(9, 11)}:${stamp.slice(11, 13)}`,
      is_current: true,
      llm_files: [],
      approved_file: '',
    };
  }

  function ensureRecentPatientOption(pid) {
    pid = String(pid || '').trim();
    if (!pid || !patientSelectEl) return;

    recentPatientIds = [pid, ...recentPatientIds.filter(x => x !== pid)];

    const optionMap = new Map();
    Array.from(patientSelectEl.querySelectorAll('option')).forEach(opt => {
      if (opt.value) optionMap.set(opt.value, opt.textContent || opt.value);
    });
    if (!optionMap.has(pid)) optionMap.set(pid, pid);

    const optionHtml = recentPatientIds
      .filter(x => optionMap.has(x))
      .map(x => `<option value="${e_(x)}">${e_(optionMap.get(x) || x)}</option>`)
      .join('');

    patientSelectEl.innerHTML = `<option value="">履歴</option>${optionHtml}`;
    patientSelectEl.value = pid;
  }

  function renderOptimisticPatientSwitch(pid, sessionTxt, patientInfo) {
    const session = buildOptimisticSession(sessionTxt);
    if (!pid || !session) return;
    renderSidebar(pid, [session], patientInfo || null);
    renderTimeline([session], {});
  }

  function hasSessionCard(sessionTxt) {
    if (!karteTimeline || !sessionTxt) return false;
    return !!karteTimeline.querySelector(`.karte-card[data-session="${CSS.escape(sessionTxt)}"]`);
  }

  function schedulePatientSessionRefresh(pid, opts = {}) {
    pid = String(pid || '').trim();
    if (!pid) return;
    clearTimeout(patientSessionRefreshTimer);
    patientSessionRefreshTimer = setTimeout(() => {
      patientSessionRefreshTimer = null;
      loadPatientSessions(pid, opts)
        .then(() => loadAllCardAsrText())
        .then(() => loadRecentPatients())
        .catch(e => log(`[patient] refresh failed: ${e}`));
    }, 120);
  }

  // ===== タイムライン描画 =====
  function renderTimeline(sessions, opts = {}) {
    if (!karteTimeline) return;

    if (!sessions.length) {
      karteTimeline.innerHTML = '<div class="karte-empty muted2">受診履歴がありません</div>';
      return;
    }

    karteTimeline.innerHTML = sessions.map(s => buildCard(s)).join('');

    // カード内イベント登録
    karteTimeline.querySelectorAll('.karte-card').forEach(card => {
      const session = card.dataset.session;

      // LLM送信ボタン
      card.querySelectorAll('.btn-llm-run').forEach(btn => {
        btn.addEventListener('click', () => runLlmForCard(card, session));
      });

      // 確定ボタン
      card.querySelectorAll('.btn-approve-cta').forEach(btn => {
        btn.addEventListener('click', async () => {
          const llmFile = btn.dataset.llmFile;
          await approveFile(card, session, llmFile);
        });
      });

      // LLM履歴セレクトボックス変更
      card.querySelectorAll('.llm-history-select').forEach(sel => {
        sel.addEventListener('change', async () => {
          await loadLlmContent(card, session, sel.value);
        });
      });

      // ASR補正ボタン
      card.querySelectorAll('.btn-correct-asr').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          e.preventDefault();
          e.stopPropagation();
          await correctAsr(card, session);
        });
      });

      // ASR元に戻すボタン
      card.querySelectorAll('.btn-rebuild-asr').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          e.preventDefault();
          e.stopPropagation();
          await rebuildAsr(card, session);
        });
      });

      // Copy ボタン (ASR)
      card.querySelectorAll('.btn-copy-asr').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          const pre = card.querySelector('.asr-text');
          if (pre) navigator.clipboard.writeText(pre.textContent).catch(() => { });
        });
      });

      // Copy ボタン (LLM)
      card.querySelectorAll('.btn-copy-llm').forEach(btn => {
        btn.addEventListener('click', () => {
          const pre = card.querySelector('.llm-text');
          if (pre) navigator.clipboard.writeText(pre.textContent).catch(() => { });
        });
      });
    });

    // 各カードのLLMを自動ロード
    // ルール:
    //   1) 確定済みがあればそれを表示
    //   2) 確定済みがなければ、最新LLMを表示（autoShowLatest時）
    //   3) LLM履歴がなければ何もしない
    karteTimeline.querySelectorAll('.karte-card').forEach(card => {
      const session = card.dataset.session;
      const approvedOpt = card.querySelector('.llm-history-select option[data-approved="true"]');
      const latestOpt = card.querySelector('.llm-history-select option');

      if (approvedOpt) {
        loadLlmContent(card, session, approvedOpt.value);
      } else if (latestOpt && opts.autoShowLatest) {
        loadLlmContent(card, session, latestOpt.value);
      }
    });
  }

  // ===== カード1枚のHTML生成 =====
  function buildCard(s) {
    const isCurrent = s.is_current;
    const approvedFile = s.approved_file || '';
    const llmFiles = s.llm_files || [];
    const hasApproved = !!approvedFile;
    const hasLlm = llmFiles.length > 0;

    // セッションヘッダー（LLM状態を優先表示）
    const badgeCls = hasApproved ? 'badge-approved' : 'badge-draft';
    const badgeLabel = hasApproved
      ? '✅ 確定済'
      : (hasLlm ? '📄 下書き' : '📝 ASRのみ');
    const recMark = isCurrent ? '<span class="muted2" style="margin-left:6px;">● 録音中</span>' : '';

    // LLMファイル一覧（セレクトボックス）
    let llmHistoryHtml = '';
    if (llmFiles.length) {
      llmHistoryHtml = `<select class="select select-sm llm-history-select">` +
        llmFiles.map(f => {
          const parts = f.split('__');
          const modelShort = (parts[1] || '').split(':')[0];
          const promptShort = parts[2] || '';
          const isApproved = (f === approvedFile);
          return `<option value="${e_(f)}" data-approved="${isApproved}" ${isApproved ? 'selected' : ''}>
            ${e_(modelShort)}/${e_(promptShort)}${isApproved ? ' [確定]' : ''}
          </option>`;
        }).join('') +
        `</select>`;
    }

    // 確定ボタン（未承認カードには常に配置し、LLM選択時に表示）
    const approveHtml = !approvedFile
      ? `<button class="btn btn-sm btn-approve-cta" data-llm-file="" style="display:none">確定</button>`
      : `<span class="badge-ts-approved" title="${e_(approvedFile)}">✅ 確定済</span>`;

    // LLMセクション（ヘッダーツールバー）
    const llmDefaultModel = selLlmModel?.value || '';
    const llmDefaultPrompt = selSoapPrompt?.value || llmDefaultPromptId || '';
    const modelOptions = llmModelList.map(m =>
      `<option value="${e_(m)}" ${m === llmDefaultModel ? 'selected' : ''}>${e_(m)}</option>`).join('');
    const promptOptions = llmPromptList.map(p =>
      `<option value="${e_(p.id)}" ${p.id === llmDefaultPrompt ? 'selected' : ''}>${e_(p.label || p.id)}</option>`).join('');

    return `
<div class="karte-card card" data-session="${e_(s.session_txt)}" data-is-current="${isCurrent}">
  <!-- カードヘッダー -->
    <div class="karte-card-hd">
    <div class="karte-card-date">
      <span class="karte-date">${e_(s.date)}</span>
      <span class="karte-time">${e_(s.time)}</span>
    </div>
    <span class="badge ${badgeCls}">${badgeLabel}</span>${recMark}
  </div>

  <!-- ASR欄（折りたたみ） -->
  <details class="asr-section" ${isCurrent ? 'open' : ''}>
    <summary class="asr-summary">
      <span>認識テキスト</span>
      <span class="asr-tools" style="display:flex; gap:0.5rem;">
        <button class="btn btn-sm btn-correct-asr" title="認識テキストをルールで補正">補正</button>
        <button class="btn btn-sm btn-rebuild-asr" title="JSONL原本から .txt を再生成して戻す">元に戻す</button>
        <button class="btn btn-sm btn-copy-asr" title="ASRテキストをコピー">Copy</button>
      </span>
    </summary>
    <pre class="asr-text" id="asr-${e_(s.session_txt)}" contenteditable="true" spellcheck="false">(読み込み中…)</pre>
  </details>

  <!-- LLM欄 -->
  <div class="llm-section">
    <div class="llm-section-hd">
      <div class="llm-section-left">
        <span class="llm-section-title">AI処理結果</span>
        ${llmHistoryHtml}
      </div>
      <div class="llm-tools">
        <select class="select select-sm card-sel-model">${modelOptions}</select>
        <select class="select select-sm card-sel-prompt">${promptOptions}</select>
        <button class="btn btn-primary btn-sm btn-llm-run">送信</button>
        <button class="btn btn-sm btn-copy-llm" title="AIテキストをコピー">Copy</button>
      </div>
    </div>
    <div class="llm-approve-bar">
      ${approveHtml}
    </div>
    <pre class="llm-text" id="llm-${e_(s.session_txt)}">${llmFiles.length ? '(履歴を選択…)' : '(AIに未送信)'}</pre>
  </div>
</div>`;
  }

  // ===== ASR補正 =====
  async function correctAsr(card, sessionTxt) {
    const btn = card.querySelector('.btn-correct-asr');
    if (btn) btn.disabled = true;
    try {
      const r = await fetch('/api/correct/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: sessionTxt })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));
      if (j.changed) {
        log(`[asr] corrected session=${sessionTxt}`);
        await loadAsrText(card, sessionTxt);
      } else {
        log(`[asr] no correction needed for ${sessionTxt}`);
      }
    } catch (e) {
      log(`[asr] correct failed: ${e}`);
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  // ===== ASR元に戻す =====
  async function rebuildAsr(card, sessionTxt) {
    if (!confirm('本当に元の認識テキストに戻しますか？(手作業の編集や補正は失われます)')) return;
    const btn = card.querySelector('.btn-rebuild-asr');
    if (btn) btn.disabled = true;
    try {
      const r = await fetch('/api/session/rebuild', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: sessionTxt })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));
      log(`[asr] rebuilt session=${sessionTxt}`);
      await loadAsrText(card, sessionTxt);
    } catch (e) {
      log(`[asr] rebuild failed: ${e}`);
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  // ===== ASRテキスト読み込み =====
  async function loadAsrText(card, sessionTxt) {
    const preEl = card.querySelector('.asr-text');
    if (!preEl) return;
    try {
      const r = await fetch('/api/session/' + encodeURIComponent(sessionTxt));
      const j = await r.json();
      preEl.textContent = j.text || '(テキストなし)';
    } catch (e) {
      preEl.textContent = `(読み込み失敗: ${e})`;
    }
  }

  // ===== LLMテキスト読み込み =====
  async function loadLlmContent(card, sessionTxt, llmFile) {
    if (!llmFile) return;
    const preEl = card.querySelector('.llm-text');
    if (!preEl) return;
    try {
      const r = await fetch('/api/llm/history/item/' + encodeURIComponent(llmFile));
      const j = await r.json();
      if (!j || !j.ok) throw new Error(j?.error || 'failed');
      preEl.textContent = j.summary || '(空)';

      // 選択ハイライト更新
      const sel = card.querySelector('.llm-history-select');
      if (sel) sel.value = llmFile;

      // 確定ボタンを対象ファイルに紐付け
      const approveBtn = card.querySelector('.btn-approve-cta');
      if (approveBtn) {
        approveBtn.dataset.llmFile = llmFile;
        approveBtn.style.display = '';
      }
    } catch (e) {
      if (preEl) preEl.textContent = `(読み込み失敗: ${e})`;
    }
  }

  // ===== LLM送信 =====
  async function runLlmForCard(card, sessionTxt) {
    const runBtn = card.querySelector('.btn-llm-run');
    const preEl = card.querySelector('.llm-text');
    const asrPre = card.querySelector('.asr-text');
    const model = card.querySelector('.card-sel-model')?.value || '';
    const promptId = card.querySelector('.card-sel-prompt')?.value || llmDefaultPromptId || 'soap_v1';
    const asrText = asrPre?.textContent?.trim() || '';

    if (!asrText || asrText.startsWith('(')) {
      log('[llm] ASRテキストが空です');
      return;
    }
    if (runBtn) runBtn.disabled = true;
    if (preEl) preEl.textContent = '(処理中…)';
    try {
      const r = await fetch('/api/llm/soap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: sessionTxt, model, prompt_id: promptId, asr_text: asrText })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));
      if (preEl) preEl.textContent = j.summary || '';
      log(`[llm] done session=${sessionTxt} model=${j.model} elapsed=${j.elapsed_sec}s`);
      // 履歴リフレッシュ
      await refreshCardLlmHistory(card, sessionTxt);
    } catch (e) {
      if (preEl) preEl.textContent = `(エラー: ${e})`;
      log(`[llm] failed: ${e}`);
    } finally {
      if (runBtn) runBtn.disabled = false;
    }
  }

  // ===== LLM履歴リフレッシュ（送信後） =====
  async function refreshCardLlmHistory(card, sessionTxt) {
    try {
      const r = await fetch('/api/llm/history?session=' + encodeURIComponent(sessionTxt));
      const j = await r.json();
      if (!j || !j.ok) return;
      const items = j.items || [];
      const selExt = card.querySelector('.llm-history-select');
      if (selExt) {
        selExt.innerHTML = items.map(it => {
          const parts = it.id.split('__');
          const modelShort = (parts[1] || '').split(':')[0];
          const promptShort = parts[2] || '';
          return `<option value="${e_(it.id)}">${e_(modelShort)}/${e_(promptShort)}</option>`;
        }).join('');
      }
      // 最新を自動表示
      if (items.length) await loadLlmContent(card, sessionTxt, items[0].id);
    } catch (e) {
      log(`[llm] refreshCardLlmHistory failed: ${e}`);
    }
  }

  // ===== LLM確定（承認） =====
  async function approveFile(card, sessionTxt, llmFile) {
    if (!llmFile) { log('[approve] no llmFile selected'); return; }
    const approveBtn = card.querySelector('.btn-approve-cta');
    if (approveBtn) approveBtn.disabled = true;
    try {
      const r = await fetch('/api/llm/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session: sessionTxt, llm_file: llmFile })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));

      // UIを確定済み表示に変更
      if (approveBtn) {
        approveBtn.style.display = 'none';
      }
      const approveBar = card.querySelector('.llm-approve-bar');
      if (approveBar) {
        approveBar.innerHTML = `<span class="badge-ts-approved">✅ 確定済</span>`;
      }
      // ヘッダーバッジも更新
      const badgeEl = card.querySelector('.badge');
      if (badgeEl) {
        badgeEl.className = 'badge badge-approved';
        badgeEl.textContent = '✅ 確定済';
      }
      // 承認済みLLM履歴アイテムにもマーク
      card.querySelectorAll('.llm-history-item').forEach(el => {
        if (el.dataset.llmFile === llmFile) el.classList.add('llm-approved');
      });
      // サイドバーも更新
      const sidebarItem = sidebarListEl?.querySelector(`[data-session="${sessionTxt}"]`);
      if (sidebarItem) {
        sidebarItem.querySelector('.sidebar-icon').textContent = '✅';
      }
      log(`[approve] confirmed session=${sessionTxt} llm_file=${llmFile}`);
    } catch (e) {
      log(`[approve] failed: ${e}`);
      if (approveBtn) approveBtn.disabled = false;
    }
  }

  // ===== 患者切替（手動入力） =====
  async function switchPatientById(pid, forceCreate = false) {
    pid = (pid || '').trim();
    if (!pid) return;
    try {
      if (btnSwitchPatient) btnSwitchPatient.disabled = true;
      const createNew = isRecording || forceCreate;
      const r = await fetch('/api/patient/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: pid, create_new: createNew })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));
      log(`[patient] switched -> ${pid} session=${j.session_txt} resumed=${j.resumed}`);
      currentPatientId = pid;
      currentSessionTxt = j.session_txt || '';
      ensureRecentPatientOption(pid);

      // ★ ここを変更：選択後は入力欄を空に戻す
      if (patientInputEl) {
        patientInputEl.value = '';
        patientInputEl.placeholder = `${pid}`;
        patientInputEl.blur();
      }
      if (patientSelectEl && recentPatientIds.includes(pid)) {
        patientSelectEl.value = pid;
      }

      await loadPatientSessions(pid, { autoShowLatest: true });
      // タイムライン内でASRテキストを読み込む
      await loadAllCardAsrText();
      await loadRecentPatients();
    } catch (e) {
      log(`[patient] switch failed: ${e}`);
    } finally {
      if (btnSwitchPatient) btnSwitchPatient.disabled = false;
    }
  }

  // タイムライン上の全カードのASRを読み込む
  async function loadAllCardAsrText() {
    if (!karteTimeline) return;
    const cards = karteTimeline.querySelectorAll('.karte-card');
    for (const card of cards) {
      await loadAsrText(card, card.dataset.session);
    }
  }

  // ===== ライブASR追記 =====
  function appendLiveAsr(text) {
    if (!text) return;
    liveAsrText += (liveAsrText ? '\n' : '') + text;
    updateHearingText(liveAsrText);

    // 現在セッションカードのASR欄を更新
    syncCurrentCardAsr(liveAsrText);
  }

  function resetLiveAsr() {
    liveAsrText = '';
    hearingLastSavedSession = currentSessionTxt || '';
    hearingLastSavedText = '';
    updateHearingText('(音声認識待機中)');
  }

  // ===== WebSocket: 常時接続（録音と独立） =====
  // ★ WSは録音中かどうかに関わらず常にサーバーと接続を保つ。
  //   これにより dyna_watch_task() の patient_changed が常に届く。
  let wsReconnectTimer = null;

  function handleWsMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }

    if (msg.type === 'level' && levelEl) levelEl.textContent = msg.dbfs;

    if (msg.type === 'status') {
      if (msg.patient_id) currentPatientId = msg.patient_id;
      if (msg.session_txt) currentSessionTxt = msg.session_txt;
      log('[status] ' + msg.msg + (msg.session_txt ? ' session=' + msg.session_txt : ''));
      if (currentPatientId) loadPatientSessions(currentPatientId, { autoShowLatest: true })
        .then(() => loadAllCardAsrText());
    }

    if (msg.type === 'patient_changed') {
      currentPatientId = msg.patient_id || '';
      currentSessionTxt = msg.session_txt || '';
      ensureRecentPatientOption(currentPatientId);
      // ナビ表示も同期
      if (patientInputEl) {
        patientInputEl.value = '';
        if (currentPatientId) patientInputEl.placeholder = `${currentPatientId}`;
      }
      if (patientSelectEl) patientSelectEl.value = currentPatientId || '';
      resetLiveAsr();
      log('[patient_changed] ' + currentPatientId);

      if (currentPatientId && currentSessionTxt) {
        renderOptimisticPatientSwitch(currentPatientId, currentSessionTxt, msg.patient_info);
      }

      schedulePatientSessionRefresh(currentPatientId, { autoShowLatest: true });
    }

    if (msg.type === 'saved') {
      if (!currentPatientId || msg.patient_id !== currentPatientId) return;
      if (!currentSessionTxt || !hasSessionCard(currentSessionTxt)) {
        schedulePatientSessionRefresh(currentPatientId, { autoShowLatest: true });
      }
    }

    if (msg.type === 'asr') {
      if ((!currentSessionTxt || !hasSessionCard(currentSessionTxt)) && currentPatientId && msg.patient_id === currentPatientId) {
        schedulePatientSessionRefresh(currentPatientId, { autoShowLatest: true });
      }
      if (msg.text) appendLiveAsr(msg.text);
    }

    if (msg.type === 'corrected' || msg.type === 'rebuilt' || msg.type === 'transcript_updated') {
      if (!msg.session_txt) return;
      if (msg.session_txt === currentSessionTxt) {
        liveAsrText = msg.text || '';
        hearingLastSavedSession = msg.session_txt;
        hearingLastSavedText = liveAsrText.trim();
        updateHearingText(liveAsrText || '(音声認識待機中)');
      }
      const preEl = karteTimeline?.querySelector(`#asr-${CSS.escape(msg.session_txt)}`);
      if (preEl) preEl.textContent = msg.text || '';
    }
  }

  function connectWs() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    clearTimeout(wsReconnectTimer);

    const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => log('[ws] connected');
    ws.onerror = e => log('[ws] error ' + e);
    ws.onclose = () => {
      log('[ws] disconnected – reconnecting in 3s...');
      // 録音中なら状態をリセット
      if (isRecording) {
        isRecording = false;
        stopRecordingWatchdog();
        renderRecState();
        if (selAsrModel) selAsrModel.disabled = false;
      }
      wsReconnectTimer = setTimeout(connectWs, 3000);
    };
    ws.onmessage = handleWsMessage;
    log('[ws] connecting...');
  }

  async function ensureWsOpen() {
    connectWs();
    if (ws && ws.readyState === WebSocket.OPEN) return;
    await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error('WS connect timeout')), 3000);
      const prevOpen = ws?.onopen;
      if (!ws) {
        clearTimeout(t);
        reject(new Error('WS unavailable'));
        return;
      }
      ws.onopen = e => {
        clearTimeout(t);
        if (prevOpen) prevOpen(e);
        resolve();
      };
    });
  }

  function startRecordingWatchdog() {
    stopRecordingWatchdog();
    recordingWatchdogTimer = setInterval(() => {
      void monitorRecordingHealth();
    }, 2000);
  }

  function stopRecordingWatchdog() {
    if (recordingWatchdogTimer) {
      clearInterval(recordingWatchdogTimer);
      recordingWatchdogTimer = null;
    }
  }

  async function cleanupAudioCapture() {
    if (procNode) procNode.disconnect();
    if (srcNode) srcNode.disconnect();
    if (audioCtx) await audioCtx.close();
    if (stream) stream.getTracks().forEach(t => t.stop());
    procNode = srcNode = audioCtx = stream = null;
  }

  async function monitorRecordingHealth() {
    if (!isRecording || recoveringAudio) return;

    const now = Date.now();
    if (audioCtx && audioCtx.state === 'suspended') {
      try {
        await audioCtx.resume();
        log('[audio] resumed suspended audio context');
      } catch (e) {
        log('[audio] resume failed: ' + e);
      }
    }

    const lastActiveAt = Math.max(lastAudioProcessAt, lastAudioChunkSentAt);
    if (!lastActiveAt || (now - lastActiveAt) < 5000) return;

    await recoverRecordingPipeline(`audio callback stalled for ${Math.round((now - lastActiveAt) / 1000)}s`);
  }

  async function createAudioPipeline() {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false, noiseSuppression: false,
        autoGainControl: false, channelCount: 1, sampleRate: 48000,
      }
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    srcNode = audioCtx.createMediaStreamSource(stream);

    const chunkSamples = 2400;
    let pcmBuf = new Float32Array(0);

    const bufferSize = 2048;
    procNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);
    procNode.onaudioprocess = ev => {
      lastAudioProcessAt = Date.now();
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = ev.inputBuffer.getChannelData(0);
      const merged = new Float32Array(pcmBuf.length + input.length);
      merged.set(pcmBuf);
      merged.set(input, pcmBuf.length);
      pcmBuf = merged;
      while (pcmBuf.length >= chunkSamples) {
        ws.send(pcmBuf.slice(0, chunkSamples).buffer);
        lastAudioChunkSentAt = Date.now();
        pcmBuf = pcmBuf.slice(chunkSamples);
      }
    };
    srcNode.connect(procNode);
    procNode.connect(audioCtx.destination);

    const [track] = stream.getAudioTracks();
    if (track) {
      track.onended = () => {
        log('[audio] input track ended');
        if (isRecording) void recoverRecordingPipeline('input track ended');
      };
      track.onmute = () => log('[audio] input track muted');
      track.onunmute = () => log('[audio] input track unmuted');
    }

    lastAudioProcessAt = Date.now();
    lastAudioChunkSentAt = Date.now();

    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }
  }

  async function recoverRecordingPipeline(reason) {
    if (!isRecording || recoveringAudio) return;
    recoveringAudio = true;
    try {
      log(`[audio] recovering: ${reason}`);
      await ensureWsOpen();
      await cleanupAudioCapture();
      await createAudioPipeline();
      log('[audio] recovery complete');
    } catch (e) {
      log('[audio] recovery failed: ' + e);
      isRecording = false;
      renderRecState();
      if (selAsrModel) selAsrModel.disabled = false;
      stopRecordingWatchdog();
    } finally {
      recoveringAudio = false;
    }
  }

  // ===== 録音: WS接続は使い回し、音声キャプチャのみ開始 =====
  async function startRecording() {
    try {
      let targetPid = (patientInputEl?.value || '').trim();
      if (!targetPid) {
        targetPid = currentPatientId || 'unknown';
      }
      if (targetPid !== currentPatientId || !currentSessionTxt) {
        await switchPatientById(targetPid, true);
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        log('ERROR: getUserMedia unavailable.');
        return;
      }
      await ensureWsOpen();
      await createAudioPipeline();

      isRecording = true;
      startRecordingWatchdog();
      renderRecState();
      if (selAsrModel) selAsrModel.disabled = true;

      let existingAsr = '';
      if (currentSessionTxt) {
        try {
          const r = await fetch('/api/session/' + encodeURIComponent(currentSessionTxt));
          const j = await r.json();
          if (j.text) existingAsr = j.text.trim();
        } catch (e) { }
      }
      liveAsrText = existingAsr;
      hearingLastSavedSession = currentSessionTxt || '';
      hearingLastSavedText = existingAsr;
      updateHearingText(liveAsrText || '(音声認識待機中)');

      log('recording start');
    } catch (e) {
      log('ERROR: ' + e);
    }
  }

  async function stopRecording() {
    try {
      await flushHearingTranscriptSave();
      stopRecordingWatchdog();
      await cleanupAudioCapture();
      // ★ WSは閉じない（常時接続を維持してpatient_changedを受信し続ける）

      isRecording = false;
      renderRecState();
      if (selAsrModel) selAsrModel.disabled = false;
      try {
        const r = await fetch('/api/auto-llm/enqueue-current', { method: 'POST' });
        const j = await r.json();
        if (j?.ok && j.enqueued) {
          log(`[auto_llm] enqueued on stop: ${j.jsonl || ''}`);
        } else if (j?.ok) {
          log(`[auto_llm] not enqueued on stop: ${j?.reason || 'skipped'} queued=${j?.queued ?? 0}`);
        }
      } catch (e) {
        log(`[auto_llm] enqueue on stop failed: ${e}`);
      }

      // 停止後にセッション一覧をリフレッシュ
      if (currentPatientId) {
        await loadPatientSessions(currentPatientId, { autoShowLatest: true });
        await loadAllCardAsrText();
      }
      log('recording stop');
    } catch (e) {
      log('ERROR: ' + e);
    }
  }

  // ===== ASRモデル切替 =====
  if (selAsrModel) {
    selAsrModel.addEventListener('change', async () => {
      const id = selAsrModel.value;
      if (!id) return;
      try {
        const r = await fetch('/api/asr/model', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id })
        });
        const j = await r.json();
        if (j.ok) log(`[ui] ASR model -> ${j.current}`);
        else { log(`[ui] ASR model switch failed: ${j.error}`); await loadAsrModels(); }
      } catch (e) { log(`[ui] ASR model error: ${e}`); await loadAsrModels(); }
    });
  }

  // ===== カルテNo手動入力 =====
  if (btnSwitchPatient) {
    btnSwitchPatient.addEventListener('click', () => {
      switchPatientById(patientInputEl?.value);
    });
  }
  if (patientSelectEl) {
    patientSelectEl.addEventListener('change', () => {
      const pid = (patientSelectEl.value || '').trim();
      if (!pid) return;
      switchPatientById(pid);
    });
  }
  if (patientInputEl) {

    // Enter即切替はやめる
    patientInputEl.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        e.preventDefault();
      }
    });
  }

  // ===== 録音ボタン =====
  if (btnRec) {
    btnRec.onclick = async () => {
      if (isRecording) await stopRecording();
      else await startRecording();
    };
  }
  // 旧Start/Stopボタン互換
  if (btnStart) btnStart.onclick = () => startRecording();
  if (btnStop) btnStop.onclick = () => stopRecording();

  // ===== 初期化 =====
  loadAppVersion();
  loadAsrModels();
  loadLlmModels().then(() => loadLlmPrompts());
  loadRecentPatients();
  connectWs();  // ★ 起動時にWS接続（常時接続）
})();
