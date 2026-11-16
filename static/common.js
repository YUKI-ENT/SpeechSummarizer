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
  const btnAutoRecording = document.getElementById('btnAutoRecording');
  const autoRecordingStatusEl = document.getElementById('autoRecordingStatus');
  const btnRecordingNotifications = document.getElementById('btnRecordingNotifications');
  const selAsrModel = document.getElementById('selAsrModel');
  // const chkHearing = document.getElementById('chkHearing');
  const hearingOverlay = document.getElementById('hearingOverlay');
  const hearingTextEl = document.getElementById('hearingText');
  const btnHearingToggle = document.getElementById('btnHearingToggle');
  const btnHearingClose = document.getElementById('btnHearingClose');
  const btnMemo = document.getElementById('btnMemo');
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  const btnHearingTranslate = document.getElementById('btnHearingTranslate');
  const selHearingModel = document.getElementById('selHearingModel');
  let savedHearingModel = '';
  const selHearingLanguage = document.getElementById('selHearingLanguage');
  const hearingBilingualEl = document.getElementById('hearingBilingual');
  const hearingJapaneseEl = document.getElementById('hearingJapanese');
  const hearingTranslatedEl = document.getElementById('hearingTranslated');
  const hearingTranslationStatusEl = document.getElementById('hearingTranslationStatus');

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
  let recoveryQueued = false;
  let trackMuteRecoveryTimer = null;

  let currentPatientId = '';   // 現在選択中の患者ID
  let currentChartBranch = null;
  let summaryObserver = null;
  let patientSessionsGeneration = 0;
  let patientContextGeneration = 0;
  let switchViewRefresh = null;
  let patientSessionsRequest = null;
  let recentPatientsRequest = null;
  let recentPatientsLoadedAt = 0;
  const uiNow = () => globalThis.performance?.now?.() ?? Date.now();
  const asrRequests = new WeakMap();
  let currentSessionTxt = '';   // 現在録音中 / 選択中のセッション(.txt)名
  let liveAsrText = '';   // ライブASRの累積テキスト
  let recentPatientIds = [];   // 新しい順・重複なし
  let hearingZoom = 1.0;
  let suppressHearingInput = false;
  let hearingSaveTimer = null;
  let hearingLastSavedSession = '';
  let hearingLastSavedText = '';
  let hearingTranslationActive = false;
  let hearingTranslationGeneration = 0;
  const hearingTranslationQueue = [];
  let hearingTranslationWorkerRunning = false;
  let hearingTranslationAbortController = null;
  let hearingTranslatedText = '';
  let latestAsrSegment = '';
  let patientSessionRefreshTimer = null;
  let flushWaiter = null;
  const recordingClientId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
  let autoRecordingWanted = true;
  try { autoRecordingWanted = localStorage.getItem('autoRecordingEnabled') !== 'false'; } catch { }
  let recordingSettings = null;
  let recordingDeadline = null;
  let recordingControlReady = false;
  let recordingFailure = '';
  let recordingControlState = null;
  let recordingOperation = Promise.resolve();
  const recordingReplies = new Map();

  // Serialize start/stop operations, including events arriving during ASR flush.
  function queueRecordingOperation(operation) {
    const pending = recordingOperation.then(operation);
    recordingOperation = pending.catch(e => log('[recording] ' + e));
    return pending;
  }

  function registerRecordingControl() {
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({
      command: 'recording_register', client_id: recordingClientId,
      auto_enabled: autoRecordingWanted, recording: isRecording,
    }));
  }

  async function requestRecordingStart(requestId = '', patientId = '', chartBranch = null) {
    const id = `${recordingClientId}-${Date.now()}-${Math.random()}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        recordingReplies.delete(id);
        reject(new Error('録音開始の応答がありません。'));
      }, 10000);
      recordingReplies.set(id, { resolve: msg => { clearTimeout(timer); resolve(msg); },
        reject: error => { clearTimeout(timer); reject(error); } });
      ws.send(JSON.stringify({ command: 'recording_begin', id,
        request_id: requestId, patient_id: patientId, chart_branch: chartBranch }));
    });
  }

  function notifyRecording(message, failure = false) {
    recordingFailure = failure ? message : '';
    log('[auto_recording] ' + message);
    renderAutoRecordingButton(message);
    if (recordingSettings?.notifications_enabled && 'Notification' in window && Notification.permission === 'granted') {
      try { new Notification('SpeechSummarizer', { body: message, tag: 'speech-recording' }); } catch { }
    }
    if (recordingSettings?.sound_enabled) {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator(), gain = ctx.createGain();
        osc.frequency.value = failure ? 330 : 660;
        gain.gain.value = 0.06;
        osc.connect(gain); gain.connect(ctx.destination);
        osc.onended = () => { void ctx.close(); };
        void ctx.resume().then(() => { osc.start(); osc.stop(ctx.currentTime + (failure ? 0.3 : 0.12)); });
      } catch { }
    }
  }

  function renderAutoRecordingButton(message = '') {
    const msg = recordingControlState;
    if (!msg) return;
    const status = message || msg.error || recordingFailure || (
      !msg.enabled ? 'スケジュール未設定' : msg.active ? '録音中' :
      msg.auto_enabled ? (msg.within_schedule ? '自動開始待機中' : '時間外') : '手動開始待機中');
    let countdown = '';
    if (msg.enabled && msg.active && msg.remaining_seconds != null) {
      const seconds = Math.max(0, Math.ceil(msg.remaining_seconds));
      countdown = ` ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    }
    const description = `自動開始 ${msg.auto_enabled ? 'ON' : 'OFF'}：${status}${countdown ? `・残り${countdown.trim()}` : ''}`;
    if (btnAutoRecording) {
      btnAutoRecording.disabled = !msg.enabled;
      btnAutoRecording.setAttribute('aria-checked', String(msg.auto_enabled));
      btnAutoRecording.textContent = '自動' + countdown;
      btnAutoRecording.title = description;
      btnAutoRecording.setAttribute('aria-label', description);
    }
    if (autoRecordingStatusEl) autoRecordingStatusEl.textContent = status;
  }

  function applyRecordingControl(msg) {
    recordingControlState = msg;
    recordingSettings = msg.settings;
    recordingControlReady = true;
    recordingDeadline = msg.remaining_seconds == null ? null : Date.now() + msg.remaining_seconds * 1000;
    renderAutoRecordingButton();
    if (btnRecordingNotifications) btnRecordingNotifications.hidden = !(
      recordingSettings?.notifications_enabled && 'Notification' in window && Notification.permission === 'default');
    if (msg.reply_to) {
      recordingReplies.get(msg.reply_to)?.resolve(msg);
      recordingReplies.delete(msg.reply_to);
    } else if (isRecording && msg.enabled && !msg.active) {
      void queueRecordingOperation(() => stopRecording());
    }
    if (msg.error) {
      autoRecordingWanted = msg.auto_enabled;
      notifyRecording(msg.error, true);
    }
  }

  if (btnAutoRecording) btnAutoRecording.onclick = async () => {
    autoRecordingWanted = btnAutoRecording.getAttribute('aria-checked') !== 'true';
    try { localStorage.setItem('autoRecordingEnabled', String(autoRecordingWanted)); } catch { }
    // Prepare microphone permission and audio playback while the page is visible.
    if (autoRecordingWanted) {
      try {
        const preparation = await navigator.mediaDevices.getUserMedia({ audio: true });
        preparation.getTracks().forEach(track => track.stop());
      } catch (e) {
        autoRecordingWanted = false;
        try { localStorage.setItem('autoRecordingEnabled', 'false'); } catch { }
        notifyRecording('マイクを利用できないため自動開始を有効にできません。', true);
      }
    }
    registerRecordingControl();
  };
  if (btnRecordingNotifications) btnRecordingNotifications.onclick = async () => {
    try {
      await Notification.requestPermission();
      btnRecordingNotifications.hidden = Notification.permission !== 'default';
    } catch (e) { log('[notification] ' + e); }
  };


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
    } else if (hearingTranslationActive) {
      setHearingTranslationMode(false);
    }
  }

  function applyHearingZoom() {
    if (hearingTextEl) {
      hearingTextEl.style.setProperty('--hearing-zoom', hearingZoom);
    }
    if (hearingBilingualEl) {
      hearingBilingualEl.style.setProperty('--hearing-zoom', hearingZoom);
    }
  }

  function setHearingTranslationStatus(text, isError = false) {
    if (!hearingTranslationStatusEl) return;
    hearingTranslationStatusEl.textContent = text || '';
    hearingTranslationStatusEl.classList.toggle('error', isError);
  }

  function latestTranscriptLine() {
    const lines = (liveAsrText || '').split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    return lines[lines.length - 1] || '';
  }

  function updateHearingJapaneseText(text) {
    if (!hearingJapaneseEl) return;
    const next = text || '';
    if (hearingJapaneseEl.textContent === next) return;
    hearingJapaneseEl.textContent = next;
    // 翻訳待ちとは独立して、最新のASR追記位置を常に表示する。
    hearingJapaneseEl.scrollTop = hearingJapaneseEl.scrollHeight;
  }

  function updateHearingTranslatedText(text) {
    if (!hearingTranslatedEl) return;
    const next = text || '';
    if (hearingTranslatedEl.textContent === next) return;
    hearingTranslatedEl.textContent = next;
    hearingTranslatedEl.scrollTop = hearingTranslatedEl.scrollHeight;
  }

  async function translateHearingSegment(sourceText, language, generation) {
    if (!hearingTranslationActive || generation !== hearingTranslationGeneration) return;
    const controller = new AbortController();
    hearingTranslationAbortController = controller;
    try {
      const r = await fetch('/api/hearing-translation/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: sourceText, language }),
        signal: controller.signal
      });
      const j = await r.json();
      if (!r.ok || !j.ok) throw new Error(j.error || String(r.status));
      if (!hearingTranslationActive || generation !== hearingTranslationGeneration) return;
      const translated = (j.text || '').trim();
      if (translated) {
        hearingTranslatedText += (hearingTranslatedText ? '\n' : '') + translated;
        updateHearingTranslatedText(hearingTranslatedText);
      }
      setHearingTranslationStatus('');
      log(`[hearing] translated language=${j.language} model=${j.model} elapsed=${j.elapsed_sec}s`);
    } catch (e) {
      if (e?.name === 'AbortError') return;
      if (!hearingTranslationActive || generation !== hearingTranslationGeneration) return;
      setHearingTranslationStatus(`翻訳できませんでした: ${e}`, true);
      log(`[hearing] translation failed: ${e}`);
    } finally {
      if (hearingTranslationAbortController === controller) {
        hearingTranslationAbortController = null;
      }
    }
  }

  async function runHearingTranslationWorker() {
    if (hearingTranslationWorkerRunning) return;
    hearingTranslationWorkerRunning = true;
    try {
      while (hearingTranslationQueue.length) {
        const job = hearingTranslationQueue.shift();
        if (!hearingTranslationActive || job.generation !== hearingTranslationGeneration) continue;
        const waiting = hearingTranslationQueue.filter(
          queued => queued.generation === hearingTranslationGeneration
        ).length;
        setHearingTranslationStatus(waiting ? `翻訳中…（${waiting}件待ち）` : '翻訳中…');
        await translateHearingSegment(job.text, job.language, job.generation);
      }
    } catch (e) {
      log(`[hearing] translation worker failed: ${e}`);
    } finally {
      hearingTranslationWorkerRunning = false;
      // finally直前にASR結果が追加された場合も取りこぼさない。
      if (hearingTranslationQueue.length) void runHearingTranslationWorker();
    }
  }

  function queueHearingTranslation(sourceText) {
    const text = (sourceText || '').trim();
    if (!hearingTranslationActive || !text || !selHearingLanguage?.value) return;
    hearingTranslationQueue.push({
      text,
      language: selHearingLanguage.value,
      generation: hearingTranslationGeneration
    });
    void runHearingTranslationWorker();
  }

  function setHearingTranslationMode(on) {
    hearingTranslationActive = Boolean(on);
    hearingTranslationGeneration += 1;
    hearingTranslationQueue.length = 0;
    hearingTranslationAbortController?.abort();
    if (hearingTranslationActive) {
      hearingTranslatedText = '';
      updateHearingTranslatedText('');
    }
    hearingOverlay?.classList.toggle('translating', hearingTranslationActive);
    btnHearingTranslate?.setAttribute('aria-pressed', hearingTranslationActive ? 'true' : 'false');
    if (btnHearingTranslate) {
      btnHearingTranslate.textContent = hearingTranslationActive ? '翻訳中' : '翻訳';
    }
    setHearingTranslationStatus('');
    if (!hearingTranslationActive) return;
    const sourceText = latestAsrSegment || latestTranscriptLine();
    updateHearingJapaneseText(liveAsrText || '音声認識待機中');
    if (sourceText) {
      queueHearingTranslation(sourceText);
    } else {
      if (hearingJapaneseEl) hearingJapaneseEl.textContent = '音声認識待機中';
      if (hearingTranslatedEl) hearingTranslatedEl.textContent = '';
    }
  }

  async function loadHearingTranslationSettings() {
    if (!btnHearingTranslate || !selHearingLanguage) return;
    try {
      const r = await fetch('/api/hearing-translation/settings');
      const j = await r.json();
      if (!r.ok || !j.ok) throw new Error(j.error || String(r.status));
      savedHearingModel = j.model || '';
      if (selHearingModel) {
        selHearingModel.replaceChildren(new Option(savedHearingModel, savedHearingModel));
        void loadHearingTranslationModels(j.enabled);
      }
      const languages = Array.isArray(j.languages) ? j.languages : [];
      selHearingLanguage.innerHTML = languages.map(item =>
        `<option value="${e_(item.id)}">${e_(item.label)}</option>`
      ).join('');
      if (j.default_language) selHearingLanguage.value = j.default_language;
      const available = Boolean(j.enabled && languages.length);
      btnHearingTranslate.disabled = !available;
      selHearingLanguage.disabled = !available;
      btnHearingTranslate.closest('.hearing-translation-controls')?.classList.toggle('is-disabled', !available);
      btnHearingTranslate.title = available ? `字幕翻訳モデル: ${j.model}` : '字幕翻訳は無効です';
    } catch (e) {
      btnHearingTranslate.closest('.hearing-translation-controls')?.classList.add('is-disabled');
      log(`[hearing] translation settings failed: ${e}`);
    }
  }

  async function loadHearingTranslationModels(enabled) {
    try {
      const r = await fetch('/api/hearing-translation/models');
      const j = await r.json();
      if (!r.ok || !j.ok) throw new Error(j.error || String(r.status));
      const models = [...new Set([savedHearingModel, ...(j.models || [])].filter(Boolean))];
      selHearingModel.replaceChildren(...models.map(model => new Option(model, model)));
      selHearingModel.value = savedHearingModel;
      selHearingModel.disabled = !enabled || !models.length;
    } catch (e) {
      selHearingModel.title = `モデル一覧を取得できませんでした: ${e}`;
      log(`[hearing] model list failed: ${e}`);
    }
  }

  selHearingModel?.addEventListener('change', async () => {
    const model = selHearingModel.value;
    const wasActive = hearingTranslationActive;
    setHearingTranslationMode(false);
    selHearingModel.disabled = true;
    btnHearingTranslate.disabled = true;
    try {
      const r = await fetch('/api/hearing-translation/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model })
      });
      const j = await r.json();
      if (!r.ok || !j.ok) throw new Error(j.error || String(r.status));
      savedHearingModel = j.model;
      btnHearingTranslate.title = `字幕翻訳モデル: ${j.model}`;
    } catch (e) {
      selHearingModel.value = savedHearingModel;
      alert(`字幕翻訳モデルを保存できませんでした。\n${e}`);
    } finally {
      selHearingModel.disabled = false;
      btnHearingTranslate.disabled = false;
      if (wasActive) setHearingTranslationMode(true);
    }
  });

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

  if (btnHearingTranslate) {
    btnHearingTranslate.addEventListener('click', () => {
      setHearingTranslationMode(!hearingTranslationActive);
    });
  }

  if (selHearingLanguage) {
    selHearingLanguage.addEventListener('change', () => {
      if (!hearingTranslationActive) return;
      hearingTranslationGeneration += 1;
      hearingTranslationQueue.length = 0;
      hearingTranslationAbortController?.abort();
      hearingTranslatedText = '';
      updateHearingTranslatedText('');
      queueHearingTranslation(latestAsrSegment || latestTranscriptLine());
    });
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
      selAsrModel.dataset.switchable = (j.switchable === false) ? 'false' : 'true';
      selAsrModel.disabled = isRecording || j.switchable === false;
    } catch (e) { log(`[ui] loadAsrModels failed: ${e}`); }
  }

  // LLMモデル一覧（カード内ドロップダウン用）
  let llmModelList = [];
  async function loadLlmModels() {
    try {
      const r = await fetch('/api/llm/models');
      const j = await r.json();
      const models = Array.isArray(j.models) ? j.models.filter(Boolean) : [];
      if (j.default_model) models.unshift(j.default_model);
      llmModelList = [...new Set(models)];
      // hidden select も更新
      if (selLlmModel) {
        selLlmModel.innerHTML = '<option value="">LLM model...</option>' +
          llmModelList.map(m => `<option value="${e_(m)}">${e_(m)}</option>`).join('');
        if (j.default_model) selLlmModel.value = j.default_model;
      }
      refreshCardLlmControls();
      if (!j.ok) log(`[ui] LLMモデル一覧を取得できないため既定モデルを使用: ${j.error || r.status}`);
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
          llmPromptList.map(it => `<option value="${e_(it.id)}">${e_(it.label || it.id)}</option>`).join('');
        if (llmDefaultPromptId) selSoapPrompt.value = llmDefaultPromptId;
      }
      refreshCardLlmControls();
    } catch (e) { log(`[ui] loadLlmPrompts failed: ${e}`); }
  }

  function refreshCardLlmControls() {
    karteTimeline?.querySelectorAll('.karte-card').forEach(card => {
      const modelSelect = card.querySelector('.card-sel-model');
      if (modelSelect && llmModelList.length) {
        const selected = modelSelect.value || selLlmModel?.value;
        modelSelect.innerHTML = llmModelList.map(m =>
          `<option value="${e_(m)}">${e_(m)}</option>`).join('');
        if (llmModelList.includes(selected)) modelSelect.value = selected;
      }
      const promptSelect = card.querySelector('.card-sel-prompt');
      if (promptSelect && llmPromptList.length) {
        const selected = promptSelect.value || selSoapPrompt?.value || llmDefaultPromptId;
        promptSelect.innerHTML = llmPromptList.map(p =>
          `<option value="${e_(p.id)}">${e_(p.label || p.id)}</option>`).join('');
        if (llmPromptList.some(p => p.id === selected)) promptSelect.value = selected;
      }
    });
  }

  function loadRecentPatients({ force = false } = {}) {
    if (!patientSelectEl) return Promise.resolve();
    if (recentPatientsRequest) return recentPatientsRequest;
    if (!force && recentPatientsLoadedAt && Date.now() - recentPatientsLoadedAt < 60000) {
      return Promise.resolve();
    }
    recentPatientsRequest = (async () => {
      const started = uiNow();
      try {
        const r = await fetch('/api/sessions');
        const j = await r.json();
        if (!r.ok || !j?.ok) throw new Error(j?.error || String(r.status));
        const received = uiNow();
        const items = Array.isArray(j.items) ? j.items : [];
        const seen = new Set();
        recentPatientIds = [];
        const options = [];
        for (const it of items) {
          const pid = String(it?.patient_id || '').trim();
          if (!pid || seen.has(pid)) continue;
          seen.add(pid);
          recentPatientIds.push(pid);
          const label = it.patient_info?.name ? `${pid} | ${it.patient_info.name}` : pid;
          options.push(`<option value="${e_(pid)}">${e_(label)}</option>`);
        }
        patientSelectEl.innerHTML = `<option value="">履歴</option>${options.join('')}`;
        if (currentPatientId) ensureRecentPatientOption(currentPatientId);
        recentPatientsLoadedAt = Date.now();
        log(`[perf] recent patients=${items.length} fetch_ms=${(received - started).toFixed(1)} render_ms=${(uiNow() - received).toFixed(1)}`);
      } catch (e) {
        log(`[patient] loadRecentPatients failed: ${e}`);
      }
    })().finally(() => { recentPatientsRequest = null; });
    return recentPatientsRequest;
  }

  function selectPatientContext(pid, branch, session) {
    const changed = currentPatientId !== pid || currentChartBranch !== branch || currentSessionTxt !== session;
    currentPatientId = pid;
    currentChartBranch = branch;
    currentSessionTxt = session;
    if (changed) {
      ++patientSessionsGeneration;
      ++patientContextGeneration;
      switchViewRefresh = null;
      patientSessionsRequest?.controller.abort();
    }
    return changed;
  }

  // The POST response and WS notification describe the same switch. They share
  // one refresh even when one arrives after the other's request has completed.
  function refreshPatientAfterSwitch(pid) {
    clearTimeout(patientSessionRefreshTimer);
    patientSessionRefreshTimer = null;
    if (switchViewRefresh?.context === patientContextGeneration) return switchViewRefresh.promise;
    const promise = loadPatientSessions(pid, { autoShowLatest: true });
    switchViewRefresh = { context: patientContextGeneration, promise };
    promise.then(ok => {
      if (!ok && switchViewRefresh?.promise === promise) switchViewRefresh = null;
    });
    return promise;
  }

  // Patient info and history travel in one response; coalesce concurrent refreshes.
  function loadPatientSessions(pid, opts = {}) {
    if (!pid) return Promise.resolve(false);
    const context = patientContextGeneration;
    if (patientSessionsRequest?.pid === pid && patientSessionsRequest.context === context) {
      return patientSessionsRequest.promise;
    }
    patientSessionsRequest?.controller.abort();
    const controller = new AbortController();
    const generation = ++patientSessionsGeneration;
    const promise = (async () => {
      const started = uiNow();
      try {
        const r = await fetch(`/api/patient/${encodeURIComponent(pid)}/sessions`, { signal: controller.signal });
        const j = await r.json();
        const received = uiNow();
        if (!r.ok || !j?.ok) throw new Error(j?.error || String(r.status));
        if (pid !== currentPatientId || generation !== patientSessionsGeneration) return false;
        ensureRecentPatientOption(pid, j.patient_info);
        renderSidebar(pid, j.sessions || [], j.patient_info || null);
        renderTimeline(j.sessions || [], opts);
        const rendered = uiNow();
        log(`[perf] patient=${pid} sessions=${j.sessions?.length || 0} sqlite_ms=${j.query_ms ?? '?'} fetch_ms=${(received - started).toFixed(1)} render_ms=${(rendered - received).toFixed(1)}`);
        await loadAllCardAsrText();
        return true;
      } catch (e) {
        if (e.name !== 'AbortError') log(`[patient] loadPatientSessions failed: ${e}`);
        return false;
      }
    })();
    patientSessionsRequest = { pid, context, controller, promise };
    promise.finally(() => {
      if (patientSessionsRequest?.promise === promise) patientSessionsRequest = null;
    });
    return promise;
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
        <span class="sidebar-branch">${e_(branchLabel(s.chart_branch))}</span>
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
    const m = String(sessionTxt || '').match(/^([^_]+?)(?:_b([0-9]))?_(\d{8}_\d{6})(?:_c[0-9a-f]{32})?\.txt$/);
    if (!m) return null;
    return { patientId: m[1], chartBranch: m[2] === undefined ? null : Number(m[2]), stamp: m[3] };
  }

  function branchLabel(branch) {
    return branch === null || branch === undefined ? '枝番不明' : `枝番 ${branch}`;
  }

  function buildOptimisticSession(sessionTxt) {
    const parsed = parseSessionStamp(sessionTxt);
    if (!parsed) return null;
    const { stamp } = parsed;
    return {
      session_txt: sessionTxt,
      chart_branch: parsed.chartBranch,
      stamp,
      date: `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}`,
      time: `${stamp.slice(9, 11)}:${stamp.slice(11, 13)}`,
      is_current: true,
      llm_files: [],
      approved_file: '',
    };
  }

  function ensureRecentPatientOption(pid, patientInfo = null) {
    pid = String(pid || '').trim();
    if (!pid || !patientSelectEl) return;

    recentPatientIds = [pid, ...recentPatientIds.filter(x => x !== pid)];

    // Move one option; rebuilding thousands of options on every switch stalls
    // the browser and is unrelated to the speed of the SQLite query.
    let option = Array.from(patientSelectEl.options).find(item => item.value === pid);
    if (!option) {
      option = document.createElement('option');
      option.value = pid;
      option.textContent = pid;
    }
    if (patientInfo?.name) option.textContent = `${pid} | ${patientInfo.name}`;
    const firstPatient = patientSelectEl.options[1] || null;
    if (option !== firstPatient) patientSelectEl.insertBefore(option, firstPatient);
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
      if (pid === currentPatientId) void loadPatientSessions(pid, opts);
    }, 120);
  }

  // ===== タイムライン描画 =====
  function renderTimeline(sessions, opts = {}) {
    if (!karteTimeline) return;
    summaryObserver?.disconnect();

    if (!sessions.length) {
      karteTimeline.innerHTML = '<div class="karte-empty muted2">受診履歴がありません</div>';
      return;
    }

    karteTimeline.innerHTML = sessions.map(s => buildCard(s)).join('');

    // カード内イベント登録
    karteTimeline.querySelectorAll('.karte-card').forEach(card => {
      const session = card.dataset.session;
      // Closed past transcripts are fetched only when opened.
      card.querySelector('.asr-section')?.addEventListener('toggle', (event) => {
        if (event.target.open && !card.dataset.asrLoaded) void loadAsrText(card, session);
      });

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
    const loadCardSummary = (card) => {
      const session = card.dataset.session;
      const approvedOpt = card.querySelector('.llm-history-select option[data-approved="true"]');
      const latestOpt = card.querySelector('.llm-history-select option');

      if (approvedOpt) {
        loadLlmContent(card, session, approvedOpt.value);
      } else if (latestOpt && opts.autoShowLatest) {
        loadLlmContent(card, session, latestOpt.value);
      }
    };
    // Keep all branches in the timeline, loading summaries as cards come into view.
    if (typeof IntersectionObserver !== 'undefined') {
      summaryObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            summaryObserver.unobserve(entry.target);
            loadCardSummary(entry.target);
          }
        });
      }, { rootMargin: '300px' });
      karteTimeline.querySelectorAll('.karte-card').forEach(card => summaryObserver.observe(card));
    } else {
      karteTimeline.querySelectorAll('.karte-card').forEach(loadCardSummary);
    }
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
      <span class="karte-branch">${e_(branchLabel(s.chart_branch))}</span>
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
  function loadAsrText(card, sessionTxt) {
    if (asrRequests.has(card)) return asrRequests.get(card);
    const preEl = card.querySelector('.asr-text');
    if (!preEl) return Promise.resolve();
    const promise = (async () => {
      try {
        const r = await fetch('/api/session/' + encodeURIComponent(sessionTxt));
        const j = await r.json();
        if (!r.ok) throw new Error(j?.error || String(r.status));
        preEl.textContent = j.text || '(テキストなし)';
        card.dataset.asrLoaded = 'true';
      } catch (e) {
        preEl.textContent = `(読み込み失敗: ${e})`;
      }
    })().finally(() => asrRequests.delete(card));
    asrRequests.set(card, promise);
    return promise;
  }

  // ===== LLMテキスト読み込み =====
  async function loadLlmContent(card, sessionTxt, llmFile) {
    if (!llmFile) return;
    const preEl = card.querySelector('.llm-text');
    if (!preEl) return;
    const started = uiNow();
    try {
      const r = await fetch('/api/llm/history/item/' + encodeURIComponent(llmFile));
      const j = await r.json();
      if (!j || !j.ok) throw new Error(j?.error || 'failed');
      preEl.textContent = j.summary || '(空)';
      log(`[perf] summary fetch_ms=${(uiNow() - started).toFixed(1)}`);

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
  async function switchPatientById(pid, forceCreate = false, chartBranch = null) {
    pid = (pid || '').trim();
    if (!pid) return;
    try {
      if (btnSwitchPatient) btnSwitchPatient.disabled = true;
      const createNew = isRecording || forceCreate;
      const r = await fetch('/api/patient/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: pid, chart_branch: chartBranch, create_new: createNew })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));
      log(`[patient] switched -> ${pid} session=${j.session_txt} resumed=${j.resumed}`);
      selectPatientContext(j.patient_id || pid, j.chart_branch ?? null, j.session_txt || '');
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

      await refreshPatientAfterSwitch(pid);
    } catch (e) {
      log(`[patient] switch failed: ${e}`);
    } finally {
      if (btnSwitchPatient) btnSwitchPatient.disabled = false;
    }
  }

  // 開いているASR欄を更新し、折りたたまれた過去本文は取得しない。
  async function loadAllCardAsrText() {
    if (!karteTimeline) return;
    const cards = karteTimeline.querySelectorAll('.karte-card');
    for (const card of cards) {
      if (card.querySelector('.asr-section')?.open) await loadAsrText(card, card.dataset.session);
    }
  }

  // ===== ライブASR追記 =====
  function appendLiveAsr(text) {
    if (!text) return;
    latestAsrSegment = text.trim();
    liveAsrText += (liveAsrText ? '\n' : '') + text;
    updateHearingText(liveAsrText);
    if (hearingTranslationActive) updateHearingJapaneseText(liveAsrText);
    queueHearingTranslation(latestAsrSegment);

    // 現在セッションカードのASR欄を更新
    syncCurrentCardAsr(liveAsrText);
  }

  function resetLiveAsr() {
    liveAsrText = '';
    latestAsrSegment = '';
    hearingTranslatedText = '';
    if (hearingTranslationActive) {
      hearingTranslationGeneration += 1;
      hearingTranslationQueue.length = 0;
      hearingTranslationAbortController?.abort();
    }
    hearingLastSavedSession = currentSessionTxt || '';
    hearingLastSavedText = '';
    updateHearingText('(音声認識待機中)');
    if (hearingTranslationActive) {
      updateHearingJapaneseText('音声認識待機中');
      updateHearingTranslatedText('');
    }
  }

  // ===== WebSocket: 常時接続（録音と独立） =====
  // ★ WSは録音中かどうかに関わらず常にサーバーと接続を保つ。
  //   これにより dyna_watch_task() の patient_changed が常に届く。
  let wsReconnectTimer = null;

  function handleWsMessage(ev) {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }

    if (msg.type === 'recording_control') applyRecordingControl(msg);
    if (msg.type === 'auto_recording_start') {
      void queueRecordingOperation(() => startRecording({ automatic: true,
        requestId: msg.request_id, patientId: msg.patient_id, chartBranch: msg.chart_branch ?? null }));
    }
    if (msg.type === 'auto_recording_stop') {
      void queueRecordingOperation(async () => {
        await stopRecording('time_limit');
      });
    }

    if (msg.type === 'level' && levelEl) levelEl.textContent = msg.dbfs;

    if (msg.type === 'status') {
      selectPatientContext(msg.patient_id || '', msg.chart_branch ?? null, msg.session_txt || '');
      switchViewRefresh = null; // Reconnects must see updates made while disconnected.
      log('[status] ' + msg.msg + (msg.session_txt ? ' session=' + msg.session_txt : ''));
      if (currentPatientId) void refreshPatientAfterSwitch(currentPatientId);
    }

    if (msg.type === 'patient_changed') {
      const changed = selectPatientContext(msg.patient_id || '', msg.chart_branch ?? null, msg.session_txt || '');
      ensureRecentPatientOption(currentPatientId, msg.patient_info);
      // ナビ表示も同期
      if (patientInputEl) {
        patientInputEl.value = '';
        if (currentPatientId) patientInputEl.placeholder = `${currentPatientId}`;
      }
      if (patientSelectEl) patientSelectEl.value = currentPatientId || '';
      resetLiveAsr();
      log('[patient_changed] ' + currentPatientId);

      // 長時間稼働でブラウザ側の入力トラックだけが実質停止する事例への予防策。
      // 録音の意思状態や WebSocket は維持し、音声キャプチャだけを作り直す。
      if (isRecording) {
        void recoverRecordingPipeline('patient changed (preventive audio refresh)');
      }

      if (changed && currentPatientId && currentSessionTxt) {
        renderOptimisticPatientSwitch(currentPatientId, currentSessionTxt, msg.patient_info);
      }

      void refreshPatientAfterSwitch(currentPatientId);
    }

    if (msg.type === 'saved') {
      if (!currentPatientId || msg.patient_id !== currentPatientId) return;
      if (!currentSessionTxt || !hasSessionCard(currentSessionTxt)) {
        schedulePatientSessionRefresh(currentPatientId, { autoShowLatest: true });
      }
    }

    if (msg.type === 'asr') {
      if (msg.target?.type === 'session') {
        const expectedTargetId = (currentSessionTxt || '').replace(/\.txt$/, '');
        if (msg.target.id !== expectedTargetId) {
          log(`[asr] completed for previous session: ${msg.target.id}`);
          return;
        }
      }
      if ((!currentSessionTxt || !hasSessionCard(currentSessionTxt)) && currentPatientId && msg.patient_id === currentPatientId) {
        schedulePatientSessionRefresh(currentPatientId, { autoShowLatest: true });
      }
      if (msg.text) appendLiveAsr(msg.text);
    }

    if (msg.type === 'flush_complete' && flushWaiter) {
      flushWaiter.resolve();
      flushWaiter = null;
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

    ws.onopen = () => {
      log('[ws] connected');
      recordingControlReady = false;
      registerRecordingControl();
      if (isRecording) {
        lastAudioChunkSentAt = Date.now();
        log('[recording] connection restored; recording continues');
      }
    };
    ws.onerror = e => log('[ws] error ' + e);
    ws.onclose = () => {
      recordingControlReady = false;
      for (const waiter of recordingReplies.values()) waiter.reject(new Error('接続が切断されました。'));
      recordingReplies.clear();
      if (flushWaiter) {
        flushWaiter.resolve();
        flushWaiter = null;
      }
      log('[ws] disconnected – reconnecting in 3s...');
      // WebSocket切断は一時障害として扱い、手動OFFまでは録音状態を維持する。
      if (isRecording) {
        log('[recording] connection lost; keeping recording ON');
      }
      wsReconnectTimer = setTimeout(connectWs, 3000);
    };
    ws.onmessage = handleWsMessage;
    log('[ws] connecting...');
  }

  async function ensureWsOpen() {
    connectWs();
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline) {
      if (ws && ws.readyState === WebSocket.OPEN) return;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error('WS connect timeout');
  }

  async function flushAsrQueue() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (flushWaiter) return flushWaiter.promise;
    let resolveWaiter;
    const promise = new Promise(resolve => { resolveWaiter = resolve; });
    const timeoutId = setTimeout(() => {
      if (!flushWaiter) return;
      flushWaiter.resolve();
      flushWaiter = null;
      log('[recording] ASR flush timeout');
    }, 45000);
    flushWaiter = {
      promise,
      resolve: () => {
        clearTimeout(timeoutId);
        resolveWaiter();
      }
    };
    ws.send(JSON.stringify({ command: 'flush' }));
    return promise;
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
    if (trackMuteRecoveryTimer) {
      clearTimeout(trackMuteRecoveryTimer);
      trackMuteRecoveryTimer = null;
    }
    const oldContext = audioCtx, oldStream = stream;
    const oldProcessor = procNode, oldSource = srcNode;
    procNode = srcNode = audioCtx = stream = null;
    if (oldStream) oldStream.getTracks().forEach(t => {
      t.onended = null;
      t.onmute = null;
      t.onunmute = null;
      t.stop();
    });
    if (oldProcessor) { oldProcessor.onaudioprocess = null; oldProcessor.disconnect(); }
    if (oldSource) oldSource.disconnect();
    if (oldContext && oldContext.state !== 'closed') await oldContext.close();
  }

  async function monitorRecordingHealth() {
    if (!isRecording || recoveringAudio) return;

    const now = Date.now();
    if (recordingSettings?.enabled && recordingDeadline && now >= recordingDeadline) {
      await queueRecordingOperation(() => stopRecording('time_limit'));
      return;
    }
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      await recoverRecordingPipeline('WebSocket unavailable');
      return;
    }

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
    let mediaTimedOut = false;
    let mediaTimer;
    const mediaRequest = navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false, noiseSuppression: false,
        autoGainControl: false, channelCount: 1, sampleRate: 48000,
      }
    }).then(result => {
      if (mediaTimedOut) result.getTracks().forEach(track => track.stop());
      return result;
    });
    try {
      stream = await Promise.race([mediaRequest, new Promise((_, reject) => {
        mediaTimer = setTimeout(() => {
          mediaTimedOut = true;
          reject(new Error('マイクの許可待ちがタイムアウトしました。'));
        }, 15000);
      })]);
    } finally { clearTimeout(mediaTimer); }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    let resumeTimer;
    try {
      await Promise.race([audioCtx.resume(), new Promise((_, reject) => {
        resumeTimer = setTimeout(() => reject(new Error('音声入力を開始できません。画面でマイクボタンを押してください。')), 5000);
      })]);
    } finally { clearTimeout(resumeTimer); }
    if (audioCtx.state !== 'running') throw new Error('音声入力を開始するには画面でマイクボタンを押してください。');
    srcNode = audioCtx.createMediaStreamSource(stream);

    const chunkSamples = 2400;
    let pcmBuf = new Float32Array(0);

    const bufferSize = 2048;
    procNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);
    procNode.onaudioprocess = ev => {
      lastAudioProcessAt = Date.now();
      if (recordingSettings?.enabled && recordingDeadline && Date.now() >= recordingDeadline) {
        void queueRecordingOperation(() => stopRecording('time_limit'));
        return;
      }
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (!recordingControlReady) return;
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
      track.onmute = () => {
        log('[audio] input track muted');
        clearTimeout(trackMuteRecoveryTimer);
        trackMuteRecoveryTimer = setTimeout(() => {
          trackMuteRecoveryTimer = null;
          if (isRecording && track === stream?.getAudioTracks()[0] && track.muted) {
            void recoverRecordingPipeline('input track remained muted');
          }
        }, 2000);
      };
      track.onunmute = () => {
        log('[audio] input track unmuted');
        clearTimeout(trackMuteRecoveryTimer);
        trackMuteRecoveryTimer = null;
      };
    }

    lastAudioProcessAt = Date.now();
    lastAudioChunkSentAt = Date.now();

    if (audioCtx.state === 'suspended') {
      await audioCtx.resume();
    }
  }

  function recoverRecordingPipeline(reason) {
    if (!isRecording || recoveryQueued) return Promise.resolve();
    recoveryQueued = true;
    return queueRecordingOperation(() => performRecordingRecovery(reason))
      .finally(() => { recoveryQueued = false; });
  }

  async function performRecordingRecovery(reason) {
    if (!isRecording || recoveringAudio) return;
    recoveringAudio = true;
    try {
      log(`[audio] recovering: ${reason}`);
      await ensureWsOpen();
      if (!isRecording) return;
      await cleanupAudioCapture();
      if (!isRecording) return;
      await createAudioPipeline();
      log('[audio] recovery complete');
    } catch (e) {
      log('[audio] recovery failed: ' + e);
      if (isRecording) {
        log('[recording] recovery will retry; recording remains ON');
      }
    } finally {
      recoveringAudio = false;
    }
  }

  // ===== 録音: WS接続は使い回し、音声キャプチャのみ開始 =====
  async function startRecording({ automatic = false, requestId = '', patientId = '', chartBranch = null } = {}) {
    if (isRecording) return;
    let granted = false;
    try {
      if (automatic && (!autoRecordingWanted || patientId !== currentPatientId || chartBranch !== currentChartBranch)) return;
      let targetPid = automatic ? patientId : (patientInputEl?.value || '').trim();
      if (!targetPid) {
        targetPid = currentPatientId || 'unknown';
      }
      if (targetPid !== currentPatientId || !currentSessionTxt) {
        await switchPatientById(targetPid, true, targetPid === currentPatientId ? currentChartBranch : null);
      }
      const targetBranch = currentChartBranch;

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('getUserMedia unavailable');
      }
      await ensureWsOpen();
      const response = await requestRecordingStart(requestId, targetPid, targetBranch);
      if (!response.granted) throw new Error(response.error || '録音を開始できませんでした。');
      granted = true;
      await createAudioPipeline();

      if (targetPid !== currentPatientId || targetBranch !== currentChartBranch ||
          (automatic && !autoRecordingWanted)) {
        await cleanupAudioCapture();
        ws.send(JSON.stringify({ command: 'recording_end' }));
        return;
      }

      if (recordingSettings?.enabled && recordingDeadline && Date.now() >= recordingDeadline) {
        throw new Error('録音開始の待機中に停止期限に達しました。');
      }

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
      if (recordingSettings?.enabled) notifyRecording(automatic ? '録音を自動開始しました。' : '録音を開始しました。');
    } catch (e) {
      isRecording = false;
      stopRecordingWatchdog();
      await cleanupAudioCapture();
      renderRecState();
      if (granted && ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ command: 'recording_end' }));
      log('ERROR: ' + e);
      notifyRecording('録音開始に失敗しました。マイクと接続を確認してください。', true);
    }
  }

  async function stopRecording(reason = 'manual') {
    if (!isRecording && !stream) return true;
    try {
      // これだけが録音をOFFにする経路。先に意思状態を落として復旧処理を止める。
      isRecording = false;
      if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ command: 'recording_end' }));
      recordingDeadline = null;
      stopRecordingWatchdog();
      renderRecState();
      if (selAsrModel) selAsrModel.disabled = selAsrModel.dataset.switchable === 'false';

      await flushHearingTranscriptSave();
      await cleanupAudioCapture();
      await flushAsrQueue();
      // ★ WSは閉じない（常時接続を維持してpatient_changedを受信し続ける）

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
      }
      log('recording stop');
      if (recordingSettings?.enabled) notifyRecording(reason === 'time_limit' ? '設定時間に達したため録音を停止しました。' : '録音を停止しました。');
      return true;
    } catch (e) {
      log('ERROR: ' + e);
      return false;
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
    patientSelectEl.addEventListener('focus', () => { void loadRecentPatients(); });
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
      await queueRecordingOperation(() => isRecording ? stopRecording() : startRecording());
    };
  }

  if (btnMemo) {
    btnMemo.addEventListener('click', async e => {
      e.preventDefault();
      if (isRecording) {
        const proceed = window.confirm('録音中です。録音を終了して音声メモを開きますか？');
        if (!proceed) return;
        const stopped = await queueRecordingOperation(() => stopRecording());
        if (!stopped) {
          window.alert('録音を正常に終了できなかったため、音声メモを開きませんでした。');
          return;
        }
      }
      window.location.href = btnMemo.href;
    });
  }
  // 旧Start/Stopボタン互換
  if (btnStart) btnStart.onclick = () => queueRecordingOperation(() => startRecording());
  if (btnStop) btnStop.onclick = () => queueRecordingOperation(() => stopRecording());

  // ===== 初期化 =====
  loadAppVersion();
  loadAsrModels();
  loadLlmModels();
  loadLlmPrompts();
  loadHearingTranslationSettings();
  loadRecentPatients();
  connectWs();  // ★ 起動時にWS接続（常時接続）
})();
