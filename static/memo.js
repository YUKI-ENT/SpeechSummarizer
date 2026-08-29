(() => {
  const memoIdEl = document.getElementById('memoId');
  const patientIdEl = document.getElementById('patientId');
  const memoStatusEl = document.getElementById('memoStatus');
  const saveStatusEl = document.getElementById('saveStatus');
  const levelEl = document.getElementById('level');
  const btnRec = document.getElementById('btnRec');
  const btnBack = document.getElementById('btnBack');
  const btnCopyRaw = document.getElementById('btnCopyRaw');
  const btnCopyDraft = document.getElementById('btnCopyDraft');
  const btnClearAsr = document.getElementById('btnClearAsr');
  const btnClearLlm = document.getElementById('btnClearLlm');
  const btnLlm = document.getElementById('btnLlm');
  const btnReplaceDraft = document.getElementById('btnReplaceDraft');
  const btnAppendDraft = document.getElementById('btnAppendDraft');
  const btnReplaceMemoFromAsr = document.getElementById('btnReplaceMemoFromAsr');
  const btnAppendMemoFromAsr = document.getElementById('btnAppendMemoFromAsr');
  const btnInsertTemplate = document.getElementById('btnInsertTemplate');
  const btnNewMemo = document.getElementById('btnNewMemo');
  const rawTranscriptEl = document.getElementById('rawTranscript');
  const draftTextEl = document.getElementById('draftText');
  const llmResultEl = document.getElementById('llmResult');
  const selAction = document.getElementById('selAction');
  const selAsrModel = document.getElementById('selAsrModel');
  const selLlmModel = document.getElementById('selLlmModel');
  const selTemplate = document.getElementById('selTemplate');
  const memoHistory = document.getElementById('memoHistory');
  const toastEl = document.getElementById('memoToast');

  let memoId = '';
  let currentPatientId = '';
  let generation = 0;
  let ws = null;
  let audioCtx = null;
  let srcNode = null;
  let procNode = null;
  let stream = null;
  let pcmBuffer = new Float32Array(0);
  let isRecording = false;
  let saveTimer = null;
  let flushWaiter = null;
  let generationWaiter = null;
  let toastTimer = null;

  const storageKey = kind => `speechsummarizer:memo:${memoId}:${kind}`;

  function toast(message) {
    toastEl.textContent = message;
    toastEl.classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove('visible'), 2200);
  }

  function setStatus(text) {
    memoStatusEl.textContent = text;
  }

  function appendText(current, addition) {
    const left = (current || '').trimEnd();
    const right = (addition || '').trim();
    if (!right) return current || '';
    return left ? `${left}\n${right}` : right;
  }

  function setSelectOptions(select, items, getValue, getLabel) {
    select.replaceChildren(...items.map(item => {
      const option = document.createElement('option');
      option.value = getValue(item);
      option.textContent = getLabel(item);
      return option;
    }));
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const body = await response.json();
    if (!response.ok || !body.ok) throw new Error(body.error || String(response.status));
    return body;
  }

  async function createOrLoadMemo() {
    const requestedId = new URLSearchParams(location.search).get('id') || '';
    const data = requestedId
      ? await fetchJson(`/api/memos/${encodeURIComponent(requestedId)}`)
      : await fetchJson('/api/memos', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ document_type: 'free_text', title: '音声メモ' })
        });

    memoId = data.memo.id;
    currentPatientId = data.memo.patient_id || '';
    memoIdEl.textContent = memoId;
    patientIdEl.textContent = currentPatientId || '未指定';
    draftTextEl.value = data.draft || '';
    rawTranscriptEl.value = sessionStorage.getItem(storageKey('asr')) || '';
    llmResultEl.value = sessionStorage.getItem(storageKey('llm')) || '';
    generation = Number.parseInt(sessionStorage.getItem(storageKey('generation')) || '0', 10) || 0;
    renderLlmActions();
    history.replaceState(null, '', `/memo?id=${encodeURIComponent(memoId)}`);
  }

  function formatMemoLabel(item) {
    let dateLabel = item.updated_at || item.created_at || '';
    if (dateLabel) {
      const parsed = new Date(dateLabel);
      if (!Number.isNaN(parsed.getTime())) {
        dateLabel = parsed.toLocaleString('ja-JP', {
          year: 'numeric', month: '2-digit', day: '2-digit',
          hour: '2-digit', minute: '2-digit'
        });
      }
    }
    const currentMark = item.id === memoId ? '（表示中）' : '';
    return `${dateLabel || item.id}　${item.title || '音声メモ'}${currentMark}`;
  }

  async function loadMemoHistory() {
    const query = new URLSearchParams({ patient_id: currentPatientId });
    const data = await fetchJson(`/api/memos?${query}`);
    const items = data.items || [];
    setSelectOptions(memoHistory, items, item => item.id, formatMemoLabel);
    if (!items.length) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = '過去のメモはありません';
      memoHistory.replaceChildren(option);
      memoHistory.disabled = true;
      return;
    }
    memoHistory.disabled = false;
    memoHistory.value = memoId;
  }

  async function loadModels() {
    try {
      const asr = await fetch('/api/asr/models').then(r => r.json());
      setSelectOptions(selAsrModel, asr.models || [], item => item.id, item => item.label || item.id);
      selAsrModel.value = asr.current || '';
      selAsrModel.dataset.switchable = asr.switchable ? 'true' : 'false';
      selAsrModel.disabled = !asr.switchable;
    } catch (_) { /* Keep recording usable with the server default. */ }

    try {
      const llm = await fetch('/api/llm/models').then(r => r.json());
      const models = llm.models || (llm.default_model ? [llm.default_model] : []);
      setSelectOptions(selLlmModel, models, model => model, model => model);
      selLlmModel.value = llm.default_model || models[0] || '';
    } catch (_) { /* The API reports the error when AI processing is requested. */ }
  }

  async function loadMemoTemplates() {
    try {
      const data = await fetchJson('/api/memo-templates');
      const templates = data.templates || [];
      const placeholder = {
        text: '',
        label: templates.length ? '定型文を選択...' : '有効な定型文がありません'
      };
      setSelectOptions(
        selTemplate,
        [placeholder, ...templates],
        item => item.text,
        item => item.label
      );
      selTemplate.disabled = !templates.length;
      btnInsertTemplate.disabled = !templates.length;
    } catch (_) {
      setSelectOptions(selTemplate, [{ value: '', label: '定型文を読み込めません' }], item => item.value, item => item.label);
      selTemplate.disabled = true;
      btnInsertTemplate.disabled = true;
    }
  }

  async function loadMemoAiPrompts() {
    try {
      const data = await fetchJson('/api/memo-ai-prompts');
      const prompts = data.prompts || [];
      setSelectOptions(selAction, prompts, item => item.id, item => item.label || item.id);
      if (!prompts.length) throw new Error('AI処理プロンプトがありません');
      selAction.value = data.default_prompt_id || prompts[0].id;
      selAction.disabled = false;
      btnLlm.disabled = false;
    } catch (_) {
      setSelectOptions(selAction, [{ id: '', label: 'AI処理を読み込めません' }], item => item.id, item => item.label);
      selAction.disabled = true;
      btnLlm.disabled = true;
    }
  }

  function handleWsMessage(event) {
    let message;
    try { message = JSON.parse(event.data); } catch (_) { return; }
    if (message.type === 'level') levelEl.textContent = message.dbfs;
    if (
      message.type === 'asr' &&
      message.target?.type === 'memo' &&
      message.target.id === memoId
    ) {
      if (Number(message.generation || 0) !== generation) {
        toast('クリア前の認識結果を除外しました');
        return;
      }
      rawTranscriptEl.value = appendText(rawTranscriptEl.value, message.text);
      rawTranscriptEl.scrollTop = rawTranscriptEl.scrollHeight;
      sessionStorage.setItem(storageKey('asr'), rawTranscriptEl.value);
    }
    if (message.type === 'generation_set' && generationWaiter) {
      generationWaiter.resolve(message.generation);
      generationWaiter = null;
    }
    if (message.type === 'flush_complete' && flushWaiter) {
      flushWaiter.resolve();
      flushWaiter = null;
    }
    if (message.type === 'error') {
      setStatus('エラー');
      toast(`音声認識エラー: ${message.error || 'unknown'}`);
    }
  }

  async function connectWs() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    if (ws && ws.readyState === WebSocket.CONNECTING) {
      await new Promise(resolve => ws.addEventListener('open', resolve, { once: true }));
      return;
    }
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const query = new URLSearchParams({ target_type: 'memo', target_id: memoId });
    ws = new WebSocket(`${scheme}//${location.host}/ws?${query}`);
    ws.binaryType = 'arraybuffer';
    ws.onmessage = handleWsMessage;
    ws.onclose = () => {
      if (flushWaiter) {
        flushWaiter.resolve();
        flushWaiter = null;
      }
      if (generationWaiter) {
        generationWaiter.resolve(generation);
        generationWaiter = null;
      }
      if (isRecording) {
        isRecording = false;
        renderRecording();
        setStatus('接続切断');
      }
    };
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('WebSocket接続タイムアウト')), 5000);
      ws.addEventListener('open', () => { clearTimeout(timeout); resolve(); }, { once: true });
      ws.addEventListener('error', () => { clearTimeout(timeout); reject(new Error('WebSocket接続失敗')); }, { once: true });
    });
    await setServerGeneration(generation);
  }

  function setServerGeneration(value) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return Promise.resolve();
    if (generationWaiter) return generationWaiter.promise;
    let complete;
    const promise = new Promise(resolve => { complete = resolve; });
    const timeout = setTimeout(() => {
      if (!generationWaiter) return;
      generationWaiter.resolve(value);
      generationWaiter = null;
    }, 5000);
    generationWaiter = {
      promise,
      resolve: result => { clearTimeout(timeout); complete(result); }
    };
    ws.send(JSON.stringify({ command: 'set_generation', generation: value }));
    return promise;
  }

  async function createAudioPipeline() {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: 48000
      }
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    srcNode = audioCtx.createMediaStreamSource(stream);
    procNode = audioCtx.createScriptProcessor(2048, 1, 1);
    pcmBuffer = new Float32Array(0);
    procNode.onaudioprocess = event => {
      if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
      const input = event.inputBuffer.getChannelData(0);
      const merged = new Float32Array(pcmBuffer.length + input.length);
      merged.set(pcmBuffer);
      merged.set(input, pcmBuffer.length);
      pcmBuffer = merged;
      while (pcmBuffer.length >= 2400) {
        ws.send(pcmBuffer.slice(0, 2400).buffer);
        pcmBuffer = pcmBuffer.slice(2400);
      }
    };
    srcNode.connect(procNode);
    procNode.connect(audioCtx.destination);
    if (audioCtx.state === 'suspended') await audioCtx.resume();
  }

  async function cleanupAudio() {
    if (procNode) procNode.disconnect();
    if (srcNode) srcNode.disconnect();
    if (audioCtx) await audioCtx.close();
    if (stream) stream.getTracks().forEach(track => track.stop());
    procNode = srcNode = audioCtx = stream = null;
    pcmBuffer = new Float32Array(0);
  }

  function requestFlush() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return Promise.resolve();
    if (flushWaiter) return flushWaiter.promise;
    let complete;
    const promise = new Promise(resolve => { complete = resolve; });
    const timeout = setTimeout(() => {
      if (!flushWaiter) return;
      flushWaiter.resolve();
      flushWaiter = null;
      toast('音声認識の完了待ちがタイムアウトしました');
    }, 45000);
    flushWaiter = {
      promise,
      resolve: () => { clearTimeout(timeout); complete(); }
    };
    ws.send(JSON.stringify({ command: 'flush' }));
    return promise;
  }

  function renderRecording() {
    btnRec.classList.toggle('recording', isRecording);
    const label = isRecording ? 'メモ録音を停止' : 'メモ録音を開始';
    btnRec.title = label;
    btnRec.setAttribute('aria-label', label);
    btnRec.setAttribute('aria-pressed', isRecording ? 'true' : 'false');
    selAsrModel.disabled = isRecording || selAsrModel.dataset.switchable === 'false';
  }

  async function startRecording() {
    try {
      await connectWs();
      isRecording = true;
      await createAudioPipeline();
      renderRecording();
      setStatus('録音中');
    } catch (error) {
      isRecording = false;
      await cleanupAudio();
      renderRecording();
      setStatus('録音開始エラー');
      toast(error.message || String(error));
    }
  }

  async function stopRecording() {
    if (!isRecording) return true;
    isRecording = false;
    renderRecording();
    setStatus('認識完了待ち');
    try {
      await cleanupAudio();
      await requestFlush();
      setStatus('入力待ち');
      return true;
    } catch (error) {
      setStatus('停止エラー');
      toast(error.message || String(error));
      return false;
    }
  }

  async function clearAsr() {
    btnClearAsr.disabled = true;
    generation += 1;
    rawTranscriptEl.value = '';
    pcmBuffer = new Float32Array(0);
    sessionStorage.setItem(storageKey('asr'), '');
    sessionStorage.setItem(storageKey('generation'), String(generation));
    try {
      if (ws?.readyState === WebSocket.OPEN) await setServerGeneration(generation);
      toast('ASR入力をクリアしました');
    } catch (error) {
      toast(`世代の更新に失敗しました: ${error.message || error}`);
    } finally {
      btnClearAsr.disabled = false;
    }
  }

  function scheduleDraftSave() {
    clearTimeout(saveTimer);
    saveStatusEl.textContent = '保存待ち…';
    saveTimer = setTimeout(saveDraft, 700);
  }

  async function saveDraft() {
    clearTimeout(saveTimer);
    saveTimer = null;
    if (!memoId) return;
    saveStatusEl.textContent = '保存中…';
    try {
      await fetchJson(`/api/memos/${encodeURIComponent(memoId)}/draft`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: draftTextEl.value })
      });
      saveStatusEl.textContent = '保存済み';
      return true;
    } catch (error) {
      saveStatusEl.textContent = '保存エラー';
      toast(error.message || String(error));
      return false;
    }
  }

  function renderLlmActions() {
    const hasResult = Boolean(llmResultEl.value.trim());
    btnReplaceDraft.disabled = !hasResult;
    btnAppendDraft.disabled = !hasResult;
  }

  async function runLlm() {
    const source = rawTranscriptEl.value.trim();
    if (!source) {
      toast('ASR入力が空です');
      return;
    }
    if (!selAction.value) {
      toast('AI処理を選択してください');
      return;
    }
    btnLlm.disabled = true;
    btnReplaceDraft.disabled = true;
    btnAppendDraft.disabled = true;
    llmResultEl.value = 'AI処理中…';
    try {
      const result = await fetchJson(`/api/memos/${encodeURIComponent(memoId)}/llm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt_id: selAction.value, model: selLlmModel.value, text: source })
      });
      llmResultEl.value = result.text || '';
      sessionStorage.setItem(storageKey('llm'), llmResultEl.value);
      toast('AI送信しました');
    } catch (error) {
      llmResultEl.value = '';
      sessionStorage.removeItem(storageKey('llm'));
      toast(`AI処理エラー: ${error.message || error}`);
    } finally {
      btnLlm.disabled = selAction.disabled;
      renderLlmActions();
    }
  }

  function replaceMemoWith(text, sourceLabel) {
    if (!text.trim()) {
      toast(`${sourceLabel}が空です`);
      return;
    }
    if (draftTextEl.value.trim() && !window.confirm(`現在のメモを${sourceLabel}で置き換えますか？`)) return;
    draftTextEl.value = text;
    scheduleDraftSave();
    toast('メモを置き換えました');
  }

  function appendToMemo(text, sourceLabel) {
    if (!text.trim()) {
      toast(`${sourceLabel}が空です`);
      return;
    }
    draftTextEl.value = appendText(draftTextEl.value, text);
    scheduleDraftSave();
    toast('メモ末尾に追加しました');
  }

  function replaceDraft() {
    replaceMemoWith(llmResultEl.value, 'AI処理結果');
  }

  function appendResultToDraft() {
    appendToMemo(llmResultEl.value, 'AI処理結果');
  }

  function insertTemplate() {
    const text = selTemplate.value;
    if (!text) {
      toast('定型文を選択してください');
      return;
    }
    const start = draftTextEl.selectionStart ?? draftTextEl.value.length;
    const end = draftTextEl.selectionEnd ?? start;
    const before = draftTextEl.value.slice(0, start);
    const after = draftTextEl.value.slice(end);
    const prefix = before && !before.endsWith('\n') ? '\n' : '';
    const suffix = after && !after.startsWith('\n') ? '\n' : '';
    draftTextEl.value = `${before}${prefix}${text}${suffix}${after}`;
    const cursor = before.length + prefix.length + text.length;
    draftTextEl.focus();
    draftTextEl.setSelectionRange(cursor, cursor);
    scheduleDraftSave();
    selTemplate.value = '';
  }

  async function copyText(text) {
    await navigator.clipboard.writeText(text || '');
    toast('コピーしました');
  }

  async function prepareMemoNavigation(message) {
    if (isRecording && !window.confirm(message)) return false;
    if (isRecording && !(await stopRecording())) return false;
    if (!(await saveDraft())) {
      window.alert('メモを保存できなかったため、別のメモへ切り替えませんでした。');
      return false;
    }
    return true;
  }

  async function switchMemo(nextMemoId) {
    if (!nextMemoId || nextMemoId === memoId) return;
    if (!(await prepareMemoNavigation('メモ録音中です。録音を終了して別のメモを開きますか？'))) {
      memoHistory.value = memoId;
      return;
    }
    location.href = `/memo?id=${encodeURIComponent(nextMemoId)}`;
  }

  async function createNewMemo() {
    if (!(await prepareMemoNavigation('メモ録音中です。録音を終了して新しいメモを作成しますか？'))) return;
    btnNewMemo.disabled = true;
    try {
      const data = await fetchJson('/api/memos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: currentPatientId,
          document_type: 'free_text',
          title: '音声メモ'
        })
      });
      location.href = `/memo?id=${encodeURIComponent(data.memo.id)}`;
    } catch (error) {
      btnNewMemo.disabled = false;
      toast(`新規メモ作成エラー: ${error.message || error}`);
    }
  }

  btnRec.addEventListener('click', () => isRecording ? stopRecording() : startRecording());
  btnClearAsr.addEventListener('click', clearAsr);
  rawTranscriptEl.addEventListener('input', () => sessionStorage.setItem(storageKey('asr'), rawTranscriptEl.value));
  draftTextEl.addEventListener('input', scheduleDraftSave);
  selAsrModel.addEventListener('change', async () => {
    try {
      await fetchJson('/api/asr/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: selAsrModel.value })
      });
    } catch (error) { toast(error.message || String(error)); }
  });
  btnLlm.addEventListener('click', runLlm);
  btnClearLlm.addEventListener('click', () => {
    llmResultEl.value = '';
    sessionStorage.removeItem(storageKey('llm'));
    renderLlmActions();
  });
  btnReplaceDraft.addEventListener('click', replaceDraft);
  btnAppendDraft.addEventListener('click', appendResultToDraft);
  btnReplaceMemoFromAsr.addEventListener('click', () => replaceMemoWith(rawTranscriptEl.value, 'ASR入力'));
  btnAppendMemoFromAsr.addEventListener('click', () => appendToMemo(rawTranscriptEl.value, 'ASR入力'));
  btnInsertTemplate.addEventListener('click', insertTemplate);
  memoHistory.addEventListener('change', () => switchMemo(memoHistory.value));
  btnNewMemo.addEventListener('click', createNewMemo);
  btnCopyRaw.addEventListener('click', () => copyText(rawTranscriptEl.value));
  btnCopyDraft.addEventListener('click', () => copyText(draftTextEl.value));
  btnBack.addEventListener('click', async event => {
    event.preventDefault();
    if (isRecording && !window.confirm('メモ録音中です。録音を終了して診療画面へ戻りますか？')) return;
    if (isRecording && !(await stopRecording())) return;
    if (!(await saveDraft())) {
      window.alert('メモを保存できなかったため、診療画面へ戻りませんでした。');
      return;
    }
    location.href = btnBack.href;
  });

  window.addEventListener('beforeunload', event => {
    if (!isRecording) return;
    event.preventDefault();
    event.returnValue = '';
  });

  (async () => {
    try {
      await createOrLoadMemo();
      await Promise.all([loadModels(), loadMemoHistory(), loadMemoTemplates(), loadMemoAiPrompts()]);
      btnRec.disabled = false;
      btnNewMemo.disabled = false;
      setStatus('入力待ち');
      saveStatusEl.textContent = '保存済み';
    } catch (error) {
      setStatus('初期化エラー');
      toast(error.message || String(error));
    }
  })();
})();
