(() => {
  // ===== DOM =====
  const logEl = document.getElementById('log');
  const levelEl = document.getElementById('level');
  const btnStart = document.getElementById('btnStart');
  const btnStop  = document.getElementById('btnStop');

  const selSession = document.getElementById('selSession');
  const selAsrModel = document.getElementById('selAsrModel');
  const chkHearing = document.getElementById('chkHearing');

  const patientIdEl = document.getElementById('patientId');
  const asrModelNameEl = document.getElementById('asrModelName');

  const txEl = document.getElementById('transcript');

  const selLlmModel = document.getElementById('selLlmModel');
  const selSoapPrompt = document.getElementById('selSoapPrompt');
  const selSoapHistory = document.getElementById('selSoapHistory');
  const btnSoap = document.getElementById('btnSoap');
  const btnCopySoap = document.getElementById('btnCopySoap');
  const summaryEl = document.getElementById('summary');

  // const wsStatusEl = document.getElementById('wsStatus');
  const dotWs = document.getElementById('dotWs');
  
  const btnRec = document.getElementById("btnRec");

  const btnCorrectAsr = document.getElementById('btnCorrectAsr');
  const btnRebuild = document.getElementById('btnRebuild');
  const btnCopyAsr = document.getElementById('btnCopyAsr');

  let isRecording = false;

  // ===== state =====
  let ws = null;
  let audioCtx = null, srcNode = null, procNode = null, stream = null;

  // いま「録音中のセッション」(serverが決めた .txt 名)
  let currentSessionTxt = '';

  function renderRecState(){
    if(!btnRec) return;

    if(isRecording){
      btnRec.classList.add("recording");
    }else{
      btnRec.classList.remove("recording");
    }
  }

  async function startRecordingWrapper(){
    await btnStart.onclick();   // 既存処理流用
    isRecording = true;
    renderRecState();
  }

  async function stopRecordingWrapper(){
    await btnStop.onclick();
    isRecording = false;
    renderRecState();
  }

  function log(s){
    if (!logEl) return;
    logEl.textContent += s + '\n';
    logEl.scrollTop = logEl.scrollHeight;
  }

  // function setWsStatus(ok){
  //   if (wsStatusEl) wsStatusEl.textContent = ok ? 'connected' : 'disconnected';
  //   if (dotWs){
  //     dotWs.classList.toggle('ok', !!ok);
  //     dotWs.classList.toggle('bad', !ok);
  //   }
  // }
  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('btnToggleLog');

    // 既定：ログは畳む（表示したいならこの1行を消す）
    document.body.classList.add('log-collapsed');

    function setLogVisible(show) {
      document.body.classList.toggle('log-collapsed', !show);
      document.body.classList.toggle('log-shown', show);
      if (btn) btn.setAttribute('aria-pressed', show ? 'true' : 'false');
    }

    if (btn) {
      btn.addEventListener('click', () => {
        const show = document.body.classList.contains('log-collapsed'); // 今畳まれてるなら開く
        setLogVisible(show);
      });
    }

    // オプション：Escでログ閉じる
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') setLogVisible(false);
    });
  });

  function setPatient(pid){
    if (patientIdEl) patientIdEl.textContent = pid || '(未設定)';
  }

  function clearTranscript(){
    if (txEl) txEl.value = '';
  }

  function clearSummary(){
    if (summaryEl) summaryEl.value = '';
  }

  function setAsrModelNameFromMeta(meta){
    // app.py は {meta:{asr:{model_name/model_path}}} を返す想定
    const name = meta?.asr?.model_name || meta?.asr?.model_path || '(未設定)';
    if (asrModelNameEl) asrModelNameEl.textContent = name;
  }

  function safeSelectValue(selectEl, value){
    if (!selectEl) return;
    if (!value) return;
    const ok = [...selectEl.options].some(o => o.value === value);
    if (ok) selectEl.value = value;
  }

  // ===== API loaders =====
  async function loadAsrModels(){
    if (!selAsrModel) return;
    try{
      const r = await fetch('/api/asr/models');
      const j = await r.json();
      selAsrModel.innerHTML = '';
      const opt0 = document.createElement('option');
      opt0.value = '';
      opt0.textContent = 'ASR model...';
      selAsrModel.appendChild(opt0);
      (j.models || []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.label;
        selAsrModel.appendChild(opt);
      });
      if (j.current) selAsrModel.value = j.current;
      log(`[ui] ASR models loaded (current=${j.current || 'n/a'})`);
    }catch(e){
      log(`[ui] loadAsrModels failed: ${e}`);
    }
  }

  async function loadLlmModels(){
    if (!selLlmModel) return;
    try{
      const r = await fetch('/api/llm/models');
      const j = await r.json();
      if (!j || !j.ok) throw new Error(j?.error || 'LLM models unavailable');

      selLlmModel.innerHTML = '';
      const opt0 = document.createElement('option');
      opt0.value = '';
      opt0.textContent = 'LLM model...';
      selLlmModel.appendChild(opt0);

      (j.models || []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        opt.textContent = m;
        selLlmModel.appendChild(opt);
      });

      safeSelectValue(selLlmModel, j.default_model);
    }catch(e){
      log(`[ui] loadLlmModels failed: ${e}`);
    }
  }

  async function loadLlmPrompts(){
    if (!selSoapPrompt) return;
    try{
      const r = await fetch('/api/llm/prompts');
      const j = await r.json();
      if (!j || !j.ok) throw new Error(j?.error || 'Prompt list unavailable');

      const cur = selSoapPrompt.value;
      selSoapPrompt.innerHTML = '';

      const opt0 = document.createElement('option');
      opt0.value = '';
      opt0.textContent = 'Prompt...';
      selSoapPrompt.appendChild(opt0);

      (j.items || []).forEach(it => {
        const o = document.createElement('option');
        o.value = it.id;
        o.textContent = it.label || it.id;
        selSoapPrompt.appendChild(o);
      });

      // restore / default
      safeSelectValue(selSoapPrompt, cur);
      if (!selSoapPrompt.value) safeSelectValue(selSoapPrompt, j.default_prompt_id);
      if (!selSoapPrompt.value && (j.items || []).length) selSoapPrompt.value = j.items[0].id;
    }catch(e){
      log(`[ui] loadLlmPrompts failed: ${e}`);
    }
  }

  function getSessionForLlm(){
    // 録音中は server が送ってくる currentSessionTxt を優先
    if (btnStart && btnStart.disabled && currentSessionTxt) return currentSessionTxt;
    return (selSession?.value || '').trim();
  }

  async function loadLlmHistory(){
    if (!selSoapHistory) return;
    const session = getSessionForLlm();
    if (!session){
      selSoapHistory.innerHTML = '<option value="">AI履歴</option>';
      return;
    }
    try{
      const r = await fetch('/api/llm/history?session=' + encodeURIComponent(session));
      const j = await r.json();
      if (!j || !j.ok) throw new Error(j?.error || 'LLM history unavailable');

      const cur = selSoapHistory.value;
      selSoapHistory.innerHTML = '';

      const opt0 = document.createElement('option');
      opt0.value = '';
      opt0.textContent = '履歴…';
      selSoapHistory.appendChild(opt0);

      (j.items || []).forEach(it => {
        const o = document.createElement('option');
        o.value = it.id;
        o.textContent = it.label || it.id;
        selSoapHistory.appendChild(o);
      });

      safeSelectValue(selSoapHistory, cur);
    }catch(e){
      log(`[ui] loadLlmHistory failed: ${e}`);
    }
  }

  async function loadLlmHistoryItem(name){
    if (!name) return;
    try{
      // console.log("DEBUG: loadLlmHistoryItem called with:", name);
      const r = await fetch('/api/llm/history/item/' + encodeURIComponent(name));
      const j = await r.json();
      if (!j || !j.ok) throw new Error(j?.error || 'LLM history item unavailable');
      if (summaryEl){
        summaryEl.value = j.summary || '';
        summaryEl.scrollTop = 0;
      }
      log(`[llm] loaded history ${name}`);
    }catch(e){
      log(`[llm] load history failed: ${e}`);
    }
  }

  async function refreshSessions(){
    if (!selSession) return;
    try{
      const res = await fetch('/api/sessions');
      const data = await res.json();
      const items = data.items || [];
      const prev = selSession.value;
      selSession.innerHTML = '<option value="">認識履歴</option>' +
        items.map(it => `<option value="${it.name}">${it.label}</option>`).join('');
      if (prev && items.some(it => it.name === prev)) selSession.value = prev;
    }catch(e){
      log('[sessions] ERROR: ' + e);
    }
  }

  async function loadSession(name){
    if (!name) return;
    try{
      const res = await fetch('/api/session/' + encodeURIComponent(name));
      const data = await res.json();
      if (data.error) {
        log('[session] ' + data.error);
        return;
      }
      setPatient(data.patient_id);
      currentSessionTxt = name;
      if (txEl){
        txEl.value = data.text || '';
        txEl.scrollTop = txEl.scrollHeight;
      }
      clearSummary();
      setAsrModelNameFromMeta(data.meta || null);
      await loadLlmHistory();
      log(`[session] loaded ${name}`);
    }catch(e){
      log('[session] ERROR: ' + e);
    }
  }

  async function applyAsrCorrection(){
    if (!txEl) return;

    try{
      // 録音中にtxtを上書きすると体験がややこしいので止める（必要なら外してOK）
      if (btnStart && btnStart.disabled){
        log('[correct] 録音中は補正できません（停止してから）');
        return;
      }

      const session = getSessionForLlm();
      if (!session){
        log('[correct] no session selected');
        return;
      }

      if (btnCorrectAsr) btnCorrectAsr.disabled = true;

      const r = await fetch('/api/correct/apply', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ session })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));

      if (txEl && typeof j.text === 'string'){
        txEl.value = j.text;
        txEl.scrollTop = txEl.scrollHeight;
      }
      clearSummary(); // transcriptが変わるので要約はクリア推奨

      const st = j.stats || {};
      log(`[correct] ${j.changed ? 'changed' : 'no change'} lines ${st.lines_in}->${st.lines_out} dropped=${st.dropped_lines} changed_lines=${st.changed_lines}`);
    }catch(e){
      log(`[correct] failed: ${e}`);
    }finally{
      if (btnCorrectAsr) btnCorrectAsr.disabled = false;
    }
  }

  async function rebuildFromJsonl(){
    const session = getSessionForLlm();
    if (!session){
      log('[rebuild] no session selected');
      return;
    }
    try{
      if (btnRebuild) btnRebuild.disabled = true;
      log(`[rebuild] start session=${session}`);

      const r = await fetch('/api/session/rebuild', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ session })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));

      if (txEl){
        txEl.value = j.text || '';
        txEl.scrollTop = txEl.scrollHeight;
      }
      log(`[rebuild] done segments=${j.stats?.segments ?? '?'} chars=${j.stats?.chars ?? '?'}`);
    }catch(e){
      log(`[rebuild] failed: ${e}`);
    }finally{
      if (btnRebuild) btnRebuild.disabled = false;
    }
  }

  async function copyAsrToClipboard(){
    if (!txEl) return;
    const text = txEl.value || '';
    if (!text){
      log('[ui] copy: empty');
      return;
    }
    try{
      await navigator.clipboard.writeText(text);
      log('[ui] copied SOAP to clipboard');
    }catch(e){
      log('[ui] copy failed: ' + e);
    }
  }

  // ===== LLM actions =====
  async function runSoapSummary(){
    if (!summaryEl || !btnSoap) return;
    try{
      btnSoap.disabled = true;
      if (btnCopySoap) btnCopySoap.disabled = true;

      const session = getSessionForLlm();
      if (!session){
        log('[llm] no session selected');
        return;
      }

      const model = (selLlmModel?.value || '').trim();
      const prompt_id = (selSoapPrompt?.value || 'soap_v1').trim();
      const asr_text = (txEl?.value || '').trim();
      if (!asr_text){
        log('[llm] transcript empty');
        return;
      }

      log(`[llm] SOAP start (${session}) model=${model || '(default)'} prompt=${prompt_id}`);

      const r = await fetch('/api/llm/soap', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ session, model, prompt_id, asr_text })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));

      summaryEl.value = j.summary || '';
      summaryEl.scrollTop = 0;
      log(`[llm] SOAP done model=${j.model || '?'} elapsed=${j.elapsed_sec}s saved=${j.saved_name || 'n/a'}`);

      await loadLlmHistory();
    }catch(e){
      log(`[llm] SOAP failed: ${e}`);
    }finally{
      btnSoap.disabled = false;
      if (btnCopySoap) btnCopySoap.disabled = false;
    }
  }

  async function copySoapToClipboard(){
    if (!summaryEl) return;
    const text = summaryEl.value || '';
    if (!text){
      log('[ui] copy: empty');
      return;
    }
    try{
      if (navigator.clipboard && navigator.clipboard.writeText){
        await navigator.clipboard.writeText(text);
      }else{
        summaryEl.focus();
        summaryEl.select();
        document.execCommand('copy');
        summaryEl.setSelectionRange(0,0);
      }
      log('[ui] copied SOAP to clipboard');
    }catch(e){
      log(`[ui] copy failed: ${e}`);
    }
  }

  // ===== recording =====
  const targetSampleRate = 48000;
  const chunkSamples = 2400; // 50ms @48k
  let pcmBuf = new Float32Array(0);

  function appendBuf(a, b){
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0);
    out.set(b, a.length);
    return out;
  }

  async function startRecording(){
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        log('ERROR: getUserMedia unavailable. Try HTTPS or localhost. protocol=' + location.protocol + ' host=' + location.host);
        return;
      }

      const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
      ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      // ws.onopen = () => { log('[ws] open'); setWsStatus(true); };
      // ws.onclose = () => { log('[ws] close'); setWsStatus(false); };
      ws.onerror = (e) => log('[ws] error ' + e);

      let transcript = '';

      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);

        if (msg.type === 'level' && levelEl) levelEl.textContent = msg.dbfs;

        if (msg.type === 'status') {
          setPatient(msg.patient_id);
          if (msg.session_txt) currentSessionTxt = msg.session_txt;
          
          // ★追加：サーバがmetaを返しているなら確定表示
          if (msg.meta) setAsrModelNameFromMeta(msg.meta);
          else if (msg.asr_model && asrModelNameEl) asrModelNameEl.textContent = msg.asr_model;

          log('[status] ' + msg.msg + (msg.session_txt ? ' session=' + msg.session_txt : ''));
          loadLlmHistory();
        }

        if (msg.type === 'patient_changed') {
          setPatient(msg.patient_id);
          if (msg.session_txt) currentSessionTxt = msg.session_txt;
          transcript = '';
          clearTranscript();
          clearSummary();
          log('[patient_changed] ' + (msg.patient_id || '(未設定)') + (msg.session_txt ? ' session=' + msg.session_txt : ''));
          loadLlmHistory();
        }

        if (msg.type === 'saved') {
          log(`[saved] ${msg.wav} dur=${msg.dur}s`);
        }

        if (msg.type === 'asr') {
          if (msg.text) {
            transcript += (transcript ? ' ' : '') + msg.text;
            if (txEl){
              txEl.value = transcript;
              txEl.scrollTop = txEl.scrollHeight;
            }
          }
        }

        if (msg.type === 'error') {
          log('[error] ' + JSON.stringify(msg));
        }
      };

      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
          sampleRate: targetSampleRate,
        }
      });

      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: targetSampleRate });
      srcNode = audioCtx.createMediaStreamSource(stream);

      const bufferSize = 2048;
      procNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      procNode.onaudioprocess = (ev) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = ev.inputBuffer.getChannelData(0);
        pcmBuf = appendBuf(pcmBuf, input);

        while (pcmBuf.length >= chunkSamples) {
          const chunk = pcmBuf.slice(0, chunkSamples);
          pcmBuf = pcmBuf.slice(chunkSamples);
          ws.send(chunk.buffer);
        }
      };

      srcNode.connect(procNode);
      procNode.connect(audioCtx.destination);

      //録音中Disable
      if (btnStart) btnStart.disabled = true;
      if (btnStop) btnStop.disabled = false;
      if (selSession) selSession.disabled = true;
      if (selAsrModel) selAsrModel.disabled = true;

      if(btnRebuild) btnRebuild.disabled = true;
      if(btnCorrectAsr) btnCorrectAsr.disabled = true;

      // 録音開始時点の選択モデルを表示（metaが来るまでの暫定表示）
      if (asrModelNameEl) {
        const label = selAsrModel?.selectedOptions?.[0]?.textContent || selAsrModel?.value || '(未設定)';
        asrModelNameEl.textContent = label;
      }

      clearTranscript();
      clearSummary();

      log('recording start');
    } catch (e) {
      log('ERROR: ' + e);
    }
  }

  async function stopRecording(){
    try {
      if (procNode) procNode.disconnect();
      if (srcNode) srcNode.disconnect();
      if (audioCtx) await audioCtx.close();
      if (stream) stream.getTracks().forEach(t => t.stop());

      procNode = null; srcNode = null; audioCtx = null; stream = null;

      if (ws) { ws.close(); ws = null; }

      if (btnStart) btnStart.disabled = false;
      if (btnStop) btnStop.disabled = true;
      if (selSession) selSession.disabled = false;
      if (selAsrModel) selAsrModel.disabled = false;

      if(btnRebuild) btnRebuild.disabled = false;
      if(btnCorrectAsr) btnCorrectAsr.disabled = false;

      isRecording = false;
      renderRecState();

      await refreshSessions();
      log('recording stop');
    } catch (e) {
      log('ERROR: ' + e);
    }
  }

  // ===== events =====
  if (chkHearing){
    chkHearing.onchange = () => document.body.classList.toggle('hearing', chkHearing.checked);
  }

  if (selAsrModel){
    selAsrModel.addEventListener('change', async () => {
      const id = selAsrModel.value;
      if (!id) return;
      try{
        const r = await fetch('/api/asr/model', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ id })
        });
        const j = await r.json();
        if (j.ok){
          log(`[ui] ASR model switched -> ${j.current}`);
        }else{
          log(`[ui] ASR model switch failed: ${j.error}`);
          await loadAsrModels();
        }
      }catch(e){
        log(`[ui] ASR model switch error: ${e}`);
        await loadAsrModels();
      }
    });
  }

  if (selSession){
    selSession.onchange = async () => {
      if (btnStart && btnStart.disabled) return; // 録音中は触らない
      await loadSession(selSession.value);
    };
  }

  if (selSoapHistory){
    selSoapHistory.onchange = async () => {
      const name = selSoapHistory.value;
      if (!name) return;
      await loadLlmHistoryItem(name);
    };
  }

  if (btnSoap) btnSoap.onclick = () => runSoapSummary();
  if (btnCopySoap) btnCopySoap.onclick = () => copySoapToClipboard();

  if (btnStart) btnStart.onclick = () => startRecording();
  if (btnStop) btnStop.onclick = () => stopRecording();

  if(btnRec) btnRec.onclick = async () => {
    if(isRecording){
      await stopRecordingWrapper();
    }else{
      await startRecordingWrapper();
    }
  };

  if (btnCorrectAsr) btnCorrectAsr.onclick = () => applyAsrCorrection();
  if (btnRebuild) btnRebuild.onclick = () => rebuildFromJsonl();
  if (btnCopyAsr) btnCopyAsr.onclick = () => copyAsrToClipboard();

  // ===== init =====
  // setWsStatus(false);
  refreshSessions();
  loadAsrModels();
  loadLlmModels();
  loadLlmPrompts();
  loadLlmHistory();
})();
