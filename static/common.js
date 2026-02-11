(() => {
  "use strict";

  const logEl = document.getElementById("log");
  const levelEl = document.getElementById("level");
  const btnStart = document.getElementById("btnStart");
  const btnStop  = document.getElementById("btnStop");
  const btnSoap  = document.getElementById("btnSoap");
  const btnCopySoap = document.getElementById("btnCopySoap");
  const selLlmModel = document.getElementById("selLlmModel");
  const selSoapPrompt = document.getElementById("selSoapPrompt");
  const txEl = document.getElementById("transcript");
  const summaryEl = document.getElementById("summary");

  const wsStatusEl = document.getElementById("wsStatus");
  const dotWs = document.getElementById("dotWs");

  const patientIdEl = document.getElementById("patientId");
  const selSession = document.getElementById("selSession");

  const chkHearing = document.getElementById("chkHearing");

  const selAsrModel = document.getElementById("selAsrModel");
  const modelEl = document.getElementById("asrModelName");

  function log(s){
    if (!logEl) return;
    logEl.textContent += s + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setWsStatus(ok){
    if (wsStatusEl) wsStatusEl.textContent = ok ? "connected" : "disconnected";
    if (dotWs){
      dotWs.classList.toggle("ok", !!ok);
      dotWs.classList.toggle("bad", !ok);
    }
  }

  function setPatient(pid){
    const v = pid || "(未設定)";
    if (patientIdEl) patientIdEl.textContent = v;
  }

  function clearTranscript(){
    if (txEl) txEl.value = "";
  }

  function clearSummary(){
    if (!summaryEl) return;
    summaryEl.value = "";
  }

  function setAsrModelNameFromMeta(meta){
    const name =
      meta?.asr?.model_name ||
      meta?.asr?.model_path ||
      "(未設定)";
    if (modelEl) modelEl.textContent = name;
  }

  // ---------- ASR models ----------
  async function loadAsrModels(){
    if (!selAsrModel) return;
    try{
      const r = await fetch("/api/asr/models");
      const j = await r.json();

      selAsrModel.innerHTML = "";
      const opt0 = document.createElement("option");
      opt0.value = "";
      opt0.textContent = "ASR model...";
      selAsrModel.appendChild(opt0);

      (j.models || []).forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label;
        selAsrModel.appendChild(opt);
      });

      if (j.current){
        selAsrModel.value = j.current;
      }
      log(`[ui] ASR models loaded (current=${j.current || "n/a"})`);
    }catch(e){
      log(`[ui] loadAsrModels failed: ${e}`);
    }
  }

  if (selAsrModel){
    selAsrModel.addEventListener("change", async () => {
      const id = (selAsrModel.value || "").trim();
      if(!id) return;
      try{
        const r = await fetch("/api/asr/model", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({id})
        });
        const j = await r.json();
        if(j.ok){
          log(`[ui] ASR model switched -> ${j.current}`);
        }else{
          log(`[ui] ASR model switch failed: ${j.error}`);
          await loadAsrModels(); // 戻す
        }
      }catch(e){
        log(`[ui] ASR model switch error: ${e}`);
        await loadAsrModels();
      }
    });
  }

  // ---------- LLM models ----------
  function setSelectOptions(sel, items, placeholderText){
    sel.innerHTML = "";

    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = placeholderText;
    sel.appendChild(opt0);

    (items || []).forEach(v => {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      sel.appendChild(opt);
    });
  }

  function trySelectValue(sel, value){
    if (!value) return false;
    const ok = [...sel.options].some(o => o.value === value);
    if (ok) sel.value = value;
    return ok;
  }

  async function loadLlmModels(){
    if (!selLlmModel) return;
    try{
      const r = await fetch("/api/llm/models");
      const j = await r.json();

      // APIの揺れ吸収：models/items のどちらでもOKにする
      const models = j.models || j.items || [];
      const defaultModel = (j.default_model || j.default || j.model_default || "").trim();
      const currentModel = (j.current || "").trim();

      setSelectOptions(selLlmModel, models, "LLM model...");

      // 優先順位：current → default → 先頭モデル
      if (!trySelectValue(selLlmModel, currentModel)) {
        if (!trySelectValue(selLlmModel, defaultModel)) {
          if (models.length) selLlmModel.value = models[0];
        }
      }

      log(`[ui] LLM models loaded (default=${defaultModel || "n/a"} selected=${selLlmModel.value || "n/a"})`);
    }catch(e){
      log(`[ui] loadLlmModels failed: ${e}`);
      // 失敗しても操作不能にならないように最低限
      if (selLlmModel && selLlmModel.options.length === 0){
        setSelectOptions(selLlmModel, [], "LLM model...");
      }
    }
  }

  // ---------- LLM prompts ----------
  async function loadLlmPrompts(){
    if (!selSoapPrompt) return;
    try{
      const r = await fetch("/api/llm/prompts");
      const j = await r.json();
      if (!j || j.ok === false) throw new Error(j?.error || "prompt list unavailable");

      // API揺れ吸収：items/prompts
      const items = j.items || j.prompts || [];
      const defaultPrompt = (j.default_prompt || "").trim();

      const cur = selSoapPrompt.value;

      selSoapPrompt.innerHTML = "";
      const opt0 = document.createElement("option");
      opt0.value = "";
      opt0.textContent = "Prompt...";
      selSoapPrompt.appendChild(opt0);

      items.forEach(it => {
        const o = document.createElement("option");
        o.value = it.id;
        o.textContent = it.label || it.id;
        selSoapPrompt.appendChild(o);
      });

      // 優先順位：維持できるなら維持 → default → 先頭
      if (!trySelectValue(selSoapPrompt, cur)) {
        if (!trySelectValue(selSoapPrompt, defaultPrompt)) {
          if (items.length) selSoapPrompt.value = items[0].id;
        }
      }

      log(`[ui] LLM prompts loaded (selected=${selSoapPrompt.value || "n/a"})`);
    }catch(e){
      log(`[ui] loadLlmPrompts failed: ${e}`);
      // fallback: 何も無いなら最低限を入れる
      if (selSoapPrompt.options.length === 0){
        selSoapPrompt.innerHTML = `
          <option value="">Prompt...</option>
          <option value="soap_v1">SOAP(推測禁止)</option>
          <option value="soap_v1_short">SOAP(短め)</option>
        `;
        selSoapPrompt.value = "soap_v1";
      }
    }
  }

  // ---------- sessions ----------
  async function refreshSessions() {
    if (!selSession) return;
    try {
      const res = await fetch("/api/sessions");
      const data = await res.json();
      const items = data.items || [];

      const prev = selSession.value;

      selSession.innerHTML =
        `<option value="">履歴を読み込み…</option>` +
        items.map(it => `<option value="${it.name}">${it.label}</option>`).join("");

      if (prev && items.some(it => it.name === prev)) selSession.value = prev;
    } catch (e) {
      log("[sessions] ERROR: " + e);
    }
  }

  async function loadSession(name) {
    if (!name) return;
    try {
      const res = await fetch("/api/session/" + encodeURIComponent(name));
      const data = await res.json();
      if (data.error) {
        log("[session] " + data.error);
        return;
      }
      setPatient(data.patient_id);
      if (txEl){
        txEl.value = data.text || "";
        txEl.scrollTop = txEl.scrollHeight;
      }
      clearSummary();
      setAsrModelNameFromMeta(data.meta || null);
      log(`[session] loaded ${name}`);
    } catch (e) {
      log("[session] ERROR: " + e);
    }
  }

  if (selSession){
    selSession.onchange = async () => {
      if (btnStart && btnStart.disabled) return; // 録音中
      await loadSession(selSession.value);
    };
  }

  // ---------- hearing mode ----------
  if (chkHearing){
    chkHearing.onchange = () => {
      document.body.classList.toggle("hearing", chkHearing.checked);
    };
  }

  // ---------- SOAP ----------
  async function runSoapSummary(){
    if (!summaryEl || !btnSoap) return;

    try{
      btnSoap.disabled = true;
      if (btnCopySoap) btnCopySoap.disabled = true;

      const session = (selSession?.value || "").trim();
      const model = (selLlmModel?.value || "").trim(); // 空ならサーバ側defaultに任せる
      const prompt_id = (selSoapPrompt?.value || "soap_v1").trim() || "soap_v1";

      log(`[llm] SOAP start ${session ? "(" + session + ")" : "(current)"} model=${model || "(default)"} prompt=${prompt_id}`);

      const r = await fetch("/api/llm/soap", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ session, model, prompt_id })
      });

      const j = await r.json();

      if (j.ok){
        summaryEl.value = j.summary || "";
        summaryEl.scrollTop = 0;
        log(`[llm] SOAP done model=${j.model || model || "default"} elapsed=${j.elapsed_sec ?? "?"}s`);
      }else{
        log(`[llm] SOAP failed: ${j.error || r.status}`);
      }
    }catch(e){
      log(`[llm] SOAP error: ${e}`);
    }finally{
      btnSoap.disabled = false;
      if (btnCopySoap) btnCopySoap.disabled = false;
    }
  }

  if (btnSoap){
    btnSoap.onclick = () => runSoapSummary();
  }

  async function copySoapToClipboard(){
    if (!summaryEl) return;
    const text = summaryEl.value || "";
    if (!text){
      log("[ui] copy: empty");
      return;
    }
    try{
      if (navigator.clipboard && navigator.clipboard.writeText){
        await navigator.clipboard.writeText(text);
      }else{
        summaryEl.focus();
        summaryEl.select();
        document.execCommand("copy");
        summaryEl.setSelectionRange(0,0);
      }
      log("[ui] copied SOAP to clipboard");
    }catch(e){
      log(`[ui] copy failed: ${e}`);
    }
  }

  if (btnCopySoap){
    btnCopySoap.onclick = () => copySoapToClipboard();
  }

  // ---------- audio/ws ----------
  let ws = null;
  let audioCtx = null, srcNode = null, procNode = null, stream = null;

  const targetSampleRate = 48000;
  const chunkSamples = 2400; // 50ms @48k
  let pcmBuf = new Float32Array(0);

  function appendBuf(a, b){
    const out = new Float32Array(a.length + b.length);
    out.set(a, 0); out.set(b, a.length);
    return out;
  }

  if (btnStart){
    btnStart.onclick = async () => {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          log("ERROR: getUserMedia unavailable. Try HTTPS or localhost. protocol=" + location.protocol + " host=" + location.host);
          return;
        }

        const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => { log("[ws] open"); setWsStatus(true); };
        ws.onclose = () => { log("[ws] close"); setWsStatus(false); };
        ws.onerror = (e) => log("[ws] error " + e);

        let transcript = "";

        ws.onmessage = (ev) => {
          const msg = JSON.parse(ev.data);

          if (msg.type === "level" && levelEl) levelEl.textContent = msg.dbfs;

          if (msg.type === "saved") {
            log(`[saved] ${msg.wav} dur=${msg.dur}s`);
          }

          if (msg.type === "asr") {
            if (msg.text) {
              transcript += (transcript ? " " : "") + msg.text;
              if (txEl){
                txEl.value = transcript;
                txEl.scrollTop = txEl.scrollHeight;
              }
            }
            // metaが長いなら必要に応じて短縮してもOK
            log(`[asr] seg#${msg.seg_id} ${msg.text || "(empty)"}`);
          }

          if (msg.type === "status") {
            setPatient(msg.patient_id);
            log("[status] " + msg.msg);
          }

          if (msg.type === "patient_changed") {
            setPatient(msg.patient_id);
            transcript = "";
            clearTranscript();
            clearSummary();
            log("[patient_changed] " + (msg.patient_id || "(未設定)"));
          }

          if (msg.type === "error") log("[error] " + JSON.stringify(msg));
        };

        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            channelCount: 1,
            sampleRate: targetSampleRate
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

        btnStart.disabled = true;
        if (btnStop) btnStop.disabled = false;

        if (selSession) selSession.disabled = true;
        if (selAsrModel) selAsrModel.disabled = true;

        clearTranscript();
        clearSummary();

        log("recording start");
      } catch (e) {
        log("ERROR: " + e);
      }
    };
  }

  if (btnStop){
    btnStop.onclick = async () => {
      try {
        if (procNode) procNode.disconnect();
        if (srcNode) srcNode.disconnect();
        if (audioCtx) await audioCtx.close();
        if (stream) stream.getTracks().forEach(t => t.stop());

        procNode=null; srcNode=null; audioCtx=null; stream=null;

        if (ws) { ws.close(); ws=null; }

        if (btnStart) btnStart.disabled = false;
        btnStop.disabled = true;

        if (selSession) selSession.disabled = false;
        if (selAsrModel) selAsrModel.disabled = false;

        await refreshSessions();

        log("recording stop");
      } catch (e) {
        log("ERROR: " + e);
      }
    };
  }

  // ---------- initial ----------
  setWsStatus(false);
  loadAsrModels();
  loadLlmModels();
  loadLlmPrompts();
  refreshSessions();

})();
