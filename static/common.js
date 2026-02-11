  const logEl = document.getElementById("log");
  const levelEl = document.getElementById("level");
  const btnStart = document.getElementById("btnStart");
  const btnStop  = document.getElementById("btnStop");
  const txEl = document.getElementById("transcript");

  const wsStatusEl = document.getElementById("wsStatus");
  const dotWs = document.getElementById("dotWs");

  const patientIdEl = document.getElementById("patientId");
//   const patientIdNavEl = document.getElementById("patientIdNav");
  const selSession = document.getElementById("selSession");


  const chkHearing = document.getElementById("chkHearing");

  const selAsrModel = document.getElementById("selAsrModel");

  const modelEl = document.getElementById("asrModelName");

  async function loadAsrModels(){
    try{
      const r = await fetch("/api/asr/models");
      const j = await r.json();

      // options rebuild
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

  selAsrModel.addEventListener("change", async () => {
    const id = selAsrModel.value;
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

  function setAsrModelNameFromMeta(meta){
    // app.py は {"meta": {"asr": {...}}} を返す
    const name =
      meta?.asr?.model_name ||
      meta?.asr?.model_path ||  // フォールバック（長いけど）
      "(未設定)";
    modelEl.textContent = name;
  }

  selSession.onchange = async () => {
    if (btnStart.disabled) { // 録音中（startがdisabled）なら触らせない
      return;
    }
    await loadSession(selSession.value);
  };


  // 起動時に読み込む
  loadAsrModels();

  function log(s){
    logEl.textContent += s + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setWsStatus(ok){
    wsStatusEl.textContent = ok ? "connected" : "disconnected";
    dotWs.classList.toggle("ok", !!ok);
    dotWs.classList.toggle("bad", !ok);
  }

  // hearing mode
  chkHearing.onchange = () => {
    document.body.classList.toggle("hearing", chkHearing.checked);
  };

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

  function setPatient(pid){
    const v = pid || "(未設定)";
    patientIdEl.textContent = v;
    // patientIdNavEl.textContent = v;
  }

  async function refreshSessions() {
    try {
      const res = await fetch("/api/sessions");
      const data = await res.json();
      const items = data.items || [];

      const prev = selSession.value;

      selSession.innerHTML = `<option value="">履歴を読み込み…</option>` +
        items.map(it => `<option value="${it.name}">${it.label}</option>`).join("");

      // できれば選択維持
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
      txEl.value = data.text || "";
      txEl.scrollTop = txEl.scrollHeight;
      // ★追加：モデル名表示
      setAsrModelNameFromMeta(j.meta || null);  
      log(`[session] loaded ${name}`);
    } catch (e) {
      log("[session] ERROR: " + e);
    }
  }


  function clearTranscript(){
    txEl.value = "";
  }

  btnStart.onclick = async () => {
    try {
      // getUserMedia availability check
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

        if (msg.type === "level") levelEl.textContent = msg.dbfs;

        if (msg.type === "saved") {
          log(`[saved] ${msg.wav} dur=${msg.dur}s`);
        }

        if (msg.type === "asr") {
          if (msg.text) {
            transcript += (transcript ? " " : "") + msg.text;
            txEl.value = transcript;
            txEl.scrollTop = txEl.scrollHeight; // 追従
          }
          log(`[asr] seg#${msg.seg_id} ${msg.text || "(empty)"} meta=${JSON.stringify(msg.meta)}`);
        }

        if (msg.type === "status") {
          setPatient(msg.patient_id);
          log("[status] " + msg.msg);
        }

        if (msg.type === "patient_changed") {
          setPatient(msg.patient_id);
          transcript = "";
          clearTranscript();
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
      btnStop.disabled = false;

      selSession.disabled = true;
      selAsrModel.disabled = true;

      clearTranscript();
      
      log("recording start");
    } catch (e) {
      log("ERROR: " + e);
    }
  };

  btnStop.onclick = async () => {
    try {
      if (procNode) procNode.disconnect();
      if (srcNode) srcNode.disconnect();
      if (audioCtx) await audioCtx.close();
      if (stream) stream.getTracks().forEach(t => t.stop());

      procNode=null; srcNode=null; audioCtx=null; stream=null;

      if (ws) { ws.close(); ws=null; }

      btnStart.disabled = false;
      btnStop.disabled = true;

      selSession.disabled = false;
      selAsrModel.disabled = false;

      await refreshSessions();

      log("recording stop");
    } catch (e) {
      log("ERROR: " + e);
    }
  };

  // initial
  setWsStatus(false);
  refreshSessions();