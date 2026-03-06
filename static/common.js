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

  const patientInputEl = document.getElementById('patientInput');
  const patientRecentListEl = document.getElementById('patientRecentList');
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

  let currentPatientId = '';   // 現在選択中の患者ID
  let currentSessionTxt = '';   // 現在録音中 / 選択中のセッション(.txt)名
  let liveAsrText = '';   // ライブASRの累積テキスト
  let recentPatientIds = [];   // datalist用: 新しい順・重複なし

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
  }

  function updateHearingText(text) {
    if (!hearingTextEl) return;
    const next = text || '(音声認識待機中)';
    // ユーザー編集中は上書きしない
    if (document.activeElement === hearingTextEl)
      return;
    // 下端にいる場合のみ自動スクロール
    const nearBottom =
      hearingTextEl.scrollHeight -
      hearingTextEl.scrollTop -
      hearingTextEl.clientHeight < 80;

    hearingTextEl.value = next;

    if (nearBottom) {
      hearingTextEl.scrollTop =
        hearingTextEl.scrollHeight;
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
    if (!patientRecentListEl) return;

    try {
      const r = await fetch('/api/sessions');
      const j = await r.json();
      const items = Array.isArray(j?.items) ? j.items : [];

      const seen = new Set();
      recentPatientIds = [];

      for (const it of items) {
        const pid = String(it?.patient_id || '').trim();
        if (!pid || seen.has(pid)) continue;
        seen.add(pid);
        recentPatientIds.push(pid);
      }

      patientRecentListEl.innerHTML = recentPatientIds
        .map(pid => `<option value="${e_(pid)}"></option>`)
        .join('');

      log(`[patient] recent list loaded: ${recentPatientIds.length}`);
    } catch (e) {
      log(`[patient] loadRecentPatients failed: ${e}`);
    }
  }

  // ===== 患者セッション一覧の読み込み →サイドバー + タイムライン描画 =====
  async function loadPatientSessions(pid, opts = {}) {
    if (!pid) return;
    try {
      const r = await fetch(`/api/patient/${encodeURIComponent(pid)}/sessions`);
      const j = await r.json();
      if (!j || !j.ok) throw new Error(j?.error || 'failed');
      renderSidebar(pid, j.sessions || []);
      renderTimeline(pid, j.sessions || [], opts);
    } catch (e) {
      log(`[patient] loadPatientSessions failed: ${e}`);
    }
  }

  // ===== サイドバー描画 =====
  function renderSidebar(pid, sessions) {
    if (sidebarPidEl) sidebarPidEl.textContent = pid || '-';
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

  // ===== タイムライン描画 =====
  function renderTimeline(pid, sessions, opts = {}) {
    if (!karteTimeline) return;

    if (!sessions.length) {
      karteTimeline.innerHTML = '<div class="karte-empty muted2">受診履歴がありません</div>';
      return;
    }

    karteTimeline.innerHTML = sessions.map(s => buildCard(pid, s)).join('');

    // カード内イベント登録
    karteTimeline.querySelectorAll('.karte-card').forEach(card => {
      const session = card.dataset.session;

      // LLM送信ボタン
      card.querySelectorAll('.btn-llm-run').forEach(btn => {
        btn.addEventListener('click', () => runLlmForCard(card, session));
      });

      // 確定ボタン
      card.querySelectorAll('.btn-approve').forEach(btn => {
        btn.addEventListener('click', async () => {
          const llmFile = btn.dataset.llmFile;
          await approveFile(card, session, llmFile);
        });
      });

      // LLM履歴クリックで内容ロード
      card.querySelectorAll('.llm-history-item').forEach(el => {
        el.addEventListener('click', async () => {
          const llmFile = el.dataset.llmFile;
          await loadLlmContent(card, session, llmFile);
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
      const approvedEl = card.querySelector('.llm-history-item[data-approved="true"]');
      const firstLlm = card.querySelector('.llm-history-item');

      if (approvedEl) {
        loadLlmContent(card, session, approvedEl.dataset.llmFile);
      } else if (firstLlm && opts.autoShowLatest) {
        loadLlmContent(card, session, firstLlm.dataset.llmFile);
      }
    });
  }

  // ===== カード1枚のHTML生成 =====
  function buildCard(pid, s) {
    const isCurrent = s.is_current;
    const approvedFile = s.approved_file || '';
    const llmFiles = s.llm_files || [];

    // セッションヘッダー
    const badgeCls = isCurrent ? 'badge-current' : (approvedFile ? 'badge-approved' : 'badge-draft');
    const badgeLabel = isCurrent ? '● 録音中' : (approvedFile ? '✅ 確定済' : (llmFiles.length ? '📄 下書き' : '📝 ASRのみ'));

    // LLMファイル一覧（ボタン群）
    let llmHistoryHtml = '';
    if (llmFiles.length) {
      llmHistoryHtml = `<div class="llm-history-list">` +
        llmFiles.map(f => {
          const parts = f.split('__');
          const modelShort = (parts[1] || '').split(':')[0];
          const promptShort = parts[2] || '';
          const isApproved = (f === approvedFile);
          return `<span class="llm-history-item ${isApproved ? 'llm-approved' : ''}"
            data-llm-file="${e_(f)}" data-approved="${isApproved}"
            title="${e_(f)}">${e_(modelShort)}/${e_(promptShort)}</span>`;
        }).join('') +
        `</div>`;
    }

    // 確定ボタン（承認前 & LLMあり のカードのみ）
    const approveHtml = (llmFiles.length && !approvedFile)
      ? `<button class="btn btn-sm btn-approve-cta" data-llm-file="" style="display:none">✓ 確定</button>`
      : (approvedFile
        ? `<span class="badge-ts-approved" title="${e_(approvedFile)}">✅ 確定済</span>`
        : '');

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
    <span class="badge ${badgeCls}">${badgeLabel}</span>
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
    <pre class="asr-text" id="asr-${e_(s.session_txt)}">(読み込み中…)</pre>
  </details>

  <!-- LLM欄 -->
  <div class="llm-section">
    <div class="llm-section-hd">
      <span class="llm-section-title">AI処理結果</span>
      <div class="llm-tools">
        ${llmHistoryHtml}
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
      card.querySelectorAll('.llm-history-item').forEach(el => {
        el.classList.toggle('llm-selected', el.dataset.llmFile === llmFile);
      });

      // 確定ボタンを対象ファイルに紐付け
      const approveBtn = card.querySelector('.btn-approve-cta');
      if (approveBtn) {
        approveBtn.dataset.llmFile = llmFile;
        approveBtn.style.display = '';
        approveBtn.classList.add('btn-approve');
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
      const listEl = card.querySelector('.llm-history-list');
      if (listEl) {
        listEl.innerHTML = items.map(it => {
          const parts = it.id.split('__');
          const modelShort = (parts[1] || '').split(':')[0];
          const promptShort = parts[2] || '';
          return `<span class="llm-history-item" data-llm-file="${e_(it.id)}" title="${e_(it.id)}">${e_(modelShort)}/${e_(promptShort)}</span>`;
        }).join('');
        // 再バインド
        listEl.querySelectorAll('.llm-history-item').forEach(el => {
          el.addEventListener('click', async () => {
            await loadLlmContent(card, sessionTxt, el.dataset.llmFile);
          });
        });
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
        approveBtn.classList.remove('btn-approve');
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
  async function switchPatientById(pid) {
    pid = (pid || '').trim();
    if (!pid) return;
    try {
      if (btnSwitchPatient) btnSwitchPatient.disabled = true;
      const r = await fetch('/api/patient/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: pid })
      });
      const j = await r.json();
      if (!j.ok) throw new Error(j.error || String(r.status));
      log(`[patient] switched -> ${pid} session=${j.session_txt} resumed=${j.resumed}`);
      currentPatientId = pid;
      currentSessionTxt = j.session_txt || '';

      // ★ ここを変更：選択後は入力欄を空に戻す
      if (patientInputEl) {
        patientInputEl.value = '';
        patientInputEl.placeholder = `現在: ${pid}`;
        patientInputEl.blur();
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
    if (currentSessionTxt && karteTimeline) {
      const preEl = karteTimeline.querySelector(`#asr-${CSS.escape(currentSessionTxt)}`);
      if (preEl) preEl.textContent = liveAsrText;
    }
  }

  function resetLiveAsr() {
    liveAsrText = '';
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
      // ナビ入力欄にも反映
      if (patientInputEl && currentPatientId) patientInputEl.value = currentPatientId;
      resetLiveAsr();
      log('[patient_changed] ' + currentPatientId);
      loadPatientSessions(currentPatientId, { autoShowLatest: true })
        .then(() => loadAllCardAsrText());
    }

    if (msg.type === 'asr') {
      if (msg.text) appendLiveAsr(msg.text);
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
        renderRecState();
        if (selAsrModel) selAsrModel.disabled = false;
      }
      wsReconnectTimer = setTimeout(connectWs, 3000);
    };
    ws.onmessage = handleWsMessage;
    log('[ws] connecting...');
  }

  // ===== 録音: WS接続は使い回し、音声キャプチャのみ開始 =====
  async function startRecording() {
    try {
      let targetPid = (patientInputEl?.value || '').trim();
      if (!targetPid) {
        targetPid = currentPatientId || 'unknown';
      }
      if (targetPid !== currentPatientId || !currentSessionTxt) {
        await switchPatientById(targetPid);
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        log('ERROR: getUserMedia unavailable.');
        return;
      }
      // ★ WSが切れていれば再接続（通常は既に接続済み）
      connectWs();
      // WS が OPEN になるまで最大3秒待つ
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        await new Promise((resolve, reject) => {
          const t = setTimeout(() => reject(new Error('WS connect timeout')), 3000);
          const prevOpen = ws.onopen;
          ws.onopen = e => { clearTimeout(t); if (prevOpen) prevOpen(e); resolve(); };
        });
      }

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
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = ev.inputBuffer.getChannelData(0);
        const merged = new Float32Array(pcmBuf.length + input.length);
        merged.set(pcmBuf); merged.set(input, pcmBuf.length);
        pcmBuf = merged;
        while (pcmBuf.length >= chunkSamples) {
          ws.send(pcmBuf.slice(0, chunkSamples).buffer);
          pcmBuf = pcmBuf.slice(chunkSamples);
        }
      };
      srcNode.connect(procNode);
      procNode.connect(audioCtx.destination);

      isRecording = true;
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
      updateHearingText(liveAsrText || '(音声認識待機中)');

      log('recording start');
    } catch (e) {
      log('ERROR: ' + e);
    }
  }

  async function stopRecording() {
    try {
      if (procNode) procNode.disconnect();
      if (srcNode) srcNode.disconnect();
      if (audioCtx) await audioCtx.close();
      if (stream) stream.getTracks().forEach(t => t.stop());
      procNode = srcNode = audioCtx = stream = null;
      // ★ WSは閉じない（常時接続を維持してpatient_changedを受信し続ける）

      isRecording = false;
      renderRecState();
      if (selAsrModel) selAsrModel.disabled = false;

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
  if (patientInputEl) {
    // 候補選択時は即移動
    patientInputEl.addEventListener('input', () => {
      const v = (patientInputEl.value || '').trim();
      if (!v) return;

      // datalist候補と完全一致した時だけ自動切替
      if (recentPatientIds.includes(v) && v !== currentPatientId) {
        switchPatientById(v);
      }
    });

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

