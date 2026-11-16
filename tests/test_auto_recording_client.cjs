// Run with: node --test tests/test_auto_recording_client.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const settle = () => new Promise(resolve => setImmediate(resolve));

function client({ notifications = false, permission = 'default', history = false, patientSelect = false,
                  respond } = {}) {
  const elements = {};
  const fetches = [];
  if (patientSelect) {
    let options = [];
    elements.patientSelect = {
      value: '', builds: 0, events: {},
      get options() { return options; },
      set innerHTML(html) {
        ++this.builds;
        options = Array.from(html.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g),
          match => ({ value: match[1], textContent: match[2] }));
      },
      insertBefore(option, before) {
        options = options.filter(item => item !== option);
        const index = before ? options.indexOf(before) : options.length;
        options.splice(index, 0, option);
      },
      addEventListener(name, handler) { this.events[name] = handler; },
    };
  }
  if (history) {
    elements.sidebarList = { innerHTML: '', querySelectorAll: () => [] };
    elements.sidebarPatientInfo = { innerHTML: '', style: {} };
    elements.karteTimeline = { innerHTML: '', cards: [], querySelectorAll() { return this.cards; } };
  }
  for (const name of ['btnRec', 'btnAutoRecording', 'btnRecordingNotifications', 'autoRecordingStatus', 'log']) {
    const attributes = {};
    const classes = new Set();
    elements[name] = { textContent: '', disabled: false,
      setAttribute: (key, value) => { attributes[key] = value; },
      getAttribute: key => attributes[key],
      classList: { toggle: (key, value) => value ? classes.add(key) : classes.delete(key),
        contains: key => classes.has(key) } };
  }
  const settings = { enabled: true, sound_enabled: false, notifications_enabled: notifications };
  class Notification {
    static permission = permission;
    static async requestPermission() { this.permission = 'granted'; return this.permission; }
  }
  const storage = new Map();
  const tracks = [];
  let socket;
  let autoEnabled = true;
  let active = false;
  let failMicrophone = false;
  class WebSocket {
    static OPEN = 1;
    static CONNECTING = 0;
    readyState = 1;
    commands = [];
    constructor() { socket = this; }
    receive(msg) { this.onmessage({ data: JSON.stringify(msg) }); }
    control(extra = {}) {
      this.receive({ type: 'recording_control', settings, enabled: true,
        auto_enabled: autoEnabled, active, within_schedule: true,
        remaining_seconds: active ? 1200 : null, ...extra });
    }
    send(data) {
      if (typeof data !== 'string') return;
      const cmd = JSON.parse(data);
      this.commands.push(cmd);
      queueMicrotask(() => {
        if (cmd.command === 'recording_register') {
          autoEnabled = cmd.auto_enabled;
          this.control();
        } else if (cmd.command === 'recording_begin') {
          active = true;
          this.control({ reply_to: cmd.id, granted: true });
        } else if (cmd.command === 'recording_end') {
          active = false;
          this.control();
        } else if (cmd.command === 'flush') {
          this.receive({ type: 'flush_complete' });
        }
      });
    }
  }
  class AudioContext {
    state = 'running';
    destination = {};
    resume() { return Promise.resolve(); }
    close() { this.state = 'closed'; return Promise.resolve(); }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { return { connect() {}, disconnect() {} }; }
  }
  const sandbox = {
    document: { getElementById: name => elements[name] || null, addEventListener() {},
      createElement: () => ({ value: '', textContent: '' }) },
    window: { AudioContext, Notification }, Notification, WebSocket, location: { protocol: 'http:', host: 'localhost' },
    localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    navigator: { mediaDevices: { async getUserMedia() {
      if (failMicrophone) throw new Error('denied');
      const track = { stopped: false, stop() { this.stopped = true; } };
      tracks.push(track);
      return { getTracks: () => [track], getAudioTracks: () => [track] };
    } } },
    fetch: async (url, options) => {
      fetches.push(url);
      if (respond) return respond(url, options, socket);
      return { ok: true, json: async () => ({ ok: true, sessions: [] }) };
    },
    setTimeout, clearTimeout, setInterval: () => 0, clearInterval() {}, console, AbortController,
  };
  const source = fs.readFileSync(path.join(__dirname, '../static/common.js'), 'utf8')
    .replace('  // ===== 初期化 =====',
      '  window.testHistory = { buildOptimisticSession, renderSidebar, buildCard, loadAllCardAsrText, loadAsrText, switchPatientById, loadPatientSessions, loadRecentPatients };\n  // ===== 初期化 =====');
  vm.runInNewContext(source, sandbox);
  socket.onopen();
  socket.receive({ type: 'status', patient_id: '1', session_txt: '1_20261116_090000.txt' });
  return { elements, socket, tracks, storage, fetches, history: sandbox.window.testHistory,
    fail: () => { failMicrophone = true; } };
}

test('auto toggle controls subsequent starts without stopping an active recording', async () => {
  const c = client();
  await settle();
  assert.equal(c.elements.btnAutoRecording.getAttribute('aria-checked'), 'true');
  assert.equal(c.elements.btnAutoRecording.textContent, '自動');
  c.socket.receive({ type: 'auto_recording_start', patient_id: '1', request_id: 'event1' });
  await settle();
  assert.equal(c.elements.btnRec.classList.contains('recording'), true);
  c.socket.control({ remaining_seconds: 1199 });
  assert.equal(c.elements.btnAutoRecording.textContent, '自動 19:59');
  await c.elements.btnRec.onclick();
  assert.equal(c.elements.btnRec.classList.contains('recording'), false);
  assert.equal(c.elements.btnAutoRecording.getAttribute('aria-checked'), 'true');
  assert.equal(c.elements.btnAutoRecording.textContent, '自動');
  c.socket.receive({ type: 'auto_recording_start', patient_id: '1', request_id: 'event2' });
  await settle();
  assert.equal(c.elements.btnRec.classList.contains('recording'), true);
  await c.elements.btnAutoRecording.onclick();
  await settle();
  assert.equal(c.elements.btnAutoRecording.getAttribute('aria-checked'), 'false');
  assert.equal(c.elements.btnRec.classList.contains('recording'), true);
  await c.elements.btnRec.onclick();
  const count = c.socket.commands.filter(cmd => cmd.command === 'recording_begin').length;
  c.socket.receive({ type: 'auto_recording_start', patient_id: '1', request_id: 'event3' });
  await settle();
  assert.equal(c.socket.commands.filter(cmd => cmd.command === 'recording_begin').length, count);
  await c.elements.btnRec.onclick();
  assert.equal(c.elements.btnRec.classList.contains('recording'), true);
  assert.equal(c.storage.get('autoRecordingEnabled'), 'false');
  c.socket.receive({ type: 'auto_recording_stop' });
  await settle();
  assert.equal(c.elements.btnRec.classList.contains('recording'), false);
  assert.ok(c.tracks.every(track => track.stopped));
});

test('failed automatic microphone start releases the server grant and remains visible', async () => {
  const c = client();
  await settle();
  c.fail();
  c.socket.receive({ type: 'auto_recording_start', patient_id: '1', request_id: 'event1' });
  await settle();
  assert.equal(c.elements.btnRec.classList.contains('recording'), false);
  assert.equal(c.socket.commands.at(-1).command, 'recording_end');
  assert.match(c.elements.autoRecordingStatus.textContent, /失敗/);
  assert.match(c.elements.btnAutoRecording.title, /失敗/);
});

test('notification permission button stays hidden once permission is granted', async () => {
  const c = client({ notifications: true });
  await settle();
  assert.equal(c.elements.btnRecordingNotifications.hidden, false);
  await c.elements.btnRecordingNotifications.onclick();
  assert.equal(c.elements.btnRecordingNotifications.hidden, true);
  c.socket.control();
  assert.equal(c.elements.btnRecordingNotifications.hidden, true);
  const returning = client({ notifications: true, permission: 'granted' });
  await settle();
  assert.equal(returning.elements.btnRecordingNotifications.hidden, true);
});

test('all branch histories display zero separately from unknown, including optimistic cards', () => {
  const c = client({ history: true });
  const sessions = ['1_20261115_090000.txt', '1_b0_20261115_090000.txt',
    '1_b1_20261115_090000_c' + 'a'.repeat(32) + '.txt']
    .map(name => c.history.buildOptimisticSession(name));
  assert.equal(sessions[0].chart_branch, null);
  assert.equal(sessions[1].chart_branch, 0);
  assert.equal(sessions[2].chart_branch, 1);
  c.history.renderSidebar('1', sessions, null);
  for (const label of ['枝番不明', '枝番 0', '枝番 1']) {
    assert.ok(c.elements.sidebarList.innerHTML.includes(label));
  }
  assert.match(c.history.buildCard(sessions[1]), /karte-branch">枝番 0/);
});

test('past ASR bodies remain unloaded until their section is opened', async () => {
  const c = client({ history: true });
  await settle();
  const pre = { textContent: '' };
  c.elements.karteTimeline.cards = [true, false].map((open, i) => ({
    dataset: { session: `1_b${i}_20261115_090000.txt` },
    querySelector: selector => selector === '.asr-section' ? { open } : pre,
  }));
  c.fetches.length = 0;
  await c.history.loadAllCardAsrText();
  assert.deepEqual(c.fetches, ['/api/session/1_b0_20261115_090000.txt']);
});

test('automatic recording rejects a stale start for another branch of the same patient', async () => {
  const c = client();
  await settle();
  c.socket.receive({ type: 'status', patient_id: '1', chart_branch: 1,
    session_txt: '1_b1_20261115_090000.txt' });
  c.socket.commands.length = 0;
  c.socket.receive({ type: 'auto_recording_start', patient_id: '1', chart_branch: 0, request_id: 'stale' });
  await settle();
  assert.ok(!c.socket.commands.some(cmd => cmd.command === 'recording_begin'));
  c.socket.receive({ type: 'auto_recording_start', patient_id: '1', chart_branch: 1, request_id: 'fresh' });
  await settle();
  assert.ok(c.socket.commands.some(cmd => cmd.command === 'recording_begin' && cmd.chart_branch === 1));
});

for (const order of ['ws-first', 'post-first']) {
  test(`manual switch and its ${order} notification share one history request`, async () => {
    const changed = { type: 'patient_changed', patient_id: '16712', chart_branch: null, session_txt: '' };
    const response = value => ({ ok: true, json: async () => value });
    const c = client({ history: true, patientSelect: true,
      respond: async (url, options, socket) => {
        if (url === '/api/patient/switch') {
          if (order === 'ws-first') {
            socket.receive(changed);
            // Let the WS refresh finish before the POST arrives, too.
            await settle();
          }
          return response({ ok: true, ...changed });
        }
        if (url === '/api/patient/16712/sessions') {
          return response({ ok: true, sessions: [], patient_info: { name: '履歴の患者情報' }, query_ms: 0.2 });
        }
        return response({ ok: true, sessions: [], items: [] });
      } });
    await settle();
    c.fetches.length = 0;
    const builds = c.elements.patientSelect.builds;
    await c.history.switchPatientById('16712');
    if (order === 'post-first') c.socket.receive(changed);
    await settle();
    assert.equal(c.fetches.filter(url => url === '/api/patient/16712/sessions').length, 1);
    assert.ok(!c.fetches.some(url => url.endsWith('/info')));
    assert.ok(!c.fetches.includes('/api/sessions'));
    assert.equal(c.elements.patientSelect.builds, builds, 'switch must not rebuild the full dropdown');
    assert.match(c.elements.sidebarPatientInfo.innerHTML, /履歴の患者情報/);
    // A deliberate later refresh must still see changes to the same patient's records.
    await c.history.loadPatientSessions('16712');
    assert.equal(c.fetches.filter(url => url === '/api/patient/16712/sessions').length, 2);
  });
}

test('rapid branch changes abort the old request and never render its late response', async () => {
  let finishOld;
  let oldSignal;
  let requests = 0;
  const c = client({ history: true, respond: async (url, options) => {
    if (url === '/api/patient/16712/sessions' && ++requests === 1) {
      oldSignal = options.signal;
      return new Promise(resolve => { finishOld = () => resolve({ ok: true,
        json: async () => ({ ok: true, sessions: [], patient_info: { name: '古い枝番' } }) }); });
    }
    return { ok: true, json: async () => ({ ok: true, sessions: [], patient_info: { name: '新しい枝番' } }) };
  } });
  await settle();
  c.socket.receive({ type: 'patient_changed', patient_id: '16712', chart_branch: 0, session_txt: '' });
  c.socket.receive({ type: 'patient_changed', patient_id: '16712', chart_branch: 1, session_txt: '' });
  await settle();
  assert.equal(oldSignal.aborted, true);
  assert.match(c.elements.sidebarPatientInfo.innerHTML, /新しい枝番/);
  finishOld();
  await settle();
  assert.match(c.elements.sidebarPatientInfo.innerHTML, /新しい枝番/);
  // Returning to a previously visited branch starts a fresh query.
  c.socket.receive({ type: 'patient_changed', patient_id: '16712', chart_branch: 0, session_txt: '' });
  await settle();
  assert.equal(requests, 3);
});

test('patient dropdown refreshes share a request and use the short-lived cache', async () => {
  const c = client({ patientSelect: true });
  await settle();
  c.fetches.length = 0;
  await Promise.all([c.history.loadRecentPatients(), c.history.loadRecentPatients()]);
  assert.equal(c.fetches.length, 0);
  await Promise.all([c.history.loadRecentPatients({ force: true }), c.history.loadRecentPatients({ force: true })]);
  assert.deepEqual(c.fetches, ['/api/sessions']);
});

test('opening the current ASR section and refreshing it share one in-flight body fetch', async () => {
  const c = client({ history: true });
  await settle();
  c.fetches.length = 0;
  const card = { dataset: {}, querySelector: () => ({ textContent: '' }) };
  await Promise.all([c.history.loadAsrText(card, '1_b0_20261115_090000.txt'),
    c.history.loadAsrText(card, '1_b0_20261115_090000.txt')]);
  assert.deepEqual(c.fetches, ['/api/session/1_b0_20261115_090000.txt']);
});

test('late prompt metadata fills existing cards before models arrive and preserves user selection', async () => {
  let resolveModels;
  let resolvePrompts;
  const models = new Promise(resolve => { resolveModels = resolve; });
  const prompts = new Promise(resolve => { resolvePrompts = resolve; });
  const response = data => ({ ok: true, json: async () => data });
  const c = client({ history: true, respond: async url => {
    if (url === '/api/llm/models') return models;
    if (url === '/api/llm/prompts') return prompts;
    return response({ ok: true, sessions: [] });
  } });
  await settle();
  assert.ok(c.fetches.includes('/api/llm/prompts'));
  const promptSelect = { value: '', innerHTML: '' };
  const modelSelect = { value: '', innerHTML: '' };
  c.elements.karteTimeline.cards = [{ querySelector: selector =>
    selector === '.card-sel-prompt' ? promptSelect : modelSelect }];
  resolvePrompts(response({ ok: true, default_prompt_id: 'soap', items: [
    { id: 'soap', label: 'SOAP記録' }, { id: 'letter', label: '紹介状' },
  ] }));
  await settle();
  assert.match(promptSelect.innerHTML, /value="soap">SOAP記録/);
  assert.match(promptSelect.innerHTML, /value="letter">紹介状/);
  assert.equal(promptSelect.value, 'soap');
  promptSelect.value = 'letter';
  resolveModels(response({ ok: true, models: ['model'], default_model: 'model' }));
  await settle();
  assert.match(modelSelect.innerHTML, /value="model"/);
  assert.equal(promptSelect.value, 'letter');
});
