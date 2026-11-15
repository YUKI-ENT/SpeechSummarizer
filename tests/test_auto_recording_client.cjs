// Run with: node --test tests/test_auto_recording_client.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const settle = () => new Promise(resolve => setImmediate(resolve));

function client({ notifications = false, permission = 'default' } = {}) {
  const elements = {};
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
    document: { getElementById: name => elements[name] || null, addEventListener() {} },
    window: { AudioContext, Notification }, Notification, WebSocket, location: { protocol: 'http:', host: 'localhost' },
    localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    navigator: { mediaDevices: { async getUserMedia() {
      if (failMicrophone) throw new Error('denied');
      const track = { stopped: false, stop() { this.stopped = true; } };
      tracks.push(track);
      return { getTracks: () => [track], getAudioTracks: () => [track] };
    } } },
    fetch: async () => ({ ok: true, json: async () => ({ ok: true, sessions: [] }) }),
    setTimeout, clearTimeout, setInterval: () => 0, clearInterval() {}, console,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/common.js'), 'utf8'), sandbox);
  socket.onopen();
  socket.receive({ type: 'status', patient_id: '1', session_txt: '1_20261116_090000.txt' });
  return { elements, socket, tracks, storage, fail: () => { failMicrophone = true; } };
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
