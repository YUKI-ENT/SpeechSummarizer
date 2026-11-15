import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from auto_recording import RecordingSession, normalize_auto_recording, within_schedule
import app as app_module


class ScheduleTests(unittest.TestCase):
    def test_weekdays_breaks_and_boundaries(self):
        cfg = normalize_auto_recording({"enabled": True, "schedule": {
            "mon": ["14:00-19:00", "09:00-12:30"], "sat": ["09:00-12:00"],
        }})
        for hour, minute, expected in [(8, 59, False), (9, 0, True), (12, 30, False),
                                       (13, 59, False), (14, 0, True), (19, 0, False)]:
            with self.subTest(hour=hour, minute=minute):
                self.assertEqual(within_schedule(cfg, datetime(2026, 11, 16, hour, minute)), expected)
        self.assertFalse(within_schedule(cfg, datetime(2026, 11, 17, 10)))
        self.assertTrue(within_schedule(cfg, datetime(2026, 11, 21, 10)))
        self.assertFalse(within_schedule(cfg, datetime(2026, 11, 22, 10)))

    def test_missing_config_keeps_automation_disabled(self):
        self.assertFalse(normalize_auto_recording()["enabled"])

    def test_rejects_bad_time_ranges_and_durations(self):
        for periods in [["9:00-19:00"], ["09:00-24:00"], ["19:00-09:00"],
                        ["09:00-12:00", "11:00-13:00"], [None]]:
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                normalize_auto_recording({"schedule": {"mon": periods}})
        for minutes in [0, -1, 1441, True, "20", float("nan")]:
            with self.subTest(minutes=minutes), self.assertRaises(ValueError):
                normalize_auto_recording({"stop_after_minutes": minutes})

    def test_timer_resets_for_next_patient_but_not_repeated_file(self):
        rec = RecordingSession(auto_enabled=True)
        rec.begin(100, 20, "1")
        rec.switch_patient(150, 20, "1")
        self.assertEqual(rec.deadline, 1300)
        rec.switch_patient(200, 20, "2")
        self.assertEqual(rec.deadline, 1400)
        self.assertFalse(rec.expired(1399))
        self.assertTrue(rec.expired(1400))
        rec.stop()
        self.assertTrue(rec.auto_enabled)
        self.assertTrue(rec.blocked)


class FakeSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


class RecordingControlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = normalize_auto_recording({"enabled": True})
        self.clients = {}
        for name, value in {"AUTO_RECORDING": self.cfg, "CLIENTS": self.clients,
                            "DISCONNECTED_RECORDINGS": {}, "CURRENT": {"patient_id": "1"}}.items():
            patcher = patch.object(app_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(app_module, "within_schedule", return_value=True)
        self.schedule = patcher.start()
        self.addCleanup(patcher.stop)

    def client(self, memo=False):
        ws = FakeSocket()
        st = SimpleNamespace(recording=RecordingSession(), active_target={"type": "memo"} if memo else None,
                             last_audio_rx=0, recording_client_id="")
        self.clients[ws] = st
        return ws, st

    async def command(self, ws, st, action, **extra):
        await app_module.handle_recording_command(ws, st, {"command": action, **extra})

    async def test_manual_stop_restarts_only_with_auto_toggle_on(self):
        ws, st = self.client()
        await self.command(ws, st, "recording_register", auto_enabled=True)
        await self.command(ws, st, "recording_begin", patient_id="1", id="manual")
        self.assertTrue(ws.messages[-1]["granted"])
        await self.command(ws, st, "recording_end")
        await app_module.request_auto_recording("1")
        request = ws.messages[-1]
        self.assertEqual(request["type"], "auto_recording_start")
        await self.command(ws, st, "recording_begin", patient_id="1", request_id=request["request_id"])
        self.assertTrue(st.recording.active)
        await self.command(ws, st, "recording_register", auto_enabled=False)
        self.assertTrue(st.recording.active)  # Toggle alone never stops recording.
        await self.command(ws, st, "recording_end")
        count = len(ws.messages)
        await app_module.request_auto_recording("1")
        self.assertEqual(len(ws.messages), count)
        await self.command(ws, st, "recording_begin", patient_id="1")
        self.assertTrue(st.recording.active)
        self.assertFalse(st.recording.auto_enabled)

    async def test_only_one_auto_owner_and_one_clinical_recorder(self):
        ws, st = self.client()
        other_ws, other = self.client()
        await self.command(ws, st, "recording_register", auto_enabled=True)
        await self.command(other_ws, other, "recording_register", auto_enabled=True)
        self.assertFalse(other.recording.auto_enabled)
        self.assertTrue(other_ws.messages[-1]["error"])
        await self.command(ws, st, "recording_begin", patient_id="1")
        await self.command(other_ws, other, "recording_begin", patient_id="1")
        self.assertFalse(other_ws.messages[-1]["granted"])

    async def test_out_of_hours_and_stale_requests_do_not_start(self):
        ws, st = self.client()
        await self.command(ws, st, "recording_register", auto_enabled=True)
        await app_module.request_auto_recording("1")
        request_id = ws.messages[-1]["request_id"]
        self.schedule.return_value = False
        await self.command(ws, st, "recording_begin", request_id=request_id, patient_id="1")
        self.assertFalse(ws.messages[-1]["granted"])
        self.schedule.return_value = True
        st.recording.request_deadline = 0
        await self.command(ws, st, "recording_begin", request_id=request_id, patient_id="1")
        self.assertFalse(ws.messages[-1]["granted"])
        # Manual recording is allowed outside the schedule.
        self.schedule.return_value = False
        await self.command(ws, st, "recording_begin", patient_id="1")
        self.assertTrue(ws.messages[-1]["granted"])

    async def test_reconnect_preserves_deadline_and_honors_offline_stop(self):
        saved = RecordingSession(auto_enabled=True)
        saved.begin(app_module.time.monotonic(), 20, "1")
        deadline = saved.deadline
        app_module.DISCONNECTED_RECORDINGS["same-tab"] = (saved, app_module.time.monotonic() + 60)
        ws, st = self.client()
        await self.command(ws, st, "recording_register", client_id="same-tab", auto_enabled=True, recording=True)
        self.assertIs(st.recording, saved)
        self.assertEqual(st.recording.deadline, deadline)
        self.clients.clear()
        app_module.DISCONNECTED_RECORDINGS["same-tab"] = (saved, app_module.time.monotonic() + 60)
        ws, st = self.client()
        await self.command(ws, st, "recording_register", client_id="same-tab", auto_enabled=True, recording=False)
        self.assertFalse(st.recording.active)
        self.assertTrue(st.recording.blocked)

    async def test_expiry_blocks_audio_and_flushes_to_original_session(self):
        ws, st = self.client()
        st.recording.begin(0, 1, "1")
        st.capture_target = Mock(return_value={"type": "session", "id": "original"})
        st.active_generation = 0
        st.finalize_segment = Mock(return_value={"target": {"id": "original"}})
        st.reset_audio = Mock()
        st.asr_q = asyncio.Queue()
        path = Path("original.jsonl")
        with patch.object(app_module, "resolve_asr_target", return_value={"jsonl_path": path}), \
             patch.object(app_module, "enqueue_auto_llm_for_jsonl") as enqueue:
            await app_module.recording_deadline_tick()
            self.assertFalse(st.recording.active)
            self.assertTrue(st.recording.blocked)
            self.assertEqual(ws.messages[-1]["type"], "auto_recording_stop")
            self.assertEqual(st.asr_q.get_nowait()["target"]["id"], "original")
            st.asr_q.task_done()
            app_module.CURRENT["patient_id"] = "2"
            await asyncio.sleep(0)
            enqueue.assert_called_once_with(path, reason="recording_stop")

    async def test_memo_is_excluded_from_automation(self):
        ws, st = self.client(memo=True)
        await self.command(ws, st, "recording_register", auto_enabled=True)
        self.assertFalse(st.recording.auto_enabled)
        st.recording.begin(0, 1, "1")
        await app_module.recording_deadline_tick()
        self.assertTrue(st.recording.active)
        self.assertEqual(ws.messages, [])

    async def test_watcher_starts_only_for_fresh_valid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "dyna-old.txt"
            fresh = Path(directory) / "dyna-new.txt"
            old.write_text("1,old", encoding="cp932")
            fresh.write_text("2,new", encoding="cp932")
            with patch.object(app_module, "list_dyna_files", side_effect=[
                    [old], [old], [fresh], asyncio.CancelledError()]), \
                 patch.object(app_module, "save_patient_info"), \
                 patch.object(app_module, "delete_all_dyna_files"), \
                 patch.object(app_module, "_do_patient_switch", new_callable=AsyncMock) as switch, \
                 patch.object(app_module, "request_auto_recording", new_callable=AsyncMock) as start, \
                 patch.object(app_module.asyncio, "sleep", new_callable=AsyncMock):
                with self.assertRaises(asyncio.CancelledError):
                    await app_module.dyna_watch_task()
                self.assertEqual(switch.await_count, 2)
                start.assert_awaited_once_with("2")

    async def test_websocket_drops_audio_after_manual_stop(self):
        received = []
        class Socket(FakeSocket):
            async def accept(self):
                pass

            async def receive(inner):
                count = len(received)
                received.append(count)
                if count == 0:
                    return {"text": json.dumps({"command": "recording_begin", "patient_id": "1"})}
                if count == 1:
                    return {"bytes": b"\x00" * 4}
                if count == 2:
                    self.assertEqual(self.clients[inner].buf.size, 1)
                    return {"text": json.dumps({"command": "recording_end"})}
                if count == 3:
                    return {"bytes": b"\x00" * 4}
                self.assertEqual(self.clients[inner].buf.size, 1)
                return {"type": "websocket.disconnect"}

        ws = Socket()
        with patch.object(app_module, "asr_worker", new_callable=AsyncMock):
            await app_module.ws_endpoint(ws)
        self.assertNotIn(ws, self.clients)


if __name__ == "__main__":
    unittest.main()
