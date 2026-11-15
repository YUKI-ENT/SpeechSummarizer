import copy
import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
import threading
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import app as app_module
import httpx
from auto_recording import RecordingSession
from tools.so_labeler.jsonl_loader import _extract_session_stamp_from_name


class ChartBranchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sessions = self.root / "sessions"
        self.llm = self.root / "llm"
        self.sessions.mkdir()
        self.llm.mkdir()
        current = copy.deepcopy(app_module.CURRENT)
        current.update(patient_id=None, chart_branch=None, text_path=None, jsonl_path=None)
        for name, value in {
            "OUTPUTS_DIR": self.sessions, "LLM_DIR": self.llm,
            "PATIENT_DATA_PATH": self.root / "patients.jsonl", "_PATIENT_INFO_CACHE": {},
            "CURRENT": current, "_SESSION_INDEX": None, "CLIENTS": {},
            "AUTO_RECORDING": {**app_module.AUTO_RECORDING, "enabled": False},
            "enqueue_auto_llm_for_jsonl": unittest.mock.Mock(return_value=0),
            "broadcast": AsyncMock(), "now_stamp": lambda: "20261115_090000",
            "_today_yyyymmdd": lambda: "20261115",
        }.items():
            patcher = patch.object(app_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_branch_switch_creates_separate_session_and_return_resumes_exact_branch(self):
        app_module.new_session("12345")
        legacy = app_module.CURRENT["text_path"]
        first = await app_module._do_patient_switch("12345", chart_branch=0, source_chart_id="123450")
        self.assertNotEqual(Path(legacy).name, first["session_txt"])
        self.assertFalse(first["resumed"])
        self.assertEqual(first["session_txt"], "12345_b0_20261115_090000.txt")
        old_target = app_module.current_session_target()
        second = await app_module._do_patient_switch("12345", chart_branch=1)
        self.assertNotEqual(first["session_txt"], second["session_txt"])
        resolved = app_module.resolve_asr_target(old_target)
        self.assertEqual(resolved["patient_id"], "12345")
        self.assertEqual(resolved["chart_branch"], 0)
        self.assertEqual(resolved["text_path"].name, first["session_txt"])
        returned = await app_module._do_patient_switch("12345", chart_branch=0)
        self.assertTrue(returned["resumed"])
        self.assertEqual(returned["session_txt"], first["session_txt"])
        history = app_module.list_patient_sessions("12345")
        self.assertEqual({item["chart_branch"] for item in history}, {None, 0, 1})
        self.assertEqual(sum(item["is_current"] for item in history), 1)
        recent = await app_module.api_sessions()
        self.assertEqual(len(recent["items"]), 1)
        self.assertEqual(recent["items"][0]["patient_id"], "12345")

    async def test_selected_branch_survives_deferred_recording_session_creation(self):
        result = await app_module._do_patient_switch("12345", chart_branch=2, create_new=False)
        self.assertFalse(result["session_txt"])
        target = app_module.current_session_target()
        self.assertEqual(target["id"], "12345_b2_20261115_090000")
        header = json.loads(Path(app_module.CURRENT["jsonl_path"]).read_text().splitlines()[0])
        self.assertEqual(header["chart_branch"], 2)
        self.assertEqual(header["patient_id"], "12345")

    async def test_deferred_session_creation_preserves_pending_automatic_start(self):
        await app_module._do_patient_switch("12345", chart_branch=2, create_new=False)
        recording = RecordingSession(request_id="fresh-request", auto_enabled=True)
        st = SimpleNamespace(active_target=None, recording=recording)
        app_module.CLIENTS["socket"] = st
        await app_module._do_patient_switch("12345", chart_branch=2, create_new=True)
        self.assertEqual(recording.request_id, "fresh-request")
        await app_module._do_patient_switch("12345", chart_branch=3, create_new=True)
        self.assertEqual(recording.request_id, "")

    async def test_inflight_asr_result_is_saved_to_original_branch(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def transcribe(*_):
            started.set()
            await release.wait()
            return SimpleNamespace(text="枝番0での会話", provider="whisper", engine="whisper",
                                   model="test", language="ja", provider_metrics={}, timing=None,
                                   request_id=None, backend=None, model_id=None, segments=None)

        app_module.new_session("12345", 0)
        original = Path(app_module.CURRENT["text_path"])
        state = app_module.State()
        target = state.capture_target()
        socket = SimpleNamespace(send_json=AsyncMock())
        with patch.object(app_module, "ASR_ENABLED", True), \
             patch.object(app_module, "get_asr_provider", return_value=SimpleNamespace(transcribe=transcribe)):
            worker = asyncio.create_task(app_module.asr_worker(socket, state))
            try:
                await state.asr_q.put({"target": target, "seg_id": 1, "wav": "dummy.wav", "dur": 1})
                await asyncio.wait_for(started.wait(), timeout=2)
                await app_module._do_patient_switch("12345", chart_branch=1)
                release.set()
                await asyncio.wait_for(state.asr_q.join(), timeout=2)
            finally:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
        self.assertIn("枝番0での会話", original.read_text())
        self.assertEqual(Path(app_module.CURRENT["text_path"]).read_text(), "")
        events = [json.loads(line) for line in original.with_suffix(".jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["target"], target)
        self.assertEqual(events[-1]["patient_id"], "12345")

    async def test_same_second_files_do_not_overwrite_and_collision_name_roundtrips(self):
        app_module.new_session("12345", 1)
        path = Path(app_module.CURRENT["text_path"])
        path.write_text("original", encoding="utf-8")
        app_module.new_session("12345", 1)
        name = Path(app_module.CURRENT["text_path"]).name
        self.assertNotEqual(path.name, name)
        self.assertEqual(path.read_text(), "original")
        self.assertEqual(app_module._parse_session_name(name), ("12345", "20261115_090000"))
        self.assertEqual(_extract_session_stamp_from_name(Path(name))[1], "12345")
        self.assertEqual(app_module.resolve_asr_target(app_module.current_session_target())["chart_branch"], 1)

    async def test_llm_history_and_approvals_do_not_cross_branches(self):
        names = []
        for branch in (0, 1):
            app_module.new_session("12345", branch)
            jsonl = Path(app_module.CURRENT["jsonl_path"])
            filename = app_module.save_llm_summary(jsonl_path=jsonl, model="m", prompt_id="soap", summary="summary")
            names.append((jsonl.with_suffix(".txt").name, filename))
            result = await app_module.api_llm_approve({"session": names[-1][0], "llm_file": filename})
            self.assertTrue(result["ok"])
            self.assertTrue(app_module._has_llm_history_for_prompt(jsonl, prompt_id="soap"))
        for session, filename in names:
            history = await app_module.api_llm_history(session=session)
            self.assertEqual([item["id"] for item in history["items"]], [filename])
        wrong = await app_module.api_llm_approve({"session": names[0][0], "llm_file": names[1][1]})
        self.assertEqual(wrong.status_code, 400)
        self.assertEqual(len((await app_module.api_llm_history(patient_id="12345"))["items"]), 2)

    async def test_emr_branch_setting_preserves_base_patient_info_and_original_source(self):
        path = self.root / "dyna.txt"
        path.write_text("123451,患者,カナ,1,2000/01/01", encoding="cp932")
        for enabled, expected in ((False, "123451"), (True, "12345")):
            with patch.object(app_module, "DYNA_ID_INCLUDES_BRANCH", enabled):
                info = app_module.read_patient_id_from_dyna(path)
            self.assertEqual(info["patient_id"], expected)
            self.assertEqual(info["source_chart_id"], "123451")
            self.assertEqual(info["chart_branch"], 1 if enabled else None)
            app_module.save_patient_info(info)
            self.assertEqual(app_module.get_patient_info(expected)["name"], "患者")
        self.assertEqual(json.loads(app_module.PATIENT_DATA_PATH.read_text().splitlines()[-1])["patient_id"], "12345")

    async def test_manual_api_uses_base_id_even_when_emr_includes_branch(self):
        with patch.object(app_module, "DYNA_ID_INCLUDES_BRANCH", True):
            result = await app_module.api_patient_switch({"patient_id": "12345", "chart_branch": 3})
        self.assertEqual(result["patient_id"], "12345")
        self.assertEqual(result["chart_branch"], 3)
        invalid = await app_module.api_patient_switch({"patient_id": "12345", "chart_branch": True})
        self.assertEqual(invalid.status_code, 400)

    async def test_history_http_response_uses_index_and_includes_info_and_timing(self):
        app_module.new_session("12345", 0)
        app_module.save_patient_info({"patient_id": "12345", "name": "テスト患者"})
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            with patch.object(Path, "glob", side_effect=AssertionError("directory scan")), \
                 patch.object(Path, "open", side_effect=AssertionError("transcript read")):
                response = await client.get("/api/patient/12345/sessions")
        self.assertEqual(response.status_code, 200)
        self.assertIn("app;dur=", response.headers["server-timing"])
        self.assertEqual(response.json()["patient_info"]["name"], "テスト患者")
        self.assertEqual(response.json()["sessions"][0]["chart_branch"], 0)
        self.assertGreaterEqual(response.json()["query_ms"], 0)

    async def test_slow_summary_disk_read_does_not_block_patient_history(self):
        app_module.new_session("12345", 0)
        name = app_module.save_llm_summary(jsonl_path=Path(app_module.CURRENT["jsonl_path"]),
                                           model="m", prompt_id="soap", summary="要約本文")
        started, release = threading.Event(), threading.Event()
        read_threads = []
        original = Path.read_text

        def slow_read(path, *args, **kwargs):
            if path.name == name:
                read_threads.append(threading.current_thread().name)
                started.set()
                release.wait(timeout=5)
            return original(path, *args, **kwargs)

        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            with patch.object(Path, "read_text", slow_read):
                summary_task = asyncio.create_task(client.get("/api/llm/history/item/" + name))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    history = await asyncio.wait_for(client.get("/api/patient/12345/sessions"), timeout=2)
                    self.assertEqual(history.status_code, 200)
                    self.assertFalse(summary_task.done())
                    self.assertNotEqual(read_threads, [threading.current_thread().name])
                finally:
                    release.set()
                    summary = await summary_task
        self.assertEqual(summary.json()["summary"], "要約本文")


if __name__ == "__main__":
    unittest.main()
