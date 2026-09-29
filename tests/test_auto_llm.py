import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module


class AutoLLMTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "patient_20260929_120000.jsonl"
        self.path.touch()
        self.queue = asyncio.Queue()
        self.item = {"model_id": "test-model", "prompt_id": "soap", "asr_correct": False}
        for name, value in {
            "AUTO_LLM_ENABLED": True,
            "AUTO_LLM_EVERY_STOP": True,
            "AUTO_LLM_PROMPTS": [self.item],
            "_auto_llm_q": self.queue,
            "_auto_llm_enqueued": set(),
            "LLM_DIR": Path(directory.name),
        }.items():
            patcher = patch.object(app_module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def append(self, record):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_asr(self, text):
        self.append({"type": "asr", "text": text, "meta": {"quality": "good"}})

    def enqueue(self, reason="recording_stop"):
        return app_module.enqueue_auto_llm_for_jsonl(self.path, reason=reason)

    async def test_each_stop_sends_accumulated_transcript_after_resume(self):
        with (
            patch.object(app_module, "generate_llm_text", return_value="要約") as generate,
            patch.object(app_module, "build_prompt_from_cfg", side_effect=lambda _, text: text),
            patch.object(app_module, "save_llm_summary", return_value="summary.txt"),
        ):
            for text in ("最初の記録", "再開後の記録", "3回目の記録"):
                self.add_asr(text)
                self.assertEqual(self.enqueue(), 1)
                await app_module._run_auto_llm_job(self.queue.get_nowait())
                self.assertEqual(self.enqueue("switch"), 0)

        self.assertEqual(generate.call_count, 3)
        self.assertEqual(generate.call_args.kwargs["prompt"], "最初の記録\n再開後の記録\n3回目の記録")

    async def test_default_mode_still_sends_only_once(self):
        with patch.object(app_module, "AUTO_LLM_EVERY_STOP", False):
            self.add_asr("最初の記録")
            self.assertEqual(self.enqueue(), 1)
            self.add_asr("再開後の記録")
            self.assertEqual(self.enqueue(), 0)

    async def test_repeated_notifications_and_non_asr_events_do_not_resend(self):
        self.assertEqual(self.enqueue(), 0)
        self.add_asr("記録")
        self.assertEqual(self.enqueue(), 1)
        for kind in ("resume", "approved", "auto_llm_done"):
            self.append({"type": kind})
            self.assertEqual(self.enqueue("switch"), 0)
        self.add_asr("追加記録")
        self.assertEqual(self.enqueue(), 1)
        self.assertEqual(self.queue.qsize(), 2)

    async def test_completed_revision_is_not_resent_after_restart(self):
        self.add_asr("記録")
        self.assertEqual(self.enqueue(), 1)
        job = self.queue.get_nowait()
        self.append({"type": "auto_llm_done", "meta": {
            **self.item, "asr_revision": job["asr_revision"],
        }})
        app_module._auto_llm_enqueued.clear()
        self.assertEqual(self.enqueue(), 0)
        self.add_asr("再開後の記録")
        self.assertEqual(self.enqueue(), 1)

    async def test_old_history_blocks_default_mode_but_allows_every_stop(self):
        self.add_asr("記録")
        self.append({"type": "auto_llm_done", "meta": self.item})
        with patch.object(app_module, "_has_llm_history_for_prompt", return_value=True):
            with patch.object(app_module, "AUTO_LLM_EVERY_STOP", False):
                self.assertEqual(self.enqueue(), 0)
            self.assertEqual(self.enqueue(), 1)

    async def test_disabled_auto_llm_never_sends(self):
        self.add_asr("記録")
        with patch.object(app_module, "AUTO_LLM_ENABLED", False):
            self.assertEqual(self.enqueue(), 0)
        self.assertTrue(self.queue.empty())


if __name__ == "__main__":
    unittest.main()
