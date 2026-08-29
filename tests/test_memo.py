import asyncio
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import app as app_module


class MemoTargetTests(unittest.IsolatedAsyncioTestCase):
    def test_target_contains_only_logical_type_and_id(self):
        self.assertEqual(
            app_module.normalize_asr_target({"type": "memo", "id": "memo_20260829_ab12cd34"}),
            {"type": "memo", "id": "memo_20260829_ab12cd34"},
        )
        with self.assertRaises(ValueError):
            app_module.normalize_asr_target({"type": "memo", "id": "../outside"})
        with self.assertRaises(ValueError):
            app_module.normalize_asr_target({"type": "unknown", "id": "item_1"})

    async def test_memo_creation_and_target_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir()
            with (
                patch.object(app_module, "MEMO_OUTPUTS_DIR", outputs),
                patch.dict(app_module.CURRENT, {"patient_id": "123456"}),
            ):
                created = await app_module.api_memo_create({"title": "紹介状メモ"})
                memo = created["memo"]
                target = {"type": "memo", "id": memo["id"]}
                destination = app_module.resolve_asr_target(target)

                self.assertEqual(destination["patient_id"], "123456")
                self.assertEqual(destination["draft_path"].parent, outputs)
                self.assertTrue((outputs / f"{memo['id']}.draft.txt").is_file())
                self.assertTrue((outputs / f"{memo['id']}.meta.json").is_file())
                self.assertFalse((outputs / f"{memo['id']}.txt").exists())
                self.assertFalse((outputs / f"{memo['id']}.jsonl").exists())

    async def test_memo_list_is_limited_to_same_patient_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "outputs"
            outputs.mkdir()
            with patch.object(app_module, "MEMO_OUTPUTS_DIR", outputs):
                older = await app_module.api_memo_create({"patient_id": "123456", "title": "古いメモ"})
                await app_module.api_memo_create({"patient_id": "999999", "title": "別患者"})
                newer = await app_module.api_memo_create({"patient_id": "123456", "title": "新しいメモ"})

                for created, updated_at in (
                    (older, "2026-08-28T10:00:00+09:00"),
                    (newer, "2026-08-29T10:00:00+09:00"),
                ):
                    meta_path = outputs / f"{created['memo']['id']}.meta.json"
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["updated_at"] = updated_at
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

                result = await app_module.api_memo_list("123456")

                self.assertEqual([item["title"] for item in result["items"]], ["新しいメモ", "古いメモ"])
                self.assertTrue(all(item["patient_id"] == "123456" for item in result["items"]))

    async def test_memo_audio_is_temporary_and_deleted_after_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir()

            class FakeProvider:
                async def transcribe(self, _wav, _config):
                    return SimpleNamespace(
                        text="紹介状の下書きです。", provider="whisper", engine="whisper",
                        model="test", language="ja", provider_metrics={}, timing=None,
                        request_id=None, backend=None, model_id=None, segments=None,
                    )

            class FakeWebSocket:
                def __init__(self):
                    self.messages = []

                async def send_json(self, value):
                    self.messages.append(value)

            with (
                patch.object(app_module, "MEMO_OUTPUTS_DIR", outputs),
                patch.object(app_module, "cleanup_expired_wavs"),
                patch.object(app_module, "get_asr_provider", return_value=FakeProvider()),
            ):
                created = await app_module.api_memo_create({"patient_id": "654321"})
                target = {"type": "memo", "id": created["memo"]["id"]}
                state = app_module.State(active_target=target)
                state.active_generation = 7
                sample_count = max(int(app_module.SR * (app_module.MIN_SEC + 0.1)), 1)
                state.utter_frames = [np.zeros(sample_count, dtype=np.float32)]
                segment = state.finalize_segment(state.capture_target(), state.active_generation)
                wav_path = Path(segment["wav"])
                self.assertTrue(wav_path.is_file())

                websocket = FakeWebSocket()
                worker = asyncio.create_task(app_module.asr_worker(websocket, state))
                await state.asr_q.put(segment)
                await asyncio.wait_for(state.asr_q.join(), timeout=2)
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker

                self.assertFalse(wav_path.exists())
                self.assertEqual(segment["target"], target)
                self.assertEqual(segment["generation"], 7)
                asr_messages = [message for message in websocket.messages if message.get("type") == "asr"]
                self.assertEqual(asr_messages[0]["generation"], 7)
                self.assertFalse((outputs / f"{target['id']}.txt").exists())
                self.assertFalse((outputs / f"{target['id']}.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
