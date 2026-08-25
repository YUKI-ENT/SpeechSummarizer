import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx

from asr_providers import (
    ASRProviderError,
    Qwen3ASRProvider,
    VibeVoiceASRProvider,
    WhisperASRProvider,
    asr_model_label,
    create_asr_provider,
)


class FakeWhisperModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, _wav_path, **kwargs):
        self.kwargs = kwargs
        return iter([
            SimpleNamespace(text=" 今日は", avg_logprob=-0.2, no_speech_prob=0.1, compression_ratio=1.1),
            SimpleNamespace(text="晴れです。", avg_logprob=-0.4, no_speech_prob=0.3, compression_ratio=1.3),
        ]), None


class WhisperProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_text_and_real_whisper_metrics(self):
        model = FakeWhisperModel()
        provider = WhisperASRProvider(lambda _path: model)
        result = await provider.transcribe("unused.wav", {
            "model_path": "small",
            "model_name": "small",
            "language": "ja",
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "prompt": "prompt",
        })

        self.assertEqual(result.text, "今日は\n晴れです。")
        self.assertEqual(result.provider, "whisper")
        self.assertEqual(result.provider_metrics, {
            "avg_logprob": -0.3,
            "no_speech_prob": 0.2,
            "compression_ratio": 1.2,
        })
        self.assertFalse(model.kwargs["vad_filter"])


class QwenProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []

        async def handler(request):
            body = await request.aread()
            self.requests.append((request, body))
            if request.url.path == "/ready":
                return httpx.Response(200, json={
                    "schema_version": 1,
                    "status": "ready",
                    "engine": "qwen3-asr",
                    "model": "1.7b",
                })
            return httpx.Response(200, json={
                "schema_version": 1,
                "request_id": "server-request-id",
                "text": "今日は右の耳が痛いです。",
                "language": "Japanese",
                "engine": "qwen3-asr",
                "model": "1.7b",
                "timing": {"inference_sec": 0.72},
                "provider_metrics": {},
            })

        self.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:8010",
        )
        self.provider = Qwen3ASRProvider("http://127.0.0.1:8010", client=self.client)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_ready_contract(self):
        payload = await self.provider.ready()
        self.assertEqual(payload["status"], "ready")

    async def test_transcribe_contract_without_fake_whisper_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "input.wav"
            wav_path.write_bytes(b"RIFFfake-wave")
            result = await self.provider.transcribe(wav_path, {
                "language": "Japanese",
                "context": "耳鼻咽喉科の診察会話",
            })

        self.assertEqual(result.text, "今日は右の耳が痛いです。")
        self.assertEqual(result.provider_metrics, {})
        self.assertNotIn("avg_logprob", result.provider_metrics)
        request, body = self.requests[-1]
        self.assertEqual(request.url.path, "/transcribe")
        self.assertEqual([request.url.path for request, _body in self.requests], ["/ready", "/transcribe"])
        self.assertIn(b'name="audio"', body)
        self.assertIn("耳鼻咽喉科の診察会話".encode(), body)
        self.assertIn(b'name="language"', body)

    async def test_common_api_error_is_preserved(self):
        async def handler(_request):
            return httpx.Response(429, json={
                "schema_version": 1,
                "error": {"code": "queue_full", "message": "queue full", "retryable": True},
            })

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8010"
        ) as client:
            provider = Qwen3ASRProvider("http://127.0.0.1:8010", client=client)
            with tempfile.TemporaryDirectory() as tmp_dir:
                wav_path = Path(tmp_dir) / "input.wav"
                wav_path.write_bytes(b"RIFFfake-wave")
                with self.assertRaises(ASRProviderError) as raised:
                    await provider.transcribe(wav_path, {})
        self.assertEqual(raised.exception.code, "queue_full")
        self.assertTrue(raised.exception.retryable)

    def test_factory_rejects_non_local_qwen_url(self):
        with self.assertRaises(ValueError):
            create_asr_provider(
                {"provider": "qwen3-asr", "qwen": {"base_url": "http://example.com:8010"}},
                lambda _path: None,
            )

    def test_provider_model_display_label(self):
        self.assertEqual(asr_model_label("qwen3-asr", "1.7b"), "qwen3-asr:1.7b")
        self.assertEqual(asr_model_label("whisper", "large-v3"), "whisper:large-v3")


class VibeVoiceProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_vibevoice_extensions_are_sent_and_returned(self):
        requests = []

        async def handler(request):
            body = await request.aread()
            requests.append((request, body))
            if request.url.path == "/ready":
                return httpx.Response(200, json={
                    "schema_version": 1,
                    "status": "ready",
                    "engine": "vibevoice-asr",
                    "model": "7b",
                })
            return httpx.Response(200, json={
                "schema_version": 1,
                "request_id": "vibe-request-id",
                "text": "今日はどうされましたか。",
                "language": "Japanese",
                "engine": "vibevoice-asr",
                "backend": "transformers",
                "model": "7b",
                "model_id": "microsoft/VibeVoice-ASR",
                "segments": [{
                    "speaker": "Speaker 0", "start_sec": 0.1, "end_sec": 1.2,
                    "text": "今日はどうされましたか。",
                }],
                "timing": {"inference_sec": 0.8},
                "provider_metrics": {},
            })

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8020"
        ) as client:
            provider = VibeVoiceASRProvider("http://127.0.0.1:8020", client=client)
            with tempfile.TemporaryDirectory() as tmp_dir:
                wav_path = Path(tmp_dir) / "input.wav"
                wav_path.write_bytes(b"RIFFfake-wave")
                result = await provider.transcribe(wav_path, {
                    "language": "Japanese",
                    "context": "耳鼻咽喉科の診察会話",
                    "hotwords": ["滲出性中耳炎", "鼓膜切開"],
                    "include_segments": True,
                })

        self.assertEqual(result.provider, "vibevoice-asr")
        self.assertEqual(result.backend, "transformers")
        self.assertEqual(result.model_id, "microsoft/VibeVoice-ASR")
        self.assertEqual(result.segments[0]["speaker"], "Speaker 0")
        body = requests[-1][1]
        self.assertIn(b'name="hotwords"', body)
        self.assertIn("滲出性中耳炎".encode(), body)
        self.assertIn(b'name="include_segments"', body)
        self.assertIn(b"true", body)

    def test_factory_accepts_local_vibevoice_and_rejects_remote_url(self):
        provider = create_asr_provider(
            {"provider": "vibevoice-asr", "vibevoice": {"base_url": "http://localhost:8020"}},
            lambda _path: None,
        )
        self.assertIsInstance(provider, VibeVoiceASRProvider)
        with self.assertRaises(ValueError):
            create_asr_provider(
                {"provider": "vibevoice-asr", "vibevoice": {"base_url": "http://example.com:8020"}},
                lambda _path: None,
            )


if __name__ == "__main__":
    unittest.main()
