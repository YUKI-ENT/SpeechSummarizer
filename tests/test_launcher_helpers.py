import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from launcher_helpers import (
    build_qwen_server_command,
    build_vibevoice_server_command,
    fetch_qwen_ready_status,
    qwen_api_is_ready,
    resolve_launcher_path,
)


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LauncherHelperTests(unittest.TestCase):
    def test_relative_managed_path_is_based_on_launcher_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            app_dir = Path(tmp_dir) / "SpeechSummarizer"
            expected = (app_dir / "../QwenASR/server.py").resolve()
            self.assertEqual(resolve_launcher_path("../QwenASR/server.py", app_dir), expected)

    def test_qwen_server_command_includes_selected_model(self):
        command = build_qwen_server_command(
            Path("C:/Qwen/QwenASR-Server.exe"),
            Path("C:/Qwen/config.json"),
            "0.6b",
        )
        self.assertEqual(command, [
            str(Path("C:/Qwen/QwenASR-Server.exe")),
            "--config",
            str(Path("C:/Qwen/config.json")),
            "--model",
            "0.6b",
        ])

    def test_qwen_server_command_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            build_qwen_server_command(Path("server.exe"), Path("config.json"), "large")

    def test_vibevoice_server_command_uses_dedicated_python_environment(self):
        command = build_vibevoice_server_command(
            Path("C:/VibeVoiceASR/.venv/Scripts/python.exe"),
            Path("C:/VibeVoiceASR/server.py"),
            Path("C:/VibeVoiceASR/config.json"),
            "7b",
        )
        self.assertEqual(command, [
            str(Path("C:/VibeVoiceASR/.venv/Scripts/python.exe")),
            str(Path("C:/VibeVoiceASR/server.py")),
            "--config",
            str(Path("C:/VibeVoiceASR/config.json")),
            "--model",
            "7b",
        ])

    @patch("launcher_helpers.urllib.request.urlopen")
    def test_qwen_ready_requires_schema_v1_ready(self, urlopen):
        urlopen.return_value = FakeResponse(200, {
            "schema_version": 1,
            "status": "ready",
            "model": "1.7b",
            "model_id": "Qwen/Qwen3-ASR-1.7B",
        })
        self.assertTrue(qwen_api_is_ready("http://127.0.0.1:8010"))
        urlopen.assert_called_once_with("http://127.0.0.1:8010/ready", timeout=1.5)

    @patch("launcher_helpers.urllib.request.urlopen")
    def test_qwen_ready_status_keeps_model_details(self, urlopen):
        payload = {
            "schema_version": 1,
            "status": "ready",
            "model": "0.6b",
            "model_id": "Qwen/Qwen3-ASR-0.6B",
            "device": "cuda:0",
            "queue_depth": 1,
            "queue_capacity": 20,
        }
        urlopen.return_value = FakeResponse(200, payload)

        status = fetch_qwen_ready_status("http://127.0.0.1:8010")

        self.assertTrue(status.reachable)
        self.assertTrue(status.ready)
        self.assertEqual(status.http_status, 200)
        self.assertEqual(status.payload, payload)

    @patch("launcher_helpers.urllib.request.urlopen")
    def test_qwen_not_ready_response_is_false(self, urlopen):
        urlopen.return_value = FakeResponse(503, {"schema_version": 1, "status": "loading"})
        self.assertFalse(qwen_api_is_ready("http://127.0.0.1:8010"))

    @patch("launcher_helpers.urllib.request.urlopen")
    def test_qwen_http_error_keeps_api_error_message(self, urlopen):
        body = json.dumps({
            "schema_version": 1,
            "error": {"code": "not_ready", "message": "モデルをロード中です。"},
        }).encode("utf-8")
        urlopen.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:8010/ready", 503, "Service Unavailable", {}, io.BytesIO(body)
        )

        status = fetch_qwen_ready_status("http://127.0.0.1:8010")

        self.assertTrue(status.reachable)
        self.assertFalse(status.ready)
        self.assertEqual(status.http_status, 503)
        self.assertEqual(status.error, "モデルをロード中です。")

    @patch("launcher_helpers.urllib.request.urlopen")
    def test_qwen_schema_mismatch_is_reported(self, urlopen):
        urlopen.return_value = FakeResponse(200, {"schema_version": 2, "status": "ready"})

        status = fetch_qwen_ready_status("http://127.0.0.1:8010")

        self.assertFalse(status.ready)
        self.assertEqual(status.error, "unsupported schema_version: 2")


if __name__ == "__main__":
    unittest.main()
