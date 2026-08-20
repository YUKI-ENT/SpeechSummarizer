import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher_helpers import qwen_api_is_ready, resolve_launcher_path


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

    @patch("launcher_helpers.urllib.request.urlopen")
    def test_qwen_ready_requires_schema_v1_ready(self, urlopen):
        urlopen.return_value = FakeResponse(200, {"schema_version": 1, "status": "ready"})
        self.assertTrue(qwen_api_is_ready("http://127.0.0.1:8010"))
        urlopen.assert_called_once_with("http://127.0.0.1:8010/ready", timeout=1.5)

    @patch("launcher_helpers.urllib.request.urlopen")
    def test_qwen_not_ready_response_is_false(self, urlopen):
        urlopen.return_value = FakeResponse(503, {"schema_version": 1, "status": "loading"})
        self.assertFalse(qwen_api_is_ready("http://127.0.0.1:8010"))


if __name__ == "__main__":
    unittest.main()
