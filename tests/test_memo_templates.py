import json
import tempfile
import unittest
from pathlib import Path

from memo_templates import load_memo_templates, save_memo_templates, validate_memo_templates


class MemoTemplatesTests(unittest.TestCase):
    def test_validate_normalizes_values_and_defaults_enabled(self):
        result = validate_memo_templates({
            "version": 1,
            "templates": [{"id": " greeting ", "label": " 挨拶 ", "text": " 本文 "}],
        })
        self.assertEqual(result["templates"], [{
            "id": "greeting", "label": "挨拶", "text": "本文", "enabled": True,
        }])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_memo_templates({
                "templates": [
                    {"id": "same", "label": "A", "text": "A"},
                    {"id": "same", "label": "B", "text": "B"},
                ],
            })

    def test_save_and_load_round_trip(self):
        data = {
            "version": 1,
            "templates": [{"id": "one", "label": "定型文", "text": "本文", "enabled": False}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memo_templates.json"
            save_memo_templates(path, data)
            self.assertEqual(load_memo_templates(path), data)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), data)
            saved = path.read_bytes()
            self.assertNotIn(b"\n", saved.replace(b"\r\n", b""))


if __name__ == "__main__":
    unittest.main()
