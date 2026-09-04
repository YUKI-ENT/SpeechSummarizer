import json
import tempfile
from pathlib import Path
import unittest
import threading
from unittest.mock import patch

import app as app_module


class HearingTranslationTests(unittest.IsolatedAsyncioTestCase):
    async def test_settings_expose_model_and_languages_without_credentials(self):
        languages = [{"id": "en", "label": "英語", "name": "English"}]
        with (
            patch.object(app_module, "HEARING_TRANSLATION_ENABLED", True),
            patch.object(app_module, "HEARING_TRANSLATION_MODEL", "cloud-translate-model"),
            patch.object(app_module, "HEARING_TRANSLATION_LANGUAGES", languages),
            patch.object(app_module, "HEARING_TRANSLATION_DEFAULT_LANGUAGE", "en"),
        ):
            result = await app_module.api_hearing_translation_settings()

        self.assertEqual(result["model"], "cloud-translate-model")
        self.assertEqual(result["languages"], [{"id": "en", "label": "英語"}])
        self.assertNotIn("api_key", result)
        self.assertNotIn("base_url", result)

    async def test_translate_uses_dedicated_model_and_returns_ephemeral_text(self):
        languages = [{"id": "en", "label": "英語", "name": "English"}]
        translation_thread_names = []

        def translate_in_worker(**_kwargs):
            translation_thread_names.append(threading.current_thread().name)
            return "How are you?", {"model": "cloud-translate-model"}

        with (
            patch.object(app_module, "HEARING_TRANSLATION_ENABLED", True),
            patch.object(app_module, "HEARING_TRANSLATION_MODEL", "cloud-translate-model"),
            patch.object(app_module, "HEARING_TRANSLATION_BASE_URL", "https://example.test/v1"),
            patch.object(app_module, "HEARING_TRANSLATION_API_KEY", "secret"),
            patch.object(app_module, "HEARING_TRANSLATION_TIMEOUT", 12.0),
            patch.object(app_module, "HEARING_TRANSLATION_REASONING_ENABLED", None),
            patch.object(app_module, "HEARING_TRANSLATION_LANGUAGES", languages),
            patch.object(app_module, "HEARING_TRANSLATION_DEFAULT_LANGUAGE", "en"),
            patch.object(
                app_module,
                "openai_chat_text",
                side_effect=translate_in_worker,
            ) as translate,
        ):
            result = await app_module.api_hearing_translation_translate({
                "text": "具合はいかがですか？",
                "language": "en",
            })

        self.assertEqual(result["text"], "How are you?")
        self.assertEqual(result["language"], "en")
        kwargs = translate.call_args.kwargs
        self.assertEqual(kwargs["model"], "cloud-translate-model")
        self.assertEqual(kwargs["base_url"], "https://example.test/v1")
        self.assertIn("English", kwargs["prompt"])
        self.assertIn("具合はいかがですか？", kwargs["prompt"])
        self.assertIsNone(kwargs["reasoning_enabled"])
        self.assertEqual(len(translation_thread_names), 1)
        self.assertTrue(translation_thread_names[0].startswith("hearing-translation"))

    async def test_model_list_uses_translation_connection(self):
        with (
            patch.object(app_module, "HEARING_TRANSLATION_BASE_URL", "https://translation.test/v1"),
            patch.object(app_module, "HEARING_TRANSLATION_API_KEY", "translation-key"),
            patch.object(app_module, "HEARING_TRANSLATION_MODEL", "saved-model"),
            patch.object(app_module, "list_openai_models", return_value=["other-model"]) as models,
        ):
            result = await app_module.api_hearing_translation_models()
        self.assertEqual(result["models"], ["saved-model", "other-model"])
        self.assertEqual(models.call_args.args, ("https://translation.test/v1",))
        self.assertEqual(models.call_args.kwargs["api_key"], "translation-key")

    async def test_model_selection_persists_and_applies_without_losing_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = {"llm": {"model_default": "summary"}, "hearing_translation": {
                "model": "old", "languages": [{"id": "id", "label": "インドネシア語", "name": "Indonesian"}]
            }}
            path.write_text(json.dumps(original), encoding="utf-8")
            with (
                patch.object(app_module, "CONFIG_PATH", path),
                patch.object(app_module, "CFG", {}),
                patch.object(app_module, "HEARING_TRANSLATION_CFG", {}),
                patch.object(app_module, "HEARING_TRANSLATION_MODEL", "old"),
            ):
                result = await app_module.api_hearing_translation_update_settings({"model": " new-model "})
                self.assertTrue(result["ok"])
                self.assertEqual((await app_module.api_hearing_translation_settings())["model"], "new-model")
            original["hearing_translation"]["model"] = "new-model"
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)

    async def test_failed_save_keeps_original_file_and_active_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"hearing_translation": {"model": "old"}}', encoding="utf-8")
            before = path.read_bytes()
            with (
                patch.object(app_module, "CONFIG_PATH", path),
                patch.object(app_module, "HEARING_TRANSLATION_MODEL", "old"),
                patch.object(app_module.os, "replace", side_effect=OSError("disk error")),
            ):
                result = await app_module.api_hearing_translation_update_settings({"model": "new"})
                self.assertEqual(result.status_code, 500)
                self.assertEqual(app_module.HEARING_TRANSLATION_MODEL, "old")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    async def test_invalid_model_does_not_write_config(self):
        with patch.object(app_module.os, "replace") as replace:
            for model in (None, "", "  ", [], 123, "x" * 257):
                result = await app_module.api_hearing_translation_update_settings({"model": model})
                self.assertEqual(result.status_code, 400)
            replace.assert_not_called()

    async def test_indonesian_language_is_used_in_prompt(self):
        with (
            patch.object(app_module, "HEARING_TRANSLATION_ENABLED", True),
            patch.object(app_module, "HEARING_TRANSLATION_LANGUAGES", [
                {"id": "id", "label": "インドネシア語", "name": "Indonesian"}
            ]),
            patch.object(app_module, "openai_chat_text", return_value=("Halo", {})) as translate,
        ):
            result = await app_module.api_hearing_translation_translate({"text": "こんにちは", "language": "id"})
        self.assertEqual(result["language"], "id")
        self.assertIn("Indonesian", translate.call_args.kwargs["prompt"])

    async def test_translate_rejects_language_not_in_config(self):
        languages = [{"id": "en", "label": "英語", "name": "English"}]
        with (
            patch.object(app_module, "HEARING_TRANSLATION_ENABLED", True),
            patch.object(app_module, "HEARING_TRANSLATION_MODEL", "model"),
            patch.object(app_module, "HEARING_TRANSLATION_LANGUAGES", languages),
            patch.object(app_module, "openai_chat_text") as translate,
        ):
            result = await app_module.api_hearing_translation_translate({
                "text": "こんにちは",
                "language": "invalid",
            })

        self.assertEqual(result.status_code, 400)
        translate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
