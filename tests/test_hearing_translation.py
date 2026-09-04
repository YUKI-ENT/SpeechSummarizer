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
