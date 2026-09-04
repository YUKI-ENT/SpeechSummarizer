import unittest

from launcher import get_hearing_translation_languages, validate_hearing_translation_languages


class HearingTranslationLauncherTests(unittest.TestCase):
    def test_uses_default_languages_for_an_existing_config_without_section(self):
        languages = get_hearing_translation_languages({})

        self.assertEqual([item["id"] for item in languages], ["en", "zh", "ko"])

    def test_validates_and_normalizes_languages(self):
        languages = validate_hearing_translation_languages(
            [(" en ", " 英語 ", " English "), ("fr", "フランス語", "French")],
            "fr",
        )

        self.assertEqual(
            languages,
            [
                {"id": "en", "label": "英語", "name": "English"},
                {"id": "fr", "label": "フランス語", "name": "French"},
            ],
        )

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "重複"):
            validate_hearing_translation_languages(
                [("en", "英語", "English"), ("en", "英語2", "English")],
                "en",
            )

    def test_rejects_unknown_default_language(self):
        with self.assertRaisesRegex(ValueError, "既定"):
            validate_hearing_translation_languages(
                [("en", "英語", "English")],
                "fr",
            )


if __name__ == "__main__":
    unittest.main()
