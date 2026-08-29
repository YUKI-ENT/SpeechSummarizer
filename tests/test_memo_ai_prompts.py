import unittest

from memo_ai_prompts import normalize_memo_ai_settings


class MemoAiPromptsTests(unittest.TestCase):
    def test_normalizes_prompts_and_default(self):
        prompts, default_id = normalize_memo_ai_settings({
            "llm": {
                "memo_default_prompt_id": "polish",
                "memo_prompts": {
                    "proofread": {"label": " 誤字修正 ", "template": "修正してください。\n{text}"},
                    "polish": {"label": "清書", "template": "清書してください。\n{text}"},
                },
            },
        })
        self.assertEqual(list(prompts), ["proofread", "polish"])
        self.assertEqual(prompts["proofread"]["label"], "誤字修正")
        self.assertEqual(default_id, "polish")

    def test_uses_fallback_when_existing_config_has_no_memo_prompts(self):
        fallback = {
            "llm": {
                "memo_default_prompt_id": "default",
                "memo_prompts": {
                    "default": {"label": "既定", "template": "処理する。\n{text}"},
                },
            },
        }
        prompts, default_id = normalize_memo_ai_settings({"llm": {}}, fallback)
        self.assertIn("default", prompts)
        self.assertEqual(default_id, "default")

    def test_requires_text_placeholder(self):
        with self.assertRaisesRegex(ValueError, "contain"):
            normalize_memo_ai_settings({
                "llm": {
                    "memo_prompts": {
                        "broken": {"label": "不正", "template": "差し込み位置なし"},
                    },
                },
            })


if __name__ == "__main__":
    unittest.main()
