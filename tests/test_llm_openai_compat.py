import unittest
from unittest.mock import Mock, patch

from llm_openai_compat import build_openai_base_url, list_openai_models, normalize_openai_base_url, openai_chat_text


class OpenAiCompatClientTests(unittest.TestCase):
    def test_builds_base_url_from_server_and_port(self):
        self.assertEqual(build_openai_base_url("127.0.0.1", 11434), "http://127.0.0.1:11434/v1")
        self.assertEqual(build_openai_base_url("api.example.com", 443, use_https=True), "https://api.example.com:443/v1")

    def test_normalizes_v1_base_url(self):
        self.assertEqual(normalize_openai_base_url("http://localhost:1234"), "http://localhost:1234/v1")
        self.assertEqual(normalize_openai_base_url("http://localhost:1234/v1/"), "http://localhost:1234/v1")

    @patch("llm_openai_compat.requests.post")
    def test_chat_completions_request(self, post):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "summary"}}]}
        post.return_value = response

        text, _ = openai_chat_text(
            base_url="http://localhost:1234/v1/", api_key="secret", model="local-model",
            prompt="hello", timeout_sec=30, temperature=0.2, top_p=0.8,
        )

        self.assertEqual(text, "summary")
        _, kwargs = post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(kwargs["json"]["messages"], [{"role": "user", "content": "hello"}])
        self.assertNotIn("response_format", kwargs["json"])
        self.assertNotIn("reasoning_effort", kwargs["json"])

    @patch("llm_openai_compat.requests.post")
    def test_chat_completions_disables_reasoning(self, post):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "summary"}}]}
        post.return_value = response

        openai_chat_text(
            base_url="http://localhost:1234/v1", api_key="", model="local-model",
            prompt="hello", timeout_sec=30, temperature=0.2, top_p=0.8,
            reasoning_enabled=False,
        )

        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "none")

    @patch("llm_openai_compat.requests.post")
    def test_chat_completions_enables_reasoning(self, post):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "summary"}}]}
        post.return_value = response

        openai_chat_text(
            base_url="http://localhost:1234/v1", api_key="", model="local-model",
            prompt="hello", timeout_sec=30, temperature=0.2, top_p=0.8,
            reasoning_enabled=True,
        )

        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "medium")

    @patch("llm_openai_compat.requests.get")
    def test_lists_models_from_openai_schema(self, get):
        response = Mock()
        response.json.return_value = {"data": [{"id": "z-model"}, {"id": "a-model"}]}
        get.return_value = response

        self.assertEqual(list_openai_models("http://localhost:1234", timeout_sec=10), ["a-model", "z-model"])


if __name__ == "__main__":
    unittest.main()
