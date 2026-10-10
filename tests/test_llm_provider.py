import json
import os
import unittest
from unittest.mock import patch, MagicMock

from dori_ai import llm_provider


class LLMProviderTests(unittest.TestCase):
    def test_provider_is_disabled_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(llm_provider.enabled())
            self.assertIsNone(llm_provider.answer("안녕"))

    def test_history_and_evidence_are_sent_to_chat_completions(self):
        fake_response = MagicMock()
        fake_response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": "피타고라스는 고대 그리스의 수학자야."}}]
        }).encode("utf-8")
        with patch.dict(os.environ, {
            "DORI_LLM_API_KEY": "test-secret",
            "DORI_LLM_MODEL": "test-model",
            "DORI_LLM_BASE_URL": "https://example.invalid/v1",
        }, clear=True), patch.object(llm_provider.urllib.request, "urlopen", return_value=fake_response) as urlopen:
            answer = llm_provider.answer(
                "그 사람은 뭘 했어?",
                history=[("user", "피타고라스가 누구야?"), ("dori", "고대 그리스의 수학자야.")],
                evidence="Test source: Pythagoras",
                language="ko",
            )
        self.assertEqual(answer, "피타고라스는 고대 그리스의 수학자야.")
        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "test-model")
        self.assertTrue(any(m["role"] == "assistant" and "고대 그리스의 수학자" in m["content"] for m in body["messages"]))
        self.assertTrue(any(m["role"] == "system" and "Test source" in m["content"] for m in body["messages"]))
        self.assertTrue(any("cite the relevant source inline" in m["content"] for m in body["messages"] if m["role"] == "system"))
        self.assertLessEqual(body["max_tokens"], 1200)
        self.assertNotIn("test-secret", request.data.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
