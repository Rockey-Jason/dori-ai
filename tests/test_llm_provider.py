import json
import os
import unittest
from unittest.mock import patch, MagicMock

from dori_ai import llm_provider


class LocalModelProviderTests(unittest.TestCase):
    def test_disabled_without_private_local_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(llm_provider.enabled())
            self.assertIsNone(llm_provider.answer("안녕"))

    def test_public_ai_endpoints_are_rejected(self):
        for url in ("https://api.openai.com/v1", "http://example.com:8080", "https://my-model.internal"):
            with patch.dict(os.environ, {"DORI_LOCAL_LLM_URL": url}, clear=True):
                self.assertFalse(llm_provider.enabled(), url)
                self.assertIsNone(llm_provider.answer("안녕"))

    def test_local_model_receives_history_and_evidence_without_api_key(self):
        fake_response = MagicMock()
        fake_response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"message": {"content": "피타고라스는 고대 그리스의 수학자야."}}]
        }).encode("utf-8")
        with patch.dict(os.environ, {
            "DORI_LOCAL_LLM_URL": "http://127.0.0.1:8080",
            "DORI_LOCAL_LLM_MODEL": "dori-local-test",
        }, clear=True), patch.object(llm_provider.urllib.request, "urlopen", return_value=fake_response) as urlopen:
            answer = llm_provider.answer(
                "그 사람은 뭘 했어?",
                history=[("user", "피타고라스가 누구야?"), ("dori", "고대 그리스의 수학자야.")],
                evidence="Test source: Pythagoras https://example.org/p",
                language="ko",
            )
        self.assertEqual(answer, "피타고라스는 고대 그리스의 수학자야.")
        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "dori-local-test")
        self.assertTrue(any(m["role"] == "assistant" and "고대 그리스의 수학자" in m["content"] for m in body["messages"]))
        self.assertTrue(any(m["role"] == "system" and "https://example.org/p" in m["content"] for m in body["messages"]))
        self.assertFalse(request.has_header("Authorization"))
        self.assertLessEqual(body["max_tokens"], 900)


if __name__ == "__main__":
    unittest.main()
