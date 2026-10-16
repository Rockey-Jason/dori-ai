import unittest
from collections import OrderedDict
from unittest.mock import patch

from dori_ai.response_engine import ResponseEngine


class FakeMemory:
    def context(self, query, k=5):
        return "사용자는 돌이 사이트를 만들고 있어."


class FakeDialogue:
    def __init__(self):
        self.history = []
        self.memory = FakeMemory()

    def add_turn(self, user, assistant):
        self.history.extend([("user", user), ("dori", assistant)])


class FakeKnowledge:
    def answer(self, query, threshold=0.70):
        return None


class FreeRoutingTests(unittest.TestCase):
    def test_creative_and_transform_requests_do_not_trigger_search(self):
        self.assertFalse(ResponseEngine._needs_web("짧은 시를 써줘?"))
        self.assertFalse(ResponseEngine._needs_web("Translate this sentence for me"))
        self.assertTrue(ResponseEngine._needs_web("오늘 날씨 어때?"))

    def test_free_offline_mode_has_useful_conversation_fallbacks(self):
        supportive = ResponseEngine._offline_conversation("기분이 조금 우울해")
        self.assertIn("이야기해 줘", supportive)
        capabilities = ResponseEngine._offline_conversation("너는 뭐 할 수 있어?")
        self.assertIn("번역", capabilities)
        joke = ResponseEngine._offline_conversation("농담해 줘")
        self.assertIn("돌이", joke)

    def test_live_query_is_classified_separately(self):
        self.assertTrue(ResponseEngine._is_time_sensitive("오늘 원달러 환율"))
        self.assertFalse(ResponseEngine._is_time_sensitive("피타고라스 정리 설명해줘"))

    def test_configured_self_hosted_model_handles_open_ended_chat_without_search(self):
        dialogue = FakeDialogue()
        engine = ResponseEngine.__new__(ResponseEngine)
        engine.dialogues = OrderedDict([("anonymous", dialogue)])
        engine.max_dialogues = 64
        engine.smalltalk = {}
        engine.kb = FakeKnowledge()
        engine.web_enabled = False
        engine.llm_enabled = True

        with patch.object(ResponseEngine, "_site_answer", return_value=None), \
             patch.object(ResponseEngine, "_reasoning_answer", return_value=None), \
             patch.object(ResponseEngine, "_builtin_answer", return_value=None), \
             patch("dori_ai.response_engine.math_answer", return_value=None), \
             patch("dori_ai.response_engine.llm_answer", return_value="그랬구나. 어떤 점이 가장 마음에 걸리는지 이야기해 줘.") as llm:
            answer = engine.reply("기분이 조금 가라앉아 있어.")

        self.assertIn("어떤 점", answer)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(llm.call_args.kwargs["memory_context"], "사용자는 돌이 사이트를 만들고 있어.")
        self.assertEqual(llm.call_args.kwargs["evidence"], None)

if __name__ == "__main__":
    unittest.main()
