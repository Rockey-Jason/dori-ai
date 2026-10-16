import json
import tempfile
import unittest
from pathlib import Path

from dori_ai.retrieval import LocalKnowledge


class RetrievalPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "knowledge.jsonl"
        rows = [
            {"question": "What is Earth?", "answer": "Earth is a planet."},
            {"question": "How does photosynthesis work?", "answer": "Plants convert light energy into chemical energy."},
        ]
        self.path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
        self.kb = LocalKnowledge(paths=[self.path])

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_match_uses_fast_exact_index(self):
        self.assertEqual(self.kb.answer("What is Earth?"), "Earth is a planet.")

    def test_unrelated_prompt_is_not_given_a_weak_match(self):
        self.assertIsNone(self.kb.answer("Explain medieval European banking laws."))


if __name__ == "__main__":
    unittest.main()
