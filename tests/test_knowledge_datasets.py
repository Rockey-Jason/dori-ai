import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

class KnowledgeDatasetTests(unittest.TestCase):
    def test_training_seed_has_question_answer_pairs(self):
        path = ROOT / "data/curriculum/qa_following/verified_general_qa.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 40)
        self.assertTrue(all(row.get("question") and row.get("answer") and row.get("source") for row in rows))
        self.assertEqual(len({row["id"] for row in rows}), len(rows))

    def test_evaluation_set_is_separate_and_has_expected_keywords(self):
        path = ROOT / "data/evaluation/general_qa.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 15)
        self.assertTrue(all(row.get("question") and row.get("keywords") and row.get("category") for row in rows))
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertFalse(any("answer" in row for row in rows))

if __name__ == "__main__": unittest.main()
