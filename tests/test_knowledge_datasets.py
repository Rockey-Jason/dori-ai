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

    def test_original_heldout_set_is_separate_and_well_formed(self):
        path = ROOT / "data/evaluation/general_qa.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 15)
        self.assertTrue(all(row.get("question") and row.get("keywords") and row.get("category") for row in rows))
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertFalse(any("answer" in row for row in rows))

    def test_response_quality_benchmark_is_held_out_and_rubric_based(self):
        path = ROOT / "data/evaluation/response_quality.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 35)
        self.assertTrue(all(row.get("question") and row.get("criteria") and row.get("category") for row in rows))
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertFalse(any("answer" in row for row in rows))
        self.assertTrue({"reasoning", "language", "instruction_following", "safety"}.issubset(
            {row["category"] for row in rows}
        ))

    def test_heldout_evaluation_data_is_not_a_training_source(self):
        # The trainer's source fingerprint intentionally hashes only approved
        # training directories; benchmark rows must remain outside training.
        from scripts.continuous_train import source_fingerprint
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data" / "evaluation").mkdir(parents=True)
            first = source_fingerprint(root)
            (root / "data" / "evaluation" / "leak.jsonl").write_text(
                '{"id":"heldout","question":"do not train","keywords":["answer"]}\\n',
                encoding="utf-8",
            )
            second = source_fingerprint(root)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
