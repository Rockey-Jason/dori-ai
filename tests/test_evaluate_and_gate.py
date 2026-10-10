import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_and_gate import normalize_answer, read_dataset, should_promote


class HeldOutGateTests(unittest.TestCase):
    def test_keyword_normalization_handles_spacing_and_superscripts(self):
        self.assertEqual(normalize_answer("A² + B² = C²"), "a^2+b^2=c^2".replace("+", "").replace("=", ""))

    def test_candidate_must_strictly_improve(self):
        self.assertTrue(should_promote({"rate": 0.4}, {"rate": 0.45}))
        self.assertFalse(should_promote({"rate": 0.4}, {"rate": 0.4}))
        self.assertFalse(should_promote({"rate": 0.4}, {"rate": 0.3}))

    def test_dataset_rejects_empty_or_malformed_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qa.jsonl"
            path.write_text(json.dumps({"question": "Q?", "keywords": ["answer"]}) + "\n", encoding="utf-8")
            self.assertEqual(len(read_dataset(path)), 1)
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_dataset(path)
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_dataset(path)


if __name__ == "__main__":
    unittest.main()
