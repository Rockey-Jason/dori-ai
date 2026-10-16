import tempfile
import unittest
from pathlib import Path

from scripts.continuous_train import source_fingerprint


class GrowthFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data" / "new").mkdir(parents=True)
        (self.root / "data" / "streaming").mkdir(parents=True)
        (self.root / "data" / "evaluation").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_training_source_change_changes_fingerprint(self):
        source = self.root / "data" / "new" / "approved.jsonl"
        source.write_text('{"question":"What is a planet?","answer":"A world orbiting a star."}\n', encoding="utf-8")
        first = source_fingerprint(self.root)
        source.write_text('{"question":"What is a planet?","answer":"A celestial body orbiting a star."}\n', encoding="utf-8")
        second = source_fingerprint(self.root)
        self.assertNotEqual(first, second)

    def test_streaming_cache_and_heldout_data_do_not_start_training_cycles(self):
        source = self.root / "data" / "new" / "approved.txt"
        source.write_text("reviewed training data", encoding="utf-8")
        first = source_fingerprint(self.root)
        (self.root / "data" / "streaming" / "train.jsonl").write_text("temporary cache", encoding="utf-8")
        (self.root / "data" / "evaluation" / "general_qa.jsonl").write_text("held-out cases", encoding="utf-8")
        self.assertEqual(first, source_fingerprint(self.root))

    def test_tokenizer_change_changes_fingerprint(self):
        source = self.root / "data" / "new" / "approved.txt"
        source.write_text("reviewed training data", encoding="utf-8")
        tokenizer = self.root / "data" / "tokenizer.json"
        tokenizer.write_text('{"vocab_size":768}', encoding="utf-8")
        first = source_fingerprint(self.root)
        tokenizer.write_text('{"vocab_size":769}', encoding="utf-8")
        self.assertNotEqual(first, source_fingerprint(self.root))


if __name__ == "__main__":
    unittest.main()
