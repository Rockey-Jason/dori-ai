import tempfile
import unittest
from pathlib import Path

from scripts import collect_world_knowledge as collector


class CollectWorldKnowledgeTests(unittest.TestCase):
    def test_corpus_output_rotates_into_numbered_shards(self):
        old_out = collector.OUT
        old_limit = collector.MAX_SHARD_BYTES
        with tempfile.TemporaryDirectory() as folder:
            collector.OUT = Path(folder) / "world_wikipedia.jsonl"
            collector.MAX_SHARD_BYTES = 64
            try:
                self.assertEqual(collector._current_output_path(), collector.OUT)
                collector.OUT.write_text("a" * 80, encoding="utf-8")
                shard1 = Path(folder) / "world_wikipedia_0001.jsonl"
                self.assertEqual(collector._current_output_path(), shard1)
                shard1.write_text("b" * 64, encoding="utf-8")
                shard2 = Path(folder) / "world_wikipedia_0002.jsonl"
                self.assertEqual(collector._current_output_path(), shard2)
                self.assertEqual(
                    [p.name for p in collector._corpus_paths()],
                    ["world_wikipedia.jsonl", "world_wikipedia_0001.jsonl", "world_wikipedia_0002.jsonl"],
                )
            finally:
                collector.OUT = old_out
                collector.MAX_SHARD_BYTES = old_limit

    def test_seed_queue_contains_both_languages_and_generated_queries(self):
        queue = collector._seed_query_queue()
        languages = {row["language"] for row in queue}
        self.assertEqual(languages, {"ko", "en"})
        self.assertGreater(len(queue), len(collector.KO_TOPICS) + len(collector.EN_TOPICS))
        self.assertTrue(any("작동 원리" in row["query"] for row in queue if row["language"] == "ko"))
        self.assertTrue(any("real-world applications" in row["query"] for row in queue if row["language"] == "en"))

    def test_discovered_title_creates_two_repeatable_follow_up_queries(self):
        first = collector._queries_for_discovered_title("Quantum mechanics", "en")
        second = collector._queries_for_discovered_title("Quantum mechanics", "en")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all("Quantum mechanics" in query for query in first))

    def test_pending_queue_deduplicates_done_and_queued_queries(self):
        queue = []
        queued = set()
        done = {collector._query_key("en", "mathematics")}
        self.assertFalse(collector._append_pending(queue, queued, done, "en", "Mathematics", 100))
        self.assertTrue(collector._append_pending(queue, queued, done, "ko", "미적분", 100))
        self.assertFalse(collector._append_pending(queue, queued, done, "ko", " 미적분 ", 100))
        self.assertEqual(len(queue), 1)


if __name__ == "__main__":
    unittest.main()
