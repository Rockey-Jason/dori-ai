import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from dori_ai import world_knowledge


class WorldKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.tmp.name) / "cache.json"
        self.old_path = world_knowledge.CACHE_PATH
        self.old_memory = dict(world_knowledge._MEMORY)
        self.old_corpus_path = world_knowledge.CORPUS_PATH
        self.old_local_cache = world_knowledge._LOCAL_CORPUS_CACHE
        self.old_local_signature = world_knowledge._LOCAL_CORPUS_SIGNATURE
        self.corpus_path = Path(self.tmp.name) / "corpus.jsonl"
        world_knowledge.CACHE_PATH = self.cache_path
        world_knowledge.CORPUS_PATH = self.corpus_path
        world_knowledge._LOCAL_CORPUS_CACHE = None
        world_knowledge._LOCAL_CORPUS_SIGNATURE = None
        world_knowledge._MEMORY.clear()

    def tearDown(self):
        world_knowledge.CACHE_PATH = self.old_path
        world_knowledge.CORPUS_PATH = self.old_corpus_path
        world_knowledge._LOCAL_CORPUS_CACHE = self.old_local_cache
        world_knowledge._LOCAL_CORPUS_SIGNATURE = self.old_local_signature
        world_knowledge._MEMORY.clear()
        world_knowledge._MEMORY.update(self.old_memory)
        self.tmp.cleanup()

    def test_clean_normalizes_whitespace(self):
        self.assertEqual(world_knowledge._clean("  Pythagoras\n  theorem  "), "Pythagoras theorem")

    def test_search_retrieves_and_caches_source_linked_summaries(self):
        hits = [{"pageid": 42, "title": "Pythagoras"}]
        pages = [{
            "title": "Pythagoras",
            "text": "Pythagoras was an ancient Greek philosopher and mathematician. " * 3,
            "url": "https://en.wikipedia.org/wiki/Pythagoras",
            "language": "en",
        }]
        with patch.object(world_knowledge, "_search_wikipedia", return_value=hits) as search, patch.object(world_knowledge, "_page_extracts", return_value=pages):
            result = world_knowledge.search("Pythagoras", language="en")
            self.assertEqual(result[0]["title"], "Pythagoras")
            self.assertTrue(self.cache_path.exists())
            search.reset_mock()
            cached = world_knowledge.search("Pythagoras", language="en")
            search.assert_not_called()
            self.assertEqual(cached[0]["url"], pages[0]["url"])

    def test_empty_search_results_are_cached_briefly(self):
        with patch.object(world_knowledge, "_search_wikipedia", side_effect=OSError("offline")) as search:
            self.assertEqual(world_knowledge.search("unavailable topic", language="en"), [])
            self.assertEqual(world_knowledge.search("unavailable topic", language="en"), [])
        search.assert_called_once()

    def test_collected_corpus_is_used_when_live_search_is_offline(self):
        row = {
            "title": "Pythagoras",
            "language": "en",
            "source": "https://en.wikipedia.org/wiki/Pythagoras",
            "license": "CC BY-SA",
            "text": (
                "Topic: Pythagoras\nLanguage: en\nEncyclopedia summary: "
                "Pythagoras was an ancient Greek philosopher and mathematician. "
                "The Pythagorean theorem relates the three sides of a right triangle. "
                "His ideas influenced later mathematics."
            ),
        }
        self.corpus_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with patch.object(world_knowledge, "_search_wikipedia", side_effect=OSError("offline")):
            results = world_knowledge.search("Pythagoras theorem", language="en")
        self.assertTrue(results)
        self.assertEqual(results[0]["title"], "Pythagoras")
        self.assertIn("Pythagorean theorem", results[0]["text"])
        self.assertEqual(results[0]["url"], row["source"])

    def test_fallback_answer_selects_relevant_sentences_and_keeps_source(self):
        answer = world_knowledge.as_answer("largest planet in the solar system", [{
            "title": "Solar System",
            "text": (
                "Mercury is the closest planet to the Sun. "
                "The largest planet in the solar system is Jupiter. "
                "Saturn has a system of rings. "
                "Some stars were recorded in ancient charts."
            ),
            "url": "https://en.wikipedia.org/wiki/Solar_System",
        }], "en")
        self.assertIn("largest planet in the solar system is Jupiter", answer)
        self.assertIn("https://en.wikipedia.org/wiki/Solar_System", answer)
        self.assertIn("Source:", answer)
        self.assertNotIn("ancient charts", answer)

    def test_evidence_contains_source_url(self):
        evidence = world_knowledge.format_evidence([{
            "title": "Earth", "text": "Earth is a planet.", "url": "https://en.wikipedia.org/wiki/Earth"
        }])
        self.assertIn("https://en.wikipedia.org/wiki/Earth", evidence)
        answer = world_knowledge.as_answer("Earth", [{
            "title": "Earth", "text": "Earth is a planet.", "url": "https://en.wikipedia.org/wiki/Earth"
        }], "en")
        self.assertIn("Earth is a planet", answer)


if __name__ == "__main__":
    unittest.main()
