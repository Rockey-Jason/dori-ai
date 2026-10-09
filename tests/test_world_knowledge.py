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
        world_knowledge.CACHE_PATH = self.cache_path
        world_knowledge._MEMORY.clear()

    def tearDown(self):
        world_knowledge.CACHE_PATH = self.old_path
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
