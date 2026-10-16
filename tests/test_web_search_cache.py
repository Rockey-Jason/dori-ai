import unittest
from unittest.mock import MagicMock, patch

from dori_ai import web_search


class WebSearchCacheTests(unittest.TestCase):
    def setUp(self):
        with web_search._CACHE_LOCK:
            web_search._CACHE.clear()

    def tearDown(self):
        with web_search._CACHE_LOCK:
            web_search._CACHE.clear()

    def test_normalized_repeat_query_uses_cache(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'<div class="result results_links">'
            b'<a class="result__a" href="https://example.org/page">Example title</a>'
            b'<div class="result__snippet">An informative snippet.</div>'
            b'</div>'
        )
        with patch.object(web_search.urllib.request, "urlopen", return_value=response) as urlopen:
            first = web_search.search("  Dori   AI  ")
            second = web_search.search("Dori AI", limit=1)

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["title"], "Example title")

    def test_empty_results_are_cached_briefly(self):
        with patch.object(web_search.urllib.request, "urlopen", side_effect=OSError("offline")) as urlopen:
            self.assertEqual(web_search.search("offline query"), [])
            self.assertEqual(web_search.search("offline query"), [])
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
