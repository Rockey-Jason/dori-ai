import unittest

from dori_ai.site_data import SiteData


class FakeSiteData(SiteData):
    def __init__(self):
        super().__init__()
        self.last_params = None

    def readable_news_limit(self, user_id=None):
        return 2

    def _get(self, table, params, timeout=6):
        self.last_params = (table, params)
        return [
            {"news_number": 2, "rockey_news": "돌이 AI와 인공지능 기술을 다루는 기사"},
            {"news_number": 1, "rockey_news": "체스 게임과 전략에 관한 기사"},
        ]


class ReadableNewspaperSearchTests(unittest.TestCase):
    def test_query_is_bounded_by_user_read_limit_at_database(self):
        site = FakeSiteData()
        rows = site.search_news("돌이신문 인공지능 기사 검색", user_id="verified-user")
        self.assertEqual(site.last_params[0], "rockey_news")
        self.assertEqual(site.last_params[1]["news_number"], "lte.2")
        self.assertTrue(all(int(row["news_number"]) <= 2 for row in rows))

    def test_relevant_article_ranks_first(self):
        site = FakeSiteData()
        rows = site.search_news("인공지능", user_id="verified-user")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["news_number"], 2)

    def test_no_access_returns_no_articles(self):
        class NoAccessSite(FakeSiteData):
            def readable_news_limit(self, user_id=None):
                return 0
        site = NoAccessSite()
        self.assertEqual(site.search_news("인공지능", user_id=None), [])
        self.assertIsNone(site.last_params)


if __name__ == "__main__":
    unittest.main()
