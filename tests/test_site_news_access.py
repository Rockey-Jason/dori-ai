import os
import unittest
from unittest.mock import patch

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

    def test_legacy_supabase_environment_names_are_supported(self):
        env = {
            "DORI_SUPABASE_URL": "https://example.supabase.co",
            "DORI_SUPABASE_KEY": "anon-test-key",
        }
        with patch.dict(os.environ, env, clear=True):
            site = SiteData()
        self.assertEqual(site.url, "https://example.supabase.co")
        self.assertEqual(site.key, "anon-test-key")

    def test_rls_fallback_reads_only_verified_users_readable_news(self):
        class RlsBlockedSite(SiteData):
            def __init__(self):
                super().__init__(access_token="verified-jwt", api_key="anon-key")
                self.url = "https://example.supabase.co"
                self.service_role_key = "server-only-key"
                self.fallback_calls = []

            def _get(self, table, params, timeout=6):
                # Simulate RLS returning an empty result for an authenticated user.
                return []

            def _get_verified_service_rows(self, table, params, user_id, timeout=6):
                self.fallback_calls.append((table, params, user_id))
                if table == "users":
                    return [{"read_dori_news": 2}]
                if table == "rockey_news":
                    self.assert_bounded = params.get("news_number") == "lte.2"
                    return [
                        {"news_number": 2, "rockey_news": "인공지능과 돌이 AI"},
                        {"news_number": 1, "rockey_news": "체스 전략"},
                        {"news_number": 3, "rockey_news": "비공개 최신 신문"},
                    ]
                return []

        site = RlsBlockedSite()
        row = site.news(2, user_id="verified-user")
        self.assertEqual(row["news_number"], 2)
        self.assertTrue(site.assert_bounded)
        self.assertTrue(all(call[2] == "verified-user" for call in site.fallback_calls))
        self.assertEqual(site.news(3, user_id="verified-user")["denied"], True)

    def test_search_fallback_never_returns_news_above_read_limit(self):
        class RlsBlockedSite(FakeSiteData):
            def __init__(self):
                super().__init__()
                self.service_role_key = "server-only-key"
                self.fallback_params = None

            def _get(self, table, params, timeout=6):
                return []

            def _get_verified_service_rows(self, table, params, user_id, timeout=6):
                self.fallback_params = params
                return [
                    {"news_number": 2, "rockey_news": "인공지능 돌이 AI"},
                    {"news_number": 3, "rockey_news": "인공지능 비공개 기사"},
                ]

        site = RlsBlockedSite()
        rows = site.search_news("인공지능", user_id="verified-user")
        self.assertEqual(site.fallback_params["news_number"], "lte.2")
        self.assertEqual([row["news_number"] for row in rows], [2])


if __name__ == "__main__":
    unittest.main()
