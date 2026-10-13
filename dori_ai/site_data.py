"""Read-only, live bridge to Dori site's Supabase data."""
import json
import os
import re
import urllib.parse
import urllib.request


class SiteData:
    def __init__(self, access_token=None, api_key=None):
        # Support both current Render names and the DORI_* names used by the
        # original local deployment. This prevents an empty config from being
        # silently interpreted as "there is no newspaper".
        self.url = (
            os.getenv("SUPABASE_URL")
            or os.getenv("DORI_SUPABASE_URL")
            or ""
        ).rstrip("/")
        self.key = (
            api_key
            or os.getenv("SUPABASE_ANON_KEY")
            or os.getenv("DORI_SUPABASE_ANON_KEY")
            or os.getenv("DORI_SUPABASE_KEY")
            or ""
        ).strip()
        self.service_role_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("DORI_SUPABASE_SERVICE_ROLE_KEY")
            or ""
        ).strip()
        self.access_token = access_token

    def with_token(self, token):
        return SiteData(token or self.access_token, self.key)

    def _request_rows(self, table, params, key, bearer, timeout):
        if not self.url or not key:
            return None
        q = urllib.parse.urlencode(params, doseq=True)
        req = urllib.request.Request(
            f"{self.url}/rest/v1/{table}?{q}",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {bearer or key}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, list) else None
        except Exception as exc:
            # Do not print tokens or API keys. The status/type is enough to
            # distinguish auth/config/RLS problems from an empty query.
            print(
                f"Dori AI Supabase read failed table={table} "
                f"error={type(exc).__name__}",
                flush=True,
            )
            return None

    def _get(self, table, params, timeout=6):
        if not self.key:
            return None
        return self._request_rows(
            table, params, self.key, self.access_token or self.key, timeout
        )

    def _get_verified_service_rows(self, table, params, user_id, timeout=6):
        """Server-only RLS fallback for a user whose JWT was verified by dori_server.

        Callers must provide the UID returned by Supabase Auth validation.
        Queries remain scoped to that UID or to the user's readable news limit.
        The service-role key is never sent to the browser.
        """
        if not user_id or not self.service_role_key or not self.url:
            return None
        return self._request_rows(
            table, params, self.service_role_key, self.service_role_key, timeout
        )

    def is_admin(self, user_id=None):
        if not user_id:
            return False
        params = {
            "select": "is_admin,user_level",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        }
        rows = self._get("users", params) or []
        if not rows:
            rows = self._get_verified_service_rows("users", params, user_id) or []
        try:
            return bool(rows and rows[0].get("is_admin")) and int(
                rows[0].get("user_level") or 0
            ) >= 10
        except (TypeError, ValueError):
            return False

    def readable_news_limit(self, user_id=None):
        if not user_id:
            return 0
        params = {
            "select": "read_dori_news",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        }
        rows = self._get("users", params) or []
        if not rows:
            rows = self._get_verified_service_rows("users", params, user_id) or []
        try:
            return max(0, int(rows[0].get("read_dori_news") or 0)) if rows else 0
        except (TypeError, ValueError):
            return 0

    def public_news_all(self):
        return self._get(
            "rockey_news",
            {
                "select": "news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5",
                "order": "news_number.asc",
            },
            timeout=10,
        ) or []

    def public_news(self, number):
        try:
            n = int(number)
        except (TypeError, ValueError):
            return None
        rows = self._get(
            "rockey_news",
            {
                "select": "news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5",
                "news_number": f"eq.{n}",
                "limit": "1",
            },
        ) or []
        return rows[0] if rows else None

    def search_news(self, query, user_id=None, limit=5):
        """Search only articles up to the verified user's readable news limit."""
        readable_limit = self.readable_news_limit(user_id)
        if readable_limit <= 0:
            return []
        params = {
            "select": "news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5",
            "news_number": f"lte.{readable_limit}",
            "order": "news_number.desc",
            "limit": "250",
        }
        rows = self._get("rockey_news", params, timeout=8) or []
        if not rows:
            # RLS may permit Auth but deny direct table reads. This fallback is
            # still constrained in PostgREST to news_number <= readable_limit.
            rows = self._get_verified_service_rows(
                "rockey_news", params, user_id, timeout=8
            ) or []

        q = str(query or "").casefold()
        terms = [x.casefold() for x in re.findall(r"[가-힣A-Za-z0-9]{2,}", q)]
        stop = {
            "돌이신문", "신문에서", "신문", "기사", "관련", "검색",
            "찾아줘", "찾아", "알려줘", "내용", "있어", "무엇", "어떤", "제목",
        }
        terms = [term for term in terms if term not in stop]
        scored = []
        for row in rows:
            try:
                number = int(row.get("news_number") or 0)
            except (TypeError, ValueError):
                continue
            if number < 1 or number > readable_limit:
                continue
            body = " ".join(
                str(row.get(k) or "")
                for k in (
                    "rockey_news", "question", "choice1", "choice2",
                    "choice3", "choice4", "choice5",
                )
            ).casefold()
            score = sum(min(4, body.count(term)) for term in terms)
            if len(q.strip()) >= 2 and q.strip() in body:
                score += 8
            if score:
                scored.append((score, number, row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            row for _, _, row in scored[:max(1, min(10, int(limit)))]
        ]

    def latest_news(self, user_id=None):
        """Return the newest existing article the verified user is allowed to read.

        read_dori_news is an access ceiling, not proof that an article with that
        exact number exists. For example, a limit of 89 with 77 stored issues
        must return issue 77 rather than report that the newspaper is missing.
        """
        readable_limit = self.readable_news_limit(user_id)
        if readable_limit <= 0:
            return {"denied": True, "max_readable": readable_limit}
        params = {
            "select": "news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5,answer",
            "news_number": f"lte.{readable_limit}",
            "order": "news_number.desc",
            "limit": "1",
        }
        rows = self._get("rockey_news", params, timeout=8) or []
        if not rows:
            rows = self._get_verified_service_rows(
                "rockey_news", params, user_id, timeout=8
            ) or []
        # Defense in depth: never return an issue above the user's access limit.
        allowed = []
        for row in rows:
            try:
                number = int(row.get("news_number") or 0)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= readable_limit:
                allowed.append(row)
        if not allowed:
            return None
        return max(allowed, key=lambda row: int(row.get("news_number") or 0))

    def news(self, number, user_id=None):
        try:
            n = int(number)
        except (TypeError, ValueError):
            return None
        readable_limit = self.readable_news_limit(user_id)
        if n < 1 or n > readable_limit:
            return {"denied": True, "number": n, "max_readable": readable_limit}
        params = {
            "select": "news_number,rockey_news,question,question_type,choice1,choice2,choice3,choice4,choice5,answer",
            "news_number": f"eq.{n}",
            "limit": "1",
        }
        rows = self._get("rockey_news", params) or []
        if not rows:
            rows = self._get_verified_service_rows(
                "rockey_news", params, user_id
            ) or []
        return rows[0] if rows else None

    def stock(self, query=None):
        rows = self._get(
            "dori_stocks",
            {
                "select": "id,ticker,name,description,risk_label,current_price,previous_price,change_amount,change_percent,is_active,characteristics,available_shares",
                "is_active": "eq.true",
                "order": "id.asc",
            },
            timeout=8,
        ) or []
        if not query:
            return rows
        q = str(query).lower()
        return [
            row for row in rows
            if q in str(row.get("name", "")).lower()
            or q in str(row.get("ticker", "")).lower()
        ]

    def profile(self, user_id=None):
        if not user_id:
            return None
        params = {
            "select": "name,user_level,exp_level,doldolcoin,exp,equipped_title,read_dori_news,quiz_right,quiz_news_number,quiz_answered",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        }
        rows = self._get("users", params) or []
        if not rows:
            rows = self._get_verified_service_rows("users", params, user_id) or []
        return rows[0] if rows else None

    def site_summary(self):
        # Fetch only article numbers for counts; do not download every article body.
        news = self._get(
            "rockey_news",
            {"select": "news_number", "order": "news_number.asc", "limit": "1000"},
            timeout=8,
        ) or []
        stocks = self.stock()
        return {
            "news_count": len(news),
            "latest_news": max(
                [int(row.get("news_number") or 0) for row in news], default=0
            ),
            "stock_count": len(stocks),
            "stocks": stocks,
        }
