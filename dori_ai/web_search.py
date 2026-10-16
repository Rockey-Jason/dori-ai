"""Optional lightweight web search adapter for Dori AI.

This module uses only Python's standard library. It returns search-result
metadata; it does not call an external AI model or generate answers.
If web search is unavailable, callers can safely fall back to local knowledge
or the from-scratch Transformer.
"""
import html
import re
import urllib.parse
import urllib.request
import threading
import time

_USER_AGENT = "DoriAI/3.0 (+https://github.com/Rockey-Jason/dori-ai)"

# Public HTML search has no paid API, but repeated identical lookups waste free-tier
# network allowance and add latency. Cache successful responses in RAM for 2 hours
# and empty/error responses briefly to avoid request storms.
_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 128
_CACHE_TTL = 2 * 60 * 60
_EMPTY_CACHE_TTL = 60


def search(query, limit=5, timeout=4):
    query = re.sub(r"\\s+", " ", str(query)).strip()
    if not query:
        return []
    key = query.casefold()
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached:
            ttl = _CACHE_TTL if cached.get("results") else _EMPTY_CACHE_TTL
            if now - cached.get("saved_at", 0) < ttl:
                return list(cached.get("results", []))[:max(0, min(10, int(limit)))]
            _CACHE.pop(key, None)

    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept-Language": "ko,en;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            body = resp.read(1_000_001).decode("utf-8", "ignore")
        if len(body) > 1_000_000:
            body = body[:1_000_000]
    except Exception:
        with _CACHE_LOCK:
            _CACHE[key] = {"saved_at": now, "results": []}
        return []

    results = []
    blocks = re.findall(r'<div[^>]+class="result[^"]*"[^>]*>(.*?)(?=<div[^>]+class="result|$)', body, re.I | re.S)

    for block in blocks:
        if len(results) >= int(limit):
            break
        m = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.I | re.S)
        if not m:
            continue
        href = html.unescape(m.group(1))
        title = re.sub(r"<[^>]+>", "", html.unescape(m.group(2))).strip()
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>', block, re.I | re.S)
        snippet = re.sub(r"<[^>]+>", "", html.unescape(sm.group(1))).strip() if sm else ""
        results.append({"title": title, "url": href, "snippet": snippet})

    with _CACHE_LOCK:
        _CACHE[key] = {"saved_at": now, "results": list(results)}
        if len(_CACHE) > _CACHE_MAX:
            oldest = sorted(_CACHE.items(), key=lambda item: item[1].get("saved_at", 0))
            for old_key, _ in oldest[:len(_CACHE) - _CACHE_MAX]:
                _CACHE.pop(old_key, None)
    return results[:max(0, min(10, int(limit)))]

def format_results(query, results):
    if not results:
        return None
    lines = [f'🔎 웹에서 확인한 "{query}" 관련 결과입니다.']
    for i, item in enumerate(results, 1):
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        snippet = item.get("snippet", "").strip()
        if title:
            lines.append(f"{i}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
        if url:
            lines.append(f"   {url}")
    return "\n".join(lines)
