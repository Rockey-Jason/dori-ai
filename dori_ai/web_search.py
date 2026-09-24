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

_USER_AGENT = "DoriAI/3.0 (+https://github.com/Rockey-Jason/dori-ai)"

def search(query, limit=5, timeout=4):
    query = str(query).strip()
    if not query:
        return []
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept-Language": "ko,en;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            body = resp.read().decode("utf-8", "ignore")
    except Exception:
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
    return results

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
