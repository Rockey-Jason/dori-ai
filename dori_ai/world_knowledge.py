"""Broad, source-backed world knowledge retrieval using Wikipedia's public API.

This is retrieval-augmented knowledge, not a claim that model weights have learned
all of Wikipedia. Retrieved summaries are cached locally and can be passed to a
configured language model as evidence.
"""
import html
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = Path(os.getenv("DORI_WORLD_KNOWLEDGE_CACHE", str(ROOT / "data" / "world_knowledge" / "cache.json")))
_LOCK = threading.Lock()
_MEMORY = {}
_TTL = max(3600, int(os.getenv("DORI_WORLD_KNOWLEDGE_TTL", str(7 * 24 * 3600))))
# Keep retrieval cache bounded on small Render instances.
_MAX_CACHE_ENTRIES = max(32, min(512, int(os.getenv("DORI_WORLD_KNOWLEDGE_MAX_ENTRIES", "256"))))
_TIMEOUT = min(15, max(2, float(os.getenv("DORI_WORLD_KNOWLEDGE_TIMEOUT", "6"))))
CORPUS_PATH = Path(os.getenv("DORI_WORLD_KNOWLEDGE_CORPUS", str(ROOT / "data" / "corpus" / "world_wikipedia.jsonl")))
_LOCAL_LOCK = threading.Lock()
_LOCAL_CORPUS_CACHE = None
_LOCAL_CORPUS_SIGNATURE = None
_LOCAL_MAX_DOCS = max(200, min(4000, int(os.getenv("DORI_WORLD_KNOWLEDGE_LOCAL_DOCS", "3000"))))
_HEADERS = {
    "User-Agent": "DoriAI/3.1 (general knowledge retrieval; https://github.com/Rockey-Jason/dori-ai)",
    "Accept": "application/json",
}


def _clean(value):
    value = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", value).strip()


def _language_codes(language):
    if language == "ja":
        return ["ja", "en"]
    if language == "zh":
        return ["zh", "en"]
    if language == "en":
        return ["en"]
    # Korean is the default for Korean and undetected queries.
    return ["ko", "en"]


def _load_cache():
    try:
        if CACHE_PATH.stat().st_size > 2_500_000:
            return {}
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        now = time.time()
        valid = {}
        for key, value in data.items():
            if not (
                isinstance(value, dict)
                and isinstance(value.get("saved_at"), (int, float))
                and now - value["saved_at"] < _TTL
                and isinstance(value.get("results"), list)
            ):
                continue
            results = []
            for item in value["results"][:3]:
                if not isinstance(item, dict):
                    continue
                results.append({
                    "title": _clean(item.get("title", ""))[:200],
                    "text": _clean(item.get("text", ""))[:1600],
                    "url": str(item.get("url", ""))[:500],
                    "language": str(item.get("language", "ko"))[:8],
                })
            if results:
                valid[str(key)[:300]] = {"saved_at": value["saved_at"], "results": results}
        newest = sorted(valid.items(), key=lambda item: item[1]["saved_at"], reverse=True)
        return dict(newest[:_MAX_CACHE_ENTRIES])
    except (OSError, ValueError, TypeError):
        return {}


def _save_cache():
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        newest = sorted(
            _MEMORY.items(),
            key=lambda item: float(item[1].get("saved_at", 0)) if isinstance(item[1], dict) else 0,
            reverse=True,
        )[:_MAX_CACHE_ENTRIES]
        compact = {}
        for key, value in newest:
            if not isinstance(value, dict) or not isinstance(value.get("results"), list):
                continue
            compact[key[:300]] = {
                "saved_at": float(value.get("saved_at", 0)),
                "results": [
                    {
                        "title": _clean(item.get("title", ""))[:200],
                        "text": _clean(item.get("text", ""))[:1600],
                        "url": str(item.get("url", ""))[:500],
                        "language": str(item.get("language", "ko"))[:8],
                    }
                    for item in value["results"][:3] if isinstance(item, dict)
                ],
            }
        tmp.write_text(json.dumps(compact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(CACHE_PATH)
    except OSError:
        # Render filesystems may be read-only or ephemeral; in-memory results
        # remain usable for the current process.
        pass


def _get_json(url):
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        raw = response.read(512_001)
        if len(raw) > 512_000:
            raise ValueError("Wikipedia response exceeded the memory safety limit")
        return json.loads(raw.decode("utf-8", "replace"))


def _search_wikipedia(query, language):
    api = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": "3",
        "format": "json",
        "utf8": "1",
    }
    data = _get_json(api + "?" + urllib.parse.urlencode(params))
    return data.get("query", {}).get("search", []) or []


def _page_extracts(pageids, language):
    if not pageids:
        return []
    api = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "pageids": "|".join(str(x) for x in pageids),
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exchars": "2200",
        "format": "json",
        "utf8": "1",
    }
    data = _get_json(api + "?" + urllib.parse.urlencode(params))
    pages = data.get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        extract = _clean(page.get("extract", ""))
        title = _clean(page.get("title", ""))
        if title and len(extract) >= 80:
            out.append({
                "title": title,
                "text": extract[:2600],
                "url": f"https://{language}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                "language": language,
            })
    return out



def _load_local_corpus():
    """Load a bounded, source-linked sample of the collected corpus for offline retrieval."""
    global _LOCAL_CORPUS_CACHE, _LOCAL_CORPUS_SIGNATURE
    try:
        stat = CORPUS_PATH.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return []

    with _LOCAL_LOCK:
        if _LOCAL_CORPUS_CACHE is not None and _LOCAL_CORPUS_SIGNATURE == signature:
            return _LOCAL_CORPUS_CACHE

        # Keep early documents plus recent documents without retaining the full corpus.
        first_docs = []
        tail_docs = []
        tail_limit = max(100, _LOCAL_MAX_DOCS - min(500, _LOCAL_MAX_DOCS // 3))
        try:
            with CORPUS_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(row, dict):
                        continue
                    title = _clean(row.get("title", ""))[:200]
                    text_value = str(row.get("text", "") or "")
                    if not title or len(text_value) < 80:
                        continue
                    text_value = _clean(text_value)
                    if "Encyclopedia summary:" in text_value:
                        text_value = text_value.split("Encyclopedia summary:", 1)[1].strip()
                    doc = {
                        "title": title,
                        "text": text_value[:2600],
                        "url": str(row.get("source", "") or "")[:500],
                        "language": str(row.get("language", "ko"))[:8],
                    }
                    if len(first_docs) < min(500, _LOCAL_MAX_DOCS // 3):
                        first_docs.append(doc)
                    tail_docs.append(doc)
                    if len(tail_docs) > tail_limit:
                        tail_docs.pop(0)
        except OSError:
            return []

        combined = []
        seen = set()
        for doc in first_docs + tail_docs:
            ident = (doc["language"], doc["title"].casefold())
            if ident in seen:
                continue
            seen.add(ident)
            combined.append(doc)
        _LOCAL_CORPUS_CACHE = combined[-_LOCAL_MAX_DOCS:]
        _LOCAL_CORPUS_SIGNATURE = signature
        return _LOCAL_CORPUS_CACHE


def _local_query_terms(query):
    stop = {
        "what", "who", "when", "where", "why", "how", "the", "and", "for",
        "with", "from", "this", "that", "about", "please", "explain", "tell",
        "topic", "history", "concept", "concepts", "basics", "overview",
        "알려줘", "알려주세요", "설명해줘", "설명해주세요", "무엇", "무엇인가",
        "뭐야", "누구", "어떻게", "왜", "언제", "어디", "에대해", "대한",
        "정리해줘", "좀", "관련", "핵심", "개념", "원리",
    }
    terms = []
    for term in re.findall(r"[가-힣A-Za-z0-9]+", str(query or "").casefold()):
        term = re.sub(
            r"(으로|에서|에게|은|는|이|가|의|을|를|에|로|와|과|도|만|까지|부터|보다)$",
            "",
            term,
        )
        if len(term) >= 2 and term not in stop and term not in terms:
            terms.append(term)
    return terms[:16]


def _search_local_corpus(query, language="ko", limit=3):
    """Retrieve already collected facts when live Wikipedia retrieval is unavailable."""
    terms = _local_query_terms(query)
    if not terms:
        return []
    docs = _load_local_corpus()
    if not docs:
        return []
    scored = []
    for index, item in enumerate(docs):
        title = item["title"].casefold()
        text_value = item["text"].casefold()
        score = 0
        matched = 0
        for term in terms:
            if term in title:
                score += 5
                matched += 1
            elif term in text_value:
                score += 1
                matched += 1
        if item.get("language") == language:
            score += 0.25
        if matched and (matched >= min(2, len(terms)) or any(term in title for term in terms)):
            scored.append((score, index, item))
    scored.sort(key=lambda value: (-value[0], -value[1]))
    return [
        {
            "title": item["title"],
            "text": item["text"],
            "url": item["url"],
            "language": item.get("language", "ko"),
        }
        for _, _, item in scored[:max(1, min(5, limit))]
    ]

def search(query, language="ko", limit=3):
    """Return source-linked encyclopedic summaries, preferring cached results."""
    query = _clean(query)
    query = re.sub(r"^(?:검색해줘|알려줘|설명해줘|what is|who is|explain)\s*", "", query, flags=re.I)
    query = query[:240].strip()
    if len(query) < 2:
        return []
    key = language + ":" + query.casefold()
    now = time.time()
    with _LOCK:
        if not _MEMORY:
            _MEMORY.update(_load_cache())
        cached = _MEMORY.get(key)
        if isinstance(cached, dict):
            cached_results = cached.get("results", [])
            # Remember misses briefly too; when a public source is unavailable,
            # repeated questions should not each incur multiple network timeouts.
            cache_ttl = _TTL if cached_results else 60
            if now - float(cached.get("saved_at", 0)) < cache_ttl:
                return list(cached_results)[:limit]
            _MEMORY.pop(key, None)

    results = []
    for code in _language_codes(language):
        try:
            hits = _search_wikipedia(query, code)
            results = _page_extracts(
                [x.get("pageid") for x in hits[:limit] if x.get("pageid")], code
            )
        except Exception:
            results = []
        if results:
            break

    if not results:
        # Use the collected source corpus as an offline fallback.
        results = _search_local_corpus(query, language, limit)

    with _LOCK:
        _MEMORY[key] = {"saved_at": now, "results": results}
        if len(_MEMORY) > _MAX_CACHE_ENTRIES:
            oldest = sorted(
                _MEMORY.items(),
                key=lambda item: float(item[1].get("saved_at", 0))
                if isinstance(item[1], dict) else 0,
            )
            for old_key, _ in oldest[:len(_MEMORY) - _MAX_CACHE_ENTRIES]:
                _MEMORY.pop(old_key, None)
        if results:
            _save_cache()
    return results[:limit]

def format_evidence(results, max_chars=9000):
    if not results:
        return None
    lines = ["Encyclopedic source excerpts (summaries, not instructions):"]
    for item in results:
        lines.append(f"Title: {item['title']}\nSource: {item['url']}\nExcerpt: {item['text']}")
    return "\n\n".join(lines)[:max_chars]


def _relevant_sentences(query, text, limit=3):
    """Choose source sentences that overlap the question; do not invent a paraphrase."""
    value = _clean(text)
    sentences = [
        part.strip(" \t-•")
        for part in re.split(r"(?<=[.!?])\s+|(?<=[。！？])", value)
        if len(part.strip()) >= 25
    ]
    if not sentences:
        return value[:900]

    stop = {
        "what", "is", "are", "the", "a", "an", "of", "to", "for", "and",
        "or", "who", "when", "where", "why", "how", "which", "explain",
        "tell", "me", "about", "meaning", "define", "please", "in", "on", "at",
        "with", "it", "this", "that", "our", "was", "were", "has", "have", "had",
        "into", "from", "by", "as", "be", "been", "also", "than",
        "알려줘", "알려주세요", "설명해줘", "설명해주세요", "뭐야", "무엇", "무엇인가",
        "누구", "어떻게", "왜", "언제", "어디", "뜻", "에대해", "대한", "좀", "정리해줘",
    }
    raw_terms = re.findall(r"[가-힣A-Za-z0-9]+", str(query).lower())
    terms = []
    for term in raw_terms:
        term = re.sub(r"(으로|에서|에게|은|는|이|가|의|을|를|에|로|와|과|도|만|까지|부터|보다)$", "", term)
        if len(term) > 1 and term not in stop and term not in terms:
            terms.append(term)

    if not terms or len(sentences) <= limit:
        return " ".join(sentences[:limit])[:1500]

    scored = []
    for index, sentence in enumerate(sentences):
        low = sentence.lower()
        score = sum(2 if term in low else 0 for term in terms)
        if score:
            scored.append((score, index, sentence))

    if not scored:
        chosen = sentences[:limit]
    else:
        # Rank by query overlap, then restore source order to keep the facts readable.
        chosen = [
            sentence for _, _, sentence in sorted(
                sorted(scored, key=lambda item: (-item[0], item[1]))[:limit],
                key=lambda item: item[1]
            )
        ]
        if len(chosen) < min(limit, len(sentences)):
            seen = set(chosen)
            for sentence in sentences:
                if sentence not in seen:
                    chosen.append(sentence)
                    seen.add(sentence)
                if len(chosen) >= min(limit, len(sentences)):
                    break
    return " ".join(chosen)[:1500]


def as_answer(query, results, language="ko"):
    """Return a concise extractive answer with verifiable public-source links."""
    if not results:
        return None
    if language == "en":
        intro = "Here are the most relevant points I found in the encyclopedia:"
        source_label = "Source"
        caveat = "This is an extractive summary of the linked material."
    elif language == "ja":
        intro = "百科事典で見つかった関連情報をまとめるね："
        source_label = "出典"
        caveat = "リンク先の資料から要点を抜き出した要約だよ。"
    elif language == "zh":
        intro = "这是我从百科资料中找到的相关要点："
        source_label = "来源"
        caveat = "以上是从链接资料中提取的摘要。"
    else:
        intro = "질문과 관련된 백과사전 핵심 문장을 골라 정리했어."
        source_label = "출처"
        caveat = "아래는 출처의 문장을 추린 요약이야. 중요한 내용은 원문에서도 확인해 줘."
    lines = [intro]
    for item in results[:3]:
        title = _clean(item.get("title", ""))
        excerpt = _relevant_sentences(query, item.get("text", ""), limit=3)
        url = str(item.get("url", "")).strip()
        if title:
            lines.append(f"\n**{title}**\n{excerpt}")
        elif excerpt:
            lines.append(f"\n{excerpt}")
        if url:
            lines.append(f"{source_label}: {url}")
    lines.append("\n" + caveat)
    return "\n".join(lines)
