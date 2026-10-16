#!/usr/bin/env python3
"""Collect a bounded, diverse Wikipedia corpus for Dori AI training.

Wikipedia text is CC BY-SA; each stored record preserves title, source URL,
language and retrieval timestamp. The collector is resumable and deduplicates
by (language, page title), so scheduled training does not repeatedly download
the same pages.
"""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "corpus" / "world_wikipedia.jsonl"
STATE = ROOT / "data" / "world_knowledge" / "collection_state.json"
USER_AGENT = "DoriAI-WorldKnowledgeCollector/1.0 (https://github.com/Rockey-Jason/dori-ai)"

KO_TOPICS = """
수학 대수학 기하학 미적분 확률 통계 정수론 논리학 컴퓨터 과학 알고리즘 자료구조 프로그래밍 언어 인공지능 기계학습 신경망 데이터베이스 인터넷 사이버보안 운영체제 로봇공학 정보이론 물리학 고전역학 양자역학 상대성이론 열역학 전자기학 광학 음향학 화학 유기화학 무기화학 주기율표 화학결합 생물학 세포 유전학 진화 생태학 해부학 신경과학 미생물학 식물학 동물학 면역학 의학 역사 고대사 중세 유럽사 한국사 조선 왕조 세계사 산업혁명 프랑스 혁명 로마 제국 고대 그리스 철학 윤리학 인식론 정치철학 경제학 미시경제학 거시경제학 금융 통화 인플레이션 무역 경영학 회계학 법학 헌법 국제법 민주주의 국제관계 지리학 지질학 기후학 해양학 기상학 환경과학 생물다양성 재생에너지 원자력 태양계 천문학 별 은하 블랙홀 우주론 우주탐사 지구 대륙 국가 수도 인구 도시 건축 미술사 회화 조각 음악 음악이론 클래식 음악 재즈 영화 문학 소설 시 희곡 신화 종교학 언어학 한국어 영어 문법 번역 교육학 심리학 사회학 인류학 고고학 미디어 철학 스포츠 축구 농구 야구 체스 올림픽 게임이론 디자인 타이포그래피 사진 요리 농업 식품과학 영양학 교통 철도 항공 우주공학 토목공학 재료과학 나노기술 전기공학 기계공학 통신 반도체 전자공학 에너지 저장 물리화학 고생물학 공룡 화석 전염병 공중보건 응급의학 의료윤리
""".split()

EN_TOPICS = """
mathematics algebra geometry calculus probability statistics number theory logic computer science algorithms data structures programming languages artificial intelligence machine learning neural networks databases internet cybersecurity operating systems robotics information theory physics classical mechanics quantum mechanics relativity thermodynamics electromagnetism optics acoustics chemistry organic chemistry periodic table chemical bonds biology cells genetics evolution ecology anatomy neuroscience microbiology botany zoology immunology history ancient history medieval history Korean history world history industrial revolution French Revolution Roman Empire ancient Greece philosophy ethics epistemology political philosophy economics microeconomics macroeconomics finance monetary policy inflation international trade business accounting law constitutional law international law democracy international relations geography geology climatology oceanography meteorology environmental science biodiversity renewable energy nuclear power astronomy stars galaxies black holes cosmology space exploration Earth continents countries capitals demographics cities architecture art history painting sculpture music theory classical music jazz cinema literature novels poetry mythology linguistics education psychology sociology anthropology archaeology media sports football basketball baseball chess Olympics game theory design typography photography cooking agriculture food science nutrition transport railways aviation aerospace engineering civil engineering materials science nanotechnology electrical engineering mechanical engineering telecommunications semiconductors electronics energy storage paleontology dinosaurs fossils epidemiology public health medical ethics
""".split()



def _query_key(lang, query):
    normalized = " ".join(str(query or "").split()).casefold()
    return lang + ":" + normalized


def _seed_query_queue(limit_topics=0):
    """Build a broad, deterministic starter queue from hundreds of seed subjects."""
    seeds = [("ko", x) for x in KO_TOPICS] + [("en", x) for x in EN_TOPICS]
    if limit_topics:
        seeds = seeds[:max(0, int(limit_topics))]
    queue = []
    for lang, topic in seeds:
        queue.append({"language": lang, "query": topic})
        if lang == "ko":
            suffixes = ("핵심 개념", "작동 원리", "실제 응용")
            queue.extend({"language": lang, "query": f"{topic} {suffix}"} for suffix in suffixes)
            queue.append({"language": lang, "query": f"역사와 발전 {topic}"})
        else:
            suffixes = ("key concepts", "how it works", "real-world applications")
            queue.extend({"language": lang, "query": f"{topic} {suffix}"} for suffix in suffixes)
            queue.append({"language": lang, "query": f"history of {topic}"})
    return queue


def _queries_for_discovered_title(title, lang):
    """Generate fresh follow-up searches from a newly discovered encyclopedia title."""
    if lang == "ko":
        templates = (
            "{topic} 핵심 개념", "{topic} 작동 원리", "{topic} 역사와 발전",
            "{topic} 실제 응용", "{topic} 사례와 영향", "{topic} 관련 개념",
        )
    else:
        templates = (
            "{topic} key concepts", "how {topic} works", "history of {topic}",
            "{topic} real-world applications", "{topic} examples and impact",
            "concepts related to {topic}",
        )
    digest = hashlib.sha256((lang + ":" + title.casefold()).encode("utf-8")).digest()
    start = digest[0] % len(templates)
    chosen = [templates[start], templates[(start + 3) % len(templates)]]
    return [template.format(topic=title) for template in chosen]


def _append_pending(queue, queued_keys, done, lang, query, max_frontier):
    query = " ".join(str(query or "").split()).strip()
    if len(query) < 2 or len(query) > 180 or len(queue) >= max_frontier:
        return False
    key = _query_key(lang, query)
    if key in done or key in queued_keys:
        return False
    queue.append({"language": lang, "query": query})
    queued_keys.add(key)
    return True

def get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "ko,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def search_pages(topic, lang, limit):
    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "list": "search", "srsearch": topic,
        "srlimit": str(limit), "format": "json", "utf8": "1",
    }
    data = get_json(base + "?" + urllib.parse.urlencode(params))
    return data.get("query", {}).get("search", [])


def extract_pages(pageids, lang):
    if not pageids:
        return []
    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "pageids": "|".join(str(x) for x in pageids),
        "prop": "extracts", "explaintext": "1", "exchars": "3800",
        "format": "json", "utf8": "1",
    }
    data = get_json(base + "?" + urllib.parse.urlencode(params))
    pages = data.get("query", {}).get("pages", {})
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for page in pages.values():
        title = str(page.get("title", "")).strip()
        text = " ".join(str(page.get("extract", "")).split())
        if len(text) < 180 or not title:
            continue
        url = f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        rows.append({
            "text": f"Topic: {title}\nLanguage: {lang}\nEncyclopedia summary: {text}",
            "title": title,
            "language": lang,
            "source": url,
            "license": "CC BY-SA (Wikipedia; verify page history and attribution for redistribution)",
            "fetched_at": now,
        })
    return rows



def main():
    parser = argparse.ArgumentParser(description="Incrementally grow a multilingual, source-attributed world-knowledge corpus.")
    parser.add_argument("--limit-topics", type=int, default=0, help="limit starter seed subjects; 0 means all configured seeds")
    parser.add_argument("--pages-per-topic", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--max-queries-per-run", type=int, default=300, help="maximum new web searches in this run")
    parser.add_argument("--max-frontier", type=int, default=8000, help="maximum queued follow-up searches")
    args = parser.parse_args()
    max_queries = max(1, min(1500, int(args.max_queries_per_run)))
    max_frontier = max(500, min(20000, int(args.max_frontier)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}

    done = set(str(x) for x in state.get("done", []) if isinstance(x, str))
    expanded_titles = set(str(x) for x in state.get("expanded_titles", []) if isinstance(x, str))
    known_titles = set(str(x) for x in state.get("discovered_titles", []) if isinstance(x, str))
    existing = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    identity = (str(row.get("language", "ko")), str(row.get("title", "")).strip())
                    if identity[1]:
                        existing.add(identity)
                        known_titles.add(identity[0] + ":" + identity[1].casefold())
                except (ValueError, TypeError):
                    continue

    raw_pending = state.get("pending")
    if isinstance(raw_pending, list):
        queue = [
            {"language": str(row.get("language", "ko")), "query": " ".join(str(row.get("query", "")).split()).strip()}
            for row in raw_pending
            if isinstance(row, dict)
            and row.get("language") in {"ko", "en"}
            and len(" ".join(str(row.get("query", "")).split()).strip()) >= 2
        ]
    else:
        queue = _seed_query_queue(args.limit_topics)
    if not queue:
        queue = _seed_query_queue(args.limit_topics)

    queued_keys = set()
    unique_queue = []
    for row in queue:
        key = _query_key(row["language"], row["query"])
        if key not in done and key not in queued_keys and len(unique_queue) < max_frontier:
            unique_queue.append(row)
            queued_keys.add(key)
    queue = unique_queue

    # Process a fixed batch; failed searches are retried later and fresh discoveries extend the frontier.
    batch = []
    while queue and len(batch) < max_queries:
        row = queue.pop(0)
        queued_keys.discard(_query_key(row["language"], row["query"]))
        key = _query_key(row["language"], row["query"])
        if key in done:
            continue
        batch.append(row)

    added = 0
    processed = 0
    errors = 0
    for item in batch:
        lang, topic = item["language"], item["query"]
        key = _query_key(lang, topic)
        try:
            hits = search_pages(topic, lang, max(1, min(5, args.pages_per_topic)))
            ids = [x.get("pageid") for x in hits if x.get("pageid")]
            rows = extract_pages(ids, lang)
            for hit in hits:
                title = str(hit.get("title", "")).strip()
                if not title:
                    continue
                title_key = lang + ":" + title.casefold()
                if title_key not in expanded_titles:
                    expanded_titles.add(title_key)
                    known_titles.add(title_key)
                    for generated_query in _queries_for_discovered_title(title, lang):
                        _append_pending(queue, queued_keys, done, lang, generated_query, max_frontier)

            with OUT.open("a", encoding="utf-8") as handle:
                for row in rows:
                    identity = (row["language"], row["title"])
                    if identity in existing:
                        continue
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    existing.add(identity)
                    known_titles.add(row["language"] + ":" + row["title"].casefold())
                    added += 1
            done.add(key)
            processed += 1
            print(f"COLLECTED {lang}:{topic} hits={len(hits)} pages={len(rows)} new_total={added}", flush=True)
        except Exception as exc:
            errors += 1
            _append_pending(queue, queued_keys, done, lang, topic, max_frontier)
            print(f"SKIP {lang}:{topic} error={type(exc).__name__}", flush=True)
        if args.delay > 0:
            time.sleep(args.delay)

    if processed or errors or added:
        STATE.write_text(json.dumps({
            "done": sorted(done),
            "pending": queue[:max_frontier],
            "expanded_titles": sorted(expanded_titles),
            "discovered_titles": sorted(known_titles),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "new_pages": added,
            "processed_queries": processed,
            "pending_queries": len(queue),
            "errors": errors,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"WORLD_CORPUS total_pages={len(existing)} new_pages={added} processed_queries={processed} pending_queries={len(queue)} failed_queries={errors}", flush=True)
    if len(existing) == 0:
        raise SystemExit("No Wikipedia pages available; refusing to continue with an empty world corpus.")


if __name__ == "__main__":
    main()
