#!/usr/bin/env python3
"""Run the held-out general QA set against a running Dori AI HTTP service."""
import argparse, json, time, urllib.error, urllib.request
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://dori-ai-u3kf.onrender.com")
    parser.add_argument("--dataset", default="data/evaluation/general_qa.jsonl")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all evaluation cases")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit: rows = rows[:max(1, args.limit)]
    results = []
    for index, row in enumerate(rows, 1):
        payload = json.dumps({"text": row["question"], "stream": False}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(args.base_url.rstrip("/") + "/chat", data=payload, headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response: data = json.loads(response.read().decode('utf-8'))
            answer = str(data.get("answer", "")).strip()
            matched = [keyword for keyword in row["keywords"] if keyword.casefold() in answer.casefold()]
            passed = bool(answer) and len(matched) >= max(1, (len(row["keywords"]) + 1) // 2)
            result = {"id": row["id"], "category": row["category"], "passed": passed, "matched_keywords": matched, "expected_keywords": row["keywords"], "latency_seconds": round(time.monotonic() - started, 2), "answer": answer[:500]}
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            result = {"id": row.get("id", str(index)), "category": row.get("category", "unknown"), "passed": False, "error": type(exc).__name__, "latency_seconds": round(time.monotonic() - started, 2)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    total = len(results); passed = sum(1 for row in results if row['passed'])
    summary = {"summary": {"total": total, "passed": passed, "failed": total-passed, "keyword_pass_rate": round(passed/total, 4) if total else 0.0, "mean_latency_seconds": round(sum(x["latency_seconds"] for x in results)/total, 2) if total else None}, "note": "Keyword coverage is a lightweight regression signal, not a full measure of understanding or factual accuracy."}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if total and passed / total >= 0.65 else 1)

if __name__ == "__main__": main()
