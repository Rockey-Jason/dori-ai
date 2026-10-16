#!/usr/bin/env python3
"""Safely evaluate end-to-end responses from the running Dori AI HTTP service.

By default this uses a stratified sample of eight questions to reduce memory
pressure on the free 512 MiB Render instance. It stops at the first request
failure unless --continue-on-errors is explicitly set. Use the offline
checkpoint evaluator for the complete held-out suite.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.response_quality import grade_response, read_rows, summarize_grades

DEFAULT_DATASETS = [
    ROOT / "data" / "evaluation" / "general_qa.jsonl",
    ROOT / "data" / "evaluation" / "response_quality.jsonl",
]
SAMPLE_CATEGORY_ORDER = [
    "reasoning",
    "language",
    "instruction_following",
    "safety",
    "dori_lore",
    "site_access",
    "math",
    "science",
    "factual",
    "security",
    "computer_science",
    "evaluation",
]


def request_answer(base_url, question, timeout):
    payload = json.dumps({"text": question, "stream": False}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    answer = data.get("answer")
    if answer is None:
        raise ValueError("Response JSON did not contain an 'answer' field")
    return str(answer).strip()


def select_stratified_rows(rows, limit):
    """Select distinct categories first, prioritizing the richer response rubric."""
    if limit <= 0 or limit >= len(rows):
        return list(rows)

    def source_priority(row):
        return 0 if "response_quality.jsonl" in str(row.get("_source_file", "")) else 1

    selected, selected_ids, selected_categories = [], set(), set()
    for category in SAMPLE_CATEGORY_ORDER:
        candidates = [row for row in rows if row.get("category") == category and row.get("id") not in selected_ids]
        candidates.sort(key=source_priority)
        if not candidates:
            continue
        chosen = candidates[0]
        selected.append(chosen)
        selected_ids.add(chosen["id"])
        selected_categories.add(category)
        if len(selected) >= limit:
            return selected

    # If there are fewer unique categories than the requested sample size,
    # fill from the held-out set in its stable original order.
    for row in rows:
        if row.get("id") in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row["id"])
        if len(selected) >= limit:
            break
    return selected


def empty_summary():
    return {
        "total": 0,
        "passed": 0,
        "pass_rate": None,
        "quality_score": None,
        "rubric_score": None,
        "language_score": None,
        "format_score": None,
        "category_scores": {},
        "category_rubric_scores": {},
        "category_language_scores": {},
        "category_pass_rates": {},
        "cases": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://dori-ai-u3kf.onrender.com")
    parser.add_argument("--dataset", action="append", help="JSONL dataset path; may be supplied more than once")
    parser.add_argument("--timeout", type=float, default=35,
                        help="Per-request timeout; allow a free service time to wake from idle")
    parser.add_argument("--limit", type=int, default=8,
                        help="Maximum number of live questions (default 8; 0 explicitly requests the full suite)")
    parser.add_argument("--continue-on-errors", action="store_true",
                        help="Continue after a request failure; not recommended for memory-limited instances")
    parser.add_argument("--report", default="runtime/live_response_quality_report.json")
    parser.add_argument("--min-quality-score", type=float, default=None,
                        help="Optional quality threshold (0-1); compared only if enough responses succeeded")
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.min_quality_score is not None and not 0 <= args.min_quality_score <= 1:
        parser.error("--min-quality-score must be between 0 and 1")

    datasets = [Path(x) for x in args.dataset] if args.dataset else DEFAULT_DATASETS
    rows = read_rows(datasets)
    rows = select_stratified_rows(rows, args.limit)

    attempts, successful_grades, latencies, errors = [], [], [], []
    for index, row in enumerate(rows, 1):
        started = time.monotonic()
        try:
            answer = request_answer(args.base_url, row["question"], args.timeout)
            error = None
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
            answer = ""
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"id": row["id"], "error": error})

        latency = round(time.monotonic() - started, 2)
        latencies.append(latency)
        grade = grade_response(answer, row)
        grade["latency_seconds"] = latency
        if error:
            grade["request_error"] = error
        attempts.append(grade)
        if not error:
            successful_grades.append(grade)

        print(json.dumps({
            "id": row["id"],
            "category": row["category"],
            "quality_score": grade["quality_score"] if not error else None,
            "rubric_score": grade["rubric_score"] if not error else None,
            "language_score": grade["language_score"] if not error else None,
            "format_score": grade["format_score"] if not error else None,
            "passed": grade["passed"] if not error else None,
            "latency_seconds": latency,
            "request_error": error,
            "answer_preview": answer[:300] if not error else "",
        }, ensure_ascii=False), flush=True)

        # A 502/timeout can signal that a low-memory instance is failing.
        # Stop instead of flooding it with follow-up requests.
        if error and not args.continue_on_errors:
            print("STOPPING_AFTER_FIRST_REQUEST_ERROR= true", flush=True)
            break

    summary = summarize_grades(successful_grades) if successful_grades else empty_summary()
    summary["planned_questions"] = len(rows)
    summary["attempted_requests"] = len(attempts)
    summary["responses_received"] = len(successful_grades)
    summary["failed_requests"] = len(errors)
    summary["response_coverage"] = round(len(successful_grades) / len(rows), 4) if rows else 0.0
    summary["complete"] = len(successful_grades) == len(rows) and not errors
    summary["mean_latency_seconds"] = (
        round(sum(g["latency_seconds"] for g in successful_grades) / len(successful_grades), 2)
        if successful_grades else None
    )

    report = {
        "base_url": args.base_url,
        "dataset_files": [str(Path(p)) for p in datasets],
        "evaluation_kind": "end_to_end_live_service",
        "sampling": {
            "requested_limit": args.limit,
            "actual_questions": len(rows),
            "selection": "stratified category sample" if args.limit and args.limit < len(read_rows(datasets)) else "full held-out suite",
            "stop_on_first_error": not args.continue_on_errors,
        },
        "summary": summary,
        "attempts": attempts,
        "request_errors": errors,
        "limitations": [
            "A live-service score includes routing, retrieval, and tools in addition to the model.",
            "A report with request failures is incomplete and must not be treated as a valid full-suite score.",
            "The low-memory Render instance is evaluated with a small sample by default; run full comparison offline.",
            "The deterministic rubric is not a perfect semantic judge or a complete grammar evaluator.",
        ],
    }
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    printable = {
        "planned_questions": summary["planned_questions"],
        "attempted_requests": summary["attempted_requests"],
        "responses_received": summary["responses_received"],
        "failed_requests": summary["failed_requests"],
        "response_coverage": summary["response_coverage"],
        "complete": summary["complete"],
        "passed": summary["passed"],
        "pass_rate": summary["pass_rate"],
        "quality_score": summary["quality_score"],
        "rubric_score": summary["rubric_score"],
        "language_score": summary["language_score"],
        "format_score": summary["format_score"],
        "mean_latency_seconds": summary["mean_latency_seconds"],
        "category_scores": summary["category_scores"],
        "report": str(report_path),
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    # Any request failure makes the overall live run incomplete. Quality scores
    # still describe successful responses only, never the failed empty bodies.
    if errors:
        raise SystemExit(2)
    if args.min_quality_score is not None and summary["quality_score"] < args.min_quality_score:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
