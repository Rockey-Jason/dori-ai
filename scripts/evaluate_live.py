#!/usr/bin/env python3
"""Evaluate actual end-to-end answers from a running Dori AI HTTP service.

This diagnostic uses the same held-out response rubric as the checkpoint
promotion gate, but measures the whole deployed answer path (routing, tools,
retrieval, and generation). It never uses a paid model judge.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.response_quality import grade_response, read_rows, summarize_grades

DEFAULT_DATASETS = [
    ROOT / "data" / "evaluation" / "general_qa.jsonl",
    ROOT / "data" / "evaluation" / "response_quality.jsonl",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://dori-ai-u3kf.onrender.com")
    parser.add_argument("--dataset", action="append", help="JSONL dataset path; may be supplied more than once")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all rows")
    parser.add_argument("--report", default="runtime/live_response_quality_report.json")
    parser.add_argument("--min-quality-score", type=float, default=None,
                        help="Optional quality score threshold (0-1); fails if the score is below it")
    args = parser.parse_args()

    datasets = [Path(x) for x in args.dataset] if args.dataset else DEFAULT_DATASETS
    rows = read_rows(datasets)
    if args.limit:
        rows = rows[:max(1, args.limit)]

    grades = []
    latencies = []
    errors = []
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
        grades.append(grade)
        print(json.dumps({
            "id": row["id"],
            "category": row["category"],
            "quality_score": grade["quality_score"],
            "rubric_score": grade["rubric_score"],
            "language_score": grade["language_score"],
            "format_score": grade["format_score"],
            "passed": grade["passed"],
            "latency_seconds": latency,
            "request_error": error,
            "answer_preview": answer[:300],
        }, ensure_ascii=False), flush=True)

    summary = summarize_grades(grades)
    summary["mean_latency_seconds"] = round(sum(latencies) / len(latencies), 2) if latencies else None
    summary["request_errors"] = errors
    report = {
        "base_url": args.base_url,
        "dataset_files": [str(Path(p)) for p in datasets],
        "evaluation_kind": "end_to_end_live_service",
        "summary": summary,
        "limitations": [
            "A live-service score includes routing, retrieval, and tools in addition to the model.",
            "Live-service results do not replace the paired checkpoint gate for promoting model weights.",
            "The deterministic rubric is not a perfect semantic judge or a complete grammar evaluator.",
        ],
    }
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "summary": {
            "total": summary["total"],
            "passed": summary["passed"],
            "pass_rate": summary["pass_rate"],
            "quality_score": summary["quality_score"],
            "rubric_score": summary["rubric_score"],
            "language_score": summary["language_score"],
            "format_score": summary["format_score"],
            "mean_latency_seconds": summary["mean_latency_seconds"],
            "request_errors": len(errors),
            "category_scores": summary["category_scores"],
        },
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))

    if errors:
        raise SystemExit(2)
    if args.min_quality_score is not None and summary["quality_score"] < args.min_quality_score:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
