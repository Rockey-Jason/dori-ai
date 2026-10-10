#!/usr/bin/env python3
"""Compare the trained candidate with the pre-training checkpoint on held-out QA.

This is a conservative promotion gate: the candidate must improve held-out
keyword coverage strictly. If it does not, restore the baseline checkpoint.
Keyword coverage is a screening metric, not proof of general intelligence.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from dori_ai.response_engine import ResponseEngine
from dori_ai import world_knowledge

DATASET = ROOT / "data" / "evaluation" / "general_qa.jsonl"
CHECKPOINTS = ROOT / "checkpoints"
BASELINE = ROOT / "runtime" / "gate_baseline.npz"
BASELINE_META = Path(str(BASELINE) + ".json")
CANDIDATE = CHECKPOINTS / "best.npz"
CANDIDATE_META = Path(str(CANDIDATE) + ".json")
REPORT = ROOT / "runtime" / "heldout_gate_report.json"


def score(checkpoint, metadata, rows, tokenizer):
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    model = load_model(str(checkpoint), meta["model_config"])
    bot = ResponseEngine(tokenizer, model)
    # Keep the benchmark repeatable and offline; this compares checkpoint
    # behavior, not changing live search results or private user data.
    bot.web_enabled = False
    answers = []
    passed = 0
    for row in rows:
        answer = str(bot.reply(row["question"], mode="fast") or "").strip()
        keys = row["keywords"]
        matched = [key for key in keys if key.casefold() in answer.casefold()]
        ok = bool(answer) and len(matched) >= max(1, (len(keys) + 1) // 2)
        passed += int(ok)
        answers.append({
            "id": row["id"],
            "category": row.get("category", "unknown"),
            "passed": ok,
            "matched_keywords": matched,
            "expected_keywords": keys,
            "answer_preview": answer[:400],
        })
    return {
        "passed": passed,
        "total": len(rows),
        "rate": passed / len(rows) if rows else 0.0,
        "cases": answers,
    }


def main():
    if not all(p.exists() for p in (DATASET, BASELINE, BASELINE_META, CANDIDATE, CANDIDATE_META)):
        raise SystemExit("Missing held-out dataset or baseline/candidate checkpoint; refusing unverified promotion.")

    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 15:
        raise SystemExit(f"Held-out evaluation set too small: {len(rows)}")
    # Suppress network-dependent knowledge lookups so both checkpoints receive
    # identical offline inputs and the workflow does not depend on external APIs.
    world_knowledge.search = lambda *args, **kwargs: []
    tokenizer = BPETokenizer.load(ROOT / "data" / "tokenizer.json")
    baseline = score(BASELINE, BASELINE_META, rows, tokenizer)
    candidate = score(CANDIDATE, CANDIDATE_META, rows, tokenizer)

    promoted = candidate["rate"] > baseline["rate"]
    if not promoted:
        shutil.copy2(BASELINE, CANDIDATE)
        shutil.copy2(BASELINE_META, CANDIDATE_META)

    report = {
        "metric": "held_out_keyword_coverage",
        "promoted_candidate": promoted,
        "baseline": {"passed": baseline["passed"], "total": baseline["total"], "rate": baseline["rate"]},
        "candidate_before_gate": {"passed": candidate["passed"], "total": candidate["total"], "rate": candidate["rate"]},
        "selected": "candidate" if promoted else "baseline",
        "warning": "Keyword coverage is a lightweight screening metric, not a full factuality or reasoning benchmark.",
        "baseline_cases": baseline["cases"],
        "candidate_cases": candidate["cases"],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Do not commit the temporary baseline copy; the report keeps the measured result.
    BASELINE.unlink(missing_ok=True)
    BASELINE_META.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ("baseline_cases", "candidate_cases")}, ensure_ascii=False, indent=2))
    print("HELDOUT_GATE_REPORT=" + str(REPORT))
    if not promoted:
        print("Candidate did not strictly improve held-out keyword coverage; baseline checkpoint restored.")
    else:
        print("Candidate improved held-out keyword coverage and was retained.")


if __name__ == "__main__":
    main()
