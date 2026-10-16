#!/usr/bin/env python3
"""Compare actual checkpoint-generated responses and keep only a better model.

The held-out scorer uses the production dialogue prompt shape and fast-mode
sampling settings, then grades the generated answer with an inspectable offline
rubric. No external judge or paid AI API is used.
"""
import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from generate_final import generate
from scripts.response_quality import grade_response, read_rows, summarize_grades

DATASETS = [
    ROOT / "data" / "evaluation" / "general_qa.jsonl",
    ROOT / "data" / "evaluation" / "response_quality.jsonl",
]
CHECKPOINTS = ROOT / "checkpoints"
BASELINE = ROOT / "runtime" / "gate_baseline.npz"
BASELINE_META = Path(str(BASELINE) + ".json")
CANDIDATE = CHECKPOINTS / "best.npz"
CANDIDATE_META = Path(str(CANDIDATE) + ".json")
REPORT = ROOT / "runtime" / "heldout_gate_report.json"
MIN_QUALITY_GAIN = 0.015
CATEGORY_REGRESSION_LIMITS = {
    "reasoning": 0.025,
    "safety": 0.025,
    "instruction_following": 0.035,
    "language": 0.035,
    "factual": 0.05,
    "dori_lore": 0.05,
}


def score(checkpoint, metadata, rows, tokenizer):
    """Generate and grade every held-out answer from a single checkpoint."""
    meta = json.loads(Path(metadata).read_text(encoding="utf-8"))
    model = load_model(str(checkpoint), meta["model_config"])
    grades = []
    raw_answers = []
    for row in rows:
        # Exact dialogue prompt shape used by DialogueManager with no memory or
        # earlier turns. This tests the checkpoint's own answer, not retrieval,
        # hand-written fallback rules, or an outside model.
        prompt = (
            "<system>너는 Dori AI다. 친절하고 정확하게 답한다. 모르면 추측하지 않는다.\n"
            "<user>" + row["question"] + "\n<dori>"
        )
        suffix = int(re.sub(r"\\D", "", str(row["id"])) or "0")
        answer = str(generate(
            prompt,
            tokenizer,
            model,
            tokens=64,              # Same fast-mode token budget as the response engine.
            temperature=0.42,       # Same fast-mode sampling settings.
            top_k=12,
            top_p=0.82,
            seed=20261010 + suffix, # Paired deterministic decoding for both checkpoints.
        ) or "").strip()
        grade = grade_response(answer, row)
        grades.append(grade)
        raw_answers.append({"id": row["id"], "answer": answer[:500]})
    summary = summarize_grades(grades)
    summary["model_config"] = meta.get("model_config", {})
    summary["checkpoint_epoch"] = meta.get("epoch")
    summary["checkpoint_val_loss"] = meta.get("val_loss")
    return summary


def _case_wins(baseline_result, candidate_result):
    baseline_cases = {str(row["id"]): row for row in baseline_result.get("cases", [])}
    candidate_cases = {str(row["id"]): row for row in candidate_result.get("cases", [])}
    common = sorted(set(baseline_cases) & set(candidate_cases))
    if not common:
        return {"wins": 0, "losses": 0, "ties": 0, "compared": 0}
    wins = losses = ties = 0
    for case_id in common:
        delta = float(candidate_cases[case_id].get("quality_score", 0.0)) - float(
            baseline_cases[case_id].get("quality_score", 0.0)
        )
        if delta > 0.01:
            wins += 1
        elif delta < -0.01:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "compared": len(common)}


def should_promote(baseline_result, candidate_result):
    """Conservative gate: meaningful improvement without important regressions.

    The legacy rate-only branch keeps compatibility with older unit tests and
    older report consumers. New reports always use the multi-axis path.
    """
    if "quality_score" not in baseline_result or "quality_score" not in candidate_result:
        return float(candidate_result.get("rate", 0.0)) > float(baseline_result.get("rate", 0.0))

    quality_gain = float(candidate_result.get("quality_score", 0.0)) - float(
        baseline_result.get("quality_score", 0.0)
    )
    rubric_gain = float(candidate_result.get("rubric_score", 0.0)) - float(
        baseline_result.get("rubric_score", 0.0)
    )
    pass_gain = float(candidate_result.get("pass_rate", 0.0)) - float(
        baseline_result.get("pass_rate", 0.0)
    )
    if quality_gain < MIN_QUALITY_GAIN:
        return False
    # Prevent "better style" from masking a decline in answer correctness.
    if rubric_gain < -0.005:
        return False
    if pass_gain < -0.025:
        return False

    baseline_categories = baseline_result.get("category_scores", {})
    candidate_categories = candidate_result.get("category_scores", {})
    for category, max_drop in CATEGORY_REGRESSION_LIMITS.items():
        if category in baseline_categories and category in candidate_categories:
            delta = float(candidate_categories[category]) - float(baseline_categories[category])
            if delta < -max_drop:
                return False

    comparison = _case_wins(baseline_result, candidate_result)
    if comparison["compared"]:
        # Require more meaningful per-question wins than losses so a large
        # outlier improvement cannot conceal broad answer regressions.
        if comparison["wins"] <= comparison["losses"]:
            return False
        if comparison["wins"] < max(2, math.ceil(comparison["compared"] * 0.05)):
            return False
    else:
        # A production report must contain case-level paired results.
        return False
    return True


def main():
    for path in (*DATASETS, BASELINE, BASELINE_META, CANDIDATE, CANDIDATE_META, ROOT / "data" / "tokenizer.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation file is missing: {path.relative_to(ROOT)}")

    rows = read_rows(DATASETS)
    if len(rows) < 40:
        raise ValueError(f"Expected at least 40 held-out cases, got {len(rows)}")
    tokenizer = BPETokenizer.load(ROOT / "data" / "tokenizer.json")

    baseline_result = score(BASELINE, BASELINE_META, rows, tokenizer)
    candidate_result = score(CANDIDATE, CANDIDATE_META, rows, tokenizer)
    comparison = _case_wins(baseline_result, candidate_result)
    promoted = should_promote(baseline_result, candidate_result)

    if not promoted:
        # The current 'best' path may contain a temporary trained candidate.
        # Restore both weights and metadata together before commit/deploy.
        shutil.copy2(BASELINE, CANDIDATE)
        shutil.copy2(BASELINE_META, CANDIDATE_META)

    report = {
        "dataset_files": [str(path.relative_to(ROOT)) for path in DATASETS],
        "evaluation_cases": len(rows),
        "metric": "rubric_quality_with_language_integrity",
        "generation": {
            "prompt": "production DialogueManager system/user/dori prompt shape; empty history",
            "mode": "fast",
            "tokens": 64,
            "temperature": 0.42,
            "top_k": 12,
            "top_p": 0.82,
            "deterministic_paired_seed": "20261010 + numeric suffix of evaluation id",
        },
        "promotion_policy": {
            "minimum_quality_gain": MIN_QUALITY_GAIN,
            "minimum_rubric_gain": -0.005,
            "minimum_pass_rate_gain": -0.025,
            "category_maximum_regressions": CATEGORY_REGRESSION_LIMITS,
            "paired_case_wins_must_exceed_losses": True,
            "minimum_case_wins": max(2, math.ceil(len(rows) * 0.05)),
        },
        "promoted_candidate": promoted,
        "selected_checkpoint": "candidate" if promoted else "baseline",
        "paired_case_comparison": comparison,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "limitations": [
            "The rubric is deterministic and inspectable; it cannot fully judge open-ended meaning or grammar.",
            "Language integrity is a heuristic, not a full grammar or fluency evaluation.",
            "Passing this gate is evidence of improvement on this held-out suite, not proof of general intelligence.",
            "No paid AI API or external model judge is used.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HELDOUT_CASES=", len(rows))
    print("BASELINE_QUALITY=", f'{baseline_result["quality_score"]:.2%}')
    print("CANDIDATE_QUALITY=", f'{candidate_result["quality_score"]:.2%}')
    print("QUALITY_GAIN=", f'{candidate_result["quality_score"] - baseline_result["quality_score"]:+.2%}')
    print("BASELINE_RUBRIC=", f'{baseline_result["rubric_score"]:.2%}')
    print("CANDIDATE_RUBRIC=", f'{candidate_result["rubric_score"]:.2%}')
    print("BASELINE_LANGUAGE_INTEGRITY=", f'{baseline_result["language_score"]:.2%}')
    print("CANDIDATE_LANGUAGE_INTEGRITY=", f'{candidate_result["language_score"]:.2%}')
    print("PAIRED_CASE_COMPARISON=", json.dumps(comparison, ensure_ascii=False))
    print("PROMOTED_CANDIDATE=", promoted)
    print("SELECTED_CHECKPOINT=", report["selected_checkpoint"])
    print("HELDOUT_REPORT=", REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
