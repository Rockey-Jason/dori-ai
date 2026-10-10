#!/usr/bin/env python3
"""Compare the trained candidate with the pre-training checkpoint on held-out QA.

This is a conservative promotion gate: the candidate must improve held-out
keyword coverage strictly. If it does not, restore the baseline checkpoint.
Keyword coverage is a screening metric, not proof of general intelligence.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from generate_final import generate

DATASET = ROOT / "data" / "evaluation" / "general_qa.jsonl"
CHECKPOINTS = ROOT / "checkpoints"
BASELINE = ROOT / "runtime" / "gate_baseline.npz"
BASELINE_META = Path(str(BASELINE) + ".json")
CANDIDATE = CHECKPOINTS / "best.npz"
CANDIDATE_META = Path(str(CANDIDATE) + ".json")
REPORT = ROOT / "runtime" / "heldout_gate_report.json"


def normalize_answer(text):
    text = str(text or "").casefold().translate(str.maketrans({
        "²": "^2", "³": "^3", "₀": "0", "₁": "1", "₂": "2",
        "₃": "3", "₄": "4", "₅": "5", "π": "pi",
    }))
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def score(checkpoint, metadata, rows, tokenizer):
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    model = load_model(str(checkpoint), meta["model_config"])
    answers = []
    passed = 0
    for row in rows:
        # Evaluate the checkpoint's own generation directly. Do not let fixed
        # rules, site retrieval, or search make a weak checkpoint look better.
        prompt = (
            "<system>너는 Dori AI다. 친절하고 정확하게 답한다. "
            "모르면 추측하지 않는다. 질문에 직접 답한다.\n"
            "<user>" + row["question"] + "\n<dori>"
        )
        suffix = int(re.sub(r"\D", "", row["id"]) or "0")
        answer = str(generate(
            prompt, tokenizer, model,
            tokens=32, temperature=0.25, top_k=12, top_p=0.82,
            seed=20261010 + suffix,
        ) or "").strip()
        normalized_answer = normalize_answer(answer)
        keys = row["keywords"]
        matched = [
            key for key in keys
            if normalize_answer(key) and normalize_answer(key) in normalized_answer
        ]
        required = max(1, (len(keys) + 1) // 2)
        ok = bool(answer) and len(matched) >= required
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



def read_dataset(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid evaluation JSON on line {line_no}") from exc
        if not isinstance(row, dict) or not row.get("question") or not isinstance(row.get("keywords"), list) or not row["keywords"]:
            raise ValueError(f"Invalid evaluation case on line {line_no}")
        rows.append(row)
    if not rows:
        raise ValueError("Held-out evaluation dataset is empty")
    return rows


def should_promote(baseline_result, candidate_result):
    """Require strict held-out improvement; ties and regressions keep baseline."""
    return candidate_result["rate"] > baseline_result["rate"]


def main():
    for path in (DATASET, BASELINE, BASELINE_META, CANDIDATE, CANDIDATE_META, ROOT / "data" / "tokenizer.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation file is missing: {path.relative_to(ROOT)}")

    rows = read_dataset(DATASET)
    tokenizer = BPETokenizer.load(ROOT / "data" / "tokenizer.json")

    baseline_result = score(BASELINE, BASELINE_META, rows, tokenizer)
    candidate_result = score(CANDIDATE, CANDIDATE_META, rows, tokenizer)
    promoted = should_promote(baseline_result, candidate_result)

    if not promoted:
        shutil.copy2(BASELINE, CANDIDATE)
        shutil.copy2(BASELINE_META, CANDIDATE_META)

    report = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "evaluation_cases": len(rows),
        "metric": "held_out_keyword_coverage",
        "promotion_rule": "candidate_rate_must_be_strictly_greater_than_baseline_rate",
        "promoted_candidate": promoted,
        "selected_checkpoint": "candidate" if promoted else "baseline",
        "baseline": baseline_result,
        "candidate": candidate_result,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("HELDOUT_BASELINE=", baseline_result["passed"], "/", baseline_result["total"], f'({baseline_result["rate"]:.1%})')
    print("HELDOUT_CANDIDATE=", candidate_result["passed"], "/", candidate_result["total"], f'({candidate_result["rate"]:.1%})')
    print("PROMOTED_CANDIDATE=", promoted)
    print("SELECTED_CHECKPOINT=", report["selected_checkpoint"])
    print("HELDOUT_REPORT=", REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
