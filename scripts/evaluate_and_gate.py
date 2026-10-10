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
            "모르면 추측하지 않는다. 질문에 직접 답한다.\\n"
            "<user>" + row["question"] + "<dori>"
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
