#!/usr/bin/env python3
"""Summarize Dori AI's free growth loop without changing model or data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("corpus", "new", "generated", "learning", "curriculum", "instructions")


def source_summary(root=ROOT):
    root = Path(root)
    data = root / "data"
    files = []
    for name in SOURCE_DIRS:
        path = data / name
        if path.is_file() and path.suffix.lower() in {".txt", ".json", ".jsonl"}:
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".json", ".jsonl"})
    files.extend(p for p in data.glob("dori_knowledge*.jsonl") if p.is_file())
    for name in ("tokenizer.json", "curriculum_plan.json"):
        p = data / name
        if p.is_file():
            files.append(p)
    files = sorted(set(files), key=lambda p: p.relative_to(root).as_posix())
    digest = hashlib.sha256()
    records = 0
    bytes_total = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8") + b"\\0")
        raw = path.read_bytes()
        digest.update(raw + b"\\0")
        bytes_total += len(raw)
        if path.suffix.lower() == ".jsonl":
            records += sum(1 for line in raw.splitlines() if line.strip())
    return {
        "fingerprint": digest.hexdigest(),
        "files": len(files),
        "bytes": bytes_total,
        "jsonl_records": records,
    }


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def build_report(root=ROOT):
    root = Path(root)
    state = read_json(root / "runtime" / "training_round.json") or {}
    meta = read_json(root / "checkpoints" / "best.npz.json") or {}
    gate = read_json(root / "runtime" / "heldout_gate_report.json") or {}
    source = source_summary(root)
    target = int(state.get("target_rounds_per_dataset", 1200) or 1200)
    current = max(0, int(state.get("round", 0) or 0))
    config = meta.get("model_config") or {}
    return {
        "mode": "free_event_driven_growth",
        "source": source,
        "cycle": {
            "round": current,
            "target_rounds": target,
            "remaining_rounds": max(0, target - current),
            "cumulative_round": max(0, int(state.get("cumulative_round", current) or 0)),
            "state_fingerprint_matches_current_sources": state.get("source_fingerprint") == source["fingerprint"],
        },
        "accepted_checkpoint": {
            "exists": (root / "checkpoints" / "best.npz").is_file(),
            "epoch": meta.get("epoch"),
            "train_loss": meta.get("train_loss"),
            "val_loss": meta.get("val_loss"),
            "model_config": config,
        },
        "last_gate": {
            "exists": bool(gate),
            "promoted_candidate": gate.get("promoted_candidate"),
            "baseline_rate": (gate.get("baseline") or {}).get("rate"),
            "candidate_rate": (gate.get("candidate") or {}).get("rate"),
            "metric_warning": "Keyword coverage is a screening metric, not proof of general reasoning.",
        },
        "free_resource_policy": {
            "scheduled_training": False,
            "paid_ai_api": False,
            "training_is_event_driven": True,
            "note": "Free compute is limited; this report does not promise infinite training or performance.",
        },
    }


def main():
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
