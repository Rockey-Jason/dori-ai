#!/usr/bin/env python3
"""Persistent trainer with bounded, fingerprint-driven growth cycles.

Progress within a data version is stored in runtime/training_round.json.
New training sources open another cycle; the held-out promotion gate decides
whether the trained candidate can replace the accepted checkpoint.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runtime" / "training_round.json"
HISTORY = ROOT / "runtime" / "continuous_rounds.jsonl"

def load_state_data():
    if STATE.exists():
        try:
            data = json.loads(STATE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            pass
    return {}

def load_state():
    return max(0, int(load_state_data().get("round", 0)))

def source_fingerprint(root=ROOT):
    """Hash actual training sources so changed knowledge opens a new cycle."""
    root = Path(root)
    data_root = root / "data"
    search_roots = [
        data_root / "corpus",
        data_root / "new",
        data_root / "generated",
        data_root / "curriculum",
        data_root / "instructions",
    ]
    paths = set()
    for source in search_roots:
        if source.is_file() and source.suffix.lower() in {".txt", ".json", ".jsonl"}:
            paths.add(source)
        elif source.is_dir():
            for path in source.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".txt", ".json", ".jsonl"}:
                    paths.add(path)
    paths.update(p for p in data_root.glob("dori_knowledge*.jsonl") if p.is_file())
    # Token IDs depend on the tokenizer, so include it in the dataset/model version.
    tokenizer = data_root / "tokenizer.json"
    if tokenizer.is_file():
        paths.add(tokenizer)
    curriculum_plan = data_root / "curriculum_plan.json"
    if curriculum_plan.is_file():
        paths.add(curriculum_plan)

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\\0")
    return digest.hexdigest()

def save_state(round_no, loss=None, val_loss=None, *, source_hash="", cumulative_round=0, cycle_started_at=None):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": int(round_no),
        "target_rounds": 1200,
        "target_rounds_per_dataset": 1200,
        "cumulative_round": int(cumulative_round),
        "source_fingerprint": source_hash,
        "cycle_started_at": cycle_started_at or time.time(),
        "updated_at": time.time(),
        "last_train_loss": loss,
        "last_val_loss": val_loss,
    }
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def load_curriculum():
    path = ROOT / "data" / "curriculum_plan.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("stages", [])
    except Exception:
        return []

def stage_for(round_no):
    for stage in load_curriculum():
        if int(stage["from"]) <= round_no <= int(stage["to"]):
            return stage
    return {"name": "default", "goal": "continuous training", "lr": 1.5e-4, "max_batches": 30, "batch_size": 2}

def main():
    p = argparse.ArgumentParser()
    # Run a bounded chunk per invocation; progress resumes from the data fingerprint.
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--max-batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    target = 1200
    stored_state = load_state_data()
    fingerprint = source_fingerprint()
    cumulative_before = max(0, int(stored_state.get("cumulative_round", stored_state.get("round", 0))))
    same_dataset = bool(stored_state.get("source_fingerprint")) and stored_state.get("source_fingerprint") == fingerprint
    current = max(0, int(stored_state.get("round", 0))) if same_dataset else 0
    cycle_started_at = stored_state.get("cycle_started_at") if same_dataset else time.time()

    if not same_dataset:
        print("GROWTH_CYCLE=NEW_DATASET_OR_LEGACY_STATE", flush=True)
        print("SOURCE_FINGERPRINT=" + fingerprint, flush=True)
        print("Starting a new bounded training cycle from the accepted best checkpoint.", flush=True)

    remaining = max(0, target - current)
    rounds = min(max(0, args.rounds), remaining)
    if rounds == 0:
        print("This dataset version has reached its training target; add reviewed new data to unlock another cycle.")
        print("CUMULATIVE_TRAINING_ROUNDS=", cumulative_before)
        return

    # Match the quality-approved architecture instead of training a larger,
    # incompatible scratch model that is unlikely to pass the held-out gate.
    best_meta_path = ROOT / "checkpoints" / "best.npz.json"
    if not best_meta_path.is_file():
        raise SystemExit("Missing checkpoints/best.npz.json; cannot choose a compatible architecture.")
    best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
    model_config = best_meta.get("model_config") or {}
    required_config = ("vocab_size", "max_len", "dim", "heads", "ff_dim", "layers")
    if not all(isinstance(model_config.get(k), int) and model_config[k] > 0 for k in required_config):
        raise SystemExit("Accepted best checkpoint has an invalid model_config.")
    if model_config["dim"] % model_config["heads"]:
        raise SystemExit("Accepted best checkpoint has incompatible dim/heads settings.")
    seq_len = min(max(1, args.seq_len), model_config["max_len"])

    for i in range(rounds):
        round_no = current + i + 1
        cumulative_round = cumulative_before + i + 1
        seed = 20261003 + round_no * 7919
        stage = stage_for(round_no)
        stage_batches = min(args.max_batches, int(stage.get("max_batches", args.max_batches)))
        stage_batch_size = min(args.batch_size, int(stage.get("batch_size", args.batch_size)))
        stage_lr = min(args.lr, float(stage.get("lr", args.lr)))
        print(f"CONTINUOUS_ROUND {round_no}/{target} cumulative={cumulative_round} stage={stage['name']} goal={stage.get('goal','')}", flush=True)
        cmd = [
            sys.executable, str(ROOT / "train_final.py"),
            "--epochs", "1",
            "--max-batches", str(stage_batches),
            "--batch-size", str(stage_batch_size),
            "--seq-len", str(seq_len),
            "--vocab-size", str(model_config["vocab_size"]),
            "--dim", str(model_config["dim"]),
            "--heads", str(model_config["heads"]),
            "--layers", str(model_config["layers"]),
            "--ff-dim", str(model_config["ff_dim"]),
            "--lr", str(stage_lr),
            "--grad-clip", str(args.grad_clip),
            "--seed", str(seed),
        ]
        stage_dirs = {
            "korean_language": "data/curriculum/korean_language",
            "site_mastery": "data/curriculum/site_mastery",
            "qa_following": "data/curriculum/qa_following",
            "reasoning": "data/curriculum/reasoning",
            "instruction": "data/curriculum/instruction",
            "worldbuilding": "data/curriculum/worldbuilding",
            "tools_retrieval": "data/curriculum/tools_retrieval",
            "fact_safety": "data/curriculum/fact_safety",
            "dialogue": "data/curriculum/dialogue",
            "adversarial": "data/curriculum/adversarial",
            "stabilization": "data/curriculum/stabilization",
        }
        if round_no == 1:
            # A changed corpus has a new validation distribution. Start the
            # accepted checkpoint with a fresh validation-loss baseline.
            cmd += ["--reset-best-loss", "--prefer-best-checkpoint"]
        # Include every curriculum directory when building the cache on the
        # first round. Later rounds reuse that complete cache; passing only the
        # active stage here would silently omit all other stages after round 1.
        # The stage still controls learning rate and batch budget, while the
        # corpus remains a stable, mixed dataset for replay-safe training.
        for relative_dir in sorted(set(stage_dirs.values())):
            curriculum_dir = ROOT / relative_dir
            if curriculum_dir.exists():
                cmd += ["--data", str(curriculum_dir)]
        # Build the cleaned streaming corpus once per workflow run. Rebuilding
        # all source files for each of 1000 micro-rounds wastes most of the
        # runner budget; later rounds safely reuse the complete manifest above.
        if round_no > current + 1:
            cmd += ["--no-build-cache"]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        meta_path = ROOT / "checkpoints" / "latest.npz.json"
        loss = val_loss = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            loss = meta.get("train_loss")
            val_loss = meta.get("val_loss")
        save_state(round_no, loss, val_loss, source_hash=fingerprint, cumulative_round=cumulative_round, cycle_started_at=cycle_started_at)
        print(f"CONTINUOUS_ROUND_DONE {round_no}/{target} stage={stage['name']} train={loss} val={val_loss}", flush=True)

if __name__ == "__main__":
    main()
