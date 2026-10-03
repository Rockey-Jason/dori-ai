#!/usr/bin/env python3
"""Persistent continuous trainer for Dori AI.

A single invocation can request all remaining rounds up to the 1000-round
target. The round counter is stored in runtime/training_round.json so the
training resumes safely after a job timeout or interruption.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runtime" / "training_round.json"
HISTORY = ROOT / "runtime" / "continuous_rounds.jsonl"

def load_state():
    if STATE.exists():
        try:
            data = json.loads(STATE.read_text(encoding="utf-8"))
            return max(0, int(data.get("round", 0)))
        except Exception:
            pass
    return 0

def save_state(round_no, loss=None, val_loss=None):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": round_no,
        "updated_at": time.time(),
        "last_train_loss": loss,
        "last_val_loss": val_loss,
        "target_rounds": 1000,
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
    # One invocation now requests every remaining round through round 1000.
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--max-batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    current = load_state()
    target = 1000
    remaining = max(0, target - current)
    rounds = min(max(0, args.rounds), remaining)
    if rounds == 0:
        print("1000 training rounds are already complete.")
        return

    for i in range(rounds):
        round_no = current + i + 1
        seed = 20261003 + round_no * 7919
        stage = stage_for(round_no)
        stage_batches = min(args.max_batches, int(stage.get("max_batches", args.max_batches)))
        stage_batch_size = min(args.batch_size, int(stage.get("batch_size", args.batch_size)))
        stage_lr = min(args.lr, float(stage.get("lr", args.lr)))
        print(f"CONTINUOUS_ROUND {round_no}/{target} stage={stage['name']} goal={stage.get('goal','')}", flush=True)
        cmd = [
            sys.executable, str(ROOT / "train_final.py"),
            "--epochs", "1",
            "--max-batches", str(stage_batches),
            "--batch-size", str(stage_batch_size),
            "--seq-len", str(args.seq_len),
            "--dim", "256",
            "--heads", "8",
            "--layers", "8",
            "--ff-dim", "1024",
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
        stage_dir = ROOT / stage_dirs.get(stage["name"], "")
        if stage_dir.exists():
            cmd += ["--data", str(stage_dir)]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        meta_path = ROOT / "checkpoints" / "latest.npz.json"
        loss = val_loss = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            loss = meta.get("train_loss")
            val_loss = meta.get("val_loss")
        save_state(round_no, loss, val_loss)
        print(f"CONTINUOUS_ROUND_DONE {round_no}/{target} stage={stage['name']} train={loss} val={val_loss}", flush=True)

if __name__ == "__main__":
    main()
