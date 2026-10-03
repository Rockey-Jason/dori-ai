#!/usr/bin/env python3
"""Persistent continuous trainer for Dori AI.

Each invocation performs a fixed number of resumable micro-rounds. The round
counter is stored in runtime/training_round.json so scheduled GitHub Actions
runs continue where the previous run stopped. Every round resumes from the
latest checkpoint and uses a different deterministic RNG seed.
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=20)
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
        print(f"CONTINUOUS_ROUND {round_no}/{target}", flush=True)
        cmd = [
            sys.executable, str(ROOT / "train_final.py"),
            "--epochs", "1",
            "--max-batches", str(args.max_batches),
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--lr", str(args.lr),
            "--grad-clip", str(args.grad_clip),
            "--seed", str(seed),
        ]
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
        print(f"CONTINUOUS_ROUND_DONE {round_no}/{target} train={loss} val={val_loss}", flush=True)

if __name__ == "__main__":
    main()
