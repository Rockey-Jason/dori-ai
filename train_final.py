"""Dori AI streaming trainer.

Features:
- recursively collects TXT/JSON/JSONL training data
- cleans, de-duplicates and chunks records
- trains/loads the custom byte-level BPE tokenizer
- streams batches instead of loading the whole token corpus into RAM
- mixes old + new data by scanning all configured roots together
- deterministic train/validation split
- resumable model + Adam optimizer checkpoints
- epoch loss history and automatic best-checkpoint promotion
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.optimizer import Adam
from dori_ai.core.transformer import DoriTransformer, load_model
from dori_ai.data_pipeline import CorpusBuilder, iter_jsonl_text, tokenizer_sample, token_batch

VERSION = "3.0.0-streaming"
ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
DATASET = ROOT / "data" / "streaming"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(path, model, optimizer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"p{i}": p.data for i, p in enumerate(model.parameters())}
    state = optimizer.state_dict()
    payload.update({f"m{i}": x for i, x in enumerate(state["m"])})
    payload.update({f"v{i}": x for i, x in enumerate(state["v"])})
    payload["optimizer_t"] = np.asarray([state["t"]], dtype=np.int64)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def load_optimizer(path, optimizer):
    data = np.load(path, allow_pickle=False)
    m = [data[k] for k in sorted((k for k in data.files if k.startswith("m")), key=lambda x: int(x[1:]))]
    v = [data[k] for k in sorted((k for k in data.files if k.startswith("v")), key=lambda x: int(x[1:]))]
    if len(m) != len(optimizer.m) or len(v) != len(optimizer.v) or "optimizer_t" not in data.files:
        return False
    try:
        optimizer.load_state_dict({"t": int(data["optimizer_t"][0]), "m": m, "v": v})
        return True
    except ValueError:
        return False


def write_meta(path, meta):
    Path(str(path) + ".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def cross_entropy_and_backward(model, optimizer, batch_x, batch_y, grad_clip):
    model.zero_grad()
    losses = []
    for x, y in zip(batch_x, batch_y):
        logits = model(x)
        z = logits.data - np.max(logits.data, axis=1, keepdims=True)
        p = np.exp(np.clip(z, -50.0, 0.0))
        p /= np.maximum(np.sum(p, axis=1, keepdims=True), 1e-12)
        idx = np.arange(len(y))
        loss = -np.log(np.maximum(p[idx, y], 1e-12)).mean()
        losses.append(float(loss))
        grad = p.copy()
        grad[idx, y] -= 1.0
        grad /= max(1, len(y))
        logits.backward(grad / len(batch_x))

    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        if not np.all(np.isfinite(param.grad)):
            raise FloatingPointError("non-finite gradient detected")
        total += float(np.sum(param.grad.astype(np.float64) ** 2))
    grad_norm = float(np.sqrt(total))
    if grad_clip > 0 and grad_norm > grad_clip:
        scale = grad_clip / (grad_norm + 1e-12)
        for param in model.parameters():
            if param.grad is not None:
                param.grad *= scale
    optimizer.step()
    return float(np.mean(losses)), grad_norm


def evaluate(model, tokenizer, path, seq_len, batch_size, max_batches, seed):
    rng = np.random.default_rng(seed)
    losses = []
    iterator = iter_jsonl_text(path)
    for _ in range(max(1, int(max_batches))):
        batch = token_batch(iterator, tokenizer, seq_len, batch_size, rng)
        if batch is None:
            break
        xs, ys = batch
        for x, y in zip(xs, ys):
            logits = model(x)
            z = logits.data - np.max(logits.data, axis=1, keepdims=True)
            p = np.exp(np.clip(z, -50.0, 0.0))
            p /= np.maximum(np.sum(p, axis=1, keepdims=True), 1e-12)
            idx = np.arange(len(y))
            losses.append(float(-np.log(np.maximum(p[idx, y], 1e-12)).mean()))
    return float(np.mean(losses)) if losses else float("inf")


def build_args():
    p = argparse.ArgumentParser(description="Dori AI large-data streaming trainer")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--data", action="append", default=[], help="extra TXT/JSON/JSONL file or directory; repeatable")
    p.add_argument("--vocab-size", type=int, default=768)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--chunk-chars", type=int, default=1800)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--tokenizer-sample-chars", type=int, default=2_000_000)
    p.add_argument("--retrain-tokenizer", action="store_true", help="rebuild tokenizer from the collected corpus sample; starts a fresh model")
    p.add_argument("--fresh", action="store_true", help="ignore resumable latest checkpoint and start a new model")
    p.add_argument("--no-build-cache", action="store_true", help="reuse existing data/streaming manifest/chunks")
    return p.parse_args()


def main():
    args = build_args()
    if args.dim % args.heads:
        raise SystemExit("dim must be divisible by heads")
    np.random.seed(args.seed)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    builder = CorpusBuilder(extra=args.data or [], out_dir=DATASET, chunk_chars=args.chunk_chars, val_fraction=args.val_fraction)
    manifest_path = DATASET / "manifest.json"
    if args.no_build_cache and manifest_path.exists() and (DATASET / "train.jsonl").exists() and (DATASET / "val.jsonl").exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("Using existing streaming dataset cache.")
    else:
        print("Collecting TXT/JSON/JSONL data...")
        manifest = builder.build()

    print(f"Files: {manifest['files']:,} | chunks: {manifest['chunks']:,} | train: {manifest['train']:,} | val: {manifest['val']:,}")
    print(f"Duplicates removed: {manifest['duplicates']:,} | garbage removed: {manifest['garbage']:,}")

    tokenizer_path = ROOT / "data" / "tokenizer.json"
    tokenizer_changed = False
    if args.retrain_tokenizer or not tokenizer_path.exists():
        sample = tokenizer_sample(builder, args.tokenizer_sample_chars)
        if not sample:
            raise RuntimeError("tokenizer를 학습할 데이터가 없어.")
        print(f"Training BPE tokenizer from a {len(sample):,}-character sample...")
        tokenizer = BPETokenizer(args.vocab_size).train(sample)
        tokenizer.save(tokenizer_path)
        tokenizer_changed = True
    else:
        tokenizer = BPETokenizer.load(tokenizer_path)

    latest = CHECKPOINTS / "latest.npz"
    latest_meta_path = Path(str(latest) + ".json")
    best = CHECKPOINTS / "best.npz"
    best_meta_path = Path(str(best) + ".json")
    resume = latest.exists() and latest_meta_path.exists() and not args.fresh and not tokenizer_changed
    # If this is the first run of the streaming trainer, continue from the
    # existing known-good best checkpoint instead of resetting the model.
    resume_path = latest if resume else (best if best.exists() and not args.fresh and not tokenizer_changed else None)

    if resume_path is not None:
        meta = json.loads(Path(str(resume_path) + ".json").read_text(encoding="utf-8"))
        resume = True
        config = meta["model_config"]
        model = load_model(str(resume_path), config)
        start_epoch = int(meta.get("epoch", 0)) + 1
        best_val = float(meta.get("best_val_loss", meta.get("val_loss", float("inf"))))
        print(f"Resuming from {resume_path.name}: epoch {start_epoch - 1}")
    else:
        config = dict(vocab_size=tokenizer.vocab_size, max_len=args.seq_len, dim=args.dim, heads=args.heads, ff_dim=args.ff_dim, layers=args.layers)
        model = DoriTransformer(**config)
        start_epoch = 1
        best_val = float("inf")
        print("Starting a fresh model.")

    if model.vocab_size != tokenizer.vocab_size:
        raise RuntimeError("tokenizer/model vocab size mismatch. Use --retrain-tokenizer with --fresh, or keep the existing tokenizer.")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    resumed_optimizer = resume and resume_path == latest and load_optimizer(latest, optimizer)
    if resume and not resumed_optimizer:
        print("Optimizer state not found; resuming model weights with a fresh Adam state.")

    if best.exists() and not (CHECKPOINTS / "best_before_streaming.npz").exists():
        shutil.copy2(best, CHECKPOINTS / "best_before_streaming.npz")
        if best_meta_path.exists():
            shutil.copy2(best_meta_path, Path(str(CHECKPOINTS / "best_before_streaming.npz") + ".json"))

    train_path = DATASET / "train.jsonl"
    val_path = DATASET / "val.jsonl"
    history_path = ROOT / "runtime" / "training_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    seq_len = min(int(config["max_len"]), int(args.seq_len))
    rng = np.random.default_rng(args.seed + max(0, start_epoch - 1))
    steps_per_epoch = max(1, int(np.ceil(manifest["train"] / max(1, args.batch_size))))
    total_start = time.time()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_losses, grad_norms = [], []
        iterator = iter_jsonl_text(train_path)
        for _ in range(steps_per_epoch):
            batch = token_batch(iterator, tokenizer, seq_len, args.batch_size, rng)
            if batch is None:
                break
            loss, grad_norm = cross_entropy_and_backward(model, optimizer, batch[0], batch[1], args.grad_clip)
            train_losses.append(loss)
            grad_norms.append(grad_norm)
        if not train_losses:
            raise RuntimeError("학습 가능한 batch가 없어.")

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate(model, tokenizer, val_path, seq_len, args.batch_size, args.val_batches, args.seed + epoch)
        save_checkpoint(latest, model, optimizer)
        meta = {
            "format_version": 6,
            "model_config": model.config(),
            "tokenizer": "data/tokenizer.json",
            "tokenizer_sha256": sha256(tokenizer_path),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gradient_norm": float(np.mean(grad_norms)),
            "best_val_loss": best_val,
            "version": VERSION,
            "dataset_manifest": "data/streaming/manifest.json",
            "dataset_counts": manifest,
            "steps_per_epoch": len(train_losses),
            "optimizer_t": optimizer.t,
        }
        write_meta(latest, meta)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": time.time(), **meta}, ensure_ascii=False) + "\n")

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            meta["best_val_loss"] = best_val
            save_checkpoint(best, model, optimizer)
            write_meta(best, meta)
            print(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f} | grad {np.mean(grad_norms):.4f} | ★ best {best_val:.4f}")
        else:
            print(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f} | grad {np.mean(grad_norms):.4f} | best {best_val:.4f}")

    print(f"Training complete in {time.time() - total_start:.1f}s")
    print(f"Best checkpoint: {best}")
    print(f"Loss history: {history_path}")


if __name__ == "__main__":
    main()
