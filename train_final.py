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
import uuid
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


def write_meta(path, meta):
    """Atomically write checkpoint metadata."""
    meta_path = Path(str(path) + ".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_name(meta_path.name + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)


def save_checkpoint(path, model, optimizer, meta):
    """Save weights with an embedded fingerprint, then atomically publish metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = model.parameters()
    checkpoint_id = uuid.uuid4().hex
    saved_meta = dict(meta)
    saved_meta.update({
        "format_version": 7,
        "checkpoint_id": checkpoint_id,
        "model_config": model.config(),
        "parameter_count": len(params),
        "parameter_shapes": [list(p.data.shape) for p in params],
    })
    payload = {f"p{i}": p.data for i, p in enumerate(params)}
    state = optimizer.state_dict()
    payload.update({f"m{i}": x for i, x in enumerate(state["m"])})
    payload.update({f"v{i}": x for i, x in enumerate(state["v"])})
    payload["optimizer_t"] = np.asarray([state["t"]], dtype=np.int64)
    # '__' keys cannot be confused with numbered Adam moment keys.
    payload["__checkpoint_id__"] = np.asarray(checkpoint_id)
    payload["__model_config_json__"] = np.asarray(json.dumps(model.config(), sort_keys=True))
    payload["__parameter_count__"] = np.asarray([len(params)], dtype=np.int64)
    payload["__parameter_shapes_json__"] = np.asarray(json.dumps(saved_meta["parameter_shapes"]))
    tmp = path.with_name(path.stem + f".{checkpoint_id}.tmp.npz")
    try:
        np.savez(tmp, **payload)
        os.replace(tmp, path)
        write_meta(path, saved_meta)
    finally:
        if tmp.exists():
            tmp.unlink()
    return saved_meta


def load_optimizer(path, optimizer):
    try:
        with np.load(path, allow_pickle=False) as data:
            m = [data[k] for k in sorted((k for k in data.files if k.startswith("m") and k[1:].isdigit()), key=lambda x: int(x[1:]))]
            v = [data[k] for k in sorted((k for k in data.files if k.startswith("v") and k[1:].isdigit()), key=lambda x: int(x[1:]))]
            if len(m) != len(optimizer.m) or len(v) != len(optimizer.v) or "optimizer_t" not in data.files:
                return False
            optimizer.load_state_dict({"t": int(data["optimizer_t"][0]), "m": m, "v": v})
            return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_checkpoint(path, requested_config, tokenizer_hash):
    """Validate sidecar AND real tensor shapes before attempting to resume."""
    path = Path(path)
    meta_path = Path(str(path) + ".json")
    if not path.is_file() or not meta_path.is_file():
        return None, None, "weights or metadata sidecar missing"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        config = meta.get("model_config")
        if not isinstance(config, dict):
            return None, meta, "model_config missing"
        if any(config.get(k) != v for k, v in requested_config.items()):
            return None, meta, f"model config differs: checkpoint={config}, requested={requested_config}"
        recorded_hash = meta.get("tokenizer_sha256")
        if recorded_hash and recorded_hash != tokenizer_hash:
            return None, meta, "tokenizer fingerprint differs"

        probe = DoriTransformer(**config)
        expected = probe.parameters()
        with np.load(path, allow_pickle=False) as data:
            keys = sorted((k for k in data.files if k.startswith("p") and k[1:].isdigit()), key=lambda k: int(k[1:]))
            if len(keys) != len(expected):
                return None, meta, f"parameter count differs: checkpoint={len(keys)}, model={len(expected)}"
            for i, param in enumerate(expected):
                key = f"p{i}"
                if key not in data.files or data[key].shape != param.data.shape:
                    actual = None if key not in data.files else data[key].shape
                    return None, meta, f"parameter {key} shape differs: checkpoint={actual}, model={param.data.shape}"
            if "__parameter_count__" in data.files and int(data["__parameter_count__"][0]) != len(expected):
                return None, meta, "embedded parameter count differs"
            if "__parameter_shapes_json__" in data.files:
                shapes = json.loads(str(data["__parameter_shapes_json__"].item()))
                if shapes != [list(p.data.shape) for p in expected]:
                    return None, meta, "embedded parameter shapes differ"
            if "__checkpoint_id__" in data.files and meta.get("checkpoint_id"):
                if str(data["__checkpoint_id__"].item()) != str(meta["checkpoint_id"]):
                    return None, meta, "checkpoint ID differs from sidecar (interrupted write)"
            if "__model_config_json__" in data.files:
                if json.loads(str(data["__model_config_json__"].item())) != config:
                    return None, meta, "embedded model config differs from sidecar"
        return load_model(str(path), config), meta, None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def archive_checkpoint(path, reason):
    """Keep rejected weights for recovery instead of deleting them."""
    path = Path(path)
    if not path.is_file():
        return None
    archive_dir = CHECKPOINTS / "incompatible"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    target = archive_dir / f"{path.stem}-{stamp}{path.suffix}"
    suffix = 1
    while target.exists():
        target = archive_dir / f"{path.stem}-{stamp}-{suffix}{path.suffix}"
        suffix += 1
    shutil.copy2(path, target)
    sidecar = Path(str(path) + ".json")
    if sidecar.is_file():
        shutil.copy2(sidecar, Path(str(target) + ".json"))
    Path(str(target) + ".reason.txt").write_text(reason, encoding="utf-8")
    print(f"CHECKPOINT_PRESERVED path={target} reason={reason}", flush=True)
    return target


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
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--ff-dim", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--data", action="append", default=[], help="extra TXT/JSON/JSONL file or directory; repeatable")
    p.add_argument("--vocab-size", type=int, default=768)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--chunk-chars", type=int, default=1800)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--patience", type=int, default=10, help="stop after this many epochs without meaningful validation improvement; 0 disables")
    p.add_argument("--min-delta", type=float, default=0.001, help="minimum validation-loss improvement counted as meaningful")
    p.add_argument("--max-batches", type=int, default=50, help="maximum training batches per epoch; 0 means all batches")
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
        print("TRAIN_STAGE dataset_cache", flush=True)
        print("Using existing streaming dataset cache.", flush=True)
    else:
        print("TRAIN_STAGE collecting", flush=True)
        print("Collecting TXT/JSON/JSONL data...", flush=True)
        manifest = builder.build()

    print("TRAIN_STAGE dataset_ready", flush=True)
    print(f"Files: {manifest['files']:,} | chunks: {manifest['chunks']:,} | train: {manifest['train']:,} | val: {manifest['val']:,}", flush=True)
    print(f"Duplicates removed: {manifest['duplicates']:,} | garbage removed: {manifest['garbage']:,}")

    tokenizer_path = ROOT / "data" / "tokenizer.json"
    tokenizer_changed = False
    if args.retrain_tokenizer or not tokenizer_path.exists():
        sample = tokenizer_sample(builder, args.tokenizer_sample_chars)
        if not sample:
            raise RuntimeError("tokenizer를 학습할 데이터가 없어.")
        print("TRAIN_STAGE tokenizer", flush=True)
        print(f"Training BPE tokenizer from a {len(sample):,}-character sample...", flush=True)
        tokenizer = BPETokenizer(args.vocab_size).train(sample)
        tokenizer.save(tokenizer_path)
        tokenizer_changed = True
    else:
        tokenizer = BPETokenizer.load(tokenizer_path)

    print("TRAIN_STAGE checkpoint_loading", flush=True)
    latest = CHECKPOINTS / "latest.npz"
    best = CHECKPOINTS / "best.npz"
    backup = CHECKPOINTS / "best_before_streaming.npz"
    requested_config = dict(vocab_size=tokenizer.vocab_size, max_len=args.seq_len, dim=args.dim, heads=args.heads, ff_dim=args.ff_dim, layers=args.layers)
    tokenizer_hash = sha256(tokenizer_path)
    resume = False
    resume_path = None
    model = None
    meta = None
    incompatible = []

    # Check newest first. A stale JSON sidecar must never be enough to trust a
    # checkpoint: inspect actual parameter counts and tensor shapes first.
    if not args.fresh and not tokenizer_changed:
        for candidate in (latest, best, backup):
            if not candidate.exists():
                continue
            candidate_model, candidate_meta, reason = validate_checkpoint(
                candidate, requested_config, tokenizer_hash
            )
            if candidate_model is not None:
                model, meta, resume_path = candidate_model, candidate_meta, candidate
                resume = True
                start_epoch = int(meta.get("epoch", 0)) + 1
                best_val = float(meta.get("best_val_loss", meta.get("val_loss", float("inf"))))
                print(
                    f"CHECKPOINT_RESUME path={candidate.name} epoch={start_epoch - 1} "
                    f"parameters={len(model.parameters())}",
                    flush=True,
                )
                break
            reason = reason or "checkpoint failed validation"
            incompatible.append((candidate, reason))
            print(f"CHECKPOINT_REJECTED path={candidate.name}: {reason}", flush=True)
    else:
        print("Checkpoint resume disabled by --fresh or tokenizer rebuild.", flush=True)

    if model is None:
        # Preserve rejected weights before the current run can replace latest/best.
        for candidate, reason in incompatible:
            archive_checkpoint(candidate, reason)
        config = requested_config
        model = DoriTransformer(**config)
        start_epoch = 1
        best_val = float("inf")
        resume = False
        resume_path = None
        print(
            "CHECKPOINT_FALLBACK fresh_model=true; original checkpoints preserved; "
            f"requested_parameters={len(model.parameters())}",
            flush=True,
        )
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
    all_steps_per_epoch = max(1, int(np.ceil(manifest["train"] / max(1, args.batch_size))))
    steps_per_epoch = all_steps_per_epoch if args.max_batches <= 0 else min(all_steps_per_epoch, max(1, int(args.max_batches)))
    print(f"TRAIN_STAGE training steps_per_epoch={steps_per_epoch}/{all_steps_per_epoch}", flush=True)
    total_start = time.time()

    stale_epochs = 0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_losses, grad_norms = [], []
        iterator = iter_jsonl_text(train_path)
        for batch_idx in range(1, steps_per_epoch + 1):
            print(
                f"TRAIN_BATCH_START epoch {epoch - start_epoch + 1}/{args.epochs} "
                f"batch {batch_idx}/{steps_per_epoch}",
                flush=True
            )
            batch = token_batch(iterator, tokenizer, seq_len, args.batch_size, rng)
            if batch is None:
                break
            loss, grad_norm = cross_entropy_and_backward(model, optimizer, batch[0], batch[1], args.grad_clip)
            train_losses.append(loss)
            grad_norms.append(grad_norm)

            # Emit periodic progress so the admin UI can show live movement
            # even when a single epoch contains thousands of CPU batches.
            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == steps_per_epoch:
                run_epoch = epoch - start_epoch + 1
                print(
                    f"TRAIN_PROGRESS epoch {run_epoch}/{args.epochs} "
                    f"batch {batch_idx}/{steps_per_epoch} loss {loss:.4f}",
                    flush=True
                )
        if not train_losses:
            raise RuntimeError("학습 가능한 batch가 없어.")

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate(model, tokenizer, val_path, seq_len, args.batch_size, args.val_batches, args.seed + epoch)
        meta = {
            "tokenizer": "data/tokenizer.json",
            "tokenizer_sha256": tokenizer_hash,
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
        meta = save_checkpoint(latest, model, optimizer, meta)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": time.time(), **meta}, ensure_ascii=False) + "\n")

        run_epoch = epoch - start_epoch + 1
        meaningful_improvement = np.isfinite(val_loss) and val_loss < best_val - max(0.0, args.min_delta)
        if meaningful_improvement:
            stale_epochs = 0
        else:
            stale_epochs += 1
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            meta["best_val_loss"] = best_val
            save_checkpoint(best, model, optimizer, meta)
            print(
                f"TRAIN_EPOCH run {run_epoch}/{args.epochs} "
                f"model_epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} "
                f"| grad {np.mean(grad_norms):.4f} | best {best_val:.4f}",
                flush=True
            )
        else:
            print(
                f"TRAIN_EPOCH run {run_epoch}/{args.epochs} "
                f"model_epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} "
                f"| grad {np.mean(grad_norms):.4f} | best {best_val:.4f}",
                flush=True
            )

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"TRAIN_EARLY_STOP patience={args.patience} best_val_loss={best_val:.4f}", flush=True)
            break

    print(f"Training complete in {time.time() - total_start:.1f}s")
    print(f"Best checkpoint: {best}")
    print(f"Loss history: {history_path}")


if __name__ == "__main__":
    main()
