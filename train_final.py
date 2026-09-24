import argparse
import hashlib
import json
import os
import shutil
import time

from pathlib import Path

import numpy as np

from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import DoriTransformer
from dori_ai.core.optimizer import Adam
from dori_ai.trainer import Trainer


VERSION = "2.6.0"


def sha256(path):

    h = hashlib.sha256()

    h.update(
        Path(path).read_bytes()
    )

    return h.hexdigest()


def backup_checkpoint(path):

    path = Path(path)

    if not path.exists():
        return

    backup = path.with_name(
        path.stem
        + "_v25_backup"
        + path.suffix
    )

    if not backup.exists():
        shutil.copy2(
            path,
            backup,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--ff-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260924,
    )

    parser.add_argument(
        "--data",
        default="data/corpus/v25_training.txt",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--retrain-tokenizer",
        action="store_true",
    )

    args = parser.parse_args()

    if args.dim % args.heads:
        raise SystemExit(
            "dim must be divisible by heads"
        )

    np.random.seed(
        args.seed
    )

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    tokenizer_path = Path(
        "data/tokenizer.json"
    )

    data_path = Path(
        args.data
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"training data not found: {data_path}"
        )

    text = data_path.read_text(
        encoding="utf-8"
    )

    if (
        args.retrain_tokenizer
        or not tokenizer_path.exists()
    ):
        tokenizer = (
            BPETokenizer(
                args.vocab_size
            )
            .train(text)
        )

        tokenizer.save(
            tokenizer_path
        )

    else:

        tokenizer = (
            BPETokenizer.load(
                tokenizer_path
            )
        )

    tokens = tokenizer.encode(
        text
    )

    print(
        f"Corpus: {len(text):,} chars"
    )

    print(
        f"Tokens: {len(tokens):,}"
    )

    print(
        f"Vocab: {tokenizer.vocab_size}"
    )

    model = DoriTransformer(
        tokenizer.vocab_size,
        args.seq_len,
        args.dim,
        args.heads,
        args.ff_dim,
        args.layers,
    )

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    trainer = Trainer(
        model,
        optimizer,
        tokens,
        args.seq_len,
        args.batch_size,
        val_fraction=0.12,
        seed=args.seed,
        grad_clip=args.grad_clip,
    )

    backup_checkpoint(
        "checkpoints/best.npz"
    )

    best = float("inf")

    start_time = time.time()

    print(
        "Training configuration:"
    )

    print(
        f"dim={args.dim}, "
        f"heads={args.heads}, "
        f"layers={args.layers}, "
        f"ff={args.ff_dim}, "
        f"context={args.seq_len}, "
        f"lr={args.lr}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss, grad_norm = (
            trainer.train_step()
        )

        if (
            epoch == 1
            or epoch % 25 == 0
            or epoch == args.epochs
        ):

            val_loss = trainer.evaluate(
                batches=8
            )

            metadata = {
                "format_version": 4,
                "model_config": model.config(),
                "tokenizer": "data/tokenizer.json",
                "tokenizer_sha256": sha256(
                    tokenizer_path
                ),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "gradient_norm": grad_norm,
                "version": VERSION,
                "data": str(data_path),
            }

            model.save(
                "checkpoints/latest.npz"
            )

            Path(
                "checkpoints/latest.npz.json"
            ).write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if (
                np.isfinite(val_loss)
                and val_loss < best
            ):

                best = val_loss

                model.save(
                    "checkpoints/best.npz"
                )

                Path(
                    "checkpoints/best.npz.json"
                ).write_text(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            print(
                f"Epoch {epoch:4d} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f} | "
                f"grad {grad_norm:.4f} | "
                f"best {best:.4f}"
            )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Training complete in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()