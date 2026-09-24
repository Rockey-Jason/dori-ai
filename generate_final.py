import argparse
import json
import codecs
from pathlib import Path

import numpy as np

from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model


def safe_softmax(logits, temperature=0.7):
    x = np.asarray(logits, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(
            x,
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        )

    temperature = max(float(temperature), 0.05)

    x = x / temperature
    x -= np.max(x)

    p = np.exp(np.clip(x, -50.0, 0.0))
    total = float(p.sum())

    if not np.isfinite(total) or total <= 0:
        return np.full(len(x), 1.0 / len(x))

    p /= total
    return p


def repetition_penalty(logits, generated, penalty=1.12, recent=64):
    logits = np.asarray(logits, dtype=np.float64).copy()

    if penalty <= 1:
        return logits

    recent_ids = generated[-recent:]

    for token_id in set(int(x) for x in recent_ids):
        if token_id < 0 or token_id >= len(logits):
            continue

        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty

    return logits


def token_bytes(tok, token_id):
    token_id = int(token_id)

    if token_id < 0 or token_id >= len(tok.itos):
        return None

    value = tok.itos.get(token_id)

    if isinstance(value, bytes):
        return value

    return None


def utf8_prefix_is_valid(data):
    """
    True if data is either:
    - valid complete UTF-8
    - or a valid incomplete UTF-8 prefix

    False if an invalid byte sequence has already occurred.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")

    try:
        decoder.decode(data, final=False)
        return True
    except UnicodeDecodeError:
        return False


def filter_utf8_candidates(logits, tok, current_bytes):
    """
    Remove token candidates that would immediately create invalid UTF-8.
    """
    scores = np.asarray(logits, dtype=np.float64).copy()

    for token_id in range(len(scores)):
        piece = token_bytes(tok, token_id)

        # Special tokens are handled separately.
        if piece is None:
            continue

        if not utf8_prefix_is_valid(current_bytes + piece):
            scores[token_id] = -np.inf

    return scores


def sample(
    logits,
    tok,
    generated,
    current_bytes,
    temperature=0.7,
    top_k=30,
    top_p=0.92,
    repetition=1.12,
    rng=None,
):
    rng = rng or np.random.default_rng()

    logits = repetition_penalty(
        logits,
        generated,
        penalty=repetition,
        recent=64,
    )

    logits = filter_utf8_candidates(
        logits,
        tok,
        current_bytes,
    )

    # Special tokens remain available.
    # In particular EOS must be able to terminate generation.
    probs = safe_softmax(logits, temperature)

    # top-k
    k = min(max(int(top_k), 0), len(probs))

    if k and k < len(probs):
        ids = np.argpartition(probs, -k)[-k:]

        mask = np.zeros_like(probs)
        mask[ids] = probs[ids]

        total = mask.sum()

        if total > 0:
            probs = mask / total

    # top-p
    nucleus = min(max(float(top_p), 0.0), 1.0)

    if nucleus < 1.0:
        ids = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[ids])

        n = int(
            np.searchsorted(
                cumulative,
                nucleus,
                side="left",
            )
        ) + 1

        keep = ids[:max(1, n)]

        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]

        total = mask.sum()

        if total > 0:
            probs = mask / total

    probs = np.maximum(probs, 0.0)

    total = probs.sum()

    if not np.isfinite(total) or total <= 0:
        # Fall back to argmax.
        return int(np.argmax(logits))

    probs /= total

    return int(
        rng.choice(
            len(probs),
            p=probs,
        )
    )


def load_bundle(checkpoint="checkpoints/best.npz"):
    checkpoint = Path(checkpoint)
    metadata_path = Path(str(checkpoint) + ".json")

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint가 없습니다: {checkpoint}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"checkpoint metadata가 없습니다: {metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    tokenizer = BPETokenizer.load(
        metadata["tokenizer"]
    )

    model = load_model(
        str(checkpoint),
        metadata["model_config"],
    )

    return tokenizer, model


def generate(
    prompt,
    tok,
    model,
    tokens=100,
    temperature=0.7,
    top_k=30,
    top_p=0.92,
    repetition=1.12,
    seed=20260924,
):
    prompt_ids = tok.encode(prompt)

    if not prompt_ids:
        return ""

    output_ids = list(prompt_ids)

    rng = np.random.default_rng(seed)

    generated_bytes = b""

    for _ in range(int(tokens)):
        context = np.asarray(
            output_ids[-model.max_len:],
            dtype=np.int64,
        )

        logits = model(context).data[-1]

        # EOS is a real token ID.
        eos_id = 2

        # Do not force EOS too early.
        if len(output_ids) <= len(prompt_ids) + 2:
            logits = np.asarray(
                logits,
                dtype=np.float64,
            ).copy()

            if 0 <= eos_id < len(logits):
                logits[eos_id] = -np.inf

        next_id = sample(
            logits,
            tok,
            output_ids,
            generated_bytes,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition=repetition,
            rng=rng,
        )

        # Real EOS token.
        if next_id == eos_id:
            break

        piece = token_bytes(tok, next_id)

        if piece is None:
            # Ignore other special tokens.
            continue

        candidate = generated_bytes + piece

        if not utf8_prefix_is_valid(candidate):
            # Should normally never happen because candidates are filtered.
            continue

        generated_bytes = candidate
        output_ids.append(next_id)

    try:
        answer = generated_bytes.decode(
            "utf-8",
            errors="strict",
        )
    except UnicodeDecodeError:
        answer = generated_bytes.decode(
            "utf-8",
            errors="replace",
        ).replace("�", "")

    return answer.strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.npz",
    )

    parser.add_argument(
        "--prompt",
        default="돌이는 뭐야?",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.92,
    )

    parser.add_argument(
        "--repetition",
        type=float,
        default=1.12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260924,
    )

    args = parser.parse_args()

    tok, model = load_bundle(
        args.checkpoint
    )

    answer = generate(
        args.prompt,
        tok,
        model,
        tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition=args.repetition,
        seed=args.seed,
    )

    print(answer)


if __name__ == "__main__":
    main()