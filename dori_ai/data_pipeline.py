"""Streaming training-data pipeline for Dori AI.

Collects TXT/JSON/JSONL recursively, cleans and de-duplicates records, writes a
small manifest/chunk store, and provides tokenized batches without loading the
whole corpus into RAM.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = (
    ROOT / "data" / "corpus",
    ROOT / "data" / "new",
    ROOT / "data" / "generated",
    ROOT / "data" / "learning",
)
IGNORED_NAMES = {"tokenizer.json", "learning_status.json"}
TEXT_KEYS = ("text", "content", "body", "document", "knowledge", "prompt")
QUESTION_KEYS = ("question", "input", "user", "query")
ANSWER_KEYS = ("answer", "output", "response", "assistant")


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t\f\v]+", " ", x).strip() for x in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_garbage(text, min_chars=20, max_chars=20000):
    if len(text) < min_chars or len(text) > max_chars:
        return True
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    controls = sum(ord(c) < 32 and c not in "\n\t" for c in text)
    if controls > max(3, len(text) // 100):
        return True
    if len(set(compact)) <= 2 and len(compact) > 30:
        return True
    return bool(re.search(r"(.)\1{11,}", compact))


def _records_from_json(obj):
    if isinstance(obj, list):
        for item in obj:
            yield from _records_from_json(item)
        return
    if not isinstance(obj, dict):
        if isinstance(obj, str):
            yield obj
        return
    q = next((obj.get(k) for k in QUESTION_KEYS if obj.get(k)), None)
    a = next((obj.get(k) for k in ANSWER_KEYS if obj.get(k)), None)
    if q and a:
        yield f"사용자: {q}\nAI: {a}"
        return
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            yield value
    messages = obj.get("messages")
    if isinstance(messages, list):
        parts = []
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                parts.append(f"{m.get('role', 'user')}: {m['content']}")
        if parts:
            yield "\n".join(parts)


def iter_source_records(path):
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    yield from _records_from_json(json.loads(line))
                except json.JSONDecodeError:
                    continue
    elif suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            yield from _records_from_json(obj)
        except (json.JSONDecodeError, OSError):
            return


def discover_files(roots=None, extra=None):
    roots = list(roots or DEFAULT_ROOTS)
    if extra:
        roots.extend(Path(x) for x in extra)
    found = []
    for root in roots:
        root = Path(root)
        if root.is_file() and root.suffix.lower() in {".txt", ".json", ".jsonl"}:
            found.append(root)
        elif root.is_dir():
            found.extend(
                p for p in root.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".txt", ".json", ".jsonl"}
                and p.name not in IGNORED_NAMES
            )
    return sorted(set(p.resolve() for p in found))


def chunk_text(text, chunk_chars=1800):
    text = normalize_text(text)
    if not text:
        return
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for para in paragraphs:
        if len(para) <= chunk_chars:
            yield para
            continue
        sentences = re.split(r"(?<=[.!?。！？])\s+", para)
        buf = ""
        for sent in sentences:
            if len(sent) <= chunk_chars and len(buf) + len(sent) + 1 <= chunk_chars:
                buf = f"{buf} {sent}".strip()
            else:
                if buf:
                    yield buf
                while len(sent) > chunk_chars:
                    yield sent[:chunk_chars]
                    sent = sent[chunk_chars:]
                buf = sent
        if buf:
            yield buf


def stable_split(text, source, validation_fraction=0.1):
    key = hashlib.sha256((str(source) + "\n" + text).encode("utf-8")).digest()
    value = int.from_bytes(key[:8], "big") / float(2**64)
    return "val" if value < validation_fraction else "train"


class CorpusBuilder:
    def __init__(self, roots=None, extra=None, out_dir=None, chunk_chars=1800, val_fraction=0.1):
        self.roots = roots or list(DEFAULT_ROOTS)
        self.extra = extra or []
        self.out_dir = Path(out_dir or ROOT / "data" / "streaming")
        self.chunk_chars = int(chunk_chars)
        self.val_fraction = float(val_fraction)

    def build(self, max_records=None):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.out_dir / "train.jsonl"
        val_path = self.out_dir / "val.jsonl"
        manifest_path = self.out_dir / "manifest.json"
        seen = set()
        counts = {"files": 0, "raw_records": 0, "chunks": 0, "train": 0, "val": 0,
                  "duplicates": 0, "garbage": 0}
        with train_path.open("w", encoding="utf-8") as train, val_path.open("w", encoding="utf-8") as val:
            for path in discover_files(self.roots, self.extra):
                counts["files"] += 1
                for raw in iter_source_records(path):
                    counts["raw_records"] += 1
                    for chunk in chunk_text(raw, self.chunk_chars):
                        if max_records and counts["chunks"] >= max_records:
                            break
                        if is_garbage(chunk):
                            counts["garbage"] += 1
                            continue
                        key = hashlib.sha256(chunk.casefold().encode("utf-8")).hexdigest()
                        if key in seen:
                            counts["duplicates"] += 1
                            continue
                        seen.add(key)
                        split = stable_split(chunk, path, self.val_fraction)
                        row = {"text": chunk, "source": str(path), "hash": key}
                        target = val if split == "val" else train
                        target.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts["chunks"] += 1
                        counts[split] += 1
                    if max_records and counts["chunks"] >= max_records:
                        break
                if max_records and counts["chunks"] >= max_records:
                    break
        manifest = {
            "version": 1,
            "chunk_chars": self.chunk_chars,
            "val_fraction": self.val_fraction,
            **counts,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


def iter_jsonl_text(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") if isinstance(row, dict) else None
            if text:
                yield text


def tokenizer_sample(builder, max_chars=2_000_000):
    pieces = []
    total = 0
    for path in (builder.out_dir / "train.jsonl", builder.out_dir / "val.jsonl"):
        if not path.exists():
            continue
        for text in iter_jsonl_text(path):
            pieces.append(text)
            total += len(text) + 1
            if total >= max_chars:
                return "\n".join(pieces)[:max_chars]
    return "\n".join(pieces)


def token_batch(iterator, tokenizer, seq_len, batch_size, rng):
    xs, ys = [], []
    while len(xs) < batch_size:
        try:
            text = next(iterator)
        except StopIteration:
            if not xs:
                return None
            break
        ids = tokenizer.encode(text)
        if len(ids) < seq_len + 1:
            continue
        start = int(rng.integers(0, len(ids) - seq_len))
        xs.append(np.asarray(ids[start:start + seq_len], dtype=np.int64))
        ys.append(np.asarray(ids[start + 1:start + seq_len + 1], dtype=np.int64))
    return np.stack(xs), np.stack(ys)
