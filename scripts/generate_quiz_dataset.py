#!/usr/bin/env python3
"""Export 100,000+ reproducible Dori quiz records as JSONL."""
import json
import sys
from pathlib import Path

from dori_ai.quiz_generator import generate_dataset

count = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/dori_quiz_bank_100k.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
rows = generate_dataset(count)
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Generated {len(rows):,} unique quiz records: {out}")
