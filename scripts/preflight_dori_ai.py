#!/usr/bin/env python3
"""Fast pre-deployment integrity and inference smoke tests for Dori AI."""
import json
from pathlib import Path
import sys

import numpy as np

# This file lives in scripts/, so explicitly add the repository root.
# That makes imports work both as:
#   python scripts/preflight_dori_ai.py
# and from other working directories.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dori_ai.bpe_tokenizer import BPETokenizer
from dori_ai.core.transformer import load_model
from generate_final import generate


CP = ROOT / "checkpoints" / "best.npz"
META = Path(str(CP) + ".json")
TOK = ROOT / "data" / "tokenizer.json"


def main():
    assert CP.exists() and META.exists(), "best checkpoint is missing"
    assert TOK.exists(), "tokenizer is missing"

    meta = json.loads(META.read_text(encoding="utf-8"))
    cfg = meta["model_config"]

    assert cfg["vocab_size"] == 768, cfg
    assert cfg["dim"] % cfg["heads"] == 0, cfg

    data = np.load(CP, allow_pickle=False)
    param_keys = [k for k in data.files if k.startswith("p") and k[1:].isdigit()]
    assert param_keys, "no model parameters"

    for key in param_keys:
        assert np.all(np.isfinite(data[key])), f"non-finite checkpoint: {key}"

    tok = BPETokenizer.load(TOK)
    model = load_model(str(CP), cfg)

    smoke = [
        "돌이는 뭐야?",
        "2 더하기 3은 얼마야?",
        "돌이사이트에는 무엇이 있어?",
    ]

    for prompt in smoke:
        ids = tok.encode(prompt)
        assert ids, prompt

        logits = model(
            np.asarray(ids[-cfg["max_len"]:], dtype=np.int64)
        ).data

        assert np.all(np.isfinite(logits)), f"non-finite logits: {prompt}"

        answer = generate(
            prompt,
            tok,
            model,
            tokens=16,
            temperature=0.25,
            top_k=8,
            top_p=0.90,
        )

        assert isinstance(answer, str) and answer.strip(), (
            f"empty generation: {prompt}"
        )
        assert "�" not in answer, (
            f"invalid UTF-8 replacement: {prompt}"
        )

    print("DORI_AI_PREFLIGHT=PASS")
    print("checkpoint_epoch=", meta.get("epoch"))
    print("val_loss=", meta.get("val_loss"))
    print("model_config=", cfg)


if __name__ == "__main__":
    main()
