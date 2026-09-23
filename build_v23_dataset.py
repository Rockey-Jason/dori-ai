"""Build the Dori AI v2.3 training corpus from all curated JSONL sources."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
OUT=DATA/"corpus"/"v23_training.txt"
SOURCES=[
    DATA/"dori_knowledge.jsonl",
    DATA/"dori_knowledge_expanded.jsonl",
    DATA/"dori_knowledge_v23.jsonl",
    DATA/"instructions"/"train.jsonl",
]

def main():
    rows=[]; seen=set()
    for path in SOURCES:
        if not path.exists(): continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d=json.loads(line)
                q=(d.get("question") or d.get("user") or "").strip()
                a=(d.get("answer") or d.get("assistant") or "").strip()
                if not q or not a: continue
                key=(q,a)
                if key in seen: continue
                seen.add(key)
                rows.append(f"<system>너는 Dori AI다. 정확하고 친절하게 답한다. 모르면 추측하지 않는다.<user>{q}<dori>{a}<eos>")
            except Exception:
                pass
    # Add the existing human-readable corpus as supplementary language material.
    for path in [DATA/"corpus"/"dori_corpus.txt",DATA/"corpus"/"dori_knowledge_corpus.txt"]:
        if path.exists():
            text=path.read_text(encoding="utf-8").strip()
            if text:
                rows.append(text)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text("\n".join(rows)+"\n",encoding="utf-8")
    print(f"v2.3 corpus: {len(rows):,} records | {OUT.stat().st_size:,} bytes")

if __name__=="__main__": main()
