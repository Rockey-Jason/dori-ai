"""Build a large, deterministic training corpus from the project data."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data'/'corpus'/'v25_training.txt'
paths=[ROOT/'data'/'dori.txt',ROOT/'data'/'corpus'/'training.txt',ROOT/'data'/'corpus'/'dori_corpus.txt',ROOT/'data'/'corpus'/'dori_knowledge_corpus.txt',ROOT/'data'/'instructions'/'train.jsonl',ROOT/'data'/'dialogue'/'train.jsonl']
rows=[]; seen=set()
for p in paths:
    if not p.exists(): continue
    if p.suffix=='.txt':
        lines=p.read_text(encoding='utf-8',errors='ignore').splitlines()
        for line in lines:
            line=line.strip()
            if len(line)>=8 and line.casefold() not in seen: seen.add(line.casefold()); rows.append(line)
    else:
        for raw in p.read_text(encoding='utf-8',errors='ignore').splitlines():
            try: d=json.loads(raw)
            except Exception: continue
            if 'turns' in d:
                turns=d['turns']; rows.append('\n'.join(f"<{t.get('role','user')}>{t.get('text','')}" for t in turns)+"<eos>")
            else:
                q=d.get('question') or d.get('input') or d.get('prompt'); a=d.get('answer') or d.get('output') or d.get('response')
                if q and a: rows.append(f"<user>{q}<dori>{a}<eos>")
gen=ROOT/'data'/'generated'
for p in sorted(gen.glob('*.jsonl')) if gen.exists() else []:
    for raw in p.read_text(encoding='utf-8',errors='ignore').splitlines():
        try:d=json.loads(raw)
        except Exception:continue
        q=d.get('question');a=d.get('answer')
        if q and a: rows.append(f"<user>{q}<dori>{a}<eos>")
# De-duplicate after all sources.
out=[];seen=set()
for r in rows:
    k=r.casefold()
    if k not in seen: seen.add(k); out.append(r)
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text('\n'.join(out)+'\n',encoding='utf-8')
print(f'v2.5 corpus: {len(out):,} records | {OUT.stat().st_size:,} bytes')
