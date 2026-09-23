"""Fast local retrieval over Dori knowledge and instruction data."""
import json,re
from pathlib import Path

def _norm(s): return re.sub(r"\s+", "", str(s).lower())
def _grams(s,n=2):
    s=_norm(s)
    if not s:return set()
    if len(s)<n:return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

class LocalKnowledge:
    def __init__(self, paths=None):
        root=Path("data")
        self.rows=[]
        if paths is None:
            paths=[root/"dori_knowledge.jsonl",root/"instructions/train.jsonl"]
        seen=set()
        for p in paths:
            p=Path(p)
            if not p.exists(): continue
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    d=json.loads(line)
                    q=(d.get("question") or d.get("user") or "").strip()
                    a=(d.get("answer") or d.get("assistant") or "").strip()
                    if q and a and (q,a) not in seen:
                        self.rows.append((q,a)); seen.add((q,a))
                except Exception: pass
        self.index=[(q,a,_grams(q)) for q,a in self.rows]

    def size(self):
        return len(self.rows)

    def answer(self,query,threshold=.46):
        qn=_norm(query); qg=_grams(query)
        if not qg:return None
        best=None
        for q,a,g in self.index:
            if qn==_norm(q): return a
            score=len(qg & g)/(len(qg | g) or 1)
            nq=_norm(q)
            if qn in nq or nq in qn: score=max(score,.82)
            if best is None or score>best[0]: best=(score,a)
        return best[1] if best and best[0]>=threshold else None
