"""Deterministic local knowledge retrieval for Dori AI v2.3."""
import json,re
from pathlib import Path

def _norm(s):
    s=str(s).lower().strip()
    s=re.sub(r"[^0-9a-z가-힣]+"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def _grams(s,n=2):
    s=_norm(s).replace(" ","")
    if not s:return set()
    if len(s)<n:return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

def _words(s):
    return set(re.findall(r"[가-힣]{2,}|[a-z0-9_]+",_norm(s)))

class LocalKnowledge:
    def __init__(self,paths=None):
        root=Path("data")
        if paths is None:
            paths=[
                root/"dori_knowledge.jsonl",
                root/"dori_knowledge_expanded.jsonl",
                root/"dori_knowledge_v23.jsonl",
                root/"instructions/train.jsonl",
            ]
        self.rows=[]; seen=set()
        for p in paths:
            p=Path(p)
            if not p.exists(): continue
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    d=json.loads(line)
                    q=(d.get("question") or d.get("user") or "").strip()
                    a=(d.get("answer") or d.get("assistant") or "").strip()
                    if q and a and (q,a) not in seen:
                        self.rows.append((q,a,d.get("category",""),d.get("topic","")))
                        seen.add((q,a))
                except Exception:
                    continue
        self.index=[(q,a,c,t,_norm(q),_grams(q),_words(q)) for q,a,c,t in self.rows]

    def _score(self,query,item):
        q,a,c,t,qn,g,w=item
        x=_norm(query); xg=_grams(query); xw=_words(query)
        if x==qn:return 1.0
        contain=.90 if (x in qn or qn in x) else 0.0
        gram=len(xg&g)/(len(xg|g) or 1)
        word=len(xw&w)/(len(xw|w) or 1)
        topic_bonus=.0
        if t and _norm(t) in x: topic_bonus=.12
        return min(1.0,.48*gram+.34*word+.10*contain+topic_bonus)

    def best(self,query):
        if not str(query).strip(): return None
        best=(0.0,None)
        for item in self.index:
            s=self._score(query,item)
            if s>best[0]:best=(s,item)
        return best[1] if best[0]>=.46 else None

    def answer(self,query,threshold=.58):
        b=self.best(query)
        return b[1] if b and self._score(query,b)>=threshold else None

    def confidence(self,query):
        b=self.best(query)
        return 0.0 if not b else self._score(query,b)

    def size(self):
        return len(self.index)
