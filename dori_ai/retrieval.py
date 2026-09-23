"""Fast local retrieval over Dori knowledge and instruction data.

The retriever is intentionally deterministic and local.  It combines exact
matches, containment, character n-grams and word/token overlap so paraphrased
Korean questions can still find the right factual answer.
"""
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
    # Korean is not whitespace-perfect, but noun-like chunks plus English/number
    # tokens give us a useful second signal beside character bigrams.
    return set(re.findall(r"[가-힣]{2,}|[a-z0-9_]+", _norm(s)))

class LocalKnowledge:
    def __init__(self,paths=None):
        root=Path("data")
        if paths is None:
            paths=[
                root/"dori_knowledge.jsonl",
                root/"dori_knowledge_expanded.jsonl",
                root/"instructions/train.jsonl",
            ]
        self.rows=[]
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
                        self.rows.append((q,a))
                        seen.add((q,a))
                except Exception:
                    continue
        self.index=[(q,a,_norm(q),_grams(q),_words(q)) for q,a in self.rows]

    def _score(self,query,item):
        q,a,qn,g,w=item
        query_n=_norm(query)
        query_g=_grams(query)
        query_w=_words(query)
        if query_n==qn:
            return 1.0
        if query_n in qn or qn in query_n:
            containment=0.90
        else:
            containment=0.0
        gram=len(query_g & g)/(len(query_g | g) or 1)
        word=len(query_w & w)/(len(query_w | w) or 1)
        # Exact meaningful token overlap should matter more than accidental
        # character overlap for longer Korean questions.
        score=0.52*gram+0.38*word+0.10*containment
        if len(query_w)>=2 and query_w.issubset(w):
            score=max(score,0.84)
        return score

    def answer(self,query,threshold=.46):
        query=str(query).strip()
        if not query:return None
        best=(0.0,None)
        for item in self.index:
            score=self._score(query,item)
            if score>best[0]:
                best=(score,item[1])
        return best[1] if best[0]>=threshold else None
