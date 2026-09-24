"""Fast local retrieval over Dori knowledge and instruction examples."""
import json, re
from difflib import SequenceMatcher
from pathlib import Path

def _norm(s):
    return re.sub(r"[\s\W_]+", "", str(s).lower(), flags=re.UNICODE)

def _tokens(s):
    return set(re.findall(r"[가-힣A-Za-z0-9]+", str(s).lower()))

def _grams(s, n=2):
    s = _norm(s)
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}

class LocalKnowledge:
    def __init__(self, paths=None):
        root = Path(__file__).resolve().parent.parent / "data"
        if paths is None:
            candidates = [
                root / "dori_knowledge.jsonl",
                root / "dori_knowledge_expanded.jsonl",
                root / "dori_knowledge_v23.jsonl",
                root / "dori_knowledge_v25.jsonl",
                root / "instructions" / "train.jsonl",
            ]
            paths = [p for p in candidates if p.exists()]
        self.rows, seen = [], set()
        for p in paths:
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line in lines:
                try: d = json.loads(line)
                except Exception: continue
                q = str(d.get("question") or d.get("user") or d.get("input") or "").strip()
                a = str(d.get("answer") or d.get("assistant") or d.get("output") or "").strip()
                if not q or not a: continue
                key = (q.casefold(), a.casefold())
                if key in seen: continue
                seen.add(key); self.rows.append((q, a))
        self.index = [(q,a,_norm(q),_tokens(q),_grams(q)) for q,a in self.rows]

    def size(self): return len(self.rows)

    def answer(self, query, threshold=.70):
        qn, qt, qg = _norm(query), _tokens(query), _grams(query)
        if not qn: return None
        best = None
        for q,a,qn2,qt2,qg2 in self.index:
            if qn == qn2: return a
            token_score = len(qt & qt2) / max(1, len(qt | qt2))
            gram_score = len(qg & qg2) / max(1, len(qg | qg2))
            contains = .94 if len(qn) >= 5 and (qn in qn2 or qn2 in qn) else 0.0
            ratio = SequenceMatcher(None, qn, qn2).ratio()
            score = max(contains, .45*token_score + .35*gram_score + .20*ratio)
            # Natural Korean paraphrases such as "누구야" vs "뭐야" often have
            # low token overlap but high character similarity.
            if len(qn) < 6 and score < .94: continue
            if best is None or score > best[0]: best=(score,a)
        if not best: return None
        dynamic = .52 if len(qn) >= 7 else threshold
        return best[1] if best[0] >= dynamic else None
