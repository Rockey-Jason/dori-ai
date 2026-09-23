"""Deterministic local knowledge retrieval for factual Dori answers."""
import json
import re
from pathlib import Path

def _norm(s):
    s = str(s).lower()
    s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
    return s

def _tokens(s):
    return set(re.findall(r"[가-힣A-Za-z0-9]+", str(s).lower()))

def _grams(s, n=2):
    s = _norm(s)
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))} if s else set()

class LocalKnowledge:
    def __init__(self, paths=None):
        root = Path(__file__).resolve().parent.parent / "data"
        if paths is None:
            candidates = [
                root / "dori_knowledge.jsonl",
                root / "dori_knowledge_expanded.jsonl",
                root / "dori_knowledge_v23.jsonl",
                root / "instructions" / "train.jsonl",
            ]
            paths = [p for p in candidates if p.exists()]
        self.rows = []
        seen = set()
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                q = str(d.get("question") or d.get("user") or d.get("input") or "").strip()
                a = str(d.get("answer") or d.get("assistant") or d.get("output") or "").strip()
                if not q or not a:
                    continue
                key = (q.casefold(), a.casefold())
                if key in seen:
                    continue
                seen.add(key)
                self.rows.append((q, a))
        self.index = [(q, a, _norm(q), _tokens(q), _grams(q)) for q, a in self.rows]

    def size(self):
        return len(self.rows)

    def answer(self, query, threshold=0.66):
        qn = _norm(query)
        qt = _tokens(query)
        qg = _grams(query)
        if not qn:
            return None

        best = None
        for q, a, q_norm, qt2, qg2 in self.index:
            if qn == q_norm:
                return a

            token_score = len(qt & qt2) / max(1, len(qt | qt2))
            gram_score = len(qg & qg2) / max(1, len(qg | qg2))
            contains = 0.90 if (len(qn) >= 4 and (qn in q_norm or q_norm in qn)) else 0.0
            score = max(contains, 0.55 * token_score + 0.45 * gram_score)

            # Short queries are dangerous: never answer from a weak fuzzy match.
            if len(qn) < 5 and score < 0.90:
                continue
            if best is None or score > best[0]:
                best = (score, a)

        return best[1] if best and best[0] >= threshold else None
