"""Deterministic response-quality checks and cleanup."""
import re

def clean(text):
    t=str(text or "").replace("\x00", "").strip()
    t=re.sub(r"<\/?(?:system|user|dori|eos)>", "", t, flags=re.I).strip()
    t=re.sub(r"\n{4,}", "\n\n", t)
    return t

def bad(text):
    t=clean(text)
    if len(t)<2 or len(t)>12000: return True
    if re.search(r"(.)\1{8,}", t): return True
    words=re.findall(r"[가-힣A-Za-z0-9]+",t)
    if len(words)>=12 and len(set(words))/len(words)<.45: return True
    if "NaN" in t or "inf" in t.lower(): return True
    return False

def confidence(text):
    t=clean(text)
    if bad(t): return 0.0
    score=.55
    if len(t)>=20: score += .10
    if any(x in t for x in ("모르", "확인할 수", "자료가 부족")): score -= .15
    if re.search(r"[.!?。！？]",t): score += .05
    return max(0.0,min(1.0,score))
