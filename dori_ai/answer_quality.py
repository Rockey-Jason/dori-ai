"""Deterministic response safety/quality filters."""
import re

_SPECIAL = re.compile(r"<\/?(?:system|assistant|user|dori|eos|bos|pad|unk)>", re.I)

def clean(text):
    t = str(text or "").replace("\x00", "").strip()
    t = _SPECIAL.sub("", t)
    t = re.sub(r"\n{4,}", "\n\n", t)
    t = re.sub(r"[ \t]{3,}", "  ", t)
    return t.strip()

def bad(text):
    t = clean(text)
    if len(t) < 2 or len(t) > 12000:
        return True
    if re.search(r"(.)\1{8,}", t):
        return True
    # A generated answer dominated by punctuation/number fragments is not useful.
    chunks = re.findall(r"[가-힣A-Za-z0-9]+", t)
    symbols = len(re.findall(r"[^가-힣A-Za-z0-9\s]", t))
    if len(chunks) >= 8 and symbols > len(chunks) * 2.5:
        return True
    if len(chunks) >= 12 and len(set(x.casefold() for x in chunks)) / len(chunks) < .42:
        return True
    if "NaN" in t or "inf" in t.lower():
        return True
    return False

def confidence(text):
    t = clean(text)
    if bad(t):
        return 0.0
    score = .55
    if len(t) >= 20: score += .10
    if re.search(r"[.!?。！？]", t): score += .05
    return max(0.0, min(1.0, score))
