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

    # Byte-BPE generation can concatenate a partial Hangul token with an
    # unrelated Latin fragment (for example, "룼formntrailing").
    if re.search(r"[가-힣][A-Za-z]{4,}|[A-Za-z]{4,}[가-힣]", t):
        return True

    chunks = re.findall(r"[가-힣A-Za-z0-9]+", t)
    symbols = len(re.findall(r"[^가-힣A-Za-z0-9\s]", t))
    if len(chunks) >= 8 and symbols > len(chunks) * 2.5:
        return True
    if len(chunks) >= 12 and len(set(x.casefold() for x in chunks)) / len(chunks) < .42:
        return True

    # Reject likely token-fragment gibberish rather than exposing it to users.
    short_chunks = sum(1 for x in chunks if len(x) <= 2)
    has_long_latin = any(len(x) >= 14 and re.search(r"[A-Za-z]", x) for x in chunks)
    if len(chunks) >= 8 and short_chunks / len(chunks) >= .45 and has_long_latin:
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
