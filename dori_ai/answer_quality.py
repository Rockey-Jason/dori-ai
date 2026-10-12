"""Deterministic response-quality filters for malformed generated text.

These checks cannot prove that a factual claim is true. They reject common
corruption/repetition patterns so the response engine can use its safe fallback.
"""
import re

_SPECIAL = re.compile(r"<\/?(?:system|assistant|user|dori|eos|bos|pad|unk)>", re.I)
_WORDS = re.compile(r"[가-힣A-Za-z0-9]+")
_NUMERIC_NONFINITE = re.compile(r"(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])", re.I)


def clean(text):
    t = str(text or "").replace("\x00", "").replace("\ufffd", "").strip()
    t = _SPECIAL.sub("", t)
    t = re.sub(r"\n{4,}", "\n\n", t)
    t = re.sub(r"[ \t]{3,}", "  ", t)
    return t.strip()


def _has_repeated_phrase(words):
    """Detect short token loops without rejecting ordinary repeated words."""
    n = len(words)
    # Look for a repeated 2–6 word sequence occurring at least three times
    # consecutively, a common autoregressive decoding failure.
    for width in range(2, 7):
        for start in range(0, n - width * 3 + 1):
            phrase = [w.casefold() for w in words[start:start + width]]
            if all(
                [w.casefold() for w in words[start + width * k:start + width * (k + 1)]] == phrase
                for k in (1, 2)
            ):
                return True
    return False


def bad(text):
    t = clean(text)
    if len(t) < 2 or len(t) > 12000:
        return True
    if re.search(r"(.)\1{8,}", t):
        return True
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in t):
        return True

    # Reject a Hangul syllable fused to an unrelated long Latin token, a
    # known byte-BPE failure mode (e.g. "룼formntrailing").
    if re.search(r"[가-힣][A-Za-z]{5,}", t):
        return True
    if re.search(r"[가-힣][a-z]{8,}", t):
        return True

    words = _WORDS.findall(t)
    symbols = len(re.findall(r"[^가-힣A-Za-z0-9\s]", t))
    if len(words) >= 8 and symbols > len(words) * 2.5:
        return True
    if len(words) >= 12 and len(set(x.casefold() for x in words)) / len(words) < .42:
        return True

    short_words = sum(1 for x in words if len(x) <= 2)
    has_long_latin = any(len(x) >= 14 and re.search(r"[A-Za-z]", x) for x in words)
    if len(words) >= 8 and short_words / len(words) >= .45 and has_long_latin:
        return True
    if len(words) >= 9 and _has_repeated_phrase(words):
        return True

    # Only reject non-finite numeric tokens, not normal words such as
    # "information" that happen to contain the letters "inf".
    if _NUMERIC_NONFINITE.search(t):
        return True
    return False


def confidence(text):
    t = clean(text)
    if bad(t):
        return 0.0
    score = .55
    if len(t) >= 20:
        score += .10
    if re.search(r"[.!?。！？]", t):
        score += .05
    return max(0.0, min(1.0, score))
