"""Lightweight multilingual detection and normalization."""
import re
LANG_PATTERNS = {
    "ko": r"[가-힣]", "en": r"[A-Za-z]", "ja": r"[ぁ-ゖァ-ヺ一-龯]", "zh": r"[\u4e00-\u9fff]",
    "es": r"[áéíóúüñ¿¡]", "fr": r"[àâçéèêëîïôûùüÿœæ]", "de": r"[äöüß]", "ru": r"[А-Яа-яЁё]",
    "pt": r"[ãõáàâçéêíóôúü]", "it": r"[àèéìíîòóù]"
}
def detect(text):
    s = str(text)
    scores = {k: len(re.findall(p, s)) for k, p in LANG_PATTERNS.items()}
    # A lone Latin variable such as x in "2x+3=11" is mathematics, not English.
    if scores["en"] <= 1 and not re.search(r"[가-힣ぁ-ゖァ-ヺ一-龯А-Яа-яЁё]", s):
        scores["en"] = 0
    best = max(scores, key=scores.get)
    return best if scores[best] else "unknown"
def normalize_query(text):
    return re.sub(r"\s+", " ", str(text).strip())
def multilingual_aliases(term):
    return {
        "dori": ["돌이", "Dori", "ドリ", "朵莉", "ドーリ"],
        "newspaper": ["신문", "newspaper", "news", "新聞", "报纸"],
        "quiz": ["퀴즈", "quiz", "クイズ", "测验", "cuestionario"],
        "stock": ["주식", "stock", "stocks", "株", "股票", "acción", "ações"],
        "hello": ["안녕", "hello", "hi", "hola", "bonjour", "hallo", "ciao", "こんにちは", "你好"],
    }.get(term, [])
