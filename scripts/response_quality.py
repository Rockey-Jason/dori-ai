"""Offline rubric-based evaluation for Dori AI's generated responses.

This scorer is deliberately deterministic and free: no paid judge/API is used.
It checks explicit answer criteria, response integrity, language/script signals,
and format instructions. It is a useful regression gate, not a perfect semantic
or grammatical judge.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+|[0-9]+")
_NUMERIC_RE = re.compile(r"^[+-]?\\d+(?:[.]\\d+)?$")
_DIGITS_RE = re.compile(r"^\\d+$")
_TRANSLATION_SPECIALS = re.compile(r"<\\/?(?:system|assistant|user|dori|eos|bos|pad|unk)>", re.I)


def normalize_answer(text):
    text = str(text or "").casefold().translate(str.maketrans({
        "²": "^2", "³": "^3", "₀": "0", "₁": "1", "₂": "2",
        "₃": "3", "₄": "4", "π": "pi",
    }))
    return re.sub(r"[\\s\\W_]+", "", text, flags=re.UNICODE)


def _contains(answer, phrase):
    answer = str(answer or "")
    phrase = str(phrase or "")
    if not phrase.strip():
        return False
    # Avoid treating e.g. "14" as the answer "4", while allowing punctuation
    # around standalone Arabic numerals.
    if _DIGITS_RE.fullmatch(phrase.strip()):
        return re.search(r"(?<!\\d)" + re.escape(phrase.strip()) + r"(?!\\d)", answer) is not None
    target = normalize_answer(phrase)
    return bool(target) and target in normalize_answer(answer)


def _criteria_for(row):
    criteria = row.get("criteria")
    if isinstance(criteria, list) and criteria:
        normalized = []
        for item in criteria:
            if isinstance(item, str):
                item = {"any_of": [item]}
            if not isinstance(item, dict):
                raise ValueError(f"Invalid rubric item in {row.get('id', '<unknown>')}")
            aliases = item.get("any_of")
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list) or not any(str(x).strip() for x in aliases):
                raise ValueError(f"Rubric item lacks any_of aliases in {row.get('id', '<unknown>')}")
            weight = float(item.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"Rubric weight must be positive in {row.get('id', '<unknown>')}")
            normalized.append({"any_of": [str(x) for x in aliases if str(x).strip()], "weight": weight})
        return normalized
    keywords = row.get("keywords")
    if isinstance(keywords, list) and keywords:
        return [{"any_of": [str(x)], "weight": 1.0} for x in keywords if str(x).strip()]
    raise ValueError(f"Evaluation row {row.get('id', '<unknown>')} needs criteria or keywords")


def read_rows(paths):
    """Load held-out cases from JSONL files and reject duplicate or malformed IDs."""
    rows, seen_ids = [], set()
    for path in paths:
        path = Path(path)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            if not isinstance(row, dict) or not str(row.get("question", "")).strip():
                raise ValueError(f"Missing question in {path}:{line_no}")
            case_id = str(row.get("id", "")).strip()
            if not case_id or case_id in seen_ids:
                raise ValueError(f"Missing or duplicate evaluation id {case_id!r} in {path}:{line_no}")
            _criteria_for(row)
            if not row.get("category"):
                raise ValueError(f"Missing category for {case_id}")
            seen_ids.add(case_id)
            row["_source_file"] = path.as_posix()
            rows.append(row)
    if not rows:
        raise ValueError("Held-out response evaluation dataset is empty")
    return rows


def _rubric_score(answer, row):
    criteria = _criteria_for(row)
    checks = []
    total_weight = sum(item["weight"] for item in criteria)
    earned = 0.0
    for item in criteria:
        matched = [alias for alias in item["any_of"] if _contains(answer, alias)]
        hit = bool(matched)
        if hit:
            earned += item["weight"]
        checks.append({
            "passed": hit,
            "weight": item["weight"],
            "matched": matched,
            "alternatives": item["any_of"],
        })
    score = earned / total_weight if total_weight else 0.0
    forbidden = [str(x) for x in row.get("forbidden_any", []) if _contains(answer, str(x))]
    if forbidden:
        # A response containing an explicitly forbidden answer is not allowed
        # to score as correct just because it also repeats expected keywords.
        score = min(score, 0.20)
    return score, checks, forbidden


def _format_score(answer, row):
    fmt = row.get("answer_format")
    text = str(answer or "").strip()
    if not fmt:
        return 1.0, []
    if fmt == "single_number":
        ok = bool(_NUMERIC_RE.fullmatch(text))
        return (1.0 if ok else 0.0), ([] if ok else ["expected_single_number"])
    if fmt == "allowed_answer":
        allowed = {normalize_answer(x) for x in row.get("allowed_answers", [])}
        ok = normalize_answer(text) in allowed
        return (1.0 if ok else 0.0), ([] if ok else ["expected_one_allowed_answer"])
    if fmt == "short_translation":
        max_words = int(row.get("max_words", 8))
        words = _TOKEN_RE.findall(text)
        ok = bool(text) and len(words) <= max_words and not _TRANSLATION_SPECIALS.search(text)
        return (1.0 if ok else 0.0), ([] if ok else ["expected_short_translation"])
    return 0.0, [f"unknown_answer_format:{fmt}"]


def _language_integrity(answer, row):
    """Heuristic integrity/readability score; not a full grammar evaluator."""
    text = str(answer or "").strip()
    if not text:
        return 0.0, ["empty_answer"]
    if "�" in text or _TRANSLATION_SPECIALS.search(text):
        return 0.0, ["invalid_unicode_or_control_token"]

    fmt = row.get("answer_format")
    if fmt == "single_number" and _NUMERIC_RE.fullmatch(text):
        return 1.0, []
    if len(text) == 1 and text.isdigit():
        return 1.0, []

    try:
        from dori_ai.answer_quality import bad as malformed_response
        is_malformed = malformed_response(text)
    except Exception:
        is_malformed = False

    score = 1.0
    flags = []
    if is_malformed:
        score -= 0.55
        flags.append("malformed_or_repetitive_output")

    tokens = _TOKEN_RE.findall(text)
    if len(text) < 2:
        score -= 0.35
        flags.append("too_short")
    if len(tokens) >= 10:
        diversity = len({token.casefold() for token in tokens}) / len(tokens)
        if diversity < 0.35:
            score -= 0.35
            flags.append("low_token_diversity")
        elif diversity < 0.50:
            score -= 0.15
            flags.append("limited_token_diversity")

    language = str(row.get("language", "ko")).lower()
    hangul = len(re.findall(r"[가-힣]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if language == "ko" and len(tokens) >= 4 and hangul / max(1, hangul + latin) < 0.10:
        score -= 0.20
        flags.append("unexpected_script_for_korean")
    elif language == "en" and len(tokens) >= 4 and latin / max(1, hangul + latin) < 0.25:
        score -= 0.20
        flags.append("unexpected_script_for_english")

    max_words = row.get("max_words")
    if max_words is not None and len(tokens) > int(max_words):
        score -= 0.25
        flags.append("over_word_limit")
    return max(0.0, min(1.0, score)), flags


def grade_response(answer, row):
    """Grade one real model response against its explicit, inspectable rubric."""
    answer = str(answer or "").strip()
    rubric, checks, forbidden = _rubric_score(answer, row)
    language, language_flags = _language_integrity(answer, row)
    format_score, format_flags = _format_score(answer, row)
    quality = 0.80 * rubric + 0.20 * language
    if row.get("answer_format"):
        quality *= 0.65 + 0.35 * format_score
    if forbidden:
        quality = min(quality, 0.35)
    quality = max(0.0, min(1.0, quality))
    passed = rubric >= 0.60 and language >= 0.50 and format_score >= 0.99 and not forbidden
    return {
        "id": row["id"],
        "category": row.get("category", "unknown"),
        "quality_score": round(quality, 4),
        "rubric_score": round(rubric, 4),
        "language_score": round(language, 4),
        "format_score": round(format_score, 4),
        "passed": passed,
        "matched_criteria": sum(1 for item in checks if item["passed"]),
        "total_criteria": len(checks),
        "criteria": checks,
        "forbidden_matches": forbidden,
        "flags": language_flags + format_flags,
        "answer_preview": answer[:500],
    }


def summarize_grades(grades):
    if not grades:
        raise ValueError("Cannot summarize an empty response evaluation")
    categories = sorted({g["category"] for g in grades})
    category_scores, category_rubric_scores, category_language_scores, category_pass_rates = {}, {}, {}, {}
    for category in categories:
        items = [g for g in grades if g["category"] == category]
        category_scores[category] = round(sum(g["quality_score"] for g in items) / len(items), 4)
        category_rubric_scores[category] = round(sum(g["rubric_score"] for g in items) / len(items), 4)
        category_language_scores[category] = round(sum(g["language_score"] for g in items) / len(items), 4)
        category_pass_rates[category] = round(sum(bool(g["passed"]) for g in items) / len(items), 4)
    total = len(grades)
    return {
        "total": total,
        "passed": sum(bool(g["passed"]) for g in grades),
        "pass_rate": round(sum(bool(g["passed"]) for g in grades) / total, 4),
        "quality_score": round(sum(g["quality_score"] for g in grades) / total, 4),
        "rubric_score": round(sum(g["rubric_score"] for g in grades) / total, 4),
        "language_score": round(sum(g["language_score"] for g in grades) / total, 4),
        "format_score": round(sum(g["format_score"] for g in grades) / total, 4),
        "category_scores": category_scores,
        "category_rubric_scores": category_rubric_scores,
        "category_language_scores": category_language_scores,
        "category_pass_rates": category_pass_rates,
        "cases": grades,
    }
