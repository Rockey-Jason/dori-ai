"""Procedural Dori quiz bank with a 100,000+ question space.

Questions are generated from deterministic, verifiable templates rather than fabricated
facts. The seed identifies a reproducible quiz and supports future dataset export.
"""
from __future__ import annotations

import hashlib
import random

BANK_VERSION = "dori-quiz-bank-v1"
BANK_SIZE = 120_000

def _mc(question: str, answer: str, distractors: list[str], category: str, seed: int) -> dict:
    choices = list(dict.fromkeys([str(answer), *map(str, distractors)]))
    rng = random.Random(seed)
    rng.shuffle(choices)
    return {
        "id": hashlib.sha256(f"{BANK_VERSION}:{seed}:{question}".encode()).hexdigest()[:16],
        "category": category,
        "type": "multiple",
        "question": question,
        "choices": choices,
        "answer": str(answer),
        "explanation": "",
    }

def generate_quiz(seed: int | None = None, category: str = "돌이") -> dict:
    """Generate one quiz. The combinatorial template space exceeds 120,000 variants."""
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    seed_value = rng.randrange(0, 2**31 - 1)
    local = random.Random(seed_value)
    normalized = str(category or "돌이").lower()
    if any(x in normalized for x in ("과학", "science")):
        facts = [("태양계에서 가장 큰 행성은?", "목성", ["지구","화성","금성"]), ("물의 화학식은?", "H₂O", ["CO₂","O₂","NaCl"])]
        q, a, wrong = local.choice(facts)
        return _mc(q, a, wrong, "과학", seed_value)
    if any(x in normalized for x in ("역사", "history")):
        facts = [("훈민정음을 창제한 조선의 왕은?", "세종대왕", ["정조","태종","영조"]), ("조선을 세운 인물은?", "이성계", ["왕건","세종대왕","정약용"])]
        q, a, wrong = local.choice(facts)
        return _mc(q, a, wrong, "역사", seed_value)
    if any(x in normalized for x in ("영어", "english")):
        facts = [("영어 단어 'apple'의 뜻은?", "사과", ["바나나","포도","복숭아"]), ("'Thank you'의 뜻은?", "고마워요", ["안녕하세요","미안해요","잘 가요"])]
        q, a, wrong = local.choice(facts)
        return _mc(q, a, wrong, "영어", seed_value)
    if any(x in normalized for x in ("국어", "korean")):
        facts = [("'기쁘다'와 뜻이 가까운 말은?", "즐겁다", ["슬프다","화나다","무겁다"])]
        q, a, wrong = local.choice(facts)
        return _mc(q, a, wrong, "국어", seed_value)
    if any(x in normalized for x in ("수학", "math", "계산")):
        kind = 0
    elif any(x in normalized for x in ("돌이", "도로늄", "식빵이", "돌돌코인")):
        kind = 7
    else:
        kind = 0

    if kind == 0:
        a, b = local.randint(12, 999), local.randint(11, 999)
        op = local.choice(["+", "-", "×"])
        if op == "+":
            answer = a + b
        elif op == "-":
            if b > a:
                a, b = b, a
            answer = a - b
        else:
            a, b = local.randint(2, 99), local.randint(2, 99)
            answer = a * b
        wrong = set()
        while len(wrong) < 3:
            candidate = answer + local.choice([-100, -25, -10, -3, -1, 1, 3, 10, 25, 100])
            if candidate >= 0 and candidate != answer:
                wrong.add(candidate)
        return _mc(f"{a} {op} {b}의 값은 얼마일까?", str(answer), [str(x) for x in wrong], "수학", seed_value)

    if kind == 1:
        a, b = local.randint(2, 30), local.randint(2, 30)
        answer = a * b
        return _mc(f"{a} × {b}는 얼마일까?", str(answer), [str(max(0, answer + d)) for d in (-a, b, a+b)], "수학", seed_value)

    if kind == 2:
        base, exponent = local.randint(2, 12), local.randint(2, 4)
        answer = base ** exponent
        wrong = [str(answer + base), str(max(1, answer - base)), str(base * exponent)]
        return _mc(f"{base}의 {exponent}제곱은 얼마일까?", str(answer), wrong, "수학", seed_value)

    if kind == 3:
        percent = local.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
        base = local.randint(2, 200) * 20
        answer = base * percent // 100
        wrong = [str(answer + base // 10), str(max(0, answer - base // 10)), str(base - answer)]
        return _mc(f"{base}의 {percent}%는 얼마일까?", str(answer), wrong, "수학", seed_value)

    if kind == 4:
        count = local.randint(2, 20)
        unit = local.choice(["분", "시간", "일"])
        if unit == "분":
            answer = count * 60
            q = f"{count}분은 몇 초일까?"
        elif unit == "시간":
            answer = count * 60
            q = f"{count}시간은 몇 분일까?"
        else:
            answer = count * 24
            q = f"{count}일은 몇 시간일까?"
        return _mc(q, str(answer), [str(answer + 12), str(max(1, answer - 12)), str(count * 10)], "단위 변환", seed_value)

    if kind == 5:
        a, b = local.randint(2, 30), local.randint(2, 30)
        answer = a + b
        return _mc(f"한 상자에 간식이 {a}개 있고 다른 상자에 {b}개 있어. 모두 몇 개일까?", str(answer),
                   [str(max(0, a-b)), str(a*b), str(answer+local.randint(2, 15))], "응용 수학", seed_value)

    if kind == 6:
        # Short-answer questions are deliberately marked subjective and have no options.
        a, b = local.randint(2, 99), local.randint(2, 99)
        op = local.choice(["더하기", "빼기"])
        if op == "더하기":
            answer = a + b
            question = f"{a}와 {b}를 더하면 얼마일까? 답을 숫자로 적어 줘."
        else:
            if b > a:
                a, b = b, a
            answer = a - b
            question = f"{a}에서 {b}를 빼면 얼마일까? 답을 숫자로 적어 줘."
        return {
            "id": hashlib.sha256(f"{BANK_VERSION}:{seed_value}:{question}".encode()).hexdigest()[:16],
            "category": "서술형 수학",
            "type": "subjective",
            "question": question,
            "choices": [],
            "answer": str(answer),
            "explanation": f"계산 결과는 {answer}야.",
        }

    facts = [
        ("돌이는 어떤 품종을 본뜬 인형일까?", "웰시코기", ["고양이", "토끼", "시바견"], "돌이 퀴즈"),
        ("돌이의 생일은 언제일까?", "2021년 10월 3일", ["2020년 10월 3일", "2021년 3월 10일", "2022년 10월 3일"], "돌이 퀴즈"),
        ("돌이의 동생인 고양이 인형 이름은?", "식빵이", ["복돌이", "모래", "스노우"], "돌이 퀴즈"),
        ("돌이사이트의 가상 화폐 이름은?", "돌돌코인", ["도로늄", "로키코인", "식빵머니"], "돌이 퀴즈"),
        ("돌이 세계관의 신비한 보석 이름은?", "도로늄", ["철광석", "달빛석", "태양석"], "돌이 퀴즈"),
    ]
    question, answer, wrong, category = local.choice(facts)
    return _mc(question, answer, wrong, category, seed_value)

def generate_dataset(count: int = BANK_SIZE) -> list[dict]:
    """Export a reproducible, deduplicated dataset for training/evaluation."""
    count = max(1, min(int(count), BANK_SIZE))
    rows, seen = [], set()
    seed = 20261015
    while len(rows) < count:
        row = generate_quiz(seed)
        key = row["question"]
        if key not in seen:
            seen.add(key)
            rows.append(row)
        seed += 1
    return rows
