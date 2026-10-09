import unittest

from dori_ai.answer_quality import bad, clean
from dori_ai.response_engine import ResponseEngine


class AnswerQualityTests(unittest.TestCase):
    def test_corrupted_korean_latin_fragment_is_rejected(self):
        sample = "줸 중보룼formntrailing 검 결이터 학습 수 중복 중를"
        self.assertTrue(bad(sample))

    def test_normal_korean_answer_is_accepted(self):
        sample = "피타고라스는 고대 그리스의 철학자이자 수학자야."
        self.assertFalse(bad(sample))
        self.assertEqual(clean(sample), sample)


class BuiltinFactTests(unittest.TestCase):
    def test_pythagoras_identity(self):
        answer = ResponseEngine._builtin_answer("피타고라스가 누구야?")
        self.assertIsNotNone(answer)
        self.assertIn("고대 그리스", answer)
        self.assertIn("수학자", answer)

    def test_pythagorean_theorem(self):
        answer = ResponseEngine._builtin_answer("피타고라스 정리가 뭐야?")
        self.assertIsNotNone(answer)
        self.assertIn("a² + b² = c²", answer)

    def test_dori_identity_short_question(self):
        answer = ResponseEngine._builtin_answer("돌이는?")
        self.assertIsNotNone(answer)
        self.assertIn("미요니", answer)
        self.assertIn("인형", answer)

    def test_capital_of_south_korea(self):
        answer = ResponseEngine._builtin_answer("대한민국의 수도는 어디야?")
        self.assertEqual(answer, "대한민국의 수도는 서울이야. 🇰🇷")


if __name__ == "__main__":
    unittest.main()
