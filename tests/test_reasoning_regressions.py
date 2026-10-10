import unittest

from dori_ai.response_engine import ResponseEngine


class FakeDialogue:
    def __init__(self, history):
        self.history = history


class NaturalLanguageRegressionTests(unittest.TestCase):
    def test_pythagorean_word_problem(self):
        answer = ResponseEngine._reasoning_answer(
            "직각삼각형의 두 직각변 길이가 3과 4라면 빗변의 길이는 얼마야? 계산 과정도 설명해 줘."
        )
        self.assertIsNotNone(answer)
        self.assertIn("3² + 4²", answer)
        self.assertIn("5", answer)

    def test_thunder_explanation(self):
        answer = ResponseEngine._reasoning_answer("번개가 친 다음 천둥소리가 들리는 이유는 뭐야?")
        self.assertIsNotNone(answer)
        self.assertIn("공기", answer)
        self.assertIn("빛은 소리보다", answer)

    def test_followup_resolves_seoul_from_previous_turn(self):
        dialogue = FakeDialogue([
            ("user", "대한민국의 수도는 어디야?"),
            ("dori", "대한민국의 수도는 서울이야."),
        ])
        answer = ResponseEngine._reasoning_answer("그럼 그 도시에서 가장 유명한 궁궐 중 하나는 뭐야?", dialogue)
        self.assertIsNotNone(answer)
        self.assertIn("경복궁", answer)

    def test_dori_description_distinguishes_plush_from_real_dog(self):
        answer = ResponseEngine._builtin_answer("돌이는 어떤 인형이고, 일반적인 웰시코기와 어떤 차이가 있어?")
        self.assertIsNotNone(answer)
        self.assertIn("인형", answer)
        self.assertIn("실제 강아지가 아니라", answer)
        self.assertIn("17cm", answer)


if __name__ == "__main__":
    unittest.main()
