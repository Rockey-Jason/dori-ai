import unittest

from scripts.evaluate_and_gate import should_promote
from scripts.response_quality import grade_response, normalize_answer, summarize_grades


def report(quality, rubric, pass_rate, category_scores, cases):
    return {
        "quality_score": quality,
        "rubric_score": rubric,
        "pass_rate": pass_rate,
        "category_scores": category_scores,
        "cases": cases,
    }


class CheckpointPromotionGateTests(unittest.TestCase):
    def test_legacy_strict_improvement_is_required(self):
        self.assertTrue(should_promote({"rate": 0.60}, {"rate": 0.65}))
        self.assertFalse(should_promote({"rate": 0.60}, {"rate": 0.60}))
        self.assertFalse(should_promote({"rate": 0.60}, {"rate": 0.55}))

    def test_meaningful_paired_quality_improvement_promotes(self):
        baseline_cases = [{"id": str(i), "quality_score": 0.40} for i in range(20)]
        candidate_cases = [
            {"id": str(i), "quality_score": 0.65 if i < 5 else 0.40}
            for i in range(20)
        ]
        baseline = report(0.40, 0.45, 0.30, {
            "reasoning": 0.4, "safety": 0.4, "language": 0.4,
            "instruction_following": 0.4, "factual": 0.4, "dori_lore": 0.4,
        }, baseline_cases)
        candidate = report(0.4625, 0.48, 0.30, {
            "reasoning": 0.45, "safety": 0.4, "language": 0.42,
            "instruction_following": 0.42, "factual": 0.45, "dori_lore": 0.4,
        }, candidate_cases)
        self.assertTrue(should_promote(baseline, candidate))

    def test_style_gain_cannot_hide_accuracy_regression(self):
        baseline = report(0.5, 0.7, 0.5, {"reasoning": 0.5, "safety": 0.5}, [
            {"id": "1", "quality_score": 0.5}, {"id": "2", "quality_score": 0.5}
        ])
        candidate = report(0.55, 0.68, 0.5, {"reasoning": 0.5, "safety": 0.5}, [
            {"id": "1", "quality_score": 0.7}, {"id": "2", "quality_score": 0.4}
        ])
        self.assertFalse(should_promote(baseline, candidate))

    def test_important_category_regression_blocks_promotion(self):
        baseline = report(0.5, 0.6, 0.5, {"reasoning": 0.5, "safety": 0.6}, [
            {"id": str(i), "quality_score": 0.5} for i in range(20)
        ])
        candidate = report(0.55, 0.61, 0.5, {"reasoning": 0.55, "safety": 0.4}, [
            {"id": str(i), "quality_score": 0.75 if i < 5 else 0.5}
            for i in range(20)
        ])
        self.assertFalse(should_promote(baseline, candidate))

    def test_too_small_gain_is_not_enough(self):
        baseline = report(0.5, 0.6, 0.5, {}, [
            {"id": str(i), "quality_score": 0.5} for i in range(20)
        ])
        candidate = report(0.505, 0.61, 0.5, {}, [
            {"id": str(i), "quality_score": 0.55 if i == 0 else 0.5}
            for i in range(20)
        ])
        self.assertFalse(should_promote(baseline, candidate))


class ResponseQualityTests(unittest.TestCase):
    def test_numeric_answer_is_correct_and_follows_format(self):
        row = {
            "id": "number", "category": "instruction_following", "language": "ko",
            "question": "7+8 숫자만 답해", "criteria": [{"any_of": ["15"]}],
            "answer_format": "single_number",
        }
        grade = grade_response("15", row)
        self.assertEqual(grade["rubric_score"], 1.0)
        self.assertEqual(grade["format_score"], 1.0)
        self.assertTrue(grade["passed"])

    def test_wrong_numeric_answer_fails(self):
        row = {
            "id": "number", "category": "instruction_following", "language": "ko",
            "question": "7+8 숫자만 답해", "criteria": [{"any_of": ["15"]}],
            "answer_format": "single_number",
        }
        grade = grade_response("16", row)
        self.assertEqual(grade["rubric_score"], 0.0)
        self.assertEqual(grade["format_score"], 1.0)
        self.assertFalse(grade["passed"])

    def test_numeric_criteria_do_not_match_inside_larger_number(self):
        row = {
            "id": "numeric-boundary", "category": "reasoning", "language": "ko",
            "question": "답은 4야?", "criteria": [{"any_of": ["4"]}],
        }
        self.assertEqual(grade_response("14", row)["rubric_score"], 0.0)

    def test_weighted_rubric_awards_partial_credit(self):
        row = {
            "id": "weighted", "category": "reasoning", "language": "ko",
            "question": "질문", "criteria": [
                {"any_of": ["4"], "weight": 2},
                {"any_of": ["3+2-1", "계산"], "weight": 1},
            ],
        }
        grade = grade_response("답은 4야.", row)
        self.assertAlmostEqual(grade["rubric_score"], 2 / 3, places=3)

    def test_forbidden_answer_caps_score(self):
        row = {
            "id": "forbidden", "category": "reasoning", "language": "ko",
            "question": "질문", "criteria": [{"any_of": ["4"]}],
            "forbidden_any": ["정답은 5"],
        }
        grade = grade_response("정답은 4가 아니라 정답은 5야.", row)
        self.assertLessEqual(grade["quality_score"], 0.35)
        self.assertEqual(grade["forbidden_matches"], ["정답은 5"])

    def test_repetitive_gibberish_is_penalized(self):
        row = {
            "id": "gibberish", "category": "language", "language": "ko",
            "question": "설명해", "criteria": [{"any_of": ["정답"]}],
        }
        grade = grade_response("정답 정답 정답 정답 정답 정답 정답", row)
        self.assertLess(grade["language_score"], 0.6)
        self.assertIn("malformed_or_repetitive_output", grade["flags"])

    def test_empty_answer_scores_zero(self):
        row = {
            "id": "empty", "category": "factual", "language": "ko",
            "question": "질문", "criteria": [{"any_of": ["서울"]}],
        }
        grade = grade_response("", row)
        self.assertEqual(grade["quality_score"], 0.0)
        self.assertFalse(grade["passed"])

    def test_summary_has_category_breakdown(self):
        row = {
            "id": "one", "category": "reasoning", "language": "ko",
            "question": "답?", "criteria": [{"any_of": ["정답"]}],
        }
        summary = summarize_grades([grade_response("정답입니다.", row)])
        self.assertEqual(summary["total"], 1)
        self.assertIn("reasoning", summary["category_scores"])

    def test_normalize_answer_handles_math_symbols(self):
        self.assertEqual(normalize_answer("πr²"), "pir2")


if __name__ == "__main__":
    unittest.main()
