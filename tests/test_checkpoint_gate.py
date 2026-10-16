import unittest

from scripts.evaluate_and_gate import should_promote


class CheckpointPromotionGateTests(unittest.TestCase):
    def test_strict_heldout_improvement_is_required(self):
        baseline = {"rate": 0.60}
        self.assertTrue(should_promote(baseline, {"rate": 0.65}))

    def test_tie_or_regression_keeps_accepted_checkpoint(self):
        baseline = {"rate": 0.60}
        self.assertFalse(should_promote(baseline, {"rate": 0.60}))
        self.assertFalse(should_promote(baseline, {"rate": 0.55}))


if __name__ == "__main__":
    unittest.main()
