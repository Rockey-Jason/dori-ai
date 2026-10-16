import json
import tempfile
import unittest
from pathlib import Path

from scripts.growth_status import build_report, source_summary


class GrowthStatusTests(unittest.TestCase):
    def test_source_fingerprint_changes_when_training_data_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "new").mkdir(parents=True)
            sample = root / "data" / "new" / "sample.jsonl"
            sample.write_text('{"question":"Q","answer":"A"}\\n', encoding="utf-8")
            first = source_summary(root)["fingerprint"]
            sample.write_text('{"question":"Q","answer":"B"}\\n', encoding="utf-8")
            second = source_summary(root)["fingerprint"]
            self.assertNotEqual(first, second)

    def test_report_handles_missing_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(Path(tmp))
            self.assertEqual(report["mode"], "free_event_driven_growth")
            self.assertFalse(report["accepted_checkpoint"]["exists"])
            self.assertFalse(report["last_gate"]["exists"])
            self.assertTrue(report["free_resource_policy"]["training_is_event_driven"])

    def test_report_reads_cycle_and_gate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "runtime").mkdir()
            (root / "checkpoints").mkdir()
            (root / "data").mkdir()
            (root / "runtime" / "training_round.json").write_text(json.dumps({
                "round": 20, "target_rounds_per_dataset": 1200,
                "cumulative_round": 55, "source_fingerprint": "not-current"
            }), encoding="utf-8")
            (root / "runtime" / "heldout_gate_report.json").write_text(json.dumps({
                "promoted_candidate": False,
                "baseline": {"rate": 0.1}, "candidate": {"rate": 0.1}
            }), encoding="utf-8")
            report = build_report(root)
            self.assertEqual(report["cycle"]["round"], 20)
            self.assertEqual(report["cycle"]["remaining_rounds"], 1180)
            self.assertEqual(report["cycle"]["cumulative_round"], 55)
            self.assertFalse(report["cycle"]["state_fingerprint_matches_current_sources"])
            self.assertFalse(report["last_gate"]["promoted_candidate"])


if __name__ == "__main__":
    unittest.main()
