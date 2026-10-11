"""Regression tests for Dori AI checkpoint compatibility and recovery."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train_final
from dori_ai.core.optimizer import Adam
from dori_ai.core.transformer import DoriTransformer


class CheckpointCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "vocab_size": 300,
            "max_len": 8,
            "dim": 8,
            "heads": 2,
            "ff_dim": 16,
            "layers": 1,
        }
        self.model = DoriTransformer(**self.config)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.path = self.root / "latest.npz"
        self.tokenizer_hash = "test-tokenizer-sha256"
        self.base_meta = {
            "tokenizer_sha256": self.tokenizer_hash,
            "epoch": 4,
            "val_loss": 2.5,
            "best_val_loss": 2.4,
        }

    def test_checkpoint_round_trip_and_optimizer_state(self):
        train_final.save_checkpoint(
            self.path, self.model, self.optimizer,
            {**self.base_meta, "model_config": self.config},
        )
        loaded, meta, reason = train_final.validate_checkpoint(
            self.path, self.config, self.tokenizer_hash
        )
        self.assertIsNone(reason)
        self.assertIsNotNone(loaded)
        self.assertEqual(meta["parameter_count"], len(self.model.parameters()))
        restored_optimizer = Adam(loaded.parameters(), lr=1e-3)
        self.assertTrue(train_final.load_optimizer(self.path, restored_optimizer))

    def test_stale_sidecar_cannot_hide_parameter_count_mismatch(self):
        # Reproduce the reported class of bug: metadata claims the current
        # config, while the NPZ contains a different number of model tensors.
        np.savez(self.path, p0=np.zeros((1,), dtype=np.float32))
        Path(str(self.path) + ".json").write_text(
            json.dumps({**self.base_meta, "model_config": self.config}),
            encoding="utf-8",
        )
        loaded, _meta, reason = train_final.validate_checkpoint(
            self.path, self.config, self.tokenizer_hash
        )
        self.assertIsNone(loaded)
        self.assertIn("parameter count differs", reason)

    def test_tokenizer_mismatch_rejects_checkpoint(self):
        train_final.save_checkpoint(
            self.path, self.model, self.optimizer,
            {**self.base_meta, "model_config": self.config},
        )
        loaded, _meta, reason = train_final.validate_checkpoint(
            self.path, self.config, "different-tokenizer"
        )
        self.assertIsNone(loaded)
        self.assertIn("tokenizer fingerprint differs", reason)

    def test_interrupted_weights_metadata_pair_is_rejected(self):
        train_final.save_checkpoint(
            self.path, self.model, self.optimizer,
            {**self.base_meta, "model_config": self.config},
        )
        sidecar = Path(str(self.path) + ".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["checkpoint_id"] = "stale-id"
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
        loaded, _meta, reason = train_final.validate_checkpoint(
            self.path, self.config, self.tokenizer_hash
        )
        self.assertIsNone(loaded)
        self.assertIn("checkpoint ID differs", reason)


if __name__ == "__main__":
    unittest.main()
