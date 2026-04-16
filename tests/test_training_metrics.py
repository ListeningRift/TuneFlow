from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.training_metrics import (
    load_training_metrics,
    prefilter_checkpoints_by_valid_loss,
    summarize_training_metrics,
    training_metrics_for_step,
)


class TrainingMetricsTests(unittest.TestCase):
    def test_load_training_metrics_derives_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "metrics.jsonl"
            rows = [
                {
                    "event": "run_start",
                    "effective_batch": 2,
                    "seq_len": 16,
                },
                {
                    "event": "train",
                    "step": 1,
                    "loss": 5.0,
                },
                {
                    "event": "train",
                    "step": 2,
                    "loss": 3.0,
                },
                {
                    "event": "eval",
                    "step": 2,
                    "valid_loss": 2.5,
                },
            ]
            metrics_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            payload = load_training_metrics(metrics_path)
            aligned = training_metrics_for_step(payload, 2)

            self.assertEqual(aligned["valid_loss_from_training"], 2.5)
            self.assertEqual(aligned["tokens_seen"], 64)
            self.assertAlmostEqual(float(aligned["train_loss_ema"]), 4.8)
            self.assertAlmostEqual(float(aligned["best_valid_loss_so_far"]), 2.5)
            self.assertAlmostEqual(float(aligned["overfit_gap"]), -2.3)

            summary = summarize_training_metrics(payload)
            self.assertEqual(summary["train_event_count"], 2)
            self.assertEqual(summary["eval_event_count"], 1)
            self.assertEqual(summary["best_valid_step"], 2)
            self.assertEqual(summary["plateau_eval_streak"], 0)
            self.assertAlmostEqual(float(summary["best_valid_loss"]), 2.5)

    def test_prefilter_checkpoints_by_valid_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "metrics.jsonl"
            rows = [
                {"event": "run_start", "effective_batch": 2, "seq_len": 16},
                {"event": "eval", "step": 250, "valid_loss": 1.4},
                {"event": "eval", "step": 500, "valid_loss": 1.1},
                {"event": "eval", "step": 750, "valid_loss": 1.3},
            ]
            metrics_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            payload = load_training_metrics(metrics_path)

            checkpoint_paths = [
                tmp_path / "step_250.pt",
                tmp_path / "step_500.pt",
                tmp_path / "step_750.pt",
                tmp_path / "best.pt",
            ]
            selected, meta = prefilter_checkpoints_by_valid_loss(
                checkpoint_paths,
                payload,
                top_k=3,
                preserve_earliest=1,
            )

            self.assertEqual([path.name for path in selected], ["step_250.pt", "step_500.pt", "step_750.pt"])
            self.assertEqual(meta["selected_count"], 3)
            self.assertEqual(meta["used_valid_loss_count"], 3)
            self.assertEqual(meta["preserved_earliest_count"], 1)


if __name__ == "__main__":
    unittest.main()
