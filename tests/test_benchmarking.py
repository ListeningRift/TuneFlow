from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.utils.benchmark_decode import discover_checkpoints
from src.utils.benchmarking import analyze_token_sequence, build_benchmark_manifest


def _long_sequence() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_60",
        "DUR_4",
        "VEL_8",
        "POS_4",
        "INST_PIANO",
        "PITCH_64",
        "DUR_4",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_67",
        "DUR_4",
        "VEL_8",
        "POS_8",
        "INST_PIANO",
        "PITCH_69",
        "DUR_4",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_72",
        "DUR_4",
        "VEL_8",
        "POS_12",
        "INST_PIANO",
        "PITCH_74",
        "DUR_4",
        "VEL_8",
        "BAR",
        "POS_0",
        "INST_PIANO",
        "PITCH_76",
        "DUR_4",
        "VEL_8",
        "POS_16",
        "INST_PIANO",
        "PITCH_77",
        "DUR_4",
        "VEL_8",
        "EOS",
    ]


class BenchmarkingTests(unittest.TestCase):
    def test_analyze_token_sequence_detects_empty_bars_and_time_order(self) -> None:
        payload = analyze_token_sequence(
            [
                "BOS",
                "BAR",
                "POS_8",
                "INST_PIANO",
                "PITCH_60",
                "DUR_4",
                "VEL_8",
                "POS_4",
                "INST_PIANO",
                "PITCH_62",
                "DUR_4",
                "VEL_8",
                "BAR",
                "BAR",
                "EOS",
            ]
        )
        self.assertFalse(payload["time_order_valid"])
        self.assertEqual(payload["empty_bar_count"], 2)
        self.assertTrue(payload["has_multi_empty_bar_run"])

    def test_build_benchmark_manifest_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eval_jsonl_path = tmp_path / "eval.jsonl"
            eval_tok_path = tmp_path / "eval.tok"
            rows = []
            token_lines = []
            for index in range(8):
                rows.append(
                    {
                        "artist": f"Artist {index}",
                        "title": f"Title {index}",
                        "family_key": f"family::{index}",
                        "midi_path": f"path/{index}.mid",
                        "note_count": 100 + (index * 10),
                        "duration_sec": 120.0 + float(index),
                    }
                )
                token_lines.append(" ".join(_long_sequence()))
            eval_jsonl_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            eval_tok_path.write_text("\n".join(token_lines) + "\n", encoding="utf-8")

            config = {
                "tier": "fast",
                "seed": 42,
                "sample_count": 4,
                "per_bucket_limit": 2,
                "min_prefix_tokens": 8,
                "continuation_prefix_ratio_min": 0.35,
                "continuation_prefix_ratio_max": 0.70,
                "infilling_hole_ratio_min": 0.10,
                "infilling_hole_ratio_max": 0.25,
            }
            first = build_benchmark_manifest(
                eval_jsonl_path=eval_jsonl_path,
                eval_tok_path=eval_tok_path,
                config=config,
                max_positions=64,
            )
            second = build_benchmark_manifest(
                eval_jsonl_path=eval_jsonl_path,
                eval_tok_path=eval_tok_path,
                config=config,
                max_positions=64,
            )
            self.assertEqual(first, second)

    def test_discover_checkpoints_ignores_aliases_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir)
            for name in ("step_250.pt", "step_500.pt", "best.pt", "last.pt", "latest.pt"):
                (checkpoint_dir / name).write_text("stub", encoding="utf-8")

            default_paths = discover_checkpoints(
                checkpoint_dir=checkpoint_dir,
                limit=None,
                policy="all",
                sample_count=6,
            )
            alias_paths = discover_checkpoints(
                checkpoint_dir=checkpoint_dir,
                limit=None,
                policy="all",
                sample_count=6,
                include_aliases=True,
            )

            self.assertEqual([path.name for path in default_paths], ["step_250.pt", "step_500.pt"])
            self.assertEqual(
                [path.name for path in alias_paths],
                ["step_250.pt", "step_500.pt", "best.pt", "last.pt", "latest.pt"],
            )


if __name__ == "__main__":
    unittest.main()
