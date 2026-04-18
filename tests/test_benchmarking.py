from __future__ import annotations

from contextlib import nullcontext
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local test env
    torch = None

from src.utils.benchmark_decode import discover_checkpoints, generate_continuation_tokens, generate_middle_tokens
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
    class _CacheAwareToyModel:
        def __init__(self, *, first_next_id: int, cached_next_id: int, vocab_size: int):
            self.first_next_id = int(first_next_id)
            self.cached_next_id = int(cached_next_id)
            self.vocab_size = int(vocab_size)
            self.calls: list[dict[str, int | bool]] = []

        def __call__(self, *, input_ids, past_key_values=None, use_cache=None, return_dict=True):
            self.calls.append(
                {
                    "seq_len": int(input_ids.shape[1]),
                    "used_cache": bool(past_key_values is not None),
                    "use_cache": bool(use_cache),
                }
            )
            logits = torch.full(
                (1, int(input_ids.shape[1]), self.vocab_size),
                fill_value=-1000.0,
                dtype=torch.float32,
                device=input_ids.device,
            )
            next_id = self.cached_next_id if past_key_values is not None else self.first_next_id
            logits[0, -1, next_id] = 1000.0
            return SimpleNamespace(logits=logits, past_key_values=("cached",))

    class _ContinuationFSM:
        def __init__(self, *, bar_id: int, eos_id: int):
            self.bar_id = int(bar_id)
            self.eos_id = int(eos_id)

        def state_after_prefix_ids(self, prefix_ids):
            return "start"

        def allowed_token_ids(self, state):
            if state == "start":
                return [self.bar_id]
            if state == "after_bar":
                return [self.eos_id]
            return [self.eos_id]

        def transition(self, state, token_id):
            if state == "start" and int(token_id) == self.bar_id:
                return "after_bar"
            if state == "after_bar" and int(token_id) == self.eos_id:
                return "done"
            if state == "done" and int(token_id) == self.eos_id:
                return "done"
            return None

    class _InfillingFSM:
        def __init__(self, *, middle_id: int, eos_id: int):
            self.middle_id = int(middle_id)
            self.eos_id = int(eos_id)

        def state_after_prefix_tokens(self, prefix_tokens):
            return "start"

        def compatible_states_for_suffix_tokens(self, suffix_tokens):
            return {"after_middle"}

        def allowed_token_ids(self, state):
            if state == "start":
                return [self.middle_id]
            if state == "after_middle":
                return [self.eos_id]
            return []

        def transition(self, state, token_id):
            if state == "start" and int(token_id) == self.middle_id:
                return "after_middle"
            if state == "after_middle" and int(token_id) == self.eos_id:
                return "done"
            return None

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

    def test_generate_continuation_tokens_uses_cached_single_token_steps(self) -> None:
        if torch is None:
            self.skipTest("torch is required for benchmark decode tests")
        token_to_id = {"BOS": 0, "BAR": 1, "EOS": 2}
        id_to_token = ["BOS", "BAR", "EOS"]
        model = self._CacheAwareToyModel(first_next_id=token_to_id["BAR"], cached_next_id=token_to_id["EOS"], vocab_size=len(id_to_token))
        generated_tokens, reached_eos, stats = generate_continuation_tokens(
            model=model,
            torch_mod=torch,
            prompt_tokens=["BOS", "BAR"],
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            grammar_fsm=self._ContinuationFSM(bar_id=token_to_id["BAR"], eos_id=token_to_id["EOS"]),
            device=torch.device("cpu"),
            use_amp=False,
            amp_dtype=None,
            autocast_context_fn=lambda **kwargs: nullcontext(),
            max_positions=8,
            max_new_tokens=2,
        )

        self.assertEqual(generated_tokens, ["BAR"])
        self.assertTrue(reached_eos)
        self.assertEqual(model.calls[0]["seq_len"], 2)
        self.assertEqual(model.calls[1]["seq_len"], 1)
        self.assertFalse(bool(model.calls[0]["used_cache"]))
        self.assertTrue(bool(model.calls[1]["used_cache"]))
        self.assertEqual(int(stats["step_count"]), 2)

    def test_generate_middle_tokens_uses_cached_single_token_steps(self) -> None:
        if torch is None:
            self.skipTest("torch is required for benchmark decode tests")
        token_to_id = {"BOS": 0, "BAR": 1, "EOS": 2}
        id_to_token = ["BOS", "BAR", "EOS"]
        model = self._CacheAwareToyModel(first_next_id=token_to_id["BAR"], cached_next_id=token_to_id["EOS"], vocab_size=len(id_to_token))
        generated_tokens, reached_eos, stats = generate_middle_tokens(
            model=model,
            torch_mod=torch,
            prompt_tokens=["BOS", "BAR"],
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            grammar_fsm=self._InfillingFSM(middle_id=token_to_id["BAR"], eos_id=token_to_id["EOS"]),
            prefix_tokens=["BOS"],
            suffix_tokens=["EOS"],
            device=torch.device("cpu"),
            use_amp=False,
            amp_dtype=None,
            autocast_context_fn=lambda **kwargs: nullcontext(),
            max_positions=8,
            max_new_tokens=2,
        )

        self.assertEqual(generated_tokens, ["BAR"])
        self.assertTrue(reached_eos)
        self.assertEqual(model.calls[0]["seq_len"], 2)
        self.assertEqual(model.calls[1]["seq_len"], 1)
        self.assertFalse(bool(model.calls[0]["used_cache"]))
        self.assertTrue(bool(model.calls[1]["used_cache"]))
        self.assertEqual(int(stats["step_count"]), 2)


if __name__ == "__main__":
    unittest.main()
