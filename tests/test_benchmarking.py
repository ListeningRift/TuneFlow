from __future__ import annotations

from contextlib import nullcontext
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - 取决于本地测试环境
    torch = None

from src.decoding import TuneFlowGrammarFSM
from src.utils.benchmark_decode import discover_checkpoints, generate_continuation_tokens, generate_middle_tokens
from src.utils.benchmarking import (
    _extract_first_unit,
    _infilling_boundary_time_order_stats,
    analyze_token_sequence,
    build_benchmark_manifest,
    enrich_infilling_consistency_record,
    enrich_local_development_record,
    enrich_long_context_record,
    enrich_structure_control_record,
)


def _long_sequence() -> list[str]:
    return [
        "BOS",
        "TEMPO_120",
        "KEY_UNCERTAIN",
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


def _sequence_with_pitches(pitches: list[int]) -> list[str]:
    tokens = ["BOS", "BAR"]
    for index, pitch in enumerate(pitches):
        tokens.extend(
            [
                f"POS_{index * 4}",
                "INST_PIANO",
                f"PITCH_{int(pitch)}",
                "DUR_4",
                "VEL_8",
            ]
        )
    tokens.append("EOS")
    return tokens


def _continuation_trace_record(
    *,
    generated_tokens: list[str],
    reconstructed_tokens: list[str] | None = None,
) -> dict[str, object]:
    """构造 continuation enrich 测试用的最小轨迹记录。"""
    return {
        "generated_tokens": list(generated_tokens),
        "reconstructed_tokens": list(reconstructed_tokens or ["BOS", *generated_tokens, "EOS"]),
        "reached_eos": True,
        "is_structurally_valid": True,
        "append_eos_would_validate": True,
        "failure_reason": "ok",
        "syntax_reason": "ok",
        "budget_stop": False,
    }


def _infilling_trace_record(
    *,
    prefix_tokens: list[str],
    generated_middle_tokens: list[str],
    suffix_tokens: list[str],
    reconstructed_tokens: list[str] | None = None,
) -> dict[str, object]:
    """构造 infilling enrich 测试用的最小轨迹记录。"""
    return {
        "prefix_tokens": list(prefix_tokens),
        "generated_middle_tokens": list(generated_middle_tokens),
        "suffix_tokens": list(suffix_tokens),
        "reconstructed_tokens": list(
            reconstructed_tokens or [*prefix_tokens, *generated_middle_tokens, *suffix_tokens, "EOS"]
        ),
        "reached_eos": True,
        "is_structurally_valid": True,
        "failure_reason": "ok",
        "syntax_reason": "ok",
    }


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

    def test_extract_first_unit_accepts_phrase_prefixed_event(self) -> None:
        unit = _extract_first_unit(
            [
                "BOS",
                "TEMPO_120",
                "KEY_UNCERTAIN",
                "PHRASE",
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
                "EOS",
            ]
        )
        self.assertEqual(
            unit,
            ("PHRASE", "POS_8", "INST_PIANO", "PITCH_64", "DUR_4", "VEL_8"),
        )

    def test_analyze_token_sequence_keeps_parsing_bar_head_phrase(self) -> None:
        payload = analyze_token_sequence(
            [
                "BOS",
                "BAR",
                "PHRASE",
                "POS_0",
                "INST_PIANO",
                "PITCH_60",
                "DUR_4",
                "VEL_8",
                "BAR",
                "POS_4",
                "INST_PIANO",
                "PITCH_62",
                "DUR_4",
                "VEL_8",
                "EOS",
            ]
        )
        self.assertEqual(payload["bar_count"], 2)
        self.assertEqual(payload["event_count"], 2)
        self.assertTrue(payload["time_order_valid"])

    def test_analyze_token_sequence_keeps_parsing_mid_bar_phrase(self) -> None:
        payload = analyze_token_sequence(
            [
                "BOS",
                "BAR",
                "POS_0",
                "INST_PIANO",
                "PITCH_60",
                "DUR_4",
                "VEL_8",
                "PHRASE",
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
                "EOS",
            ]
        )
        self.assertEqual(payload["bar_count"], 1)
        self.assertEqual(payload["event_count"], 2)
        self.assertTrue(payload["time_order_valid"])

    def test_infilling_boundary_time_order_stats_detect_phrase_prefixed_boundaries(self) -> None:
        stats = _infilling_boundary_time_order_stats(
            prefix_tokens=[
                "BOS",
                "BAR",
                "POS_12",
                "INST_PIANO",
                "PITCH_60",
                "DUR_4",
                "VEL_8",
            ],
            generated_middle_tokens=[
                "PHRASE",
                "POS_8",
                "INST_PIANO",
                "PITCH_62",
                "DUR_4",
                "VEL_8",
                "POS_28",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
            ],
            suffix_tokens=[
                "PHRASE",
                "POS_20",
                "INST_PIANO",
                "PITCH_65",
                "DUR_4",
                "VEL_8",
            ],
        )
        self.assertEqual(stats["prefix_to_middle_violation_count"], 1)
        self.assertEqual(stats["middle_to_suffix_violation_count"], 1)
        self.assertEqual(stats["boundary_violation_count"], 2)

    def test_enrich_structure_control_record_emits_task_level_fields(self) -> None:
        record = enrich_structure_control_record(
            _continuation_trace_record(
                generated_tokens=[
                    "PHRASE",
                    "POS_8",
                    "INST_PIANO",
                    "PITCH_64",
                    "DUR_4",
                    "VEL_8",
                    "POS_12",
                    "INST_PIANO",
                    "PITCH_67",
                    "DUR_4",
                    "VEL_8",
                ]
            ),
            target_tokens=[
                "PHRASE",
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
                "POS_12",
                "INST_PIANO",
                "PITCH_67",
                "DUR_4",
                "VEL_8",
                "EOS",
            ],
        )

        self.assertIn("boundary_type_hit", record)
        self.assertIn("boundary_timing_hit", record)
        self.assertIn("post_boundary_realization_score", record)
        self.assertEqual(record["boundary_type_hit"], 1.0)
        self.assertEqual(record["boundary_timing_hit"], 1.0)
        self.assertEqual(record["post_boundary_realization_score"], 1.0)

    def test_enrich_task_records_cover_other_task_metric_fields(self) -> None:
        continuation_record = _continuation_trace_record(
            generated_tokens=[
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
                "POS_12",
                "INST_PIANO",
                "PITCH_67",
                "DUR_4",
                "VEL_8",
            ]
        )
        local_record = enrich_local_development_record(
            continuation_record,
            target_tokens=[
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
                "POS_12",
                "INST_PIANO",
                "PITCH_67",
                "DUR_4",
                "VEL_8",
                "EOS",
            ],
        )
        long_context_record = enrich_long_context_record(
            continuation_record,
            target_tokens=[
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
                "POS_12",
                "INST_PIANO",
                "PITCH_67",
                "DUR_4",
                "VEL_8",
                "EOS",
            ],
        )
        infilling_record = enrich_infilling_consistency_record(
            _infilling_trace_record(
                prefix_tokens=[
                    "BOS",
                    "BAR",
                    "POS_0",
                    "INST_PIANO",
                    "PITCH_60",
                    "DUR_4",
                    "VEL_8",
                ],
                generated_middle_tokens=[
                    "POS_8",
                    "INST_PIANO",
                    "PITCH_64",
                    "DUR_4",
                    "VEL_8",
                ],
                suffix_tokens=[
                    "POS_12",
                    "INST_PIANO",
                    "PITCH_67",
                    "DUR_4",
                    "VEL_8",
                ],
            ),
            target_hole_tokens=[
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
            ],
        )

        self.assertIn("motif_relation_hit", local_record)
        self.assertIn("copy_overuse_penalty", local_record)
        self.assertIn("unrelated_drift_penalty", local_record)
        self.assertIn("quality_score", local_record)
        self.assertIn("completion_rate", long_context_record)
        self.assertIn("theme_retention_score", long_context_record)
        self.assertIn("section_continuity_score", long_context_record)
        self.assertIn("degeneration_penalty", long_context_record)
        self.assertIn("bridge_validity", infilling_record)
        self.assertIn("boundary_compatibility_hit", infilling_record)
        self.assertIn("rhythmic_connection_score", infilling_record)
        self.assertIn("pitch_connection_score", infilling_record)
        self.assertIn("structural_fit_score", infilling_record)

    def test_task_level_quality_scores_stay_low_for_empty_outputs(self) -> None:
        target_tokens = [
            "POS_8",
            "INST_PIANO",
            "PITCH_64",
            "DUR_4",
            "VEL_8",
            "POS_12",
            "INST_PIANO",
            "PITCH_67",
            "DUR_4",
            "VEL_8",
            "EOS",
        ]
        local_record = enrich_local_development_record(
            _continuation_trace_record(generated_tokens=[]),
            target_tokens=target_tokens,
        )
        long_context_record = enrich_long_context_record(
            _continuation_trace_record(generated_tokens=[]),
            target_tokens=target_tokens,
        )
        infilling_record = enrich_infilling_consistency_record(
            _infilling_trace_record(
                prefix_tokens=[
                    "BOS",
                    "BAR",
                    "POS_0",
                    "INST_PIANO",
                    "PITCH_60",
                    "DUR_4",
                    "VEL_8",
                ],
                generated_middle_tokens=[],
                suffix_tokens=[
                    "POS_12",
                    "INST_PIANO",
                    "PITCH_67",
                    "DUR_4",
                    "VEL_8",
                ],
            ),
            target_hole_tokens=[
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
            ],
        )

        self.assertGreaterEqual(float(local_record["copy_overuse_penalty"]), 0.8)
        self.assertGreaterEqual(float(local_record["unrelated_drift_penalty"]), 0.8)
        self.assertLess(float(local_record["quality_score"]), 0.2)
        self.assertLess(float(long_context_record["theme_retention_score"]), 0.2)
        self.assertLess(float(long_context_record["section_continuity_score"]), 0.2)
        self.assertGreaterEqual(float(long_context_record["degeneration_penalty"]), 0.8)
        self.assertEqual(float(infilling_record["bridge_validity"]), 0.0)
        self.assertEqual(float(infilling_record["boundary_compatibility_hit"]), 0.0)
        self.assertLess(float(infilling_record["rhythmic_connection_score"]), 0.2)
        self.assertLess(float(infilling_record["pitch_connection_score"]), 0.2)
        self.assertLess(float(infilling_record["structural_fit_score"]), 0.2)

    def test_long_context_theme_retention_stays_low_for_empty_output_even_with_single_pitch_target(self) -> None:
        record = enrich_long_context_record(
            _continuation_trace_record(generated_tokens=[]),
            target_tokens=[
                "POS_8",
                "INST_PIANO",
                "PITCH_64",
                "DUR_4",
                "VEL_8",
                "EOS",
            ],
        )
        self.assertLess(float(record["theme_retention_score"]), 0.2)

    def test_benchmark_runner_uses_task_specific_case_payloads(self) -> None:
        from scripts.eval.benchmark_runner import (
            _continuation_task_specs_for_case,
            _infilling_task_spec_for_case,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eval_jsonl_path = tmp_path / "eval.jsonl"
            eval_tok_path = tmp_path / "eval.tok"
            eval_jsonl_path.write_text(
                json.dumps(
                    {
                        "artist": "Artist 0",
                        "title": "Title 0",
                        "family_key": "family::0",
                        "midi_path": "path/0.mid",
                        "note_count": 120,
                        "duration_sec": 128.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            eval_tok_path.write_text(" ".join(_long_sequence()) + "\n", encoding="utf-8")
            manifest = build_benchmark_manifest(
                eval_jsonl_path=eval_jsonl_path,
                eval_tok_path=eval_tok_path,
                config={
                    "tier": "fast",
                    "seed": 42,
                    "sample_count": None,
                    "per_bucket_limit": None,
                    "min_prefix_tokens": 8,
                    "continuation_prefix_ratio_min": 0.35,
                    "continuation_prefix_ratio_max": 0.70,
                    "infilling_hole_ratio_min": 0.10,
                    "infilling_hole_ratio_max": 0.25,
                },
                max_positions=64,
            )
            case = manifest["cases"][0]

        continuation_specs = _continuation_task_specs_for_case(case)
        self.assertEqual(
            [spec["task_name"] for spec in continuation_specs],
            [
                case["structure_control_case"]["task_name"],
                case["local_development_case"]["task_name"],
                case["long_context_case"]["task_name"],
            ],
        )
        self.assertEqual(continuation_specs[0]["prompt_tokens"], case["structure_control_case"]["prompt_tokens"])
        self.assertEqual(continuation_specs[1]["prompt_tokens"], case["local_development_case"]["prompt_tokens"])
        self.assertEqual(continuation_specs[2]["prompt_tokens"], case["long_context_case"]["prompt_tokens"])
        self.assertEqual(continuation_specs[1]["target_tokens"], case["local_development_case"]["target_tokens"])
        self.assertEqual(continuation_specs[2]["target_tokens"], case["long_context_case"]["target_tokens"])

        infilling_spec = _infilling_task_spec_for_case(case)
        self.assertEqual(infilling_spec["task_name"], case["infilling_consistency_case"]["task_name"])
        self.assertEqual(infilling_spec["prompt_tokens"], case["infilling_consistency_case"]["prompt_tokens"])
        self.assertEqual(infilling_spec["target_hole_tokens"], case["infilling_consistency_case"]["target_hole_tokens"])

    def test_task_specific_rates_use_their_own_attempt_denominators(self) -> None:
        from scripts.eval.benchmark_runner import _evaluate_checkpoint_on_manifest
        from src.utils.benchmark_decode import build_continuation_trace, build_infilling_trace

        class _FakeConfig:
            def __init__(self, max_position_embeddings: int = 128):
                self.max_position_embeddings = int(max_position_embeddings)

            @classmethod
            def from_dict(cls, payload):
                return cls()

            @classmethod
            def from_yaml(cls, path):
                return cls()

        class _FakeModel:
            def to(self, device):
                return self

            def load_state_dict(self, state_dict):
                return None

            def eval(self):
                return self

        class _FakeDecoderForCausalLM:
            def __new__(cls, config):
                return _FakeModel()

        class _FakeGrammarFSM:
            def inspect_complete_tokens(self, tokens):
                return (bool(tokens) and str(tokens[0]) == "BOS" and str(tokens[-1]) == "EOS", "ok")

        class _FakeDevice:
            type = "cpu"

        structure_prompt = ["BOS", "BAR"]
        structure_target = [
            "PHRASE",
            "POS_0",
            "INST_PIANO",
            "PITCH_64",
            "DUR_4",
            "VEL_8",
            "EOS",
        ]
        local_prompt = [
            "BOS",
            "BAR",
            "POS_0",
            "INST_PIANO",
            "PITCH_60",
            "DUR_4",
            "VEL_8",
        ]
        local_target = [
            "POS_8",
            "INST_PIANO",
            "PITCH_64",
            "DUR_4",
            "VEL_8",
            "EOS",
        ]
        long_prompt = [
            "BOS",
            "BAR",
            "POS_4",
            "INST_PIANO",
            "PITCH_62",
            "DUR_4",
            "VEL_8",
        ]
        long_target = [
            "POS_12",
            "INST_PIANO",
            "PITCH_67",
            "DUR_4",
            "VEL_8",
            "EOS",
        ]
        infill_prompt = [
            "BOS",
            "BAR",
            "POS_0",
            "INST_PIANO",
            "PITCH_60",
            "DUR_4",
            "VEL_8",
            "FIM_HOLE",
            "POS_12",
            "INST_PIANO",
            "PITCH_67",
            "DUR_4",
            "VEL_8",
            "FIM_MID",
        ]
        infill_prefix = [
            "BOS",
            "BAR",
            "POS_0",
            "INST_PIANO",
            "PITCH_60",
            "DUR_4",
            "VEL_8",
        ]
        infill_suffix = [
            "POS_12",
            "INST_PIANO",
            "PITCH_67",
            "DUR_4",
            "VEL_8",
        ]
        infill_target = [
            "POS_8",
            "INST_PIANO",
            "PITCH_64",
            "DUR_4",
            "VEL_8",
        ]

        def _generate_continuation_stub(**kwargs):
            prompt_tokens = list(kwargs["prompt_tokens"])
            if prompt_tokens == structure_prompt:
                generated_tokens = structure_target[:-1]
            elif prompt_tokens == local_prompt:
                generated_tokens = []
            elif prompt_tokens == long_prompt:
                generated_tokens = long_target[:-1]
            else:
                raise AssertionError(f"unexpected continuation prompt: {prompt_tokens}")
            stats = {
                "step_count": max(1, len(generated_tokens)),
                "illegal_top1_count": 0,
                "mask_intervention_count": 0,
                "legal_mass_sum": float(max(1, len(generated_tokens))),
                "dead_end_count": 0,
                "auto_close_count": 0,
            }
            return generated_tokens, True, stats

        def _generate_middle_stub(**kwargs):
            prompt_tokens = list(kwargs["prompt_tokens"])
            self.assertEqual(prompt_tokens, infill_prompt)
            stats = {
                "step_count": 1,
                "illegal_top1_count": 0,
                "mask_intervention_count": 0,
                "legal_mass_sum": 1.0,
                "dead_end_count": 0,
                "auto_close_count": 0,
            }
            return list(infill_target), True, stats

        manifest = {
            "tier": "fast",
            "cases": [
                {
                    "row_id": 0,
                    "bucket": "bucket",
                    "meta": {"artist": "Artist", "title": "Title"},
                    "structure_control_case": {
                        "task_name": "structure_control",
                        "prompt_tokens": structure_prompt,
                        "target_tokens": structure_target,
                        "window_tokens": [*structure_prompt, *structure_target],
                    },
                    "local_development_case": {
                        "task_name": "local_development",
                        "prompt_tokens": local_prompt,
                        "target_tokens": local_target,
                        "window_tokens": [*local_prompt, *local_target],
                    },
                    "long_context_case": {
                        "task_name": "long_context_coherence",
                        "prompt_tokens": long_prompt,
                        "target_tokens": long_target,
                        "window_tokens": [*long_prompt, *long_target],
                    },
                    "infilling_consistency_case": {
                        "task_name": "infilling_consistency",
                        "prompt_tokens": infill_prompt,
                        "prefix_tokens": infill_prefix,
                        "suffix_tokens": infill_suffix,
                        "target_hole_tokens": infill_target,
                        "window_tokens": [*infill_prefix, *infill_target, *infill_suffix, "EOS"],
                    },
                }
            ],
        }

        result, _ = _evaluate_checkpoint_on_manifest(
            ckpt_path=Path("fake.pt"),
            manifest=manifest,
            capture_row_ids=None,
            task_scope="all",
            token_to_id={},
            id_to_token=[],
            grammar_fsm=_FakeGrammarFSM(),
            training_metrics_payload={},
            args=SimpleNamespace(
                device="cpu",
                precision="fp32",
                max_new_tokens=32,
                temperature=0.0,
                top_p=1.0,
            ),
            fallback_model_config_path=Path("fake.yaml"),
            torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
            DecoderConfig=_FakeConfig,
            DecoderForCausalLM=_FakeDecoderForCausalLM,
            load_checkpoint_fn=lambda torch_mod, ckpt_path: {
                "model_config": {},
                "model_state_dict": {},
                "step": 10,
            },
            autocast_context_fn=lambda **kwargs: nullcontext(),
            resolve_precision_fn=lambda **kwargs: ("fp32", False, None, None),
            resolve_torch_device_fn=lambda torch_mod, requested: _FakeDevice(),
            generate_continuation_tokens_fn=_generate_continuation_stub,
            build_continuation_trace_fn=build_continuation_trace,
            generate_middle_tokens_fn=_generate_middle_stub,
            build_infilling_trace_fn=build_infilling_trace,
        )

        self.assertEqual(float(result["structure_control_boundary_type_hit_rate"]), 1.0)
        self.assertEqual(float(result["structure_control_boundary_timing_hit_rate"]), 1.0)
        self.assertEqual(float(result["local_development_motif_relation_hit_rate"]), 0.0)
        self.assertEqual(float(result["long_context_completion_rate"]), 1.0)

    def test_analyze_token_sequence_detects_most_common_pitch_ratio(self) -> None:
        payload = analyze_token_sequence(_sequence_with_pitches([60, 60, 60, 60, 60, 64, 67, 69]))
        self.assertAlmostEqual(float(payload["most_common_pitch_ratio"]), 5.0 / 8.0)
        self.assertGreater(float(payload["most_common_pitch_ratio"]), 0.6)

    def test_analyze_token_sequence_ignores_key_control_tokens(self) -> None:
        payload = analyze_token_sequence(
            [
                "BOS",
                "TEMPO_120",
                "KEY_C_MAJ",
                "BAR",
                "KEY_G_MAJ",
                "POS_0",
                "INST_PIANO",
                "PITCH_60",
                "DUR_4",
                "VEL_8",
                "EOS",
            ]
        )
        self.assertEqual(payload["bar_count"], 1)
        self.assertEqual(payload["event_count"], 1)
        self.assertTrue(payload["time_order_valid"])

    def test_analyze_token_sequence_detects_long_same_pitch_run_ratio(self) -> None:
        payload = analyze_token_sequence(_sequence_with_pitches([60, 62, 64, 64, 64, 64, 64, 67]))
        self.assertAlmostEqual(float(payload["longest_same_pitch_run_ratio"]), 5.0 / 8.0)
        self.assertGreater(float(payload["longest_same_pitch_run_ratio"]), 0.6)

    def test_analyze_token_sequence_pitch_diversity_distinguishes_low_and_high_diversity(self) -> None:
        low_diversity = analyze_token_sequence(_sequence_with_pitches([60, 60, 60, 60, 60, 61, 60, 60]))
        high_diversity = analyze_token_sequence(_sequence_with_pitches([60, 62, 64, 65, 67, 69, 71, 72]))
        self.assertIsNotNone(low_diversity["pitch_diversity_score"])
        self.assertIsNotNone(high_diversity["pitch_diversity_score"])
        self.assertLess(float(low_diversity["pitch_diversity_score"]), float(high_diversity["pitch_diversity_score"]))

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

    def test_build_benchmark_manifest_emits_four_task_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eval_jsonl_path = tmp_path / "eval.jsonl"
            eval_tok_path = tmp_path / "eval.tok"
            row = {
                "artist": "Artist 0",
                "title": "Title 0",
                "family_key": "family::0",
                "midi_path": "path/0.mid",
                "note_count": 120,
                "duration_sec": 128.0,
            }
            eval_jsonl_path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            eval_tok_path.write_text(" ".join(_long_sequence()) + "\n", encoding="utf-8")

            manifest = build_benchmark_manifest(
                eval_jsonl_path=eval_jsonl_path,
                eval_tok_path=eval_tok_path,
                config={
                    "tier": "fast",
                    "seed": 42,
                    "sample_count": None,
                    "per_bucket_limit": None,
                    "min_prefix_tokens": 8,
                    "continuation_prefix_ratio_min": 0.35,
                    "continuation_prefix_ratio_max": 0.70,
                    "infilling_hole_ratio_min": 0.10,
                    "infilling_hole_ratio_max": 0.25,
                },
                max_positions=64,
            )

            case = manifest["cases"][0]
            self.assertIn("structure_control_case", case)
            self.assertIn("local_development_case", case)
            self.assertIn("long_context_case", case)
            self.assertIn("infilling_consistency_case", case)
            self.assertIn("continuation_case", case)
            self.assertIn("infilling_case", case)
            self.assertEqual(case["long_context_case"]["task_name"], "long_context_coherence")
            self.assertEqual(case["long_context_case"]["section_label"], "continue_section")

    def test_build_benchmark_manifest_continuation_scope_does_not_require_infilling_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eval_jsonl_path = tmp_path / "eval.jsonl"
            eval_tok_path = tmp_path / "eval.tok"
            row = {
                "artist": "Artist 0",
                "title": "Title 0",
                "family_key": "family::0",
                "midi_path": "path/0.mid",
                "note_count": 120,
                "duration_sec": 128.0,
            }
            eval_jsonl_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            eval_tok_path.write_text(" ".join(_long_sequence()) + "\n", encoding="utf-8")

            with patch("src.utils.benchmarking.build_infilling_consistency_case", return_value=None):
                manifest = build_benchmark_manifest(
                    eval_jsonl_path=eval_jsonl_path,
                    eval_tok_path=eval_tok_path,
                    config={
                        "tier": "fast",
                        "seed": 42,
                        "sample_count": None,
                        "per_bucket_limit": None,
                        "min_prefix_tokens": 8,
                        "continuation_prefix_ratio_min": 0.35,
                        "continuation_prefix_ratio_max": 0.70,
                        "infilling_hole_ratio_min": 0.10,
                        "infilling_hole_ratio_max": 0.25,
                    },
                    max_positions=64,
                    task_scope="continuation",
                )

            self.assertEqual(manifest["case_count"], 1)
            self.assertIsNone(manifest["cases"][0]["infilling_consistency_case"])
            self.assertIsNone(manifest["cases"][0]["infilling_case"])

    def test_build_benchmark_manifest_infilling_scope_does_not_require_continuation_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            eval_jsonl_path = tmp_path / "eval.jsonl"
            eval_tok_path = tmp_path / "eval.tok"
            row = {
                "artist": "Artist 0",
                "title": "Title 0",
                "family_key": "family::0",
                "midi_path": "path/0.mid",
                "note_count": 120,
                "duration_sec": 128.0,
            }
            eval_jsonl_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            eval_tok_path.write_text(" ".join(_long_sequence()) + "\n", encoding="utf-8")

            with patch("src.utils.benchmarking.build_structure_control_case", return_value=None), patch(
                "src.utils.benchmarking.build_local_development_case",
                return_value=None,
            ), patch(
                "src.utils.benchmarking.build_long_context_case",
                return_value=None,
            ):
                manifest = build_benchmark_manifest(
                    eval_jsonl_path=eval_jsonl_path,
                    eval_tok_path=eval_tok_path,
                    config={
                        "tier": "fast",
                        "seed": 42,
                        "sample_count": None,
                        "per_bucket_limit": None,
                        "min_prefix_tokens": 8,
                        "continuation_prefix_ratio_min": 0.35,
                        "continuation_prefix_ratio_max": 0.70,
                        "infilling_hole_ratio_min": 0.10,
                        "infilling_hole_ratio_max": 0.25,
                    },
                    max_positions=64,
                    task_scope="infilling",
                )

            self.assertEqual(manifest["case_count"], 1)
            self.assertIsNone(manifest["cases"][0]["structure_control_case"])
            self.assertIsNone(manifest["cases"][0]["continuation_case"])

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

    def test_bridgeable_states_for_suffix_include_incomplete_note_states(self) -> None:
        vocab_tokens = [
            "BOS",
            "EOS",
            "BAR",
            "POS_0",
            "POS_4",
            "INST_PIANO",
            "PITCH_60",
            "PITCH_64",
            "DUR_4",
            "VEL_8",
        ]
        token_to_id = {token: index for index, token in enumerate(vocab_tokens)}
        grammar_fsm = TuneFlowGrammarFSM.from_vocab(token_to_id)

        suffix_tokens = ["POS_4", "INST_PIANO", "PITCH_64", "DUR_4", "VEL_8"]
        prefix_state = grammar_fsm.state_after_prefix_tokens(["BOS", "BAR"])
        bridgeable_states = grammar_fsm.bridgeable_states_for_suffix_tokens(suffix_tokens)

        self.assertEqual(prefix_state, "after_bar")
        self.assertIn("after_bar", bridgeable_states)
        self.assertIn("after_pos", bridgeable_states)
        self.assertIn("after_inst", bridgeable_states)
        self.assertIn("after_pitch", bridgeable_states)
        self.assertIn("after_dur", bridgeable_states)
        self.assertEqual(
            grammar_fsm.transition(prefix_state, token_to_id["POS_0"]),
            "after_pos",
        )

    def test_grammar_fsm_accepts_sparse_key_control_tokens(self) -> None:
        vocab_tokens = [
            "BOS",
            "EOS",
            "BAR",
            "TEMPO_120",
            "KEY_C_MAJ",
            "KEY_G_MAJ",
            "POS_0",
            "INST_PIANO",
            "PITCH_60",
            "DUR_4",
            "VEL_8",
        ]
        token_to_id = {token: index for index, token in enumerate(vocab_tokens)}
        grammar_fsm = TuneFlowGrammarFSM.from_vocab(token_to_id)

        valid, reason = grammar_fsm.inspect_complete_tokens(
            ["BOS", "TEMPO_120", "KEY_C_MAJ", "BAR", "KEY_G_MAJ", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8", "EOS"]
        )
        self.assertTrue(valid, msg=reason)
        self.assertEqual(reason, "ok")

    def test_generate_middle_tokens_allows_note_prefix_that_becomes_suffix_compatible_later(self) -> None:
        if torch is None:
            self.skipTest("torch is required for benchmark decode tests")

        vocab_tokens = [
            "BOS",
            "EOS",
            "BAR",
            "FIM_HOLE",
            "FIM_MID",
            "POS_0",
            "POS_4",
            "INST_PIANO",
            "PITCH_60",
            "PITCH_64",
            "DUR_4",
            "VEL_8",
        ]
        token_to_id = {token: index for index, token in enumerate(vocab_tokens)}
        id_to_token = list(vocab_tokens)

        class _PlannedToyModel:
            def __init__(self, plan: list[int], vocab_size: int):
                self.plan = [int(token_id) for token_id in plan]
                self.vocab_size = int(vocab_size)
                self.calls = 0

            def __call__(self, *, input_ids, past_key_values=None, use_cache=None, return_dict=True):
                next_id = self.plan[min(self.calls, len(self.plan) - 1)]
                self.calls += 1
                logits = torch.full(
                    (1, int(input_ids.shape[1]), self.vocab_size),
                    fill_value=-1000.0,
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                logits[0, -1, next_id] = 1000.0
                return SimpleNamespace(logits=logits, past_key_values=("cached",))

        model = _PlannedToyModel(
            plan=[
                token_to_id["POS_0"],
                token_to_id["INST_PIANO"],
                token_to_id["PITCH_60"],
                token_to_id["DUR_4"],
                token_to_id["VEL_8"],
                token_to_id["EOS"],
            ],
            vocab_size=len(id_to_token),
        )
        grammar_fsm = TuneFlowGrammarFSM.from_vocab(token_to_id)

        generated_tokens, reached_eos, stats = generate_middle_tokens(
            model=model,
            torch_mod=torch,
            prompt_tokens=["BOS", "BAR", "FIM_HOLE", "POS_4", "INST_PIANO", "PITCH_64", "DUR_4", "VEL_8", "FIM_MID"],
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            grammar_fsm=grammar_fsm,
            prefix_tokens=["BOS", "BAR"],
            suffix_tokens=["POS_4", "INST_PIANO", "PITCH_64", "DUR_4", "VEL_8"],
            device=torch.device("cpu"),
            use_amp=False,
            amp_dtype=None,
            autocast_context_fn=lambda **kwargs: nullcontext(),
            max_positions=32,
            max_new_tokens=8,
        )

        self.assertEqual(
            generated_tokens,
            ["POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8"],
        )
        self.assertTrue(reached_eos)
        self.assertEqual(int(stats["dead_end_count"]), 0)


if __name__ == "__main__":
    unittest.main()
