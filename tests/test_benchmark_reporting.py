from __future__ import annotations

import unittest
from pathlib import Path

from scripts.eval.benchmark_runner import (
    _absolute_plot_metric_specs_v2,
    _attach_baseline_win_rates,
    _build_summary_markdown_v3,
    _core_metric_specs_v2,
    _plot_metric_specs_v2,
    _resolve_baseline_checkpoint_name,
    _task_capability_plot_filename,
    _to_jsonable_result,
)


_BATTLE_KEYS = (
    "task_capability_score",
    "task_control_score",
    "task_realization_score",
    "structure_control_score",
    "local_development_score",
    "long_context_coherence_score",
    "infilling_consistency_score",
)


def _recommended_checkpoint() -> dict[str, object]:
    """构造用于摘要测试的推荐 checkpoint 数据。"""
    return {
        "checkpoint_name": "step_120.pt",
        "checkpoint_path": "outputs/checkpoints/run/step_120.pt",
        "step": 120,
        "evaluation_tier": "formal",
        "recommendation_source": "formal",
        "task_capability_score": 0.82,
        "task_control_score": 0.79,
        "task_realization_score": 0.84,
        "vs_baseline_win_rate": 0.67,
        "structure_control_score": 0.80,
        "local_development_score": 0.78,
        "long_context_coherence_score": 0.81,
        "infilling_consistency_score": 0.76,
        "gate_details": {},
        "selection_breakdown": {
            "ranking_basis": "task_capability_score",
            "tie_breakers": [
                {"metric_key": "vs_baseline_win_rate", "value": 0.67},
            ],
        },
        "primary_score_breakdown": {"dimensions": {}},
        "task_capability_score_breakdown": {"dimensions": {}},
        "legacy_score_fields": {
            "balanced_score": 0.12,
            "balanced_score_coverage": 0.10,
            "balanced_score_breakdown": {},
            "absolute_score": 48.6,
            "absolute_score_coverage": 0.88,
            "absolute_score_breakdown": {"dimensions": {}},
        },
        "failure_reason_counts": {},
        "syntax_reason_counts": {},
    }


def _build_summary(*, plot_artifacts: dict[str, str] | None = None) -> str:
    """构造最小摘要输入，便于聚焦推荐文案断言。"""
    recommended = _recommended_checkpoint()
    return _build_summary_markdown_v3(
        run_id="unit-test-run",
        task_scope="all",
        benchmark_root=Path("outputs/benchmark/unit-test-run"),
        recommended=recommended,
        fast_results=[recommended],
        formal_results=[recommended],
        top_results=[recommended],
        training_summary={"run": {}},
        plot_artifacts={} if plot_artifacts is None else plot_artifacts,
        sample_artifacts={"final_top3": {}, "formal_candidates": {}},
        exported_samples={"final_top3": {}, "formal_candidates": {}},
        manifest_stats={
            "fast_case_count": 4,
            "formal_case_count": 8,
            "candidate_count": 2,
        },
        checkpoint_prefilter={"enabled": False},
        evaluation_context={
            "config_paths": {},
            "benchmark_configs": {"fast": {}, "formal": {}},
            "decoding": {},
        },
    )


class BenchmarkReportingTests(unittest.TestCase):
    def test_summary_v3_displays_task_capability_and_baseline_sections(self) -> None:
        summary = _build_summary()

        self.assertIn("任务型能力分", summary)
        self.assertIn("基线胜率", summary)

    def test_summary_v3_explains_baseline_metric_as_task_battle_proxy(self) -> None:
        summary = _build_summary()

        self.assertIn("任务维度对战率代理", summary)
        self.assertNotIn("case-level", summary)

    def test_summary_v3_final_recommendation_prefers_task_fields(self) -> None:
        summary = _build_summary()
        final_section = summary.split("## 最终推荐", 1)[1].split("## 评估上下文", 1)[0]

        self.assertIn("任务型能力分", final_section)
        self.assertIn("任务控制分", final_section)
        self.assertIn("音乐实现分", final_section)
        self.assertIn("结构控制能力", final_section)
        self.assertIn("局部发展能力", final_section)
        self.assertIn("长程连贯能力", final_section)
        self.assertIn("补全一致性能力", final_section)
        self.assertNotIn("相对分：", final_section)
        self.assertNotIn("绝对分：", final_section)

    def test_summary_v3_uses_new_breakdown_sections_for_benchmark_overall(self) -> None:
        summary = _build_summary()

        self.assertIn("推荐 Checkpoint 任务型主分拆解", summary)
        self.assertIn("推荐 Checkpoint 选择排序拆解", summary)
        self.assertNotIn("推荐 Checkpoint 绝对分拆解", summary)
        self.assertNotIn("legacy_score_fields", summary)

    def test_summary_v3_renders_nested_task_metric_details_instead_of_component_shells(self) -> None:
        recommended = _recommended_checkpoint()
        recommended["primary_score_breakdown"] = {
            "dimensions": {
                "structure_control_score": {
                    "label": "结构控制",
                    "score": 82.0,
                    "coverage": 1.0,
                    "weight": 0.30,
                    "submetrics": {
                        "control_hit": {
                            "submetrics": {
                                "structure_control_boundary_type_hit_rate": {
                                    "raw_value": 0.80,
                                    "normalized_value": 0.80,
                                    "weight": 0.50,
                                }
                            }
                        },
                        "music_realization": {
                            "submetrics": {
                                "structure_control_post_boundary_realization_score": {
                                    "raw_value": 0.84,
                                    "normalized_value": 0.84,
                                    "weight": 1.00,
                                }
                            }
                        },
                    },
                }
            }
        }
        recommended["task_capability_score_breakdown"] = recommended["primary_score_breakdown"]
        summary = _build_summary_markdown_v3(
            run_id="unit-test-run",
            task_scope="all",
            benchmark_root=Path("outputs/benchmark/unit-test-run"),
            recommended=recommended,
            fast_results=[recommended],
            formal_results=[recommended],
            top_results=[recommended],
            training_summary={"run": {}},
            plot_artifacts={},
            sample_artifacts={"final_top3": {}, "formal_candidates": {}},
            exported_samples={"final_top3": {}, "formal_candidates": {}},
            manifest_stats={"fast_case_count": 1, "formal_case_count": 1, "candidate_count": 1},
            checkpoint_prefilter={"enabled": False},
            evaluation_context={"config_paths": {}, "benchmark_configs": {"fast": {}, "formal": {}}, "decoding": {}},
        )

        self.assertIn("控制命中", summary)
        self.assertIn("音乐实现", summary)
        self.assertIn("structure_control_boundary_type_hit_rate", summary)
        self.assertIn("structure_control_post_boundary_realization_score", summary)
        self.assertNotIn("| control_hit |", summary)
        self.assertNotIn("| music_realization |", summary)

    def test_fast_and_formal_baseline_are_resolved_independently(self) -> None:
        fast_results = [
            {"checkpoint_name": "step_10.pt", "step": 10},
            {"checkpoint_name": "step_30.pt", "step": 30},
        ]
        formal_results = [
            {"checkpoint_name": "step_20.pt", "step": 20},
            {"checkpoint_name": "step_40.pt", "step": 40},
        ]

        self.assertEqual(
            _resolve_baseline_checkpoint_name(fast_results, requested_baseline=None),
            "step_10.pt",
        )
        self.assertEqual(
            _resolve_baseline_checkpoint_name(formal_results, requested_baseline=None),
            "step_20.pt",
        )

    def test_explicit_baseline_missing_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "fast.*missing.pt"):
            _attach_baseline_win_rates(
                [
                    {"checkpoint_name": "step_10.pt", "task_capability_score": 0.7},
                ],
                baseline_checkpoint_name="missing.pt",
                stage_name="fast",
            )

    def test_vs_baseline_win_rate_supports_zero_half_one_and_fractional_values(self) -> None:
        baseline = {
            "checkpoint_name": "base.pt",
            "task_capability_score": 0.50,
            "task_control_score": 0.50,
            "task_realization_score": 0.50,
            "structure_control_score": 0.50,
            "local_development_score": 0.50,
            "long_context_coherence_score": 0.50,
            "infilling_consistency_score": 0.50,
        }
        all_lower = {"checkpoint_name": "low.pt", **{key: 0.40 for key in _BATTLE_KEYS}}
        all_equal = {"checkpoint_name": "equal.pt", **{key: 0.50 for key in _BATTLE_KEYS}}
        all_higher = {"checkpoint_name": "high.pt", **{key: 0.60 for key in _BATTLE_KEYS}}
        mixed = {
            "checkpoint_name": "mixed.pt",
            "task_capability_score": 0.60,
            "task_control_score": 0.50,
            "task_realization_score": 0.40,
            "structure_control_score": 0.60,
            "local_development_score": 0.50,
            "long_context_coherence_score": 0.40,
            "infilling_consistency_score": 0.60,
        }

        enriched = _attach_baseline_win_rates(
            [baseline, all_lower, all_equal, all_higher, mixed],
            baseline_checkpoint_name="base.pt",
            stage_name="formal",
        )
        score_by_name = {item["checkpoint_name"]: item["vs_baseline_win_rate"] for item in enriched}

        self.assertEqual(score_by_name["low.pt"], 0.0)
        self.assertEqual(score_by_name["equal.pt"], 0.5)
        self.assertEqual(score_by_name["high.pt"], 1.0)
        self.assertAlmostEqual(float(score_by_name["mixed.pt"]), 4.0 / 7.0, places=6)

    def test_attach_baseline_win_rates_does_not_mutate_other_stage_results(self) -> None:
        fast_results = [
            {
                "checkpoint_name": "fast_base.pt",
                "task_capability_score": 0.50,
                "task_control_score": 0.50,
                "task_realization_score": 0.50,
                "structure_control_score": 0.50,
                "local_development_score": 0.50,
                "long_context_coherence_score": 0.50,
                "infilling_consistency_score": 0.50,
            },
            {
                "checkpoint_name": "fast_top.pt",
                "task_capability_score": 0.60,
                "task_control_score": 0.60,
                "task_realization_score": 0.60,
                "structure_control_score": 0.60,
                "local_development_score": 0.60,
                "long_context_coherence_score": 0.60,
                "infilling_consistency_score": 0.60,
            },
        ]
        formal_results = [
            {
                "checkpoint_name": "formal_base.pt",
                "task_capability_score": 0.40,
                "task_control_score": 0.40,
                "task_realization_score": 0.40,
                "structure_control_score": 0.40,
                "local_development_score": 0.40,
                "long_context_coherence_score": 0.40,
                "infilling_consistency_score": 0.40,
            },
            {
                "checkpoint_name": "formal_top.pt",
                "task_capability_score": 0.50,
                "task_control_score": 0.50,
                "task_realization_score": 0.50,
                "structure_control_score": 0.50,
                "local_development_score": 0.50,
                "long_context_coherence_score": 0.50,
                "infilling_consistency_score": 0.50,
            },
        ]

        fast_scored = _attach_baseline_win_rates(
            fast_results,
            baseline_checkpoint_name="fast_base.pt",
            stage_name="fast",
        )
        formal_scored = _attach_baseline_win_rates(
            formal_results,
            baseline_checkpoint_name="formal_base.pt",
            stage_name="formal",
        )

        self.assertEqual(
            next(item for item in fast_scored if item["checkpoint_name"] == "fast_top.pt")["vs_baseline_win_rate"],
            1.0,
        )
        self.assertEqual(
            next(item for item in formal_scored if item["checkpoint_name"] == "formal_top.pt")["vs_baseline_win_rate"],
            1.0,
        )
        self.assertEqual(
            next(item for item in fast_scored if item["checkpoint_name"] == "fast_base.pt")["vs_baseline_win_rate"],
            0.5,
        )

    def test_task_capability_plot_filename_and_summary_use_new_naming(self) -> None:
        filename = _task_capability_plot_filename("all")
        summary = _build_summary(
            plot_artifacts={"任务型能力面板": f"outputs/benchmark/unit-test-run/{filename}"}
        )
        metric_specs = _absolute_plot_metric_specs_v2("all")

        self.assertIn("task_capability", filename)
        self.assertNotIn("absolute_capabilities", filename)
        self.assertIn("任务型能力面板", summary)
        self.assertTrue(any(item["key"] == "task_capability_score" for item in metric_specs))

    def test_scope_specific_tables_and_plots_hide_irrelevant_task_dimensions(self) -> None:
        continuation_labels = [label for _key, label in _core_metric_specs_v2("continuation")]
        infilling_labels = [label for _key, label in _core_metric_specs_v2("infilling")]
        continuation_core_plot_keys = [item["key"] for item in _plot_metric_specs_v2("continuation", diagnostics=False)]
        infilling_core_plot_keys = [item["key"] for item in _plot_metric_specs_v2("infilling", diagnostics=False)]
        continuation_plot_keys = [item["key"] for item in _absolute_plot_metric_specs_v2("continuation")]
        infilling_plot_keys = [item["key"] for item in _absolute_plot_metric_specs_v2("infilling")]

        self.assertNotIn("补全一致性能力", continuation_labels)
        self.assertIn("长程连贯能力", continuation_labels)
        self.assertEqual(infilling_labels.count("补全一致性能力"), 1)
        self.assertNotIn("长程连贯能力", infilling_labels)
        self.assertNotIn("infilling_consistency_score", continuation_core_plot_keys)
        self.assertNotIn("long_context_coherence_score", infilling_core_plot_keys)
        self.assertNotIn("local_development_score", infilling_plot_keys)
        self.assertNotIn("long_context_coherence_score", infilling_plot_keys)
        self.assertIn("long_context_coherence_score", continuation_plot_keys)

    def test_to_jsonable_result_moves_legacy_scores_into_compatibility_container(self) -> None:
        jsonable = _to_jsonable_result(_recommended_checkpoint())

        self.assertNotIn("balanced_score", jsonable)
        self.assertNotIn("absolute_score", jsonable)
        self.assertTrue(jsonable["legacy_score_fields_present"])
        self.assertTrue(jsonable["compatibility_only"])
        self.assertEqual(jsonable["legacy_score_fields"]["balanced_score"], 0.12)
        self.assertEqual(jsonable["legacy_score_fields"]["absolute_score"], 48.6)
        self.assertEqual(jsonable["task_capability_score"], 0.82)


if __name__ == "__main__":
    unittest.main()
