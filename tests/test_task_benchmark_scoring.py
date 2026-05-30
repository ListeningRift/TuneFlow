from __future__ import annotations

import math
import unittest

from src.utils import attach_task_capability_scores as attach_task_capability_scores_from_utils
from src.utils import score_task_capabilities as score_task_capabilities_from_utils
from src.utils.task_benchmark_scoring import attach_task_capability_scores, score_task_capabilities


def _base_result() -> dict[str, float]:
    return {
        "structure_control_boundary_type_hit_rate": 0.82,
        "structure_control_boundary_timing_hit_rate": 0.76,
        "structure_control_post_boundary_realization_score": 0.71,
        "local_development_motif_relation_hit_rate": 0.74,
        "local_development_copy_overuse_penalty": 0.18,
        "local_development_unrelated_drift_penalty": 0.12,
        "local_development_quality_score": 0.69,
        "long_context_completion_rate": 0.88,
        "long_context_theme_retention_score": 0.72,
        "long_context_section_continuity_score": 0.75,
        "long_context_degeneration_penalty": 0.14,
        "infilling_bridge_validity_rate": 0.79,
        "infilling_boundary_compatibility_hit_rate": 0.83,
        "infilling_rhythmic_connection_score": 0.73,
        "infilling_pitch_connection_score": 0.70,
        "infilling_structural_fit_score": 0.77,
    }


class TaskBenchmarkScoringTests(unittest.TestCase):
    def test_score_task_capabilities_outputs_required_scores(self) -> None:
        scored = score_task_capabilities(_base_result())
        self.assertIn("task_capability_score", scored)
        self.assertIn("task_capability_score_coverage", scored)
        self.assertIn("task_control_score", scored)
        self.assertIn("task_realization_score", scored)
        for key in (
            "structure_control_score",
            "local_development_score",
            "long_context_coherence_score",
            "infilling_consistency_score",
        ):
            self.assertIn(key, scored)
            self.assertIn(f"{key}_coverage", scored)

    def test_task_scores_stay_within_0_100(self) -> None:
        scored = score_task_capabilities(_base_result())
        for key in (
            "task_capability_score",
            "task_control_score",
            "task_realization_score",
            "structure_control_score",
            "local_development_score",
            "long_context_coherence_score",
            "infilling_consistency_score",
        ):
            self.assertGreaterEqual(float(scored[key]), 0.0)
            self.assertLessEqual(float(scored[key]), 100.0)

    def test_weighted_scores_match_expected_baseline_values(self) -> None:
        scored = score_task_capabilities(_base_result())

        self.assertAlmostEqual(float(scored["structure_control_score"]), 74.2, places=6)
        self.assertAlmostEqual(float(scored["local_development_score"]), 74.84, places=6)
        self.assertAlmostEqual(float(scored["long_context_coherence_score"]), 80.8, places=6)
        self.assertAlmostEqual(float(scored["infilling_consistency_score"]), 76.4, places=6)
        self.assertAlmostEqual(float(scored["task_capability_score"]), 76.45, places=6)
        self.assertAlmostEqual(float(scored["task_control_score"]), 80.4, places=6)
        self.assertAlmostEqual(float(scored["task_realization_score"]), 73.81666666666666, places=6)

    def test_penalty_metrics_are_inverted_in_local_development(self) -> None:
        baseline = score_task_capabilities(_base_result())
        worsened = score_task_capabilities(
            {
                **_base_result(),
                "local_development_copy_overuse_penalty": 0.90,
                "local_development_unrelated_drift_penalty": 0.85,
            }
        )

        self.assertLess(float(worsened["local_development_score"]), float(baseline["local_development_score"]))
        local_breakdown = baseline["task_capability_score_breakdown"]["dimensions"]["local_development_score"]["submetrics"]
        self.assertAlmostEqual(
            float(local_breakdown["music_realization"]["submetrics"]["local_development_copy_overuse_penalty"]["normalized_value"]),
            0.82,
            places=6,
        )

    def test_missing_task_metrics_reduce_coverage_and_mark_missing_task(self) -> None:
        partial = {
            **_base_result(),
            "infilling_bridge_validity_rate": None,
            "infilling_boundary_compatibility_hit_rate": None,
            "infilling_rhythmic_connection_score": None,
            "infilling_pitch_connection_score": None,
            "infilling_structural_fit_score": None,
        }
        scored = score_task_capabilities(partial)

        self.assertIsNone(scored["infilling_consistency_score"])
        self.assertAlmostEqual(float(scored["infilling_consistency_score_coverage"]), 0.0, places=6)
        self.assertAlmostEqual(float(scored["infilling_consistency_score_weighted_coverage"]), 0.0, places=6)
        self.assertAlmostEqual(float(scored["task_capability_score_coverage"]), 0.8, places=6)
        self.assertAlmostEqual(float(scored["task_capability_score_weighted_coverage"]), 0.8, places=6)
        self.assertIn("infilling_consistency_score", scored["task_capability_score_missing_tasks"])
        self.assertIn("infilling_consistency_score", scored["task_capability_score_unscored_tasks"])

    def test_partial_component_coverage_is_exposed_in_flat_and_breakdown_fields(self) -> None:
        partial = {
            **_base_result(),
            "long_context_degeneration_penalty": None,
        }
        scored = score_task_capabilities(partial)

        self.assertAlmostEqual(float(scored["long_context_coherence_score_coverage"]), 0.88, places=6)
        self.assertAlmostEqual(float(scored["long_context_coherence_score_weighted_coverage"]), 0.88, places=6)
        breakdown = scored["task_capability_score_breakdown"]["dimensions"]["long_context_coherence_score"]
        self.assertAlmostEqual(float(breakdown["coverage"]), 0.88, places=6)
        self.assertAlmostEqual(float(breakdown["weighted_coverage"]), 0.88, places=6)
        self.assertIn("long_context_degeneration_penalty", breakdown["missing_metrics"])

    def test_partial_metric_missing_keeps_score_and_does_not_mark_task_unscored(self) -> None:
        partial = {
            **_base_result(),
            "long_context_degeneration_penalty": None,
        }
        scored = score_task_capabilities(partial)

        self.assertIsNotNone(scored["long_context_coherence_score"])
        self.assertNotIn("long_context_coherence_score", scored["task_capability_score_unscored_tasks"])
        self.assertIn("long_context_coherence_score", scored["task_capability_score_missing_tasks"])

    def test_nan_is_not_exposed_in_external_output_fields(self) -> None:
        partial = {
            **_base_result(),
            "structure_control_boundary_type_hit_rate": None,
            "structure_control_boundary_timing_hit_rate": None,
            "structure_control_post_boundary_realization_score": None,
            "local_development_motif_relation_hit_rate": None,
            "local_development_copy_overuse_penalty": None,
            "local_development_unrelated_drift_penalty": None,
            "local_development_quality_score": None,
            "long_context_completion_rate": None,
            "long_context_theme_retention_score": None,
            "long_context_section_continuity_score": None,
            "long_context_degeneration_penalty": None,
            "infilling_bridge_validity_rate": None,
            "infilling_boundary_compatibility_hit_rate": None,
            "infilling_rhythmic_connection_score": None,
            "infilling_pitch_connection_score": None,
            "infilling_structural_fit_score": None,
        }
        scored = score_task_capabilities(partial)

        for key in (
            "task_capability_score",
            "task_control_score",
            "task_realization_score",
            "structure_control_score",
            "local_development_score",
            "long_context_coherence_score",
            "infilling_consistency_score",
        ):
            self.assertIsNone(scored[key], key)

    def test_breakdown_uses_dimensions_as_only_formal_entry(self) -> None:
        scored = score_task_capabilities(_base_result())
        breakdown = scored["task_capability_score_breakdown"]

        self.assertIn("dimensions", breakdown)
        self.assertNotIn("tasks", breakdown)

    def test_top_level_and_subblocks_expose_coverage_semantics(self) -> None:
        scored = score_task_capabilities(_base_result())
        dimension = scored["task_capability_score_breakdown"]["dimensions"]["structure_control_score"]
        control_block = dimension["submetrics"]["control_hit"]
        music_block = dimension["submetrics"]["music_realization"]

        self.assertEqual(scored["task_capability_score_coverage_semantics"], "weighted_task_coverage_ratio")
        self.assertEqual(dimension["coverage_semantics"], "weighted_component_coverage_ratio")
        self.assertEqual(control_block["coverage_semantics"], "metric_weight_coverage_ratio")
        self.assertEqual(music_block["coverage_semantics"], "metric_weight_coverage_ratio")

    def test_metric_breakdown_sanitizes_non_finite_raw_value(self) -> None:
        scored = score_task_capabilities(
            {
                **_base_result(),
                "local_development_copy_overuse_penalty": math.nan,
                "long_context_degeneration_penalty": math.inf,
            }
        )
        local_metric = scored["task_capability_score_breakdown"]["dimensions"]["local_development_score"]["submetrics"]["music_realization"]["submetrics"]["local_development_copy_overuse_penalty"]
        long_context_metric = scored["task_capability_score_breakdown"]["dimensions"]["long_context_coherence_score"]["submetrics"]["music_realization"]["submetrics"]["long_context_degeneration_penalty"]

        self.assertIsNone(local_metric["raw_value"])
        self.assertIsNone(long_context_metric["raw_value"])
        self.assertIsNone(local_metric["normalized_value"])
        self.assertIsNone(long_context_metric["normalized_value"])

    def test_attach_task_capability_scores_enriches_results(self) -> None:
        results = [_base_result(), {**_base_result(), "long_context_completion_rate": 0.92}]
        enriched_results = attach_task_capability_scores(results)

        self.assertEqual(len(enriched_results), 2)
        for enriched in enriched_results:
            self.assertIn("task_capability_score", enriched)
            self.assertIn("structure_control_score", enriched)

    def test_new_interfaces_are_available_from_src_utils(self) -> None:
        scored = score_task_capabilities_from_utils(_base_result())
        enriched = attach_task_capability_scores_from_utils([_base_result()])

        self.assertIn("task_capability_score", scored)
        self.assertEqual(len(enriched), 1)
        self.assertIn("task_capability_score", enriched[0])


if __name__ == "__main__":
    unittest.main()
