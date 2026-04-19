from __future__ import annotations

import unittest

from src.utils.absolute_benchmark_scoring import attach_absolute_capability_scores, score_absolute_capabilities


def _base_result() -> dict[str, float]:
    return {
        "continuation_stop_success_rate": 0.28,
        "continuation_budget_stop_rate": 0.68,
        "continuation_missing_eos_rate": 0.68,
        "append_eos_recoverable_rate": 0.18,
        "continuation_structural_validity_rate": 0.52,
        "continuation_time_order_validity_rate": 0.97,
        "continuation_empty_bar_rate": 0.04,
        "low_density_bar_rate": 0.08,
        "continuation_syntax_invalid_rate": 0.48,
        "infilling_structural_validity_rate": 0.86,
        "infilling_time_order_validity_rate": 0.56,
        "infilling_syntax_invalid_rate": 0.10,
        "infilling_pitch_collapse_coverage": 0.90,
        "continuation_first_event_hit_rate": 0.26,
        "duration_bin_l1_distance": 0.78,
        "continuation_pitch_diversity_score": 0.56,
        "infilling_pitch_diversity_score": 0.51,
        "continuation_most_common_pitch_ratio": 0.34,
        "infilling_most_common_pitch_ratio": 0.39,
        "multi_empty_bar_run_rate": 0.03,
        "continuation_longest_same_pitch_run_ratio": 0.28,
        "infilling_longest_same_pitch_run_ratio": 0.30,
        "overall_most_common_pitch_ratio": 0.36,
        "overall_pitch_diversity_score": 0.54,
        "valid_loss_from_training": 0.82,
        "best_valid_loss_so_far": 0.80,
        "train_loss_ema": 0.95,
        "overfit_gap": -0.12,
    }


class AbsoluteBenchmarkScoringTests(unittest.TestCase):
    def test_absolute_score_is_stable_across_candidate_sets(self) -> None:
        target = _base_result()
        other = {**_base_result(), "continuation_stop_success_rate": 0.10, "absolute_score": None}
        solo = attach_absolute_capability_scores([target])[0]
        with_other = attach_absolute_capability_scores([target, other])[0]
        self.assertAlmostEqual(float(solo["absolute_score"]), float(with_other["absolute_score"]))
        self.assertAlmostEqual(
            float(solo["continuation_closure_score"]),
            float(with_other["continuation_closure_score"]),
        )

    def test_improving_metrics_raises_associated_dimension(self) -> None:
        baseline = score_absolute_capabilities(_base_result())
        improved_payload = {
            **_base_result(),
            "continuation_stop_success_rate": 0.42,
            "continuation_budget_stop_rate": 0.42,
            "continuation_missing_eos_rate": 0.40,
            "append_eos_recoverable_rate": 0.26,
        }
        improved = score_absolute_capabilities(improved_payload)
        self.assertGreater(
            float(improved["continuation_closure_score"]),
            float(baseline["continuation_closure_score"]),
        )

    def test_dimension_scores_and_total_score_stay_within_0_100(self) -> None:
        scored = score_absolute_capabilities(_base_result())
        self.assertGreaterEqual(float(scored["absolute_score"]), 0.0)
        self.assertLessEqual(float(scored["absolute_score"]), 100.0)
        for key in (
            "continuation_closure_score",
            "continuation_structure_score",
            "infilling_integrity_score",
            "phrase_coherence_score",
            "long_context_stability_score",
            "training_health_score",
        ):
            self.assertGreaterEqual(float(scored[key]), 0.0)
            self.assertLessEqual(float(scored[key]), 100.0)

    def test_baseline_result_stays_in_mid_band_after_tightening(self) -> None:
        scored = score_absolute_capabilities(_base_result())
        self.assertGreater(float(scored["absolute_score"]), 50.0)
        self.assertLess(float(scored["absolute_score"]), 70.0)

    def test_missing_metrics_do_not_crash_and_report_coverage(self) -> None:
        scored = score_absolute_capabilities(
            {
                "continuation_stop_success_rate": 0.30,
                "continuation_budget_stop_rate": 0.55,
            }
        )
        self.assertIsNotNone(scored["absolute_score"])
        self.assertLess(float(scored["absolute_score_coverage"]), 1.0)
        self.assertIn("infilling_integrity_score", scored["absolute_score_missing_dimensions"])
        self.assertLess(float(scored["continuation_closure_score_coverage"]), 1.0)

    def test_center_goal_metric_for_overfit_gap_does_not_raise(self) -> None:
        scored = score_absolute_capabilities(
            {
                **_base_result(),
                "overfit_gap": -0.08,
            }
        )
        self.assertIsNotNone(scored["training_health_score"])
        breakdown = scored["absolute_score_breakdown"]["dimensions"]["training_health_score"]["submetrics"]["overfit_gap"]
        self.assertIsNotNone(breakdown["score"])


if __name__ == "__main__":
    unittest.main()
