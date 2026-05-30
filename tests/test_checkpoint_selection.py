from __future__ import annotations

import unittest

from src.utils.checkpoint_selection import score_checkpoint_results


class CheckpointSelectionTests(unittest.TestCase):
    def test_continuation_profile_prefers_stronger_stop_behavior(self) -> None:
        results = [
            {
                "checkpoint_name": "step_1.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_1.pt",
                "step": 1,
                "absolute_score": 42.0,
                "continuation_stop_success_rate": 0.10,
                "continuation_budget_stop_rate": 0.90,
                "continuation_structural_validity_rate": 0.40,
                "continuation_time_order_validity_rate": 0.70,
                "continuation_empty_bar_rate": 0.30,
                "continuation_first_event_hit_rate": 0.20,
                "valid_loss_from_training": 4.0,
                "task_capability_score": 55.0,
                "task_capability_score_coverage": 0.78,
                "task_control_score": 58.0,
                "task_realization_score": 54.0,
                "structure_control_score": 52.0,
                "local_development_score": 56.0,
                "long_context_coherence_score": 57.0,
                "vs_baseline_win_rate": 0.50,
            },
            {
                "checkpoint_name": "step_2.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_2.pt",
                "step": 2,
                "absolute_score": 61.5,
                "continuation_stop_success_rate": 0.60,
                "continuation_budget_stop_rate": 0.20,
                "continuation_structural_validity_rate": 0.80,
                "continuation_time_order_validity_rate": 0.95,
                "continuation_empty_bar_rate": 0.05,
                "continuation_first_event_hit_rate": 0.70,
                "valid_loss_from_training": 3.0,
                "task_capability_score": 71.0,
                "task_capability_score_coverage": 0.82,
                "task_control_score": 69.0,
                "task_realization_score": 72.0,
                "structure_control_score": 68.0,
                "local_development_score": 70.0,
                "long_context_coherence_score": 73.0,
                "vs_baseline_win_rate": 0.64,
            },
        ]

        scored, selection = score_checkpoint_results(results, profile="continuation")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_2.pt")
        top = next(item for item in scored if item["checkpoint_name"] == "step_2.pt")
        self.assertTrue(top["gate_passed"])
        self.assertEqual(selection["primary_score_key"], "task_capability_score")
        self.assertAlmostEqual(float(selection["recommended_checkpoint"]["task_capability_score"]), 71.0)
        self.assertAlmostEqual(float(selection["recommended_checkpoint"]["absolute_score"]), 61.5)
        self.assertAlmostEqual(float(selection["leaderboard"][0]["absolute_score"]), 61.5)

    def test_infilling_profile_prefers_structural_validity(self) -> None:
        results = [
            {
                "checkpoint_name": "step_3.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_3.pt",
                "step": 3,
                "infilling_structural_validity_rate": 0.55,
                "infilling_time_order_validity_rate": 0.80,
                "fsm_structural_validity_rate": 0.90,
                "valid_loss_from_training": 2.5,
                "task_capability_score": 63.0,
                "task_capability_score_coverage": 0.20,
                "task_realization_score": 64.0,
                "infilling_consistency_score": 63.0,
                "vs_baseline_win_rate": 0.50,
            },
            {
                "checkpoint_name": "step_4.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_4.pt",
                "step": 4,
                "infilling_structural_validity_rate": 0.75,
                "infilling_time_order_validity_rate": 0.92,
                "fsm_structural_validity_rate": 0.94,
                "valid_loss_from_training": 2.8,
                "task_capability_score": 78.0,
                "task_capability_score_coverage": 0.20,
                "task_realization_score": 79.0,
                "infilling_consistency_score": 78.0,
                "vs_baseline_win_rate": 0.71,
            },
        ]

        _scored, selection = score_checkpoint_results(results, profile="infilling")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_4.pt")
        self.assertEqual(selection["primary_score_key"], "task_capability_score")
        self.assertAlmostEqual(float(selection["recommended_checkpoint"]["task_capability_score"]), 78.0)

    def test_continuation_profile_no_longer_uses_legacy_balanced_score_as_primary(self) -> None:
        results = [
            {
                "checkpoint_name": "legacy_favored.pt",
                "checkpoint_path": "outputs/checkpoints/run/legacy_favored.pt",
                "step": 20,
                "continuation_stop_success_rate": 0.60,
                "continuation_budget_stop_rate": 0.20,
                "continuation_structural_validity_rate": 0.90,
                "continuation_time_order_validity_rate": 0.96,
                "balanced_score": 0.90,
                "task_capability_score": 58.0,
                "task_capability_score_coverage": 0.80,
                "task_realization_score": 57.0,
                "structure_control_score": 56.0,
                "local_development_score": 58.0,
                "long_context_coherence_score": 59.0,
                "vs_baseline_win_rate": 0.55,
            },
            {
                "checkpoint_name": "task_favored.pt",
                "checkpoint_path": "outputs/checkpoints/run/task_favored.pt",
                "step": 21,
                "continuation_stop_success_rate": 0.61,
                "continuation_budget_stop_rate": 0.21,
                "continuation_structural_validity_rate": 0.91,
                "continuation_time_order_validity_rate": 0.97,
                "balanced_score": 0.20,
                "task_capability_score": 72.0,
                "task_capability_score_coverage": 0.82,
                "task_realization_score": 71.0,
                "structure_control_score": 70.0,
                "local_development_score": 72.0,
                "long_context_coherence_score": 73.0,
                "vs_baseline_win_rate": 0.66,
            },
        ]

        _scored, selection = score_checkpoint_results(results, profile="continuation")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "task_favored.pt")

    def test_benchmark_overall_prefers_higher_task_capability_score(self) -> None:
        results = [
            {
                "checkpoint_name": "step_5.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_5.pt",
                "step": 5,
                "continuation_stop_success_rate": 0.72,
                "continuation_budget_stop_rate": 0.18,
                "continuation_structural_validity_rate": 0.90,
                "continuation_time_order_validity_rate": 0.99,
                "continuation_empty_bar_rate": 0.01,
                "continuation_syntax_invalid_rate": 0.02,
                "infilling_structural_validity_rate": 0.92,
                "infilling_time_order_validity_rate": 0.96,
                "phrase_coherence_score": 0.45,
                "long_context_stability_score": 0.65,
                "overall_pitch_diversity_score": 0.44,
                "overall_rhythm_diversity_score": 0.40,
                "continuation_first_event_hit_rate": 0.34,
                "duration_bin_l1_distance": 0.86,
                "overall_same_pitch_overlap_rate": 0.16,
                "continuation_event_ngram_repeat_ratio": 0.12,
                "continuation_rhythm_ngram_repeat_ratio": 0.40,
                "overall_event_ngram_repeat_ratio": 0.10,
                "overall_rhythm_ngram_repeat_ratio": 0.35,
                "valid_loss_from_training": 0.80,
                "task_capability_score": 0.61,
                "task_control_score": 0.72,
                "task_realization_score": 0.60,
                "structure_control_score": 0.66,
                "local_development_score": 0.63,
                "long_context_coherence_score": 0.62,
                "infilling_consistency_score": 0.71,
                "vs_baseline_win_rate": 0.53,
            },
            {
                "checkpoint_name": "step_6.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_6.pt",
                "step": 6,
                "continuation_stop_success_rate": 0.68,
                "continuation_budget_stop_rate": 0.24,
                "continuation_structural_validity_rate": 0.83,
                "continuation_time_order_validity_rate": 0.95,
                "continuation_empty_bar_rate": 0.03,
                "continuation_syntax_invalid_rate": 0.05,
                "infilling_structural_validity_rate": 0.78,
                "infilling_time_order_validity_rate": 0.88,
                "phrase_coherence_score": 0.72,
                "long_context_stability_score": 0.73,
                "overall_pitch_diversity_score": 0.79,
                "overall_rhythm_diversity_score": 0.69,
                "continuation_first_event_hit_rate": 0.61,
                "duration_bin_l1_distance": 0.38,
                "overall_same_pitch_overlap_rate": 0.04,
                "continuation_event_ngram_repeat_ratio": 0.03,
                "continuation_rhythm_ngram_repeat_ratio": 0.12,
                "overall_event_ngram_repeat_ratio": 0.02,
                "overall_rhythm_ngram_repeat_ratio": 0.18,
                "valid_loss_from_training": 0.86,
                "task_capability_score": 0.78,
                "task_control_score": 0.81,
                "task_realization_score": 0.77,
                "structure_control_score": 0.75,
                "local_development_score": 0.73,
                "long_context_coherence_score": 0.74,
                "infilling_consistency_score": 0.76,
                "vs_baseline_win_rate": 0.66,
            },
        ]

        scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_6.pt")
        top = next(item for item in scored if item["checkpoint_name"] == "step_6.pt")
        self.assertTrue(top["gate_passed"])
        self.assertAlmostEqual(
            float(selection["recommended_checkpoint"]["task_capability_score"]),
            0.78,
        )
        self.assertAlmostEqual(
            float(selection["leaderboard"][0]["task_realization_score"]),
            0.77,
        )

    def test_benchmark_overall_gate_still_filters_higher_task_capability(self) -> None:
        results = [
            {
                "checkpoint_name": "gate_failed.pt",
                "checkpoint_path": "outputs/checkpoints/run/gate_failed.pt",
                "step": 10,
                "continuation_stop_success_rate": 0.19,
                "continuation_budget_stop_rate": 0.20,
                "continuation_structural_validity_rate": 0.95,
                "continuation_time_order_validity_rate": 0.95,
                "infilling_structural_validity_rate": 0.95,
                "task_capability_score": 0.99,
                "task_control_score": 0.90,
                "task_realization_score": 0.98,
                "structure_control_score": 0.91,
                "local_development_score": 0.89,
                "long_context_coherence_score": 0.87,
                "infilling_consistency_score": 0.88,
                "vs_baseline_win_rate": 0.80,
            },
            {
                "checkpoint_name": "gate_passed.pt",
                "checkpoint_path": "outputs/checkpoints/run/gate_passed.pt",
                "step": 11,
                "continuation_stop_success_rate": 0.21,
                "continuation_budget_stop_rate": 0.30,
                "continuation_structural_validity_rate": 0.88,
                "continuation_time_order_validity_rate": 0.90,
                "infilling_structural_validity_rate": 0.70,
                "task_capability_score": 0.60,
                "task_control_score": 0.61,
                "task_realization_score": 0.62,
                "structure_control_score": 0.63,
                "local_development_score": 0.64,
                "long_context_coherence_score": 0.65,
                "infilling_consistency_score": 0.66,
                "vs_baseline_win_rate": 0.55,
            },
        ]

        scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "gate_passed.pt")
        failed = next(item for item in scored if item["checkpoint_name"] == "gate_failed.pt")
        self.assertFalse(failed["gate_passed"])
        self.assertIn("continuation_stop_success_rate<0.2000", failed["gate_failed_reasons"])

    def test_benchmark_overall_does_not_require_absolute_score(self) -> None:
        results = [
            {
                "checkpoint_name": "step_7.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_7.pt",
                "step": 7,
                "continuation_stop_success_rate": 0.30,
                "continuation_budget_stop_rate": 0.20,
                "continuation_structural_validity_rate": 0.86,
                "continuation_time_order_validity_rate": 0.90,
                "infilling_structural_validity_rate": 0.72,
                "task_capability_score": 0.70,
                "task_control_score": 0.68,
                "task_realization_score": 0.69,
                "structure_control_score": 0.67,
                "local_development_score": 0.66,
                "long_context_coherence_score": 0.65,
                "infilling_consistency_score": 0.64,
                "vs_baseline_win_rate": 0.60,
            },
            {
                "checkpoint_name": "step_8.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_8.pt",
                "step": 8,
                "continuation_stop_success_rate": 0.31,
                "continuation_budget_stop_rate": 0.19,
                "continuation_structural_validity_rate": 0.87,
                "continuation_time_order_validity_rate": 0.91,
                "infilling_structural_validity_rate": 0.74,
                "task_capability_score": 0.71,
                "task_control_score": 0.69,
                "task_realization_score": 0.70,
                "structure_control_score": 0.68,
                "local_development_score": 0.67,
                "long_context_coherence_score": 0.66,
                "infilling_consistency_score": 0.65,
                "vs_baseline_win_rate": 0.61,
            },
        ]

        _scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_8.pt")
        self.assertEqual(selection["leaderboard"][0]["checkpoint_name"], "step_8.pt")

    def test_benchmark_overall_excludes_results_missing_task_capability_score(self) -> None:
        results = [
            {
                "checkpoint_name": "missing_primary.pt",
                "checkpoint_path": "outputs/checkpoints/run/missing_primary.pt",
                "step": 20,
                "continuation_stop_success_rate": 0.40,
                "continuation_budget_stop_rate": 0.20,
                "continuation_structural_validity_rate": 0.90,
                "continuation_time_order_validity_rate": 0.95,
                "infilling_structural_validity_rate": 0.80,
                "task_realization_score": 0.99,
                "structure_control_score": 0.98,
                "local_development_score": 0.97,
                "long_context_coherence_score": 0.96,
                "infilling_consistency_score": 0.95,
                "vs_baseline_win_rate": 0.94,
            },
            {
                "checkpoint_name": "valid_primary.pt",
                "checkpoint_path": "outputs/checkpoints/run/valid_primary.pt",
                "step": 21,
                "continuation_stop_success_rate": 0.41,
                "continuation_budget_stop_rate": 0.21,
                "continuation_structural_validity_rate": 0.91,
                "continuation_time_order_validity_rate": 0.96,
                "infilling_structural_validity_rate": 0.81,
                "task_capability_score": 0.70,
                "task_capability_score_coverage": 0.95,
                "task_control_score": 0.71,
                "task_realization_score": 0.72,
                "structure_control_score": 0.73,
                "local_development_score": 0.74,
                "long_context_coherence_score": 0.75,
                "infilling_consistency_score": 0.76,
                "vs_baseline_win_rate": 0.77,
            },
        ]

        scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "valid_primary.pt")
        self.assertEqual(selection["eligible_checkpoint_count"], 1)
        self.assertEqual(len(selection["leaderboard"]), 1)
        missing = next(item for item in scored if item["checkpoint_name"] == "missing_primary.pt")
        self.assertIsNone(missing["balanced_rank"])

    def test_benchmark_overall_prefers_higher_task_capability_coverage_when_scores_close(self) -> None:
        results = [
            {
                "checkpoint_name": "low_coverage.pt",
                "checkpoint_path": "outputs/checkpoints/run/low_coverage.pt",
                "step": 30,
                "continuation_stop_success_rate": 0.45,
                "continuation_budget_stop_rate": 0.18,
                "continuation_structural_validity_rate": 0.92,
                "continuation_time_order_validity_rate": 0.97,
                "infilling_structural_validity_rate": 0.83,
                "task_capability_score": 78.10,
                "task_capability_score_coverage": 0.55,
                "task_control_score": 80.0,
                "task_realization_score": 79.0,
                "structure_control_score": 78.0,
                "local_development_score": 77.0,
                "long_context_coherence_score": 76.0,
                "infilling_consistency_score": 75.0,
                "vs_baseline_win_rate": 0.74,
            },
            {
                "checkpoint_name": "high_coverage.pt",
                "checkpoint_path": "outputs/checkpoints/run/high_coverage.pt",
                "step": 31,
                "continuation_stop_success_rate": 0.46,
                "continuation_budget_stop_rate": 0.19,
                "continuation_structural_validity_rate": 0.93,
                "continuation_time_order_validity_rate": 0.98,
                "infilling_structural_validity_rate": 0.84,
                "task_capability_score": 78.05,
                "task_capability_score_coverage": 0.95,
                "task_control_score": 70.0,
                "task_realization_score": 69.0,
                "structure_control_score": 68.0,
                "local_development_score": 67.0,
                "long_context_coherence_score": 66.0,
                "infilling_consistency_score": 65.0,
                "vs_baseline_win_rate": 0.64,
            },
        ]

        _scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "high_coverage.pt")
        self.assertAlmostEqual(
            float(selection["recommended_checkpoint"]["task_capability_score_coverage"]),
            0.95,
        )
        self.assertEqual(selection["leaderboard"][0]["primary_score_key"], "task_capability_score")

    def test_benchmark_overall_selection_metadata_matches_task_capability_semantics(self) -> None:
        results = [
            {
                "checkpoint_name": "meta.pt",
                "checkpoint_path": "outputs/checkpoints/run/meta.pt",
                "step": 40,
                "continuation_stop_success_rate": 0.42,
                "continuation_budget_stop_rate": 0.22,
                "continuation_structural_validity_rate": 0.88,
                "continuation_time_order_validity_rate": 0.93,
                "infilling_structural_validity_rate": 0.79,
                "task_capability_score": 0.73,
                "task_capability_score_coverage": 0.91,
                "task_control_score": 0.74,
                "task_realization_score": 0.75,
                "structure_control_score": 0.76,
                "local_development_score": 0.77,
                "long_context_coherence_score": 0.78,
                "infilling_consistency_score": 0.79,
                "vs_baseline_win_rate": 0.80,
            },
        ]

        _scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        self.assertEqual(selection["primary_score_key"], "task_capability_score")
        self.assertEqual(selection["selection_version"], "task_capability_v1")
        self.assertIn("task-capability", " ".join(selection["notes"]))

    def test_benchmark_overall_recommended_prefers_task_fields_as_primary_skeleton(self) -> None:
        results = [
            {
                "checkpoint_name": "skeleton.pt",
                "checkpoint_path": "outputs/checkpoints/run/skeleton.pt",
                "step": 50,
                "continuation_stop_success_rate": 0.43,
                "continuation_budget_stop_rate": 0.21,
                "continuation_structural_validity_rate": 0.89,
                "continuation_time_order_validity_rate": 0.94,
                "infilling_structural_validity_rate": 0.82,
                "task_capability_score": 0.79,
                "task_capability_score_coverage": 0.93,
                "task_control_score": 0.80,
                "task_realization_score": 0.81,
                "structure_control_score": 0.82,
                "local_development_score": 0.83,
                "long_context_coherence_score": 0.84,
                "infilling_consistency_score": 0.85,
                "vs_baseline_win_rate": 0.73,
            },
        ]

        _scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        recommended = selection["recommended_checkpoint"]
        self.assertEqual(recommended["primary_score_key"], "task_capability_score")
        self.assertEqual(recommended["task_capability_score"], 0.79)
        self.assertEqual(recommended["vs_baseline_win_rate"], 0.73)
        self.assertIn("primary_score_breakdown", recommended)
        self.assertIn("selection_breakdown", recommended)
        self.assertNotIn("score_breakdown", recommended)
        self.assertTrue(recommended["compatibility_only"])
        self.assertIn("balanced_score", recommended["legacy_score_fields"])

        recommended_keys = list(recommended.keys())
        self.assertLess(
            recommended_keys.index("task_capability_score"),
            recommended_keys.index("legacy_score_fields"),
        )
        self.assertLess(
            recommended_keys.index("task_control_score"),
            recommended_keys.index("legacy_score_fields"),
        )

    def test_benchmark_overall_keeps_balanced_score_only_as_compatibility_field(self) -> None:
        results = [
            {
                "checkpoint_name": "compat.pt",
                "checkpoint_path": "outputs/checkpoints/run/compat.pt",
                "step": 60,
                "continuation_stop_success_rate": 0.47,
                "continuation_budget_stop_rate": 0.20,
                "continuation_structural_validity_rate": 0.90,
                "continuation_time_order_validity_rate": 0.95,
                "infilling_structural_validity_rate": 0.84,
                "task_capability_score": 0.77,
                "task_capability_score_coverage": 0.91,
                "task_control_score": 0.76,
                "task_realization_score": 0.75,
                "structure_control_score": 0.74,
                "local_development_score": 0.73,
                "long_context_coherence_score": 0.72,
                "infilling_consistency_score": 0.71,
                "vs_baseline_win_rate": 0.70,
                "absolute_score_version": "absolute_v1",
                "absolute_score": 61.5,
                "absolute_score_coverage": 0.90,
                "absolute_score_breakdown": {"dimensions": {}},
            },
        ]

        _scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        leaderboard_row = selection["leaderboard"][0]
        self.assertEqual(leaderboard_row["primary_score_key"], "task_capability_score")
        self.assertNotIn("balanced_score", leaderboard_row)
        self.assertNotIn("balanced_score_breakdown", leaderboard_row)
        self.assertNotIn("absolute_score", leaderboard_row)
        self.assertNotIn("absolute_score_breakdown", leaderboard_row)
        self.assertNotIn("score_breakdown", leaderboard_row)
        self.assertTrue(leaderboard_row["compatibility_only"])
        self.assertIn("legacy_score_fields", leaderboard_row)
        self.assertIn("balanced_score", leaderboard_row["legacy_score_fields"])
        self.assertIn("balanced_score_breakdown", leaderboard_row["legacy_score_fields"])
        self.assertIn("absolute_score", leaderboard_row["legacy_score_fields"])
        self.assertIn("absolute_score_breakdown", leaderboard_row["legacy_score_fields"])
        self.assertNotIn("legacy_balanced_score", leaderboard_row["selection_breakdown"])
        self.assertNotIn("legacy_balanced_score_coverage", leaderboard_row["selection_breakdown"])
        self.assertEqual(
            leaderboard_row["primary_score_breakdown"],
            leaderboard_row["task_capability_score_breakdown"],
        )
        recommended = selection["recommended_checkpoint"]
        self.assertNotIn("balanced_score", recommended)
        self.assertNotIn("balanced_score_coverage", recommended)
        self.assertNotIn("absolute_score", recommended)
        self.assertNotIn("absolute_score_coverage", recommended)
        self.assertTrue(recommended["compatibility_only"])
        self.assertIn("legacy_score_fields", recommended)
        self.assertIn("balanced_score", recommended["legacy_score_fields"])
        self.assertIn("absolute_score", recommended["legacy_score_fields"])
        self.assertNotIn("legacy_balanced_score", recommended["selection_breakdown"])
        self.assertNotIn("legacy_balanced_score_coverage", recommended["selection_breakdown"])


if __name__ == "__main__":
    unittest.main()
