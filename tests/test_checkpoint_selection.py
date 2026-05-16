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
            },
        ]

        scored, selection = score_checkpoint_results(results, profile="continuation")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_2.pt")
        top = next(item for item in scored if item["checkpoint_name"] == "step_2.pt")
        self.assertTrue(top["gate_passed"])
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
            },
            {
                "checkpoint_name": "step_4.pt",
                "checkpoint_path": "outputs/checkpoints/run/step_4.pt",
                "step": 4,
                "infilling_structural_validity_rate": 0.75,
                "infilling_time_order_validity_rate": 0.92,
                "fsm_structural_validity_rate": 0.94,
                "valid_loss_from_training": 2.8,
            },
        ]

        _scored, selection = score_checkpoint_results(results, profile="infilling")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_4.pt")

    def test_overall_profile_prefers_expression_after_gates_pass(self) -> None:
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
            },
        ]

        scored, selection = score_checkpoint_results(results, profile="benchmark_overall")

        self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_6.pt")
        top = next(item for item in scored if item["checkpoint_name"] == "step_6.pt")
        self.assertTrue(top["gate_passed"])


if __name__ == "__main__":
    unittest.main()
