from __future__ import annotations

import random
import unittest

from src.music_analysis import (
    PhraseAnalysisConfig,
    PhraseWindowPolicy,
    analyze_phrase_candidates,
    extract_phrase,
    sample_phrase_window,
)
from src.training.train_base import PhraseSamplingConfig, TokenBinDataset
from src.utils.eval_windows import sample_bar_aligned_subsequence


def _bar(*events: tuple[int, int, int], tempo: str | None = None) -> list[str]:
    tokens = ["BAR"]
    if tempo is not None:
        tokens.append(tempo)
    for pos, pitch, dur in events:
        tokens.extend(
            [
                f"POS_{pos}",
                "INST_PIANO",
                f"PITCH_{pitch}",
                f"DUR_{dur}",
                "VEL_8",
            ]
        )
    return tokens


def _phrase_source_tokens() -> list[str]:
    tokens = ["BOS", "TEMPO_120"]
    tokens.extend(_bar((0, 60, 4), (8, 64, 4)))
    tokens.extend(_bar((0, 62, 4), (8, 65, 4)))
    tokens.extend(_bar())
    tokens.extend(_bar((0, 72, 8), (16, 76, 4), tempo="TEMPO_132"))
    tokens.extend(_bar((0, 74, 8), (12, 77, 4)))
    tokens.extend(_bar())
    tokens.extend(_bar((0, 67, 4), (8, 71, 4), (16, 74, 4)))
    tokens.extend(_bar((0, 69, 4), (8, 72, 4)))
    tokens.append("EOS")
    return tokens


def _long_phrase_source_tokens() -> list[str]:
    tokens = ["BOS", "TEMPO_120"]
    phrase_specs = [
        [(0, 60, 4), (8, 64, 4)],
        [(0, 62, 4), (8, 65, 4)],
        [(0, 64, 4), (8, 67, 4)],
        [],
        [(0, 72, 8), (16, 76, 4)],
        [(0, 74, 8), (12, 77, 4)],
        [(0, 76, 8), (16, 79, 4)],
        [],
        [(0, 67, 4), (8, 71, 4), (16, 74, 4)],
        [(0, 69, 4), (8, 72, 4)],
        [(0, 71, 4), (8, 74, 4)],
        [],
        [(0, 55, 8), (16, 59, 4)],
        [(0, 57, 8), (12, 60, 4)],
        [(0, 59, 8), (16, 62, 4)],
    ]
    for index, events in enumerate(phrase_specs):
        tempo = "TEMPO_132" if index == 4 else None
        tokens.extend(_bar(*events, tempo=tempo))
    tokens.append("EOS")
    return tokens


class MusicAnalysisTests(unittest.TestCase):
    def test_analyze_phrase_candidates_detects_boundaries_and_lengths(self) -> None:
        analysis = analyze_phrase_candidates(_phrase_source_tokens(), config=PhraseAnalysisConfig())
        self.assertEqual(len(analysis.bars), 8)
        self.assertTrue(any(score.bar_index in {2, 3, 5, 6} for score in analysis.boundary_scores if score.score > 0.5))
        self.assertTrue(all(2 <= (span.end_bar - span.start_bar) <= 8 for span in analysis.phrase_spans))

    def test_extract_phrase_rebuilds_single_tempo_view(self) -> None:
        tokens = _phrase_source_tokens()
        analysis = analyze_phrase_candidates(tokens)
        phrase = extract_phrase(tokens, analysis, 2)
        self.assertEqual(phrase.tokens[0], "BOS")
        self.assertEqual(phrase.tokens[-1], "EOS")
        self.assertEqual(phrase.tempo_token, "TEMPO_132")
        self.assertEqual(phrase.tokens[1], "TEMPO_132")
        self.assertEqual(sum(1 for token in phrase.tokens if token.startswith("TEMPO_")), 1)

    def test_analyze_phrase_candidates_accepts_missing_terminal_eos(self) -> None:
        tokens = _phrase_source_tokens()[:-1]
        analysis = analyze_phrase_candidates(tokens)
        self.assertEqual(len(analysis.bars), 8)
        self.assertTrue(analysis.phrase_spans)

    def test_sample_phrase_window_supports_cross_boundary_and_long_context(self) -> None:
        tokens = _long_phrase_source_tokens()
        analysis = analyze_phrase_candidates(tokens)
        rng = random.Random(42)

        cross_window = sample_phrase_window(
            tokens,
            analysis,
            PhraseWindowPolicy(kind="cross_boundary", min_bars=4, max_bars=8, max_tokens=128),
            rng,
        )
        self.assertIsNotNone(cross_window)
        assert cross_window is not None
        self.assertEqual(cross_window.source_kind, "cross_boundary")
        self.assertGreaterEqual(cross_window.boundary_count, 1)

        long_window = sample_phrase_window(
            tokens,
            analysis,
            PhraseWindowPolicy(kind="long_context", min_bars=6, max_bars=12, max_tokens=256),
            rng,
        )
        self.assertIsNotNone(long_window)
        assert long_window is not None
        self.assertEqual(long_window.source_kind, "long_context")
        self.assertGreaterEqual(long_window.boundary_count, 1)

    def test_eval_window_keeps_only_window_start_tempo(self) -> None:
        tokens = _phrase_source_tokens()
        rng = random.Random(7)
        window = sample_bar_aligned_subsequence(tokens, max_core_tokens=48, min_core_tokens=12, rng=rng)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(window[0], "BOS")
        self.assertEqual(window[-1], "EOS")
        self.assertLessEqual(sum(1 for token in window if token.startswith("TEMPO_")), 1)

    def test_phrase_fim_builder_falls_back_to_generic_structure(self) -> None:
        base_tokens_text = ["BOS", "TEMPO_120", "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_1", "VEL_8", "BAR", "EOS"]
        vocab_tokens = [
            "BOS",
            "EOS",
            "FIM_HOLE",
            "FIM_MID",
            "BAR",
            "POS_0",
            "INST_PIANO",
            "PITCH_60",
            "DUR_1",
            "VEL_8",
            "TEMPO_120",
        ]
        token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
        id_to_token = list(vocab_tokens)
        fim_input, fim_labels, _source_kind, used_phrase_hole = TokenBinDataset._build_phrase_or_fallback_fim_example(
            base_tokens_text=base_tokens_text,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            rng=random.Random(123),
            source_kind="single_phrase",
            phrase_sampling=PhraseSamplingConfig(enabled=True),
            fim_hole_token_id=token_to_id["FIM_HOLE"],
            fim_mid_token_id=token_to_id["FIM_MID"],
            fim_min_span=4,
            fim_max_span=16,
            append_eos=False,
            eos_token_id=token_to_id["EOS"],
        )
        self.assertFalse(used_phrase_hole)
        self.assertIn(token_to_id["FIM_HOLE"], fim_input)
        self.assertIn(token_to_id["FIM_MID"], fim_input)
        self.assertIn(-100, fim_labels)

    def test_phrase_fim_builder_uses_phrase_hole_on_rich_window(self) -> None:
        base_tokens_text = _long_phrase_source_tokens()
        vocab_tokens = ["FIM_HOLE", "FIM_MID", *dict.fromkeys(base_tokens_text)]
        token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
        id_to_token = list(vocab_tokens)
        fim_input, fim_labels, _source_kind, used_phrase_hole = TokenBinDataset._build_phrase_or_fallback_fim_example(
            base_tokens_text=base_tokens_text,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            rng=random.Random(42),
            source_kind="cross_boundary",
            phrase_sampling=PhraseSamplingConfig(enabled=True),
            fim_hole_token_id=token_to_id["FIM_HOLE"],
            fim_mid_token_id=token_to_id["FIM_MID"],
            fim_min_span=16,
            fim_max_span=128,
            append_eos=False,
            eos_token_id=token_to_id["EOS"],
        )
        self.assertTrue(used_phrase_hole)
        self.assertIn(token_to_id["FIM_HOLE"], fim_input)
        self.assertIn(token_to_id["FIM_MID"], fim_input)
        self.assertIn(-100, fim_labels)

    def test_phrase_fim_builder_keeps_phrase_hole_when_reappending_eos(self) -> None:
        base_tokens_text = _long_phrase_source_tokens()
        vocab_tokens = ["FIM_HOLE", "FIM_MID", *dict.fromkeys(base_tokens_text)]
        token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
        id_to_token = list(vocab_tokens)
        fim_input, fim_labels, _source_kind, used_phrase_hole = TokenBinDataset._build_phrase_or_fallback_fim_example(
            base_tokens_text=base_tokens_text,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            rng=random.Random(7),
            source_kind="cross_boundary",
            phrase_sampling=PhraseSamplingConfig(enabled=True),
            fim_hole_token_id=token_to_id["FIM_HOLE"],
            fim_mid_token_id=token_to_id["FIM_MID"],
            fim_min_span=16,
            fim_max_span=128,
            append_eos=True,
            eos_token_id=token_to_id["EOS"],
        )
        self.assertTrue(used_phrase_hole)
        self.assertEqual(fim_input[-1], token_to_id["EOS"])
        self.assertIn(token_to_id["FIM_HOLE"], fim_input)
        self.assertIn(token_to_id["FIM_MID"], fim_input)
        self.assertIn(-100, fim_labels)


if __name__ == "__main__":
    unittest.main()
