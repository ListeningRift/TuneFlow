from __future__ import annotations

import random
import unittest

from src.music_analysis import (
    KeyAnalysisConfig,
    PhraseAnalysisConfig,
    PhraseWindowPolicy,
    analyze_key_timeline,
    analyze_phrase_candidates,
    extract_phrase,
    sample_phrase_window,
)
from src.tokenizer.midi_codec import inject_key_tokens
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


def _bars_to_tokens(bar_specs: list[list[tuple[int, int, int]]], *, append_eos: bool = True) -> list[str]:
    tokens = ["BOS", "TEMPO_120"]
    for bar_events in bar_specs:
        tokens.extend(_bar(*bar_events))
    if append_eos:
        tokens.append("EOS")
    return tokens


def _c_major_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 65, 12), (8, 69, 8), (16, 72, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
        ]
    )


def _a_minor_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 57, 12), (8, 64, 8), (16, 69, 12)],
            [(0, 55, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
        ]
    )


def _c_to_g_major_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 65, 12), (8, 69, 8), (16, 72, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 67, 12), (8, 71, 8), (16, 74, 12)],
            [(0, 64, 12), (8, 67, 8), (16, 71, 12)],
            [(0, 60, 12), (8, 67, 8), (16, 71, 12)],
            [(0, 67, 12), (8, 71, 8), (16, 74, 12)],
        ]
    )


def _single_misleading_bar_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 65, 12), (8, 69, 8), (16, 72, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 67, 12), (8, 71, 8), (16, 66, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 65, 12), (8, 69, 8), (16, 72, 12)],
        ]
    )


def _ambiguous_tokens() -> list[str]:
    chromatic_bar = [(index * 2, 60 + index, 1) for index in range(12)]
    return _bars_to_tokens([chromatic_bar, chromatic_bar, chromatic_bar, chromatic_bar])


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

    def test_extract_phrase_keeps_only_window_start_key_token(self) -> None:
        tokens = inject_key_tokens(_c_to_g_major_tokens())
        analysis = analyze_phrase_candidates(tokens)
        phrase = extract_phrase(tokens, analysis, 1)
        self.assertEqual(phrase.key_token, "KEY_G_MAJ")
        self.assertEqual(phrase.tokens[0], "BOS")
        self.assertEqual(phrase.tokens[1], "TEMPO_120")
        self.assertEqual(phrase.tokens[2], "KEY_G_MAJ")
        self.assertEqual(sum(1 for token in phrase.tokens if token.startswith("KEY_")), 1)

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

    def test_eval_window_keeps_only_window_start_key(self) -> None:
        tokens = inject_key_tokens(_c_to_g_major_tokens())
        rng = random.Random(7)
        window = sample_bar_aligned_subsequence(tokens, max_core_tokens=80, min_core_tokens=24, rng=rng)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(window[0], "BOS")
        self.assertEqual(window[-1], "EOS")
        self.assertLessEqual(sum(1 for token in window if token.startswith("KEY_")), 1)


    def test_phrase_fim_builder_falls_back_to_generic_structure(self) -> None:
        base_tokens_text = ["BOS", "TEMPO_120", "KEY_UNCERTAIN", "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_1", "VEL_8", "BAR", "EOS"]
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
            "KEY_UNCERTAIN",
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

    def test_key_timeline_detects_single_major_key(self) -> None:
        analysis = analyze_key_timeline(_c_major_tokens())
        self.assertEqual(analysis.initial_key, "C:maj")
        self.assertEqual(len(analysis.segments), 1)
        self.assertEqual(analysis.segments[0].key, "C:maj")
        self.assertEqual(len(analysis.modulation_points), 0)

    def test_key_timeline_detects_single_minor_key(self) -> None:
        analysis = analyze_key_timeline(_a_minor_tokens())
        self.assertEqual(analysis.initial_key, "A:min")
        self.assertEqual(len(analysis.segments), 1)
        self.assertEqual(analysis.segments[0].key, "A:min")

    def test_key_timeline_detects_modulation_point(self) -> None:
        analysis = analyze_key_timeline(_c_to_g_major_tokens())
        self.assertEqual(analysis.initial_key, "C:maj")
        self.assertEqual([segment.key for segment in analysis.segments], ["C:maj", "G:maj"])
        self.assertEqual(len(analysis.modulation_points), 1)
        self.assertEqual(analysis.modulation_points[0].from_key, "C:maj")
        self.assertEqual(analysis.modulation_points[0].to_key, "G:maj")
        self.assertGreaterEqual(analysis.modulation_points[0].bar_index, 3)

    def test_key_timeline_ignores_single_misleading_bar(self) -> None:
        analysis = analyze_key_timeline(_single_misleading_bar_tokens())
        self.assertEqual(len(analysis.segments), 1)
        self.assertEqual(analysis.segments[0].key, "C:maj")
        self.assertEqual(len(analysis.modulation_points), 0)

    def test_key_timeline_marks_ambiguous_sequence_uncertain(self) -> None:
        analysis = analyze_key_timeline(_ambiguous_tokens())
        self.assertEqual(analysis.initial_key, "uncertain")
        self.assertFalse(analysis.segments)
        self.assertTrue(any(frame.is_uncertain for frame in analysis.frames))

    def test_key_timeline_accepts_missing_terminal_eos(self) -> None:
        tokens = _c_major_tokens()[:-1]
        analysis = analyze_key_timeline(tokens)
        self.assertEqual(analysis.initial_key, "C:maj")
        self.assertEqual(len(analysis.segments), 1)

    def test_key_timeline_keeps_finer_frames_than_segments(self) -> None:
        analysis = analyze_key_timeline(_c_major_tokens(), config=KeyAnalysisConfig(window_bars=1.0, hop_bars=0.5))
        self.assertGreater(len(analysis.frames), len(analysis.segments))
        self.assertEqual(len(analysis.frames), 7)
        self.assertEqual(analysis.segments[0].start_bar, 0)
        self.assertEqual(analysis.segments[0].start_pos, 0)
        self.assertEqual(analysis.segments[0].end_bar, 4)
        self.assertEqual(analysis.segments[0].end_pos, 0)

    def test_key_timeline_ignores_existing_key_tokens(self) -> None:
        analysis = analyze_key_timeline(inject_key_tokens(_c_to_g_major_tokens()))
        self.assertEqual(analysis.initial_key, "C:maj")
        self.assertEqual([segment.key for segment in analysis.segments], ["C:maj", "G:maj"])


if __name__ == "__main__":
    unittest.main()
