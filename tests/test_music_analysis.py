from __future__ import annotations

import random
import unittest

from src.music_analysis import (
    KeyAnalysisConfig,
    PhraseAnalysisConfig,
    PhraseBoundary,
    analyze_key_timeline,
    analyze_phrase_candidates,
)
from src.tokenizer.midi_codec import inject_key_tokens
from src.utils.eval_windows import sample_phrase_aligned_subsequence


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

    def test_bar_info_exposes_onset_positions(self) -> None:
        analysis = analyze_phrase_candidates(_phrase_source_tokens())
        self.assertEqual(analysis.bars[0].onset_positions, (0, 8))
        self.assertEqual(analysis.bars[3].onset_positions, (0, 16))

    def test_analyze_phrase_candidates_returns_boundaries(self) -> None:
        analysis = analyze_phrase_candidates(_phrase_source_tokens())
        self.assertTrue(analysis.boundaries)
        first_content_bar = next(i for i, bar in enumerate(analysis.bars) if bar.note_count > 0)
        self.assertEqual(analysis.boundaries[0], PhraseBoundary(first_content_bar, 0))

    def test_phrase_spans_align_with_boundaries(self) -> None:
        analysis = analyze_phrase_candidates(_phrase_source_tokens())
        expected_starts = tuple(b.bar_index for b in analysis.boundaries)
        actual_starts = tuple(span.start_bar for span in analysis.phrase_spans)
        self.assertEqual(actual_starts, expected_starts)

    def test_mid_bar_anchor_picks_first_onset_when_rest_threshold_met(self) -> None:
        from src.music_analysis.phrase_analysis import _pick_in_bar_anchor, BarInfo
        cfg = PhraseAnalysisConfig()
        right = BarInfo(
            start_token=0, end_token=0, note_count=2, onset_count=2,
            rest_ratio=0.5, pitch_span=0, mean_duration=4.0,
            effective_tempo_token=None, effective_key_token=None,
            onset_positions=(16, 24),
        )
        left = BarInfo(
            start_token=0, end_token=0, note_count=4, onset_count=4,
            rest_ratio=0.0, pitch_span=0, mean_duration=4.0,
            effective_tempo_token=None, effective_key_token=None,
            onset_positions=(0, 8, 16, 24),
        )
        self.assertEqual(_pick_in_bar_anchor(left, right, cfg), 16)

    def test_mid_bar_anchor_returns_zero_when_no_lead_rest(self) -> None:
        from src.music_analysis.phrase_analysis import _pick_in_bar_anchor, BarInfo
        cfg = PhraseAnalysisConfig()
        right = BarInfo(
            start_token=0, end_token=0, note_count=2, onset_count=2,
            rest_ratio=0.0, pitch_span=0, mean_duration=4.0,
            effective_tempo_token=None, effective_key_token=None,
            onset_positions=(0, 8),
        )
        left = BarInfo(
            start_token=0, end_token=0, note_count=4, onset_count=4,
            rest_ratio=0.0, pitch_span=0, mean_duration=4.0,
            effective_tempo_token=None, effective_key_token=None,
            onset_positions=(0, 8, 16, 24),
        )
        self.assertEqual(_pick_in_bar_anchor(left, right, cfg), 0)

    def test_analyze_phrase_candidates_accepts_missing_terminal_eos(self) -> None:
        tokens = _phrase_source_tokens()[:-1]
        analysis = analyze_phrase_candidates(tokens)
        self.assertEqual(len(analysis.bars), 8)
        self.assertTrue(analysis.phrase_spans)

    def test_eval_window_keeps_only_window_start_tempo(self) -> None:
        tokens = _phrase_source_tokens()
        rng = random.Random(7)
        window = sample_phrase_aligned_subsequence(tokens, max_core_tokens=48, min_core_tokens=12, rng=rng)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(window[0], "BOS")
        self.assertEqual(window[-1], "EOS")
        self.assertLessEqual(sum(1 for token in window if token.startswith("TEMPO_")), 1)

    def test_eval_window_keeps_only_window_start_key(self) -> None:
        tokens = inject_key_tokens(_c_to_g_major_tokens())
        rng = random.Random(7)
        window = sample_phrase_aligned_subsequence(tokens, max_core_tokens=80, min_core_tokens=24, rng=rng)
        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual(window[0], "BOS")
        self.assertEqual(window[-1], "EOS")
        self.assertLessEqual(sum(1 for token in window if token.startswith("KEY_")), 1)

    def test_eval_window_starts_on_bar_after_header(self) -> None:
        from src.tokenizer import TokenizerConfig, build_vocab
        from src.tokenizer.midi_codec import inject_phrase_tokens, validate_token_order
        raw = _phrase_source_tokens()
        tokens = inject_phrase_tokens(inject_key_tokens(raw))
        vocab = build_vocab(TokenizerConfig())
        rng = random.Random(0)
        for _ in range(8):
            window = sample_phrase_aligned_subsequence(
                tokens, max_core_tokens=120, min_core_tokens=12, rng=rng,
            )
            if window is None:
                continue
            self.assertEqual(window[0], "BOS")
            self.assertEqual(window[-1], "EOS")
            # First non-header body token must be BAR (validate_token_order requirement)
            idx = 1
            while idx < len(window) and (
                window[idx].startswith("TEMPO_") or window[idx].startswith("KEY_")
            ):
                idx += 1
            self.assertEqual(window[idx], "BAR", msg=f"window: {window}")
            # The window itself must be a valid full sequence
            valid, oov = validate_token_order(window, vocab)
            self.assertTrue(valid, msg=f"window failed validate: {window}")
            self.assertEqual(oov, 0)

    def test_eval_window_keeps_inline_phrases(self) -> None:
        from src.tokenizer.midi_codec import inject_phrase_tokens
        raw = _long_phrase_source_tokens()
        tokens = inject_phrase_tokens(inject_key_tokens(raw))
        # confirm source contains PHRASE
        self.assertIn("PHRASE", tokens)
        survived = 0
        for seed in range(16):
            window = sample_phrase_aligned_subsequence(
                tokens, max_core_tokens=240, min_core_tokens=24, rng=random.Random(seed),
            )
            if window is not None and "PHRASE" in window:
                survived += 1
        self.assertGreater(survived, 0, msg="No window preserved an inline PHRASE")


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
