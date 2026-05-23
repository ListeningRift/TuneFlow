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
from src.music_analysis.phrase_analysis import (
    AnalysisAnchor,
    BarInfo,
    BoundaryFeature,
    HierarchicalBoundaryScore,
    NoteInfo,
    _assemble_phrase_boundaries_from_scores,
    _best_core_repeat_similarity,
    _build_note_info,
    _collect_core_trim_candidate_ranges,
    _detect_adjacent_repeated_bar_span_targets,
    _derive_phrase_spans,
    _postprocess_boundary_scores,
    _promote_salient_gap_boundaries,
    _score_boundary_features,
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


def _no_clear_first_phrase_tokens() -> list[str]:
    """构造首句缺少强结构证据的连续上行片段。"""

    return _bars_to_tokens(
        [
            [(0, 60, 4), (4, 62, 4), (8, 64, 4), (12, 65, 4)],
            [(0, 67, 4), (4, 69, 4), (8, 71, 4), (12, 72, 4)],
            [(0, 74, 4), (4, 76, 4), (8, 77, 4), (12, 79, 4)],
        ]
    )


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


def _mid_bar_anchor_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 4), (8, 64, 4)],
            [(0, 61, 4), (8, 65, 4)],
            [(0, 62, 4), (8, 66, 4)],
            [(0, 63, 4), (8, 67, 4)],
            [(0, 64, 4), (8, 68, 4)],
            [(16, 75, 4)],
        ]
    )


def _repeated_four_bar_phrase_tokens() -> list[str]:
    """构造相邻四小节重复段，验证真实 phrase 边界会被保留。"""

    repeated_phrase = [
        [(0, 60, 4), (16, 64, 4)],
        [(0, 62, 4), (16, 65, 4)],
        [(0, 64, 4), (16, 67, 4)],
        [(0, 65, 4), (16, 69, 4)],
    ]
    return _bars_to_tokens(
        [
            *repeated_phrase,
            *repeated_phrase,
            [(0, 72, 8), (16, 76, 8)],
            [(0, 74, 8), (16, 77, 8)],
        ]
    )


def _head_variant_repeated_four_bar_phrase_tokens() -> list[str]:
    """构造前缀不同但主体重复的四小节对照段。"""

    return _bars_to_tokens(
        [
            [(0, 72, 2), (2, 69, 2), (4, 65, 2), (6, 62, 2), (8, 60, 2)],
            [(0, 60, 4), (8, 62, 4), (16, 64, 4)],
            [(0, 65, 4), (8, 67, 4), (16, 69, 4)],
            [(0, 67, 4), (8, 69, 4), (16, 71, 4)],
            [(0, 55, 2), (2, 58, 2), (4, 61, 2), (6, 64, 2), (8, 67, 2)],
            [(0, 60, 4), (8, 62, 4), (16, 64, 4)],
            [(0, 65, 4), (8, 67, 4), (16, 69, 4)],
            [(0, 67, 4), (8, 69, 4), (16, 71, 4)],
        ]
    )


def _three_repeated_four_bar_phrase_tokens() -> list[str]:
    """构造连续三次重复的四小节链式结构。"""

    repeated_phrase = [
        [(0, 60, 4), (16, 64, 4)],
        [(0, 62, 4), (16, 65, 4)],
        [(0, 64, 4), (16, 67, 4)],
        [(0, 65, 4), (16, 69, 4)],
    ]
    return _bars_to_tokens(
        [
            *repeated_phrase,
            *repeated_phrase,
            *repeated_phrase,
            [(0, 72, 8), (16, 76, 8)],
            [(0, 74, 8), (16, 77, 8)],
        ]
    )

def _partial_repeated_bar_span_tokens() -> list[str]:
    """构造只在相邻两段 bar-span 中局部重复的样例。"""

    return _bars_to_tokens(
        [
            [(0, 72, 4), (8, 74, 4), (16, 76, 4), (24, 77, 4)],
            [(0, 79, 4), (8, 81, 4), (16, 83, 4), (24, 84, 4)],
            [(0, 55, 2), (4, 67, 6), (16, 50, 2), (20, 62, 4)],
            [(0, 60, 4), (8, 62, 4), (16, 64, 4), (24, 65, 4)],
            [(0, 60, 4), (8, 62, 4), (16, 64, 4), (24, 65, 4)],
            [(0, 80, 2), (4, 78, 2), (12, 73, 6), (24, 70, 2)],
            [(0, 74, 4), (8, 76, 4), (16, 77, 4), (24, 79, 4)],
            [(0, 81, 4), (8, 83, 4), (16, 84, 4), (24, 86, 4)],
        ]
    )


def _dense_note_fragment(*, onset_count: int, notes_per_onset: int = 2) -> tuple[NoteInfo, ...]:
    """构造高密度长片段，便于约束长片段裁剪搜索规模。"""

    notes: list[NoteInfo] = []
    note_index = 0
    for onset_index in range(onset_count):
        start_unit = onset_index * 2
        bar_index = start_unit // 32
        pos_in_bar = start_unit % 32
        for chord_offset in range(notes_per_onset):
            pitch = 60 + chord_offset + (onset_index % 5)
            notes.append(
                NoteInfo(
                    note_index=note_index,
                    start_unit=start_unit,
                    end_unit=start_unit + 2,
                    duration=2,
                    pitch=pitch,
                    bar_index=bar_index,
                    pos_in_bar=pos_in_bar,
                    effective_key_token=None,
                )
            )
            note_index += 1
    return tuple(notes)


class MusicAnalysisTests(unittest.TestCase):
    def test_analyze_phrase_candidates_detects_boundaries_and_lengths(self) -> None:
        analysis = analyze_phrase_candidates(_phrase_source_tokens(), config=PhraseAnalysisConfig())
        self.assertEqual(len(analysis.bars), 8)
        self.assertEqual(len(analysis.boundary_scores), len(analysis.notes) - 1)
        self.assertTrue(any(item.boundary_type in {"motif", "subphrase", "phrase"} for item in analysis.boundary_scores))
        self.assertTrue(analysis.phrase_spans)
        self.assertTrue(all(span.end_bar > span.start_bar for span in analysis.phrase_spans))

    def test_analyze_phrase_candidates_exposes_note_level_results(self) -> None:
        """验证乐句分析会暴露 note-level 特征与分层评分结果。"""

        analysis = analyze_phrase_candidates(_phrase_source_tokens(), config=PhraseAnalysisConfig())
        self.assertGreater(len(analysis.notes), 0)
        self.assertEqual(len(analysis.boundary_features), len(analysis.notes) - 1)
        self.assertEqual(len(analysis.boundary_scores), len(analysis.notes) - 1)
        self.assertFalse(hasattr(analysis.boundary_scores[0], "score"))
        self.assertTrue(
            all(
                score.motif_score >= 0.0
                and score.subphrase_score >= 0.0
                and score.phrase_score >= 0.0
                for score in analysis.boundary_scores
            )
        )

    def test_phrase_analysis_exposes_analysis_start_separately(self) -> None:
        """验证分析起点会通过独立字段暴露。"""

        analysis = analyze_phrase_candidates(_phrase_source_tokens(), config=PhraseAnalysisConfig())
        self.assertIsNotNone(analysis.analysis_start)
        self.assertEqual(analysis.analysis_start.bar_index, 0)
        self.assertEqual(analysis.analysis_start.anchor_pos, 0)

    def test_analysis_start_is_not_forced_into_boundaries(self) -> None:
        """验证存在真实边界时，分析起点不会被强制写入真实结构边界。"""

        analysis = analyze_phrase_candidates(_repeated_four_bar_phrase_tokens(), config=PhraseAnalysisConfig())
        self.assertTrue(analysis.boundaries)
        self.assertIsNotNone(analysis.analysis_start)
        self.assertNotEqual(
            (analysis.boundaries[0].bar_index, analysis.boundaries[0].anchor_pos),
            (analysis.analysis_start.bar_index, analysis.analysis_start.anchor_pos),
        )

    def test_no_forced_first_phrase_boundary_when_no_strong_evidence(self) -> None:
        """验证首句缺少强证据时，不会再强制补真实结构边界。"""

        analysis = analyze_phrase_candidates(_no_clear_first_phrase_tokens(), config=PhraseAnalysisConfig())
        self.assertEqual(tuple(analysis.boundaries), tuple())

    def test_first_note_fields_are_serialized_from_tokens(self) -> None:
        """验证首个音符字段会按 token 正确序列化。"""

        analysis = analyze_phrase_candidates(_phrase_source_tokens(), config=PhraseAnalysisConfig())
        first = analysis.notes[0]
        self.assertEqual(first.note_index, 0)
        self.assertEqual(first.start_unit, 0)
        self.assertEqual(first.duration, 4)
        self.assertEqual(first.bar_index, 0)
        self.assertEqual(first.pos_in_bar, 0)

    def test_bar_info_exposes_onset_positions(self) -> None:
        analysis = analyze_phrase_candidates(_phrase_source_tokens())
        self.assertEqual(analysis.bars[0].onset_positions, (0, 8))
        self.assertEqual(analysis.bars[3].onset_positions, (0, 16))

    def test_analyze_phrase_candidates_returns_boundaries(self) -> None:
        analysis = analyze_phrase_candidates(_repeated_four_bar_phrase_tokens())
        self.assertTrue(analysis.boundaries)
        self.assertTrue(
            all(
                (left.bar_index, left.anchor_pos) <= (right.bar_index, right.anchor_pos)
                for left, right in zip(analysis.boundaries, analysis.boundaries[1:])
            )
        )

    def test_phrase_boundary_keeps_core_repeat_when_prefix_differs(self) -> None:
        """验证前缀不同但主体重复时，仍能保留真实乐句边界。"""

        analysis = analyze_phrase_candidates(_head_variant_repeated_four_bar_phrase_tokens())
        boundary_positions = {(item.bar_index, item.anchor_pos) for item in analysis.boundaries}
        self.assertIn((4, 0), boundary_positions)

    def test_repeated_phrase_end_is_not_auto_promoted_to_boundary(self) -> None:
        """验证相邻重复只提升下一段起点，不会自动把重复段结尾也当成乐句边界。"""

        analysis = analyze_phrase_candidates(_repeated_four_bar_phrase_tokens())
        boundary_positions = {(item.bar_index, item.anchor_pos) for item in analysis.boundaries}
        self.assertIn((4, 0), boundary_positions)
        self.assertNotIn((8, 0), boundary_positions)

    def test_phrase_boundary_keeps_chain_repeat_boundaries(self) -> None:
        """验证链式重复仍会保留后续重复单元起点，但不再自动补重复段结尾。"""

        analysis = analyze_phrase_candidates(_three_repeated_four_bar_phrase_tokens())
        boundary_positions = {(item.bar_index, item.anchor_pos) for item in analysis.boundaries}
        self.assertIn((4, 0), boundary_positions)
        self.assertIn((8, 0), boundary_positions)
        self.assertNotIn((12, 0), boundary_positions)

    def test_long_fragment_core_trim_candidates_stay_bounded(self) -> None:
        """验证长片段核心裁剪候选数量会被限制在可控范围。"""

        notes = _dense_note_fragment(onset_count=20, notes_per_onset=2)
        candidates = _collect_core_trim_candidate_ranges(notes, min_core_notes=3)

        self.assertLessEqual(len(candidates), 16)
        self.assertIn((0, len(notes)), candidates)

    def test_long_fragment_core_repeat_still_matches_after_dense_prefix_shift(self) -> None:
        """验证长片段在前缀不同但主体相同的情况下仍能命中核心重复。"""

        left = _dense_note_fragment(onset_count=12, notes_per_onset=2)
        right_prefix = _dense_note_fragment(onset_count=2, notes_per_onset=2)
        right = right_prefix + tuple(
            NoteInfo(
                note_index=len(right_prefix) + note.note_index,
                start_unit=note.start_unit + 4,
                end_unit=note.end_unit + 4,
                duration=note.duration,
                pitch=note.pitch,
                bar_index=(note.start_unit + 4) // 32,
                pos_in_bar=(note.start_unit + 4) % 32,
                effective_key_token=None,
            )
            for note in left[4:]
        )

        similarity, overlap_ratio, core_length = _best_core_repeat_similarity(
            left,
            right,
            min_core_notes=3,
        )

        self.assertGreaterEqual(similarity, 0.78)
        self.assertGreaterEqual(overlap_ratio, 0.60)
        self.assertGreater(core_length, 0)

    def test_core_repeat_similarity_allows_partial_bar_span_repeat(self) -> None:
        """验证局部重复核心不再要求覆盖完整 bar-span 主体。"""

        config = PhraseAnalysisConfig()
        notes = _build_note_info(_partial_repeated_bar_span_tokens(), config)
        left = tuple(note for note in notes if note.bar_index in {2, 3})
        right = tuple(note for note in notes if note.bar_index in {4, 5})

        similarity, overlap_ratio, core_length = _best_core_repeat_similarity(
            left,
            right,
            min_core_notes=config.repeat_min_notes,
            allow_partial_core=True,
        )

        self.assertGreaterEqual(similarity, 0.78)
        self.assertGreaterEqual(overlap_ratio, 0.50)
        self.assertEqual(core_length, 4)

    def test_adjacent_repeated_bar_span_targets_allow_partial_repeat_core(self) -> None:
        """验证相邻 bar-span 的局部重复也能提升后段起点。"""

        config = PhraseAnalysisConfig()
        notes = _build_note_info(_partial_repeated_bar_span_tokens(), config)

        targets = _detect_adjacent_repeated_bar_span_targets(notes, config)

        self.assertEqual(targets.get(4), "phrase")

    def test_phrase_spans_align_with_boundaries(self) -> None:
        analysis = analyze_phrase_candidates(_repeated_four_bar_phrase_tokens())
        self.assertTrue(analysis.phrase_spans)
        self.assertTrue(all(span.end_bar > span.start_bar for span in analysis.phrase_spans))
        self.assertTrue(
            all(
                (left.start_bar, left.end_bar) <= (right.start_bar, right.end_bar)
                for left, right in zip(analysis.phrase_spans, analysis.phrase_spans[1:])
            )
        )
        self.assertIsNotNone(analysis.analysis_start)
        self.assertTrue(analysis.boundaries)
        first_span = analysis.phrase_spans[0]
        self.assertEqual(first_span.start_bar, analysis.analysis_start.bar_index)
        self.assertEqual(first_span.end_bar, analysis.boundaries[0].bar_index)

    def test_phrase_spans_can_be_derived_from_analysis_start_without_boundaries(self) -> None:
        """验证没有真实边界时，仍可基于分析起点推导整段跨度。"""

        analysis = analyze_phrase_candidates(_no_clear_first_phrase_tokens(), config=PhraseAnalysisConfig())
        self.assertIsNotNone(analysis.analysis_start)
        self.assertEqual(len(analysis.boundaries), 0)
        self.assertEqual(len(analysis.phrase_spans), 1)
        self.assertEqual(analysis.phrase_spans[0].start_bar, analysis.analysis_start.bar_index)

    def test_phrase_spans_keep_leading_segment_when_analysis_start_and_boundary_share_bar(self) -> None:
        """验证分析起点与真实边界同小节时，标准化跨度不会丢掉前导片段。"""

        bars = (
            BarInfo(
                start_token=1,
                end_token=11,
                note_count=2,
                onset_count=2,
                rest_ratio=0.5,
                pitch_span=4,
                mean_duration=4.0,
                effective_tempo_token="TEMPO_120",
                effective_key_token=None,
                onset_positions=(8, 16),
            ),
            BarInfo(
                start_token=12,
                end_token=22,
                note_count=1,
                onset_count=1,
                rest_ratio=0.75,
                pitch_span=0,
                mean_duration=4.0,
                effective_tempo_token="TEMPO_120",
                effective_key_token=None,
                onset_positions=(0,),
            ),
            BarInfo(
                start_token=23,
                end_token=33,
                note_count=1,
                onset_count=1,
                rest_ratio=0.75,
                pitch_span=0,
                mean_duration=4.0,
                effective_tempo_token="TEMPO_120",
                effective_key_token=None,
                onset_positions=(0,),
            ),
        )
        analysis_start = AnalysisAnchor(0, 8)
        boundaries = (
            PhraseBoundary(0, 16),
            PhraseBoundary(1, 0),
        )

        spans = _derive_phrase_spans(bars, boundaries, analysis_start)

        self.assertTrue(spans)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].start_bar, 0)
        self.assertEqual(spans[0].end_bar, 1)
        self.assertEqual(spans[1].start_bar, 1)

    def test_analyze_phrase_candidates_exposes_mid_bar_anchor_boundary(self) -> None:
        """验证 note-level 候选仍保留小节内锚点信息。"""

        analysis = analyze_phrase_candidates(_mid_bar_anchor_tokens(), config=PhraseAnalysisConfig())
        # Task 4 之后最终 boundaries 只保留 phrase 级结果；
        # 这里单独断言 note-level 候选仍然保留小节内锚点信息。
        self.assertTrue(any(score.anchor_pos > 0 for score in analysis.boundary_scores))
        self.assertTrue(
            all(
                (left.bar_index, left.anchor_pos) <= (right.bar_index, right.anchor_pos)
                for left, right in zip(analysis.boundaries, analysis.boundaries[1:])
            )
        )

    def test_phrase_analysis_exposes_mid_bar_anchor_boundary(self) -> None:
        """兼容外部按旧名字调用的 mid-bar 锚点测试。"""

        self.test_analyze_phrase_candidates_exposes_mid_bar_anchor_boundary()

    def test_mid_bar_anchor_picks_first_onset_when_rest_threshold_met(self) -> None:
        from src.music_analysis.phrase_analysis import _pick_in_bar_anchor, BarInfo

        cfg = PhraseAnalysisConfig()
        right = BarInfo(
            start_token=0,
            end_token=0,
            note_count=2,
            onset_count=2,
            rest_ratio=0.5,
            pitch_span=0,
            mean_duration=4.0,
            effective_tempo_token=None,
            effective_key_token=None,
            onset_positions=(16, 24),
        )
        left = BarInfo(
            start_token=0,
            end_token=0,
            note_count=4,
            onset_count=4,
            rest_ratio=0.0,
            pitch_span=0,
            mean_duration=4.0,
            effective_tempo_token=None,
            effective_key_token=None,
            onset_positions=(0, 8, 16, 24),
        )
        self.assertEqual(_pick_in_bar_anchor(left, right, cfg), 16)

    def test_mid_bar_anchor_returns_zero_when_no_lead_rest(self) -> None:
        from src.music_analysis.phrase_analysis import _pick_in_bar_anchor, BarInfo

        cfg = PhraseAnalysisConfig()
        right = BarInfo(
            start_token=0,
            end_token=0,
            note_count=2,
            onset_count=2,
            rest_ratio=0.0,
            pitch_span=0,
            mean_duration=4.0,
            effective_tempo_token=None,
            effective_key_token=None,
            onset_positions=(0, 8),
        )
        left = BarInfo(
            start_token=0,
            end_token=0,
            note_count=4,
            onset_count=4,
            rest_ratio=0.0,
            pitch_span=0,
            mean_duration=4.0,
            effective_tempo_token=None,
            effective_key_token=None,
            onset_positions=(0, 8, 16, 24),
        )
        self.assertEqual(_pick_in_bar_anchor(left, right, cfg), 0)

    def test_analyze_phrase_candidates_accepts_missing_terminal_eos(self) -> None:
        tokens = _phrase_source_tokens()[:-1]
        analysis = analyze_phrase_candidates(tokens)
        self.assertEqual(len(analysis.bars), 8)
        self.assertTrue(analysis.phrase_spans)

    def test_final_boundaries_keep_any_thresholded_level_candidate(self) -> None:
        """验证三层分数任一达阈值时，最终都会落成真实乐句边界。"""

        notes = (
            NoteInfo(
                note_index=0,
                start_unit=8,
                end_unit=12,
                duration=4,
                pitch=60,
                bar_index=0,
                pos_in_bar=8,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=1,
                start_unit=16,
                end_unit=20,
                duration=4,
                pitch=62,
                bar_index=0,
                pos_in_bar=16,
                effective_key_token=None,
            ),
        )
        scores = (
            HierarchicalBoundaryScore(
                note_index=0,
                unit=8,
                bar_index=0,
                anchor_pos=8,
                motif_score=0.68,
                subphrase_score=0.22,
                phrase_score=0.18,
                boundary_type="motif",
                sequence_role="none",
                reasons=("motive_end",),
            ),
            HierarchicalBoundaryScore(
                note_index=1,
                unit=32,
                bar_index=1,
                anchor_pos=0,
                motif_score=0.34,
                subphrase_score=0.66,
                phrase_score=0.44,
                boundary_type="subphrase",
                sequence_role="none",
                reasons=("gap_break",),
            ),
            HierarchicalBoundaryScore(
                note_index=5,
                unit=64,
                bar_index=2,
                anchor_pos=0,
                motif_score=0.30,
                subphrase_score=0.58,
                phrase_score=0.88,
                boundary_type="phrase",
                sequence_role="none",
                reasons=("gap_break", "cadence"),
            ),
        )

        boundaries = _assemble_phrase_boundaries_from_scores(notes, scores, PhraseAnalysisConfig())

        self.assertEqual(
            boundaries,
            (
                PhraseBoundary(bar_index=0, anchor_pos=8),
                PhraseBoundary(bar_index=1, anchor_pos=0),
                PhraseBoundary(bar_index=2, anchor_pos=0),
            ),
        )

    def test_first_content_bar_keeps_only_bar_head_boundary(self) -> None:
        """验证首个有内容小节的 phrase 候选会保留其真实锚点位置。"""

        notes = (
            NoteInfo(
                note_index=0,
                start_unit=8,
                end_unit=12,
                duration=4,
                pitch=60,
                bar_index=1,
                pos_in_bar=8,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=1,
                start_unit=16,
                end_unit=20,
                duration=4,
                pitch=62,
                bar_index=1,
                pos_in_bar=16,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=2,
                start_unit=32,
                end_unit=36,
                duration=4,
                pitch=64,
                bar_index=2,
                pos_in_bar=0,
                effective_key_token=None,
            ),
        )
        scores = (
            HierarchicalBoundaryScore(
                note_index=0,
                unit=16,
                bar_index=1,
                anchor_pos=16,
                motif_score=0.30,
                subphrase_score=0.58,
                phrase_score=0.91,
                boundary_type="phrase",
                sequence_role="none",
                reasons=("gap_break", "cadence"),
            ),
            HierarchicalBoundaryScore(
                note_index=1,
                unit=32,
                bar_index=2,
                anchor_pos=0,
                motif_score=0.28,
                subphrase_score=0.54,
                phrase_score=0.82,
                boundary_type="phrase",
                sequence_role="none",
                reasons=("gap_break",),
            ),
        )

        boundaries = _assemble_phrase_boundaries_from_scores(notes, scores, PhraseAnalysisConfig())

        self.assertEqual(boundaries, (PhraseBoundary(bar_index=1, anchor_pos=16), PhraseBoundary(bar_index=2, anchor_pos=0)))

    def test_final_boundaries_keep_nearby_cross_level_candidates(self) -> None:
        """验证没有 phrase 分数达阈值时，其它层级过阈值的候选仍会落成边界。"""

        notes = (
            NoteInfo(
                note_index=0,
                start_unit=0,
                end_unit=4,
                duration=4,
                pitch=60,
                bar_index=0,
                pos_in_bar=0,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=1,
                start_unit=8,
                end_unit=12,
                duration=4,
                pitch=62,
                bar_index=1,
                pos_in_bar=0,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=2,
                start_unit=16,
                end_unit=20,
                duration=4,
                pitch=64,
                bar_index=2,
                pos_in_bar=0,
                effective_key_token=None,
            ),
        )
        scores = (
            HierarchicalBoundaryScore(
                note_index=0,
                unit=8,
                bar_index=1,
                anchor_pos=0,
                motif_score=0.71,
                subphrase_score=0.20,
                phrase_score=0.18,
                boundary_type="motif",
                sequence_role="none",
                reasons=("repeat_end",),
            ),
            HierarchicalBoundaryScore(
                note_index=1,
                unit=16,
                bar_index=2,
                anchor_pos=0,
                motif_score=0.30,
                subphrase_score=0.68,
                phrase_score=0.40,
                boundary_type="subphrase",
                sequence_role="none",
                reasons=("gap_break", "cadence"),
            ),
        )

        boundaries = _assemble_phrase_boundaries_from_scores(notes, scores, PhraseAnalysisConfig())

        self.assertEqual(
            boundaries,
            (
                PhraseBoundary(bar_index=1, anchor_pos=0),
                PhraseBoundary(bar_index=2, anchor_pos=0),
            ),
        )

    def test_same_bar_nearby_candidates_are_both_preserved(self) -> None:
        """验证同小节近邻候选只要各自达阈值，最终都会保留为边界。"""

        notes = (
            NoteInfo(
                note_index=0,
                start_unit=0,
                end_unit=4,
                duration=4,
                pitch=60,
                bar_index=0,
                pos_in_bar=0,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=1,
                start_unit=28,
                end_unit=31,
                duration=3,
                pitch=62,
                bar_index=0,
                pos_in_bar=28,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=2,
                start_unit=32,
                end_unit=35,
                duration=3,
                pitch=64,
                bar_index=1,
                pos_in_bar=0,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=3,
                start_unit=36,
                end_unit=39,
                duration=3,
                pitch=64,
                bar_index=1,
                pos_in_bar=4,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=4,
                start_unit=60,
                end_unit=63,
                duration=3,
                pitch=62,
                bar_index=1,
                pos_in_bar=28,
                effective_key_token=None,
            ),
            NoteInfo(
                note_index=5,
                start_unit=64,
                end_unit=67,
                duration=3,
                pitch=60,
                bar_index=2,
                pos_in_bar=0,
                effective_key_token=None,
            ),
        )
        scores = (
            HierarchicalBoundaryScore(
                note_index=1,
                unit=32,
                bar_index=1,
                anchor_pos=0,
                motif_score=0.42,
                subphrase_score=0.56,
                phrase_score=0.22,
                boundary_type="subphrase",
                sequence_role="none",
                reasons=("repeat_end", "motive_end", "repeat_start", "adjacent_repeated_bar_span"),
            ),
            HierarchicalBoundaryScore(
                note_index=3,
                unit=60,
                bar_index=1,
                anchor_pos=28,
                motif_score=0.42,
                subphrase_score=0.56,
                phrase_score=0.38,
                boundary_type="subphrase",
                sequence_role="none",
                reasons=("gap_break",),
            ),
        )

        boundaries = _assemble_phrase_boundaries_from_scores(notes, scores, PhraseAnalysisConfig())

        self.assertEqual(
            boundaries,
            (
                PhraseBoundary(bar_index=1, anchor_pos=0),
                PhraseBoundary(bar_index=1, anchor_pos=28),
            ),
        )

    def test_phrase_boundaries_remain_compatible_with_phrase_token_injection(self) -> None:
        """验证新的最终边界仍可被 PHRASE token 注入逻辑直接消费。"""

        from src.tokenizer.midi_codec import inject_phrase_tokens

        with_key = inject_key_tokens(_repeated_four_bar_phrase_tokens())
        analysis = analyze_phrase_candidates(with_key, config=PhraseAnalysisConfig())
        with_phrase = inject_phrase_tokens(with_key)

        self.assertTrue(analysis.boundaries)
        self.assertIn("PHRASE", with_phrase)
        self.assertEqual(with_phrase.count("PHRASE"), len(analysis.boundaries))

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
            # 第一个非头部的正文 token 必须是 BAR。
            idx = 1
            while idx < len(window) and (
                window[idx].startswith("TEMPO_") or window[idx].startswith("KEY_")
            ):
                idx += 1
            self.assertEqual(window[idx], "BAR", msg=f"窗口内容异常: {window}")
            # 窗口本身也必须是合法完整序列。
            valid, oov = validate_token_order(window, vocab)
            self.assertTrue(valid, msg=f"窗口序列校验失败: {window}")
            self.assertEqual(oov, 0)

    def test_eval_window_keeps_inline_phrases(self) -> None:
        from src.tokenizer.midi_codec import inject_phrase_tokens

        raw = _long_phrase_source_tokens()
        tokens = inject_phrase_tokens(inject_key_tokens(raw))
        # 先确认源序列里本来就包含 PHRASE。
        self.assertIn("PHRASE", tokens)
        survived = 0
        for seed in range(16):
            window = sample_phrase_aligned_subsequence(
                tokens, max_core_tokens=240, min_core_tokens=24, rng=random.Random(seed),
            )
            if window is not None and "PHRASE" in window:
                survived += 1
        self.assertGreater(survived, 0, msg="没有任何窗口保留内联 PHRASE")

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
