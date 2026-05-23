"""TuneFlow 的乐句分析工具。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


@dataclass(frozen=True)
class PhraseAnalysisConfig:
    """乐句边界分析使用的启发式配置。"""

    positions_per_bar: int = 32
    min_phrase_bars: int = 2
    max_phrase_bars: int = 8
    preferred_phrase_bars: int = 4
    min_boundary_gap_bars: int = 2
    mid_bar_min_rest_pos: int = 8
    rest_weight: float = 0.40
    note_density_weight: float = 0.24
    onset_density_weight: float = 0.18
    pitch_span_weight: float = 0.12
    duration_weight: float = 0.06
    local_window_notes: int = 4
    phrase_threshold: float = 0.72
    subphrase_threshold: float = 0.56
    motif_threshold: float = 0.42
    near_boundary_note_gap: int = 2
    repeat_min_notes: int = 3
    repeat_max_notes: int = 8
    strong_beat_positions: tuple[int, ...] = (0, 8, 16, 24)
    stronger_beat_positions: tuple[int, ...] = (0, 16)


@dataclass(frozen=True)
class PhraseBoundary:
    """PHRASE 落点。`anchor_pos==0` 表示对齐小节，否则表示小节内锚点。"""

    bar_index: int
    anchor_pos: int


@dataclass(frozen=True)
class BarInfo:
    """token 序列中单个小节的统计信息。"""

    start_token: int
    end_token: int
    note_count: int
    onset_count: int
    rest_ratio: float
    pitch_span: int
    mean_duration: float
    effective_tempo_token: str | None
    effective_key_token: str | None
    onset_positions: tuple[int, ...]


@dataclass(frozen=True)
class NoteInfo:
    """乐句分析使用的音符级事件。"""

    note_index: int
    start_unit: int
    end_unit: int
    duration: int
    pitch: int
    bar_index: int
    pos_in_bar: int
    effective_key_token: str | None


@dataclass(frozen=True)
class BoundaryFeature:
    """相邻音符之间的边界特征。"""

    note_index: int
    left_end_unit: int
    right_start_unit: int
    bar_index: int
    anchor_pos: int
    gap: int
    local_gap_mean: float
    local_duration_mean: float
    gap_break_score: float
    duration_release_score: float
    cadence_score: float
    motive_end_score: float
    repeat_start_score: float
    repeat_end_score: float
    sequence_stop_score: float
    continuity_penalty: float
    bar_hint_score: float
    sequence_role: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RepeatSignal:
    """重复或动机复现在线性边界上的语义信号。"""

    motive_end: float = 0.0
    repeat_start: float = 0.0
    repeat_end: float = 0.0


@dataclass(frozen=True)
class BoundaryScore:
    """内部过渡结构，承载旧的小节级边界得分。"""

    bar_index: int
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class HierarchicalBoundaryScore:
    """单个音符候选点的三层边界分数。"""

    note_index: int
    unit: int
    bar_index: int
    anchor_pos: int
    motif_score: float
    subphrase_score: float
    phrase_score: float
    boundary_type: str
    sequence_role: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PhraseSpan:
    """带标准化乐句视图的类乐句小节区间。"""

    start_bar: int
    end_bar: int
    start_token: int
    end_token: int
    tempo_token: str | None
    key_token: str | None
    tokens: tuple[str, ...]
    source_kind: str


@dataclass(frozen=True)
class AnalysisAnchor:
    """表示分析起点，仅用于说明分析从哪里开始，不表示真实结构边界。"""

    bar_index: int
    anchor_pos: int


@dataclass(frozen=True)
class PhraseAnalysis:
    """单条 token 序列的乐句分析结果。

    当前主链路已经统一到 note-level：
    - `notes`、`boundary_features`、`boundary_scores` 来自 note-level 特征提取与评分流程。
    - `boundary_scores` 表示三层初始评分经过 Task 4 第一轮后处理后的结果。
    - `boundaries`、`phrase_spans` 来自后处理后的 note-level 最终边界组装结果。
    """

    bars: tuple[BarInfo, ...]
    notes: tuple[NoteInfo, ...]
    boundary_features: tuple[BoundaryFeature, ...]
    boundary_scores: tuple[HierarchicalBoundaryScore, ...]
    boundaries: tuple[PhraseBoundary, ...]
    phrase_spans: tuple[PhraseSpan, ...]
    analysis_start: AnalysisAnchor | None = None


def _safe_ratio(delta: float, left: float, right: float) -> float:
    """计算带保护的归一化差值。"""

    denom = max(1.0, abs(left), abs(right))
    return min(1.0, max(0.0, float(delta) / denom))


def _local_mean(values: Sequence[float], center: int, radius: int) -> float:
    """计算局部窗口均值，用于比较当前候选点是否显著。"""

    if not values:
        return 0.0
    start = max(0, center - max(0, radius))
    end = min(len(values), center + max(0, radius) + 1)
    window = values[start:end]
    if not window:
        return 0.0
    return float(sum(window) / len(window))


def _is_strong_beat(unit: int, config: PhraseAnalysisConfig) -> bool:
    """判断绝对时间位置是否落在当前拍号网格的强拍上。"""

    return (unit % config.positions_per_bar) in config.strong_beat_positions


def _stable_degree_score(pitch: int, key_token: str | None) -> float:
    """根据调性估计音级稳定性，主音与属音得分最高。"""

    if not key_token or not key_token.startswith("KEY_") or key_token == "KEY_UNCERTAIN":
        return 0.0

    parts = key_token.split("_")
    if len(parts) != 3:
        return 0.0

    root_map = {
        "C": 0,
        "CS": 1,
        "DF": 1,
        "D": 2,
        "DS": 3,
        "EF": 3,
        "E": 4,
        "F": 5,
        "FS": 6,
        "GF": 6,
        "G": 7,
        "GS": 8,
        "AF": 8,
        "A": 9,
        "AS": 10,
        "BF": 10,
        "B": 11,
        "CF": 11,
    }
    root_pc = root_map.get(parts[1])
    if root_pc is None:
        return 0.0

    degree = (pitch - root_pc) % 12
    if parts[2] == "MAJ":
        if degree in {0, 7}:
            return 1.0
        if degree in {4, 2, 5, 11}:
            return 0.65
        return 0.25
    if parts[2] == "MIN":
        if degree in {0, 7}:
            return 1.0
        if degree in {3, 2, 5, 10}:
            return 0.65
        return 0.25
    return 0.0


def _iter_bar_slices(
    tokens: Sequence[str],
) -> tuple[list[tuple[int, int, str | None, str | None]], str | None, str | None] | None:
    """按小节切分 token，并带出当前生效的速度与调性 token。"""

    if not tokens or tokens[0] != "BOS":
        return None

    effective_end = len(tokens) - 1 if tokens[-1] == "EOS" else len(tokens)
    current_tempo: str | None = None
    current_key: str | None = None
    idx = 1
    if idx < effective_end and str(tokens[idx]).startswith("TEMPO_"):
        current_tempo = str(tokens[idx])
        idx += 1
    if idx < effective_end and str(tokens[idx]).startswith("KEY_"):
        current_key = str(tokens[idx])
        idx += 1

    bars: list[tuple[int, int, str | None, str | None]] = []
    while idx < effective_end:
        if tokens[idx] != "BAR":
            return None
        bar_start = idx
        idx += 1
        if idx < effective_end and str(tokens[idx]).startswith("TEMPO_"):
            current_tempo = str(tokens[idx])
            idx += 1
        if idx < effective_end and str(tokens[idx]).startswith("KEY_"):
            current_key = str(tokens[idx])
            idx += 1
        while idx < effective_end and tokens[idx] != "BAR":
            idx += 1
        bars.append((bar_start, idx, current_tempo, current_key))
    return bars, current_tempo, current_key


def _build_bar_info(tokens: Sequence[str], config: PhraseAnalysisConfig) -> tuple[BarInfo, ...]:
    """从 token 序列构造小节级统计信息。"""

    parsed = _iter_bar_slices(tokens)
    if parsed is None:
        return tuple()
    raw_bars, _, _ = parsed

    bars: list[BarInfo] = []
    for start_token, end_token, effective_tempo, effective_key in raw_bars:
        idx = start_token + 1
        if idx < end_token and str(tokens[idx]).startswith("TEMPO_"):
            idx += 1
        if idx < end_token and str(tokens[idx]).startswith("KEY_"):
            idx += 1

        note_count = 0
        onset_positions: set[int] = set()
        occupied_positions: set[int] = set()
        pitches: list[int] = []
        durations: list[int] = []

        while idx < end_token:
            token = str(tokens[idx])
            if not token.startswith("POS_"):
                idx += 1
                continue
            if idx + 4 >= end_token:
                break
            try:
                pos_value = int(token.split("_", 1)[1])
            except ValueError:
                idx += 1
                continue
            inst_token = str(tokens[idx + 1])
            pitch_token = str(tokens[idx + 2])
            dur_token = str(tokens[idx + 3])
            vel_token = str(tokens[idx + 4])
            if (
                not inst_token.startswith("INST_")
                or not pitch_token.startswith("PITCH_")
                or not dur_token.startswith("DUR_")
                or not vel_token.startswith("VEL_")
            ):
                idx += 1
                continue
            try:
                pitch_value = int(pitch_token.split("_", 1)[1])
                dur_value = int(dur_token.split("_", 1)[1])
            except ValueError:
                idx += 5
                continue
            note_count += 1
            onset_positions.add(pos_value)
            pitches.append(pitch_value)
            durations.append(dur_value)
            for occupied in range(pos_value, min(config.positions_per_bar, pos_value + max(1, dur_value))):
                occupied_positions.add(occupied)
            idx += 5

        pitch_span = 0 if len(pitches) < 2 else (max(pitches) - min(pitches))
        mean_duration = float(sum(durations) / len(durations)) if durations else 0.0
        rest_ratio = 1.0 - (len(occupied_positions) / float(max(1, config.positions_per_bar)))
        bars.append(
            BarInfo(
                start_token=start_token,
                end_token=end_token,
                note_count=note_count,
                onset_count=len(onset_positions),
                rest_ratio=min(1.0, max(0.0, rest_ratio)),
                pitch_span=pitch_span,
                mean_duration=mean_duration,
                effective_tempo_token=effective_tempo,
                effective_key_token=effective_key,
                onset_positions=tuple(sorted(onset_positions)),
            )
        )
    return tuple(bars)


def _build_note_info(tokens: Sequence[str], config: PhraseAnalysisConfig) -> tuple[NoteInfo, ...]:
    """从 token 序列提取稳定的音符级信息。"""

    parsed = _iter_bar_slices(tokens)
    if parsed is None:
        return tuple()
    raw_bars, _, _ = parsed

    notes: list[NoteInfo] = []
    note_index = 0
    for bar_index, (start_token, end_token, _, effective_key) in enumerate(raw_bars):
        idx = start_token + 1
        if idx < end_token and str(tokens[idx]).startswith("TEMPO_"):
            idx += 1
        if idx < end_token and str(tokens[idx]).startswith("KEY_"):
            idx += 1

        while idx < end_token:
            token = str(tokens[idx])
            if not token.startswith("POS_"):
                idx += 1
                continue
            if idx + 4 >= end_token:
                break

            inst_token = str(tokens[idx + 1])
            pitch_token = str(tokens[idx + 2])
            dur_token = str(tokens[idx + 3])
            vel_token = str(tokens[idx + 4])
            if (
                not inst_token.startswith("INST_")
                or not pitch_token.startswith("PITCH_")
                or not dur_token.startswith("DUR_")
                or not vel_token.startswith("VEL_")
            ):
                idx += 1
                continue

            try:
                pos_value = int(token.split("_", 1)[1])
                pitch_value = int(pitch_token.split("_", 1)[1])
                duration_value = max(1, int(dur_token.split("_", 1)[1]))
            except ValueError:
                idx += 5
                continue

            start_unit = (bar_index * config.positions_per_bar) + pos_value
            end_unit = start_unit + duration_value
            notes.append(
                NoteInfo(
                    note_index=note_index,
                    start_unit=start_unit,
                    end_unit=end_unit,
                    duration=duration_value,
                    pitch=pitch_value,
                    bar_index=bar_index,
                    pos_in_bar=pos_value,
                    effective_key_token=effective_key,
                )
            )
            note_index += 1
            idx += 5

    return tuple(notes)


def _build_legacy_boundary_scores(
    bars: Sequence[BarInfo],
    config: PhraseAnalysisConfig,
) -> tuple[BoundaryScore, ...]:
    """保留现有小节级候选打分逻辑。

    这组结果只在过渡阶段用于推导 `boundaries` 与 `phrase_spans`，
    不会暴露到 `PhraseAnalysis.boundary_scores` 中。
    """

    if len(bars) < 2:
        return tuple()

    scores: list[BoundaryScore] = []
    for left_index in range(len(bars) - 1):
        left = bars[left_index]
        right = bars[left_index + 1]

        rest_signal = max(left.rest_ratio, right.rest_ratio)
        if left.note_count == 0 or right.note_count == 0:
            rest_signal = max(rest_signal, 1.0)
        note_density_delta = _safe_ratio(abs(right.note_count - left.note_count), left.note_count, right.note_count)
        onset_density_delta = _safe_ratio(abs(right.onset_count - left.onset_count), left.onset_count, right.onset_count)
        pitch_span_delta = _safe_ratio(abs(right.pitch_span - left.pitch_span), left.pitch_span, right.pitch_span)
        duration_delta = _safe_ratio(abs(right.mean_duration - left.mean_duration), left.mean_duration, right.mean_duration)

        contributions = {
            "rest_gap": config.rest_weight * rest_signal,
            "note_density_delta": config.note_density_weight * note_density_delta,
            "onset_density_delta": config.onset_density_weight * onset_density_delta,
            "pitch_span_delta": config.pitch_span_weight * pitch_span_delta,
            "duration_delta": config.duration_weight * duration_delta,
        }
        score = float(sum(contributions.values()))
        reasons = tuple(label for label, contribution in contributions.items() if contribution >= 0.10)
        scores.append(BoundaryScore(bar_index=left_index + 1, score=score, reasons=reasons))
    return tuple(scores)


def _duration_pattern(notes: Sequence[NoteInfo]) -> tuple[int, ...]:
    """提取片段的时值模式，用于比较节奏是否一致。"""

    return tuple(note.duration for note in notes)


def _interval_pattern(notes: Sequence[NoteInfo]) -> tuple[int, ...]:
    """提取相邻音高差模式，用于比较是否属于同一音程骨架。"""

    return tuple(notes[index + 1].pitch - notes[index].pitch for index in range(max(0, len(notes) - 1)))


def _contour_pattern(notes: Sequence[NoteInfo]) -> tuple[int, ...]:
    """提取旋律轮廓模式，上行记为 1，下行记为 -1，重复记为 0。"""

    contour: list[int] = []
    for interval in _interval_pattern(notes):
        if interval > 0:
            contour.append(1)
        elif interval < 0:
            contour.append(-1)
        else:
            contour.append(0)
    return tuple(contour)


def _fragment_onset_pattern(notes: Sequence[NoteInfo]) -> tuple[int, ...]:
    """提取片段内部相对起音位置，允许后续比较 humanize 后的节奏骨架。"""

    if not notes:
        return tuple()
    first_unit = notes[0].start_unit
    return tuple(int(note.start_unit - first_unit) for note in notes)


def _tolerant_pattern_similarity(
    left: Sequence[int],
    right: Sequence[int],
    tolerance: int,
) -> float:
    """按给定容差比较两个数值模式，相同越多得分越高。"""

    if len(left) != len(right):
        return 0.0
    if not left:
        return 1.0
    safe_tolerance = max(1, int(tolerance))
    scores: list[float] = []
    for left_value, right_value in zip(left, right, strict=True):
        distance = abs(int(left_value) - int(right_value))
        scores.append(max(0.0, 1.0 - (float(distance) / float(safe_tolerance))))
    return float(sum(scores) / len(scores))


def _contour_similarity(left: Sequence[NoteInfo], right: Sequence[NoteInfo]) -> float:
    """比较两个片段的旋律轮廓一致度。"""

    left_contour = _contour_pattern(left)
    right_contour = _contour_pattern(right)
    if len(left_contour) != len(right_contour):
        return 0.0
    if not left_contour:
        return 1.0
    match_count = sum(
        1
        for left_value, right_value in zip(left_contour, right_contour, strict=True)
        if int(left_value) == int(right_value)
    )
    return float(match_count / len(left_contour))


def _pitch_shape_similarity(
    left: Sequence[NoteInfo],
    right: Sequence[NoteInfo],
    tolerance: int = 2,
) -> float:
    """比较两个片段的相对音高骨架，避免只凭轮廓就误判为重复。"""

    if len(left) != len(right) or not left:
        return 0.0
    left_base = left[0].pitch
    right_base = right[0].pitch
    left_offsets = [int(note.pitch - left_base) for note in left]
    right_offsets = [int(note.pitch - right_base) for note in right]
    return _tolerant_pattern_similarity(left_offsets, right_offsets, tolerance=tolerance)


def _as_note_tuple(notes: Sequence[NoteInfo]) -> tuple[NoteInfo, ...]:
    """把任意音符序列规范化为可缓存的 tuple。"""

    return notes if isinstance(notes, tuple) else tuple(notes)


def _required_core_note_count(total_notes: int, min_core_notes: int) -> int:
    """计算满足 60% 主体约束时所需保留的最小核心音符数。"""

    return max(int(min_core_notes), ((int(total_notes) * 3) + 4) // 5)


@lru_cache(maxsize=65536)
def _collect_onset_group_boundaries(notes: tuple[NoteInfo, ...]) -> tuple[int, ...]:
    """收集片段内部每个 onset 组的起始下标。"""

    if not notes:
        return tuple()
    boundaries = [0]
    previous_start_unit = int(notes[0].start_unit)
    for index, note in enumerate(notes[1:], start=1):
        current_start_unit = int(note.start_unit)
        if current_start_unit == previous_start_unit:
            continue
        boundaries.append(index)
        previous_start_unit = current_start_unit
    return tuple(boundaries)


def _collect_core_trim_candidate_ranges(
    notes: Sequence[NoteInfo],
    *,
    min_core_notes: int,
    allow_partial_core: bool = False,
) -> tuple[tuple[int, int], ...]:
    """为核心重复修正收集受控的裁剪候选区间。"""

    note_items = _as_note_tuple(notes)
    total_notes = len(note_items)
    if total_notes < min_core_notes:
        return tuple()

    required_core_notes = (
        max(1, int(min_core_notes))
        if allow_partial_core
        else _required_core_note_count(total_notes, min_core_notes)
    )
    max_trim_notes = max(0, total_notes - required_core_notes)
    if max_trim_notes == 0:
        return ((0, total_notes),)

    if total_notes <= 16:
        candidates: list[tuple[int, int]] = []
        for trim_start in range(max_trim_notes + 1):
            for trim_end in range(max_trim_notes + 1):
                end_index = total_notes - trim_end
                if end_index - trim_start < required_core_notes:
                    continue
                candidates.append((trim_start, end_index))
        return tuple(
            sorted(
                set(candidates),
                key=lambda item: (-(item[1] - item[0]), item[0], -item[1]),
            )
        )

    onset_boundaries = _collect_onset_group_boundaries(note_items)
    if not onset_boundaries:
        return ((0, total_notes),)

    start_candidates = {0}
    start_added = 0
    for boundary in onset_boundaries[1:]:
        if boundary > max_trim_notes:
            break
        start_candidates.add(int(boundary))
        start_added += 1
        if start_added >= 2:
            break
    start_candidates.add(
        max((int(boundary) for boundary in onset_boundaries if int(boundary) <= max_trim_notes), default=0)
    )

    end_candidates = {total_notes}
    eligible_end_boundaries = [
        int(boundary)
        for boundary in onset_boundaries[1:]
        if (total_notes - int(boundary)) <= max_trim_notes
    ]
    for boundary in eligible_end_boundaries[-2:]:
        end_candidates.add(boundary)
    end_candidates.add(
        min((int(boundary) for boundary in onset_boundaries if int(boundary) >= required_core_notes), default=total_notes)
    )

    candidates = {
        (start_index, end_index)
        for start_index in start_candidates
        for end_index in end_candidates
        if end_index - start_index >= required_core_notes
    }
    if not candidates:
        return ((0, total_notes),)
    return tuple(
        sorted(
            candidates,
            key=lambda item: (-(item[1] - item[0]), item[0], -item[1]),
        )
    )


@lru_cache(maxsize=65536)
def _fragment_similarity_score_cached(
    left: tuple[NoteInfo, ...],
    right: tuple[NoteInfo, ...],
) -> float:
    """缓存片段相似度，避免重复重算相同候选。"""

    if len(left) != len(right) or len(left) < 3:
        return 0.0

    duration_similarity = _tolerant_pattern_similarity(
        _duration_pattern(left),
        _duration_pattern(right),
        tolerance=2,
    )
    onset_similarity = _tolerant_pattern_similarity(
        _fragment_onset_pattern(left),
        _fragment_onset_pattern(right),
        tolerance=2,
    )
    interval_similarity = _tolerant_pattern_similarity(
        _interval_pattern(left),
        _interval_pattern(right),
        tolerance=2,
    )
    strict_interval_similarity = _tolerant_pattern_similarity(
        _interval_pattern(left),
        _interval_pattern(right),
        tolerance=1,
    )
    contour_similarity = _contour_similarity(left, right)
    pitch_shape_similarity = _pitch_shape_similarity(left, right, tolerance=2)
    strict_pitch_shape_similarity = _pitch_shape_similarity(left, right, tolerance=1)

    if duration_similarity < 0.55 or onset_similarity < 0.55:
        return 0.0
    if contour_similarity < 0.75:
        return 0.0
    if len(left) == 3 and strict_interval_similarity < 0.99 and strict_pitch_shape_similarity < 0.99:
        return 0.0
    if interval_similarity < 0.55 and pitch_shape_similarity < 0.65:
        return 0.0

    return float(
        (0.30 * duration_similarity)
        + (0.20 * onset_similarity)
        + (0.20 * interval_similarity)
        + (0.15 * contour_similarity)
        + (0.15 * pitch_shape_similarity)
    )


def _fragment_similarity_score(
    left: Sequence[NoteInfo],
    right: Sequence[NoteInfo],
) -> float:
    """比较两个相邻片段的整体相似度，允许轻微 humanize 与个别发展变化。"""

    return _fragment_similarity_score_cached(_as_note_tuple(left), _as_note_tuple(right))


def _is_structurally_similar_fragment(left: Sequence[NoteInfo], right: Sequence[NoteInfo]) -> bool:
    """判断两个片段是否在节奏骨架与旋律形态上足够相似。"""

    return _fragment_similarity_score(left, right) >= 0.72


def _best_core_repeat_similarity(
    left: Sequence[NoteInfo],
    right: Sequence[NoteInfo],
    *,
    min_core_notes: int,
    max_edge_trim: int | None = None,
    allow_partial_core: bool = False,
) -> tuple[float, float, int]:
    """允许裁掉首尾少量非核心音后，寻找最佳主体重复相似度。"""

    if len(left) < min_core_notes or len(right) < min_core_notes:
        return 0.0, 0.0, 0
    if (float(min(len(left), len(right))) / float(max(len(left), len(right)))) < 0.60:
        return 0.0, 0.0, 0

    best_score = 0.0
    best_overlap_ratio = 0.0
    best_core_length = 0
    if max_edge_trim is None:
        left_ranges = _collect_core_trim_candidate_ranges(
            left,
            min_core_notes=min_core_notes,
            allow_partial_core=allow_partial_core,
        )
        right_ranges = _collect_core_trim_candidate_ranges(
            right,
            min_core_notes=min_core_notes,
            allow_partial_core=allow_partial_core,
        )
    else:
        left_max_trim = min(max_edge_trim, max(0, len(left) - min_core_notes))
        right_max_trim = min(max_edge_trim, max(0, len(right) - min_core_notes))
        left_ranges = tuple(
            sorted(
                {
                    (trim_start, len(left) - trim_end)
                    for trim_start in range(left_max_trim + 1)
                    for trim_end in range(left_max_trim + 1)
                    if (len(left) - trim_end) - trim_start >= min_core_notes
                },
                key=lambda item: (-(item[1] - item[0]), item[0], -item[1]),
            )
        )
        right_ranges = tuple(
            sorted(
                {
                    (trim_start, len(right) - trim_end)
                    for trim_start in range(right_max_trim + 1)
                    for trim_end in range(right_max_trim + 1)
                    if (len(right) - trim_end) - trim_start >= min_core_notes
                },
                key=lambda item: (-(item[1] - item[0]), item[0], -item[1]),
            )
        )

    for left_start_index, left_end_index in left_ranges:
        left_core = left[left_start_index:left_end_index]
        if len(left_core) < min_core_notes:
            continue
        for right_start_index, right_end_index in right_ranges:
            right_core = right[right_start_index:right_end_index]
            if len(right_core) != len(left_core) or len(right_core) < min_core_notes:
                continue

            similarity = _fragment_similarity_score(left_core, right_core)
            if similarity <= 0.0:
                continue

            overlap_ratio = float(len(left_core)) / float(max(len(left), len(right)))
            if (
                similarity > best_score
                or (similarity == best_score and overlap_ratio > best_overlap_ratio)
                or (
                    similarity == best_score
                    and overlap_ratio == best_overlap_ratio
                    and len(left_core) > best_core_length
                )
            ):
                best_score = similarity
                best_overlap_ratio = overlap_ratio
                best_core_length = len(left_core)
                if best_score >= 0.999 and best_overlap_ratio >= 0.95:
                    return best_score, best_overlap_ratio, best_core_length

    return best_score, best_overlap_ratio, best_core_length


def _detect_repeat_signals(
    notes: Sequence[NoteInfo],
    config: PhraseAnalysisConfig,
) -> tuple[RepeatSignal, ...]:
    """检测允许 humanize 的不定长度相邻重复片段，并输出语义信号。"""

    signals = [RepeatSignal() for _ in range(max(0, len(notes) - 1))]
    if len(notes) < config.repeat_min_notes * 2:
        return tuple(signals)

    max_fragment_notes = min(config.repeat_max_notes, len(notes) // 2)

    best_by_split: dict[int, tuple[float, int]] = {}
    for split_index in range(config.repeat_min_notes - 1, len(notes) - config.repeat_min_notes):
        max_length = min(
            max_fragment_notes,
            split_index + 1,
            len(notes) - (split_index + 1),
        )
        best_score = 0.0
        best_length = 0
        for fragment_len in range(config.repeat_min_notes, max_length + 1):
            left = notes[(split_index - fragment_len) + 1 : split_index + 1]
            right = notes[split_index + 1 : split_index + 1 + fragment_len]
            similarity = _fragment_similarity_score(left, right)
            effective_length = fragment_len
            if similarity < 0.78:
                similarity, _overlap_ratio, core_length = _best_core_repeat_similarity(
                    left,
                    right,
                    min_core_notes=config.repeat_min_notes,
                )
                if similarity < 0.78 or _overlap_ratio < 0.60:
                    continue
                effective_length = core_length

            length_bonus = min(0.12, max(0, effective_length - config.repeat_min_notes) * 0.04)
            score = min(0.75, 0.40 + max(0.0, (similarity - 0.78) * 1.4) + length_bonus)
            if score > best_score or (score == best_score and fragment_len > best_length):
                best_score = score
                best_length = fragment_len
        if best_score > 0.0:
            best_by_split[split_index] = (best_score, best_length)

    ranked_candidates = sorted(
        (
            (
                split_index,
                score,
                fragment_len,
                2
                if notes[split_index + 1].bar_index != notes[split_index].bar_index
                else (1 if notes[split_index + 1].pos_in_bar in config.strong_beat_positions else 0),
                max(0, notes[split_index + 1].start_unit - notes[split_index].end_unit),
            )
            for split_index, (score, fragment_len) in best_by_split.items()
        ),
        key=lambda item: (
            int(item[3]),
            float(item[1]),
            int(item[2]),
            int(item[4]),
            1 if notes[item[0] + 1].pos_in_bar == 0 else 0,
            int(item[0]),
        ),
        reverse=True,
    )
    selected: list[tuple[int, float, int]] = []
    for split_index, score, fragment_len, _, _ in ranked_candidates:
        if any(abs(split_index - kept_index) <= max(fragment_len, kept_length) for kept_index, _, kept_length in selected):
            continue
        selected.append((split_index, score, fragment_len))

    for split_index, score, _ in selected:
        
        current = signals[split_index]
        signals[split_index] = RepeatSignal(
            motive_end=max(current.motive_end, score),
            repeat_start=max(current.repeat_start, score),
            repeat_end=max(current.repeat_end, score),
        )
    return tuple(signals)


def _detect_sequence_roles(notes: Sequence[NoteInfo]) -> tuple[str, ...]:
    """用滑动起点检测顺序模进区域，并标记内部与停止边界。"""

    roles = ["none"] * max(0, len(notes) - 1)
    if len(notes) < 6:
        return tuple(roles)

    max_fragment_notes = min(4, len(notes) // 2)
    for fragment_len in range(3, max_fragment_notes + 1):
        max_start = len(notes) - (fragment_len * 3)
        for start_index in range(0, max_start + 1):
            shift_values: list[int] = []
            cursor = start_index
            while cursor + (fragment_len * 2) <= len(notes):
                left = notes[cursor:cursor + fragment_len]
                right = notes[cursor + fragment_len:cursor + (fragment_len * 2)]
                if not _is_structurally_similar_fragment(left, right):
                    break
                shift = right[0].pitch - left[0].pitch
                if shift == 0:
                    break
                if any((right[offset].pitch - left[offset].pitch) != shift for offset in range(fragment_len)):
                    break
                if shift_values and shift_values[-1] != shift:
                    break
                shift_values.append(shift)
                cursor += fragment_len

            if len(shift_values) < 2:
                continue

            for step_index in range(len(shift_values)):
                boundary_index = start_index + ((step_index + 1) * fragment_len) - 1
                if boundary_index >= len(roles):
                    continue
                if roles[boundary_index] != "sequence_stop":
                    roles[boundary_index] = "sequence_inside"

            stop_index = start_index + ((len(shift_values) + 1) * fragment_len) - 1
            if stop_index < len(roles):
                roles[stop_index] = "sequence_stop"

    return tuple(roles)


def _build_default_boundary_features(
    notes: Sequence[NoteInfo],
    config: PhraseAnalysisConfig,
) -> tuple[BoundaryFeature, ...]:
    """为相邻音符构造真实边界特征，并保留 Task 2 的重复与模进信号。"""

    repeat_signals = _detect_repeat_signals(notes, config)
    sequence_roles = _detect_sequence_roles(notes)
    gaps = [max(0, notes[index + 1].start_unit - notes[index].end_unit) for index in range(max(0, len(notes) - 1))]
    durations = [float(note.duration) for note in notes]
    items: list[BoundaryFeature] = []
    for note_index in range(max(0, len(notes) - 1)):
        left = notes[note_index]
        right = notes[note_index + 1]
        repeat_signal = repeat_signals[note_index]
        sequence_role = sequence_roles[note_index]
        gap = gaps[note_index]
        local_gap_mean = _local_mean(gaps, note_index, config.local_window_notes)
        local_duration_mean = _local_mean(durations, note_index, config.local_window_notes)
        crossed_empty_bars = max(0, right.bar_index - left.bar_index - 1)

        gap_break_score = 0.0
        if gap > 0:
            gap_ratio = float(gap) / max(1.0, local_gap_mean)
            if crossed_empty_bars > 0:
                gap_break_score = 1.0
            elif gap_ratio >= 2.0 and gap >= 4:
                gap_break_score = 1.0
            elif gap_ratio >= 1.35 and gap >= 6:
                gap_break_score = min(1.0, 0.45 + ((gap_ratio - 1.35) * 0.8))

        duration_release_score = 0.0
        if left.duration > 0 and left.duration > (2.2 * local_duration_mean):
            duration_release_score = 1.0

        left_stability = _stable_degree_score(left.pitch, left.effective_key_token)
        beat_strength = 1.0 if _is_strong_beat(left.end_unit, config) else 0.0
        cadence_score = min(1.0, beat_strength * left_stability)

        continuity_penalty = 0.0
        if gap == 0 and abs(right.duration - left.duration) <= 1 and abs(right.pitch - left.pitch) <= 2:
            continuity_penalty = 1.0

        bar_hint_score = 0.0
        if crossed_empty_bars > 0:
            bar_hint_score = min(1.0, 0.55 + (0.20 * crossed_empty_bars))
        elif right.bar_index != left.bar_index:
            bar_hint_score = 0.15

        reasons: list[str] = []
        if gap_break_score > 0:
            reasons.append("gap_break")
        if duration_release_score > 0:
            reasons.append("duration_release")
        if cadence_score > 0:
            reasons.append("cadence")
        if repeat_signal.motive_end > 0:
            reasons.append("motive_end")
        if repeat_signal.repeat_start > 0:
            reasons.append("repeat_start")
        if repeat_signal.repeat_end > 0:
            reasons.append("repeat_end")
        if continuity_penalty > 0:
            reasons.append("continuity_penalty")
        if bar_hint_score > 0:
            reasons.append("bar_hint")
        if sequence_role != "none":
            reasons.append(sequence_role)
        items.append(
            BoundaryFeature(
                note_index=note_index,
                left_end_unit=left.end_unit,
                right_start_unit=right.start_unit,
                bar_index=right.bar_index,
                anchor_pos=right.pos_in_bar,
                gap=gap,
                local_gap_mean=local_gap_mean,
                local_duration_mean=local_duration_mean,
                gap_break_score=gap_break_score,
                duration_release_score=duration_release_score,
                cadence_score=cadence_score,
                motive_end_score=repeat_signal.motive_end,
                repeat_start_score=repeat_signal.repeat_start,
                repeat_end_score=repeat_signal.repeat_end,
                sequence_stop_score=1.0 if sequence_role == "sequence_stop" else 0.0,
                continuity_penalty=continuity_penalty,
                bar_hint_score=bar_hint_score,
                sequence_role=sequence_role,
                reasons=tuple(reasons),
            )
        )
    return tuple(items)


def _score_boundary_features(
    boundary_features: Sequence[BoundaryFeature],
    config: PhraseAnalysisConfig,
) -> tuple[HierarchicalBoundaryScore, ...]:
    """把边界特征映射为三层分数，并给出初始边界类型。"""

    items: list[HierarchicalBoundaryScore] = []
    for feature in boundary_features:
        repeat_score = max(feature.repeat_start_score, feature.repeat_end_score)
        sequence_inside_penalty = 1.0 if feature.sequence_role == "sequence_inside" else 0.0
        motif_score = (
            0.34 * feature.gap_break_score
            + 0.08 * feature.duration_release_score
            + 0.30 * feature.motive_end_score
            + 0.22 * repeat_score
            + 0.08 * feature.sequence_stop_score
            + 0.12 * feature.bar_hint_score
            - 0.28 * feature.continuity_penalty
            - 0.18 * sequence_inside_penalty
        )
        motif_score = min(1.0, max(0.0, motif_score))

        subphrase_score = (
            0.32 * feature.gap_break_score
            + 0.16 * feature.duration_release_score
            + 0.16 * feature.cadence_score
            + 0.22 * feature.motive_end_score
            + 0.22 * repeat_score
            + 0.16 * feature.sequence_stop_score
            + 0.08 * feature.bar_hint_score
            - 0.26 * feature.continuity_penalty
            - 0.22 * sequence_inside_penalty
        )
        subphrase_score = min(1.0, max(0.0, subphrase_score))

        phrase_score = (
            0.38 * feature.gap_break_score
            + 0.22 * feature.duration_release_score
            + 0.26 * feature.cadence_score
            + 0.10 * feature.motive_end_score
            + 0.16 * feature.repeat_end_score
            + 0.16 * feature.sequence_stop_score
            + 0.12 * feature.bar_hint_score
            - 0.34 * feature.continuity_penalty
            - 0.30 * sequence_inside_penalty
        )
        phrase_score = min(1.0, max(0.0, phrase_score))

        boundary_type = "none"
        if phrase_score >= config.phrase_threshold and feature.sequence_role != "sequence_inside":
            boundary_type = "phrase"
        elif subphrase_score >= config.subphrase_threshold:
            boundary_type = "subphrase"
        elif motif_score >= config.motif_threshold:
            boundary_type = "motif"

        items.append(
            HierarchicalBoundaryScore(
                note_index=feature.note_index,
                unit=feature.right_start_unit,
                bar_index=feature.bar_index,
                anchor_pos=feature.anchor_pos,
                motif_score=motif_score,
                subphrase_score=subphrase_score,
                phrase_score=phrase_score,
                boundary_type=boundary_type,
                sequence_role=feature.sequence_role,
                reasons=feature.reasons,
            )
        )
    return tuple(items)


def _replace_boundary_type(
    score: HierarchicalBoundaryScore,
    boundary_type: str,
    extra_reason: str | None = None,
) -> HierarchicalBoundaryScore:
    """复制一份边界分数，并可在升级或降级时追加单条原因。"""

    reasons = score.reasons
    if extra_reason is not None and extra_reason not in reasons:
        reasons = (*reasons, extra_reason)
    return HierarchicalBoundaryScore(
        note_index=score.note_index,
        unit=score.unit,
        bar_index=score.bar_index,
        anchor_pos=score.anchor_pos,
        motif_score=score.motif_score,
        subphrase_score=score.subphrase_score,
        phrase_score=score.phrase_score,
        boundary_type=boundary_type,
        sequence_role=score.sequence_role,
        reasons=reasons,
    )


def _promote_score_to_boundary_type(
    score: HierarchicalBoundaryScore,
    boundary_type: str,
    config: PhraseAnalysisConfig,
    extra_reason: str | None,
) -> HierarchicalBoundaryScore:
    """把边界候选提升到指定层级，并同步把对应分数抬到阈值以上。"""

    reasons = score.reasons
    if extra_reason is not None and extra_reason not in reasons:
        reasons = (*reasons, extra_reason)

    motif_score = score.motif_score
    subphrase_score = score.subphrase_score
    phrase_score = score.phrase_score
    if boundary_type == "motif":
        motif_score = max(motif_score, config.motif_threshold)
    elif boundary_type == "subphrase":
        motif_score = max(motif_score, config.motif_threshold)
        subphrase_score = max(subphrase_score, config.subphrase_threshold)
    elif boundary_type == "phrase":
        motif_score = max(motif_score, config.motif_threshold)
        subphrase_score = max(subphrase_score, config.subphrase_threshold)
        phrase_score = max(phrase_score, config.phrase_threshold)

    return HierarchicalBoundaryScore(
        note_index=score.note_index,
        unit=score.unit,
        bar_index=score.bar_index,
        anchor_pos=score.anchor_pos,
        motif_score=motif_score,
        subphrase_score=subphrase_score,
        phrase_score=phrase_score,
        boundary_type=boundary_type,
        sequence_role=score.sequence_role,
        reasons=reasons,
    )


def _is_salient_gap_boundary(score: HierarchicalBoundaryScore) -> bool:
    """判断当前候选是否属于应被优先保留的强空白断句点。"""

    reason_set = {str(reason) for reason in score.reasons}
    if "gap_break" not in reason_set:
        return False
    return float(score.phrase_score) >= 0.35


def _promote_salient_gap_boundaries(
    scores: Sequence[HierarchicalBoundaryScore],
    config: PhraseAnalysisConfig,
) -> tuple[HierarchicalBoundaryScore, ...]:
    """把明显大空挡提升为正式候选，避免自然呼吸点停留在 none。"""

    items: list[HierarchicalBoundaryScore] = []
    for score in scores:
        if score.sequence_role == "sequence_inside":
            items.append(score)
            continue
        if score.boundary_type != "none" or not _is_salient_gap_boundary(score):
            items.append(score)
            continue
        items.append(
            _promote_score_to_boundary_type(
                score,
                "subphrase",
                config,
                None,
            )
        )
    return tuple(items)


def _postprocess_boundary_scores(
    scores: Sequence[HierarchicalBoundaryScore],
    config: PhraseAnalysisConfig,
) -> tuple[HierarchicalBoundaryScore, ...]:
    """执行 Task 4 第一轮后处理。

    当前仅保留 sequence 相关约束：
    - `sequence_inside` 上若初始类型是 `phrase`，降级为 `motif` 或 `none`

    其余近邻强度合并逻辑先全部关闭，避免真实候选被提前吞掉。
    """

    items = [score for score in scores]
    for index, score in enumerate(items):
        if score.sequence_role != "sequence_inside" or score.boundary_type != "phrase":
            continue
        downgraded_type = "motif" if score.motif_score >= config.motif_threshold else "none"
        items[index] = _replace_boundary_type(score, downgraded_type)

    return tuple(items)


def _collect_bar_span_notes(
    notes_by_bar: dict[int, list[NoteInfo]],
    start_bar: int,
    span_bars: int,
) -> tuple[NoteInfo, ...]:
    """收集连续多小节片段的音符，用于判断长跨度重复结构。"""

    collected: list[NoteInfo] = []
    for bar_index in range(start_bar, start_bar + span_bars):
        bar_notes = notes_by_bar.get(bar_index)
        if not bar_notes:
            return tuple()
        collected.extend(bar_notes)
    return tuple(collected)


def _span_overlap_is_excessive(
    selected_spans: Sequence[tuple[int, int]],
    *,
    start_bar: int,
    span_bars: int,
) -> bool:
    """判断候选长跨度重复是否与已选跨度发生过度重叠。"""

    candidate_end = int(start_bar) + (int(span_bars) * 2)
    for kept_start_bar, kept_span_bars in selected_spans:
        kept_end = int(kept_start_bar) + (int(kept_span_bars) * 2)
        overlap = max(0, min(candidate_end, kept_end) - max(int(start_bar), int(kept_start_bar)))
        if overlap <= min(int(span_bars), int(kept_span_bars)):
            continue
        return True
    return False


def _detect_adjacent_repeated_bar_span_targets(
    notes: Sequence[NoteInfo],
    config: PhraseAnalysisConfig,
) -> dict[int, str]:
    """检测相邻重复的多小节结构，并返回应提升的后续重复单元起点边界层级。"""

    notes_by_bar: dict[int, list[NoteInfo]] = {}
    for note in notes:
        notes_by_bar.setdefault(note.bar_index, []).append(note)
    if not notes_by_bar:
        return {}

    content_bars = sorted(notes_by_bar)
    first_content_bar = content_bars[0]
    last_content_bar = content_bars[-1]
    max_span_bars = min(config.max_phrase_bars, max(2, config.preferred_phrase_bars))
    targets: dict[int, tuple[int, str]] = {}
    selected_spans: list[tuple[int, int]] = []

    for span_bars in range(max_span_bars, 1, -1):
        max_start_bar = last_content_bar - (span_bars * 2) + 1
        for start_bar in range(first_content_bar, max_start_bar + 1):
            if _span_overlap_is_excessive(
                selected_spans,
                start_bar=start_bar,
                span_bars=span_bars,
            ):
                continue
            left_notes = _collect_bar_span_notes(notes_by_bar, start_bar, span_bars)
            right_notes = _collect_bar_span_notes(notes_by_bar, start_bar + span_bars, span_bars)
            if len(left_notes) < config.repeat_min_notes or len(right_notes) < config.repeat_min_notes:
                continue
            if not _is_structurally_similar_fragment(left_notes, right_notes):
                similarity, _overlap_ratio, _ = _best_core_repeat_similarity(
                    left_notes,
                    right_notes,
                    min_core_notes=config.repeat_min_notes,
                    allow_partial_core=True,
                )
                if similarity < 0.78:
                    continue

            target_type = "phrase" if span_bars >= config.preferred_phrase_bars else "subphrase"
            repeated_start_bar = start_bar + span_bars
            previous = targets.get(repeated_start_bar)
            if previous is None or span_bars > previous[0]:
                targets[repeated_start_bar] = (span_bars, target_type)
            selected_spans.append((start_bar, span_bars))

    return {bar_index: target_type for bar_index, (_, target_type) in targets.items()}


def _promote_repeated_bar_span_boundaries(
    notes: Sequence[NoteInfo],
    scores: Sequence[HierarchicalBoundaryScore],
    config: PhraseAnalysisConfig,
) -> tuple[HierarchicalBoundaryScore, ...]:
    """把相邻重复多小节结构的段尾提升为更合理的最终边界层级。"""

    targets = _detect_adjacent_repeated_bar_span_targets(notes, config)
    if not targets:
        return tuple(scores)

    items = [score for score in scores]
    strongest_bar_head_index: dict[int, int] = {}
    for index, score in enumerate(items):
        if score.anchor_pos != 0:
            continue
        previous_index = strongest_bar_head_index.get(score.bar_index)
        if previous_index is None or score.phrase_score > items[previous_index].phrase_score:
            strongest_bar_head_index[score.bar_index] = index

    for bar_index, target_type in targets.items():
        score_index = strongest_bar_head_index.get(bar_index)
        if score_index is None:
            continue
        score = items[score_index]
        if score.sequence_role == "sequence_inside":
            continue
        if target_type == "phrase":
            items[score_index] = _promote_score_to_boundary_type(
                score,
                "phrase",
                config,
                "adjacent_repeated_bar_span",
            )
            continue
        if score.boundary_type in {"none", "motif"}:
            items[score_index] = _promote_score_to_boundary_type(
                score,
                "subphrase",
                config,
                "adjacent_repeated_bar_span",
            )

    return tuple(items)


def _assemble_phrase_boundaries_from_scores(
    notes: Sequence[NoteInfo],
    scores: Sequence[HierarchicalBoundaryScore],
    config: PhraseAnalysisConfig,
) -> tuple[PhraseBoundary, ...]:
    """从最终评分中提取真实结构边界，任一层分数达阈值都统一落成最终边界。"""

    del notes
    boundary_map: dict[tuple[int, int], PhraseBoundary] = {}

    for score in scores:
        if not (
            score.phrase_score >= config.phrase_threshold
            or score.subphrase_score >= config.subphrase_threshold
            or score.motif_score >= config.motif_threshold
        ):
            continue
        key = (score.bar_index, score.anchor_pos)
        boundary_map[key] = PhraseBoundary(bar_index=score.bar_index, anchor_pos=score.anchor_pos)

    return tuple(sorted(boundary_map.values(), key=lambda item: (item.bar_index, item.anchor_pos)))


def _resolve_analysis_start(
    notes: Sequence[NoteInfo],
) -> AnalysisAnchor | None:
    """解析分析起点，仅表示分析起始位置，不表示真实结构边界。"""

    if not notes:
        return None
    first = notes[0]
    return AnalysisAnchor(
        bar_index=first.bar_index,
        anchor_pos=first.pos_in_bar,
    )


def _quantile_threshold(values: Sequence[float], ratio: float) -> float:
    """计算启发式候选阈值。"""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def _pick_candidate_boundaries(
    boundary_scores: Sequence[BoundaryScore],
    config: PhraseAnalysisConfig,
) -> list[int]:
    """从小节级得分中挑出边界候选。"""

    if not boundary_scores:
        return []

    threshold = _quantile_threshold([item.score for item in boundary_scores], 0.75)
    candidates: list[BoundaryScore] = []
    for idx, item in enumerate(boundary_scores):
        prev_score = boundary_scores[idx - 1].score if idx > 0 else float("-inf")
        next_score = boundary_scores[idx + 1].score if idx + 1 < len(boundary_scores) else float("-inf")
        if item.score >= threshold and item.score >= prev_score and item.score >= next_score:
            candidates.append(item)

    filtered: list[int] = []
    for item in candidates:
        if not filtered or (item.bar_index - filtered[-1]) >= config.min_boundary_gap_bars:
            filtered.append(item.bar_index)
        elif boundary_scores[item.bar_index - 1].score > boundary_scores[filtered[-1] - 1].score:
            filtered[-1] = item.bar_index
    return filtered


def _pick_in_bar_anchor(
    left_bar: BarInfo,
    right_bar: BarInfo,
    config: PhraseAnalysisConfig,
) -> int:
    """只在右侧小节前半拍为空时，把边界落到首个 onset。"""

    del left_bar
    if not right_bar.onset_positions:
        return 0
    first_onset = right_bar.onset_positions[0]
    if first_onset >= config.mid_bar_min_rest_pos:
        return first_onset
    return 0


def _assemble_final_boundaries(
    bars: Sequence[BarInfo],
    candidate_boundary_bars: Sequence[int],
    config: PhraseAnalysisConfig,
) -> tuple[PhraseBoundary, ...]:
    """沿用当前小节级逻辑装配最终边界。"""

    if not bars:
        return tuple()

    first_content_bar = next((i for i, bar in enumerate(bars) if bar.note_count > 0), None)
    if first_content_bar is None:
        return tuple()

    boundary_set: dict[tuple[int, int], PhraseBoundary] = {}
    forced = PhraseBoundary(bar_index=first_content_bar, anchor_pos=0)
    boundary_set[(forced.bar_index, forced.anchor_pos)] = forced

    for bar_index in candidate_boundary_bars:
        if bar_index <= first_content_bar or bar_index >= len(bars):
            continue
        if bars[bar_index].note_count == 0:
            continue
        anchor_pos = _pick_in_bar_anchor(bars[bar_index - 1], bars[bar_index], config)
        key = (bar_index, anchor_pos)
        if key not in boundary_set:
            boundary_set[key] = PhraseBoundary(bar_index=bar_index, anchor_pos=anchor_pos)

    ordered = sorted(boundary_set.values(), key=lambda item: (item.bar_index, item.anchor_pos))

    expanded: list[PhraseBoundary] = []
    for idx, current in enumerate(ordered):
        expanded.append(current)
        next_bar = ordered[idx + 1].bar_index if idx + 1 < len(ordered) else len(bars)
        cursor = current.bar_index
        gap = next_bar - cursor
        while gap > config.max_phrase_bars:
            synth_bar = cursor + config.preferred_phrase_bars
            if synth_bar >= next_bar:
                break
            while synth_bar < next_bar and bars[synth_bar].note_count == 0:
                synth_bar += 1
            if synth_bar >= next_bar:
                break
            expanded.append(PhraseBoundary(bar_index=synth_bar, anchor_pos=0))
            cursor = synth_bar
            gap = next_bar - cursor

    merged: list[PhraseBoundary] = []
    for boundary in expanded:
        if not merged:
            merged.append(boundary)
            continue
        prev = merged[-1]
        if boundary.bar_index - prev.bar_index < config.min_phrase_bars and boundary.bar_index != first_content_bar:
            continue
        merged.append(boundary)

    bar_indices = [item.bar_index for item in merged]
    assert len(bar_indices) == len(set(bar_indices)), "同一小节中生成了多个乐句边界"
    return tuple(merged)


def _derive_phrase_spans(
    bars: Sequence[BarInfo],
    boundaries: Sequence[PhraseBoundary],
    analysis_start: AnalysisAnchor | None,
) -> tuple[PhraseSpan, ...]:
    """根据分析起点与真实结构边界推导乐句跨度。"""

    if not bars or analysis_start is None:
        return tuple()
    sorted_boundaries = sorted(boundaries, key=lambda item: (item.bar_index, item.anchor_pos))
    cut_bars = [analysis_start.bar_index, *(item.bar_index for item in sorted_boundaries)]
    unique_cut_bars: list[int] = []
    for bar_index in cut_bars:
        if not unique_cut_bars or unique_cut_bars[-1] != bar_index:
            unique_cut_bars.append(bar_index)
    if unique_cut_bars[-1] != len(bars):
        unique_cut_bars.append(len(bars))
    spans: list[PhraseSpan] = []
    for start_bar, end_bar in zip(unique_cut_bars, unique_cut_bars[1:]):
        if end_bar <= start_bar:
            continue
        spans.append(
            PhraseSpan(
                start_bar=start_bar,
                end_bar=end_bar,
                start_token=bars[start_bar].start_token,
                end_token=bars[end_bar - 1].end_token,
                tempo_token=bars[start_bar].effective_tempo_token,
                key_token=bars[start_bar].effective_key_token,
                tokens=tuple(),
                source_kind="single_phrase",
            )
        )
    return tuple(spans)


def analyze_phrase_candidates(
    tokens: Sequence[str],
    config: PhraseAnalysisConfig | None = None,
) -> PhraseAnalysis:
    """从单条 token 序列中分析乐句候选区间。

    当前真实链路如下：
    - `boundary_scores` 来自 note-level 特征构造、三层评分与 Task 4 第一轮后处理。
    - `boundaries`、`phrase_spans` 已来自 note-level 主链路上的后处理与最终边界组装。
    - 当前仍只实现 Task 4 的第一轮后处理，不额外扩展到计划外规则。
    """

    config = PhraseAnalysisConfig() if config is None else config
    bars = _build_bar_info(tokens, config)
    notes = _build_note_info(tokens, config)
    boundary_features = _build_default_boundary_features(notes, config)
    raw_scores = _score_boundary_features(boundary_features, config)
    boundary_scores = _postprocess_boundary_scores(raw_scores, config)
    boundary_scores = _promote_repeated_bar_span_boundaries(notes, boundary_scores, config)
    boundary_scores = _promote_salient_gap_boundaries(boundary_scores, config)
    analysis_start = _resolve_analysis_start(notes)
    boundaries = _assemble_phrase_boundaries_from_scores(notes, boundary_scores, config)
    phrase_spans = _derive_phrase_spans(bars, boundaries, analysis_start)
    return PhraseAnalysis(
        bars=bars,
        notes=notes,
        boundary_features=boundary_features,
        boundary_scores=boundary_scores,
        boundaries=boundaries,
        phrase_spans=phrase_spans,
        analysis_start=analysis_start,
    )
