"""TuneFlow token 序列的乐句分析工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PhraseAnalysisConfig:
    """乐句边界分析的启发式配置。"""

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


@dataclass(frozen=True)
class PhraseBoundary:
    """PHRASE 落点。anchor_pos==0 表示 bar-aligned；>0 表示 mid-bar POS 槽前。"""

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
class BoundaryScore:
    """附着在 `bar_index` 之前边界上的得分。"""

    bar_index: int
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PhraseSpan:
    """带标准化乐句视图 token 的类乐句小节区间。"""

    start_bar: int
    end_bar: int
    start_token: int
    end_token: int
    tempo_token: str | None
    key_token: str | None
    tokens: tuple[str, ...]
    source_kind: str


@dataclass(frozen=True)
class PhraseAnalysis:
    """单条 token 序列的乐句分析结果。"""

    bars: tuple[BarInfo, ...]
    boundary_scores: tuple[BoundaryScore, ...]
    boundaries: tuple[PhraseBoundary, ...]
    phrase_spans: tuple[PhraseSpan, ...]


def _safe_ratio(delta: float, left: float, right: float) -> float:
    denom = max(1.0, abs(left), abs(right))
    return min(1.0, max(0.0, float(delta) / denom))


def _iter_bar_slices(
    tokens: Sequence[str],
) -> tuple[list[tuple[int, int, str | None, str | None]], str | None, str | None] | None:
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


def _build_boundary_scores(bars: Sequence[BarInfo], config: PhraseAnalysisConfig) -> tuple[BoundaryScore, ...]:
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
        reasons = tuple(
            label for label, contribution in contributions.items() if contribution >= 0.10
        )
        scores.append(BoundaryScore(bar_index=left_index + 1, score=score, reasons=reasons))
    return tuple(scores)


def _quantile_threshold(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def _pick_candidate_boundaries(
    boundary_scores: Sequence[BoundaryScore],
    config: PhraseAnalysisConfig,
) -> list[int]:
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
    """v1：只在右 bar 起首有 >=1 拍留白时把 PHRASE 推迟到首个 onset。"""
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

    ordered = sorted(boundary_set.values(), key=lambda b: (b.bar_index, b.anchor_pos))

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
        if (
            boundary.bar_index - prev.bar_index < config.min_phrase_bars
            and boundary.bar_index != first_content_bar
        ):
            continue
        merged.append(boundary)

    # Invariant: at most one boundary per bar after merging.
    bar_indices = [b.bar_index for b in merged]
    assert len(bar_indices) == len(set(bar_indices)), (
        f"_assemble_final_boundaries produced multiple boundaries in the same bar: {merged}"
    )
    return tuple(merged)


def _derive_phrase_spans(
    bars: Sequence[BarInfo],
    boundaries: Sequence[PhraseBoundary],
) -> tuple[PhraseSpan, ...]:
    if not bars or not boundaries:
        return tuple()
    sorted_boundaries = sorted(boundaries, key=lambda b: (b.bar_index, b.anchor_pos))
    spans: list[PhraseSpan] = []
    for idx, current in enumerate(sorted_boundaries):
        start_bar = current.bar_index
        end_bar = sorted_boundaries[idx + 1].bar_index if idx + 1 < len(sorted_boundaries) else len(bars)
        if end_bar <= start_bar:
            continue
        start_token = bars[start_bar].start_token
        end_token = bars[end_bar - 1].end_token
        spans.append(
            PhraseSpan(
                start_bar=start_bar,
                end_bar=end_bar,
                start_token=start_token,
                end_token=end_token,
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
    """从单条 token 序列中分析乐句候选区间。"""
    config = PhraseAnalysisConfig() if config is None else config
    bars = _build_bar_info(tokens, config)
    boundary_scores = _build_boundary_scores(bars, config)
    candidate_boundary_bars = _pick_candidate_boundaries(boundary_scores, config)
    boundaries = _assemble_final_boundaries(bars, candidate_boundary_bars, config)
    phrase_spans = _derive_phrase_spans(bars, boundaries)
    return PhraseAnalysis(
        bars=bars,
        boundary_scores=boundary_scores,
        boundaries=boundaries,
        phrase_spans=phrase_spans,
    )
