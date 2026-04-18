"""Phrase-oriented analysis and sampling for TuneFlow token sequences."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


@dataclass(frozen=True)
class PhraseAnalysisConfig:
    """Heuristic config for phrase boundary analysis."""

    positions_per_bar: int = 32
    min_phrase_bars: int = 2
    max_phrase_bars: int = 8
    preferred_phrase_bars: int = 4
    min_boundary_gap_bars: int = 2
    rest_weight: float = 0.40
    note_density_weight: float = 0.24
    onset_density_weight: float = 0.18
    pitch_span_weight: float = 0.12
    duration_weight: float = 0.06


@dataclass(frozen=True)
class BarInfo:
    """Statistics for one bar in a token sequence."""

    start_token: int
    end_token: int
    note_count: int
    onset_count: int
    rest_ratio: float
    pitch_span: int
    mean_duration: float
    effective_tempo_token: str | None


@dataclass(frozen=True)
class BoundaryScore:
    """Boundary score attached to the boundary before `bar_index`."""

    bar_index: int
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PhraseSpan:
    """A phrase-like bar span with normalized phrase-view tokens."""

    start_bar: int
    end_bar: int
    start_token: int
    end_token: int
    tempo_token: str | None
    tokens: tuple[str, ...]
    source_kind: str


@dataclass(frozen=True)
class PhraseAnalysis:
    """Result of phrase analysis for a token sequence."""

    bars: tuple[BarInfo, ...]
    boundary_scores: tuple[BoundaryScore, ...]
    phrase_spans: tuple[PhraseSpan, ...]


@dataclass(frozen=True)
class PhraseWindowPolicy:
    """Sampling policy for phrase-aware windows."""

    kind: str
    min_bars: int
    max_bars: int
    max_tokens: int


@dataclass(frozen=True)
class SampledWindow:
    """A normalized sampled window ready for training or eval."""

    tokens: tuple[str, ...]
    source_kind: str
    start_bar: int
    end_bar: int
    tempo_token: str | None
    boundary_count: int


def _safe_ratio(delta: float, left: float, right: float) -> float:
    denom = max(1.0, abs(left), abs(right))
    return min(1.0, max(0.0, float(delta) / denom))


def _iter_bar_slices(tokens: Sequence[str]) -> tuple[list[tuple[int, int, str | None]], str | None] | None:
    if not tokens or tokens[0] != "BOS":
        return None

    effective_end = len(tokens) - 1 if tokens[-1] == "EOS" else len(tokens)
    current_tempo: str | None = None
    idx = 1
    if idx < effective_end and str(tokens[idx]).startswith("TEMPO_"):
        current_tempo = str(tokens[idx])
        idx += 1

    bars: list[tuple[int, int, str | None]] = []
    while idx < effective_end:
        if tokens[idx] != "BAR":
            return None
        bar_start = idx
        idx += 1
        if idx < effective_end and str(tokens[idx]).startswith("TEMPO_"):
            current_tempo = str(tokens[idx])
            idx += 1
        while idx < effective_end and tokens[idx] != "BAR":
            idx += 1
        bars.append((bar_start, idx, current_tempo))
    return bars, current_tempo


def _build_bar_info(tokens: Sequence[str], config: PhraseAnalysisConfig) -> tuple[BarInfo, ...]:
    parsed = _iter_bar_slices(tokens)
    if parsed is None:
        return tuple()
    raw_bars, _ = parsed

    bars: list[BarInfo] = []
    for start_token, end_token, effective_tempo in raw_bars:
        idx = start_token + 1
        if idx < end_token and str(tokens[idx]).startswith("TEMPO_"):
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


def _find_best_split(
    start_bar: int,
    end_bar: int,
    candidate_boundaries: Sequence[int],
    config: PhraseAnalysisConfig,
) -> int:
    ideal = start_bar + config.preferred_phrase_bars
    valid_candidates = [
        boundary
        for boundary in candidate_boundaries
        if start_bar + config.min_phrase_bars <= boundary <= end_bar - config.min_phrase_bars
    ]
    if valid_candidates:
        return min(valid_candidates, key=lambda boundary: (abs(boundary - ideal), abs(boundary - start_bar)))
    fallback = min(end_bar - config.min_phrase_bars, max(start_bar + config.min_phrase_bars, ideal))
    return fallback


def _merge_short_spans(
    spans: list[tuple[int, int]],
    config: PhraseAnalysisConfig,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    idx = 0
    while idx < len(spans):
        start_bar, end_bar = spans[idx]
        length = end_bar - start_bar
        if length >= config.min_phrase_bars or len(spans) == 1:
            merged.append((start_bar, end_bar))
            idx += 1
            continue
        if merged:
            prev_start, _prev_end = merged[-1]
            merged[-1] = (prev_start, end_bar)
            idx += 1
            continue
        if idx + 1 < len(spans):
            _next_start, next_end = spans[idx + 1]
            merged.append((start_bar, next_end))
            idx += 2
            continue
        merged.append((start_bar, end_bar))
        idx += 1
    return merged


def _build_phrase_spans(
    tokens: Sequence[str],
    bars: Sequence[BarInfo],
    boundary_scores: Sequence[BoundaryScore],
    config: PhraseAnalysisConfig,
) -> tuple[PhraseSpan, ...]:
    if not bars:
        return tuple()

    candidate_boundaries = _pick_candidate_boundaries(boundary_scores, config)
    boundaries = [0, *candidate_boundaries, len(bars)]
    raw_spans: list[tuple[int, int]] = []
    for idx in range(len(boundaries) - 1):
        start_bar = boundaries[idx]
        end_bar = boundaries[idx + 1]
        if end_bar > start_bar:
            raw_spans.append((start_bar, end_bar))

    merged_spans = _merge_short_spans(raw_spans, config)
    normalized_spans: list[tuple[int, int]] = []
    for start_bar, end_bar in merged_spans:
        cursor = start_bar
        while end_bar - cursor > config.max_phrase_bars:
            split_bar = _find_best_split(cursor, end_bar, candidate_boundaries, config)
            normalized_spans.append((cursor, split_bar))
            cursor = split_bar
        normalized_spans.append((cursor, end_bar))

    phrase_spans: list[PhraseSpan] = []
    for start_bar, end_bar in normalized_spans:
        phrase_spans.append(
            _build_phrase_span(
                tokens=tokens,
                bars=bars,
                start_bar=start_bar,
                end_bar=end_bar,
                source_kind="single_phrase",
            )
        )
    return tuple(phrase_spans)


def _normalized_bar_tokens(tokens: Sequence[str], bar: BarInfo) -> list[str]:
    raw_tokens = [str(token) for token in tokens[bar.start_token : bar.end_token]]
    if len(raw_tokens) >= 2 and raw_tokens[0] == "BAR" and raw_tokens[1].startswith("TEMPO_"):
        return ["BAR", *raw_tokens[2:]]
    return raw_tokens


def _build_phrase_view_tokens(
    tokens: Sequence[str],
    bars: Sequence[BarInfo],
    start_bar: int,
    end_bar: int,
) -> tuple[str, ...]:
    if not (0 <= start_bar < end_bar <= len(bars)):
        raise IndexError("invalid bar span")
    phrase_tokens: list[str] = ["BOS"]
    tempo_token = bars[start_bar].effective_tempo_token
    if tempo_token is not None:
        phrase_tokens.append(tempo_token)
    for bar_index in range(start_bar, end_bar):
        phrase_tokens.extend(_normalized_bar_tokens(tokens, bars[bar_index]))
    phrase_tokens.append("EOS")
    return tuple(phrase_tokens)


def _build_phrase_span(
    tokens: Sequence[str],
    bars: Sequence[BarInfo],
    start_bar: int,
    end_bar: int,
    source_kind: str,
) -> PhraseSpan:
    start_token = bars[start_bar].start_token
    end_token = bars[end_bar - 1].end_token
    tempo_token = bars[start_bar].effective_tempo_token
    return PhraseSpan(
        start_bar=start_bar,
        end_bar=end_bar,
        start_token=start_token,
        end_token=end_token,
        tempo_token=tempo_token,
        tokens=_build_phrase_view_tokens(tokens, bars, start_bar, end_bar),
        source_kind=source_kind,
    )


def analyze_phrase_candidates(
    tokens: Sequence[str],
    config: PhraseAnalysisConfig | None = None,
) -> PhraseAnalysis:
    """Analyze phrase candidates from one token sequence."""
    config = PhraseAnalysisConfig() if config is None else config
    bars = _build_bar_info(tokens, config)
    boundary_scores = _build_boundary_scores(bars, config)
    phrase_spans = _build_phrase_spans(tokens, bars, boundary_scores, config)
    return PhraseAnalysis(
        bars=bars,
        boundary_scores=boundary_scores,
        phrase_spans=phrase_spans,
    )


def extract_phrase(
    tokens: Sequence[str],
    analysis: PhraseAnalysis,
    phrase_index: int,
    tempo_mode: str = "phrase_start",
) -> PhraseSpan:
    """Extract one analyzed phrase span."""
    if tempo_mode != "phrase_start":
        raise ValueError(f"Unsupported tempo_mode: {tempo_mode!r}")
    if phrase_index < 0 or phrase_index >= len(analysis.phrase_spans):
        raise IndexError(f"phrase_index out of range: {phrase_index}")
    span = analysis.phrase_spans[phrase_index]
    rebuilt_tokens = _build_phrase_view_tokens(tokens, analysis.bars, span.start_bar, span.end_bar)
    return PhraseSpan(
        start_bar=span.start_bar,
        end_bar=span.end_bar,
        start_token=span.start_token,
        end_token=span.end_token,
        tempo_token=span.tempo_token,
        tokens=rebuilt_tokens,
        source_kind=span.source_kind,
    )


def _phrase_boundaries_from_spans(analysis: PhraseAnalysis) -> set[int]:
    return {span.start_bar for span in analysis.phrase_spans[1:]}


def _count_phrase_boundaries(analysis: PhraseAnalysis, start_bar: int, end_bar: int) -> int:
    boundaries = _phrase_boundaries_from_spans(analysis)
    return sum(1 for boundary in boundaries if start_bar < boundary < end_bar)


def _choose_single_phrase_window(
    tokens: Sequence[str],
    analysis: PhraseAnalysis,
    policy: PhraseWindowPolicy,
    rng: random.Random,
) -> SampledWindow | None:
    candidates: list[SampledWindow] = []
    for span in analysis.phrase_spans:
        span_len = span.end_bar - span.start_bar
        if policy.min_bars <= span_len <= policy.max_bars and len(span.tokens) <= policy.max_tokens:
            candidates.append(
                SampledWindow(
                    tokens=span.tokens,
                    source_kind="single_phrase",
                    start_bar=span.start_bar,
                    end_bar=span.end_bar,
                    tempo_token=span.tempo_token,
                    boundary_count=0,
                )
            )
            continue
        if span_len <= policy.max_bars:
            continue
        for sub_len in range(policy.min_bars, policy.max_bars + 1):
            if sub_len > span_len:
                break
            max_start = span.end_bar - sub_len
            for start_bar in range(span.start_bar, max_start + 1):
                end_bar = start_bar + sub_len
                sampled_tokens = _build_phrase_view_tokens(tokens, analysis.bars, start_bar, end_bar)
                if len(sampled_tokens) > policy.max_tokens:
                    continue
                candidates.append(
                    SampledWindow(
                        tokens=sampled_tokens,
                        source_kind="single_phrase",
                        start_bar=start_bar,
                        end_bar=end_bar,
                        tempo_token=analysis.bars[start_bar].effective_tempo_token,
                        boundary_count=0,
                    )
                )
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _choose_cross_boundary_window(
    tokens: Sequence[str],
    analysis: PhraseAnalysis,
    policy: PhraseWindowPolicy,
    rng: random.Random,
) -> SampledWindow | None:
    if len(analysis.phrase_spans) < 2:
        return None

    candidates: list[SampledWindow] = []
    for left, right in zip(analysis.phrase_spans[:-1], analysis.phrase_spans[1:], strict=True):
        boundary_bar = right.start_bar
        left_available = boundary_bar - left.start_bar
        right_available = right.end_bar - boundary_bar
        for total_bars in range(policy.min_bars, policy.max_bars + 1):
            for left_take in range(1, total_bars):
                right_take = total_bars - left_take
                if left_take > left_available or right_take > right_available:
                    continue
                start_bar = boundary_bar - left_take
                end_bar = boundary_bar + right_take
                sampled_tokens = _build_phrase_view_tokens(tokens, analysis.bars, start_bar, end_bar)
                if len(sampled_tokens) > policy.max_tokens:
                    continue
                candidates.append(
                    SampledWindow(
                        tokens=sampled_tokens,
                        source_kind="cross_boundary",
                        start_bar=start_bar,
                        end_bar=end_bar,
                        tempo_token=analysis.bars[start_bar].effective_tempo_token,
                        boundary_count=1,
                    )
                )
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _choose_long_context_window(
    tokens: Sequence[str],
    analysis: PhraseAnalysis,
    policy: PhraseWindowPolicy,
    rng: random.Random,
) -> SampledWindow | None:
    if not analysis.bars:
        return None

    candidates: list[SampledWindow] = []
    phrase_boundaries = [span.start_bar for span in analysis.phrase_spans]
    phrase_ends = [span.end_bar for span in analysis.phrase_spans]
    for start_bar in phrase_boundaries:
        for end_bar in phrase_ends:
            if end_bar <= start_bar:
                continue
            span_len = end_bar - start_bar
            if span_len < policy.min_bars or span_len > policy.max_bars:
                continue
            boundary_count = _count_phrase_boundaries(analysis, start_bar, end_bar)
            if boundary_count < 1:
                continue
            sampled_tokens = _build_phrase_view_tokens(tokens, analysis.bars, start_bar, end_bar)
            if len(sampled_tokens) > policy.max_tokens:
                continue
            candidates.append(
                SampledWindow(
                    tokens=sampled_tokens,
                    source_kind="long_context",
                    start_bar=start_bar,
                    end_bar=end_bar,
                    tempo_token=analysis.bars[start_bar].effective_tempo_token,
                    boundary_count=boundary_count,
                )
            )
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def sample_phrase_window(
    tokens: Sequence[str],
    analysis: PhraseAnalysis,
    policy: PhraseWindowPolicy,
    rng: random.Random,
) -> SampledWindow | None:
    """Sample one phrase-aware window from a token sequence."""
    if policy.kind == "single_phrase":
        return _choose_single_phrase_window(tokens, analysis, policy, rng)
    if policy.kind == "cross_boundary":
        return _choose_cross_boundary_window(tokens, analysis, policy, rng)
    if policy.kind == "long_context":
        return _choose_long_context_window(tokens, analysis, policy, rng)
    raise ValueError(f"Unsupported phrase window policy kind: {policy.kind!r}")
