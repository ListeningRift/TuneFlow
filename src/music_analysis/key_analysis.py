"""基于加权 pitch-class histogram 与 HMM 平滑的调性时间线分析。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Sequence


_PITCH_CLASS_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_UNCERTAIN_KEY = "uncertain"
_ALL_KEY_NAMES = tuple([f"{name}:maj" for name in _PITCH_CLASS_NAMES] + [f"{name}:min" for name in _PITCH_CLASS_NAMES])


@dataclass(frozen=True)
class KeyAnalysisConfig:
    """加权局部调性分析与 HMM 平滑配置。"""

    positions_per_bar: int = 32
    window_bars: float = 1.0
    hop_bars: float = 0.5
    bar_start_weight: float = 1.60
    strong_beat_weight: float = 1.30
    weak_beat_weight: float = 1.00
    strong_beat_stride: int = 8
    min_best_score: float = 0.30
    min_score_margin: float = 0.10
    neighborhood_radius_frames: int = 3
    neighborhood_decay: float = 0.65
    modulation_confirmation_frames: int = 2
    global_key_bias: float = 0.18
    key_change_penalty: float = 0.45


@dataclass(frozen=True)
class KeyFrame:
    """一个重叠局部窗口的调性判别结果。"""

    start_bar: int
    start_pos: int
    end_bar: int
    end_pos: int
    best_key: str
    best_score: float
    margin_to_second: float
    is_uncertain: bool
    raw_key: str
    smoothed_support: float


@dataclass(frozen=True)
class KeySegment:
    """发布给上层使用的稳定调性段。"""

    key: str
    start_bar: int
    start_pos: int
    end_bar: int
    end_pos: int
    mean_score: float


@dataclass(frozen=True)
class ModulationPoint:
    """稀疏转调点。"""

    bar_index: int
    pos_in_bar: int
    from_key: str
    to_key: str
    support: float


@dataclass(frozen=True)
class KeyTimelineAnalysis:
    """单条 token 序列的完整调性分析结果。"""

    frames: tuple[KeyFrame, ...]
    segments: tuple[KeySegment, ...]
    modulation_points: tuple[ModulationPoint, ...]
    initial_key: str


@dataclass(frozen=True)
class _TokenEvent:
    bar_index: int
    pos: int
    pitch: int
    dur: int


@dataclass(frozen=True)
class _ParsedTokenSequence:
    events: tuple[_TokenEvent, ...]
    bar_count: int
    total_units: int


@dataclass(frozen=True)
class _RawFrame:
    start_unit: int
    end_unit: int
    raw_key: str
    best_score: float
    margin_to_second: float
    is_uncertain: bool
    score_by_key: tuple[tuple[str, float], ...]


def _parse_prefixed_int(token: str, prefix: str) -> int | None:
    if not token.startswith(prefix):
        return None
    try:
        return int(token[len(prefix) :])
    except ValueError:
        return None


def _key_name(root: int, mode: str) -> str:
    suffix = "maj" if mode == "major" else "min"
    return f"{_PITCH_CLASS_NAMES[root % 12]}:{suffix}"


def _rotate_profile(profile: Sequence[float], root: int) -> tuple[float, ...]:
    return tuple(float(profile[(index - root) % 12]) for index in range(12))


def _pearson_correlation(values: Sequence[float], reference: Sequence[float]) -> float:
    if len(values) != len(reference) or not values:
        return 0.0
    value_mean = sum(float(item) for item in values) / float(len(values))
    reference_mean = sum(float(item) for item in reference) / float(len(reference))

    numerator = 0.0
    value_var = 0.0
    reference_var = 0.0
    for value, ref_value in zip(values, reference, strict=True):
        centered_value = float(value) - value_mean
        centered_ref = float(ref_value) - reference_mean
        numerator += centered_value * centered_ref
        value_var += centered_value * centered_value
        reference_var += centered_ref * centered_ref

    denom = math.sqrt(value_var) * math.sqrt(reference_var)
    if denom <= 0.0:
        return 0.0
    return float(numerator / denom)


def _unit_to_bar_pos(unit: int, positions_per_bar: int) -> tuple[int, int]:
    safe_positions = max(1, int(positions_per_bar))
    clamped_unit = max(0, int(unit))
    return (clamped_unit // safe_positions, clamped_unit % safe_positions)


def _parse_token_events(tokens: Sequence[str], config: KeyAnalysisConfig) -> _ParsedTokenSequence:
    values = [str(token) for token in tokens]
    if not values or values[0] != "BOS":
        return _ParsedTokenSequence(events=tuple(), bar_count=0, total_units=0)

    effective_end = len(values) - 1 if values[-1] == "EOS" else len(values)
    idx = 1
    if idx < effective_end and values[idx].startswith("TEMPO_"):
        idx += 1

    current_bar = -1
    bar_count = 0
    max_event_end_unit = 0
    events: list[_TokenEvent] = []

    while idx < effective_end:
        token = values[idx]
        if token == "BAR":
            current_bar += 1
            bar_count = max(bar_count, current_bar + 1)
            idx += 1
            if idx < effective_end and values[idx].startswith("TEMPO_"):
                idx += 1
            continue
        if token.startswith("POS_") and current_bar >= 0:
            if idx + 4 >= effective_end:
                break
            pos_value = _parse_prefixed_int(token, "POS_")
            inst_token = values[idx + 1]
            pitch_value = _parse_prefixed_int(values[idx + 2], "PITCH_")
            dur_value = _parse_prefixed_int(values[idx + 3], "DUR_")
            vel_token = values[idx + 4]
            if (
                pos_value is None
                or pitch_value is None
                or dur_value is None
                or not inst_token.startswith("INST_")
                or not vel_token.startswith("VEL_")
            ):
                idx += 1
                continue
            clamped_pos = min(max(0, int(pos_value)), max(0, config.positions_per_bar - 1))
            clamped_dur = max(1, int(dur_value))
            events.append(
                _TokenEvent(
                    bar_index=current_bar,
                    pos=clamped_pos,
                    pitch=int(pitch_value),
                    dur=clamped_dur,
                )
            )
            max_event_end_unit = max(
                max_event_end_unit,
                (current_bar * max(1, config.positions_per_bar)) + clamped_pos + clamped_dur,
            )
            idx += 5
            continue
        idx += 1

    total_units = max(bar_count * max(1, config.positions_per_bar), max_event_end_unit)
    return _ParsedTokenSequence(events=tuple(events), bar_count=bar_count, total_units=total_units)


def _onset_weight(pos: int, config: KeyAnalysisConfig) -> float:
    if pos == 0:
        return float(config.bar_start_weight)
    if config.strong_beat_stride > 0 and pos % int(config.strong_beat_stride) == 0:
        return float(config.strong_beat_weight)
    return float(config.weak_beat_weight)


def _weighted_pitch_class_histogram(
    parsed: _ParsedTokenSequence,
    *,
    start_unit: int,
    end_unit: int,
    config: KeyAnalysisConfig,
) -> tuple[float, ...]:
    histogram = [0.0] * 12
    if end_unit <= start_unit:
        return tuple(histogram)

    positions_per_bar = max(1, config.positions_per_bar)
    for event in parsed.events:
        event_start = (event.bar_index * positions_per_bar) + event.pos
        event_end = max(event_start + 1, event_start + event.dur)
        overlap = min(end_unit, event_end) - max(start_unit, event_start)
        if overlap <= 0:
            continue
        histogram[event.pitch % 12] += float(overlap) * _onset_weight(event.pos, config)
    return tuple(histogram)


def _rank_key_scores(histogram: Sequence[float]) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for root in range(12):
        scores.append((_key_name(root, "major"), _pearson_correlation(histogram, _rotate_profile(_MAJOR_PROFILE, root))))
    for root in range(12):
        scores.append((_key_name(root, "minor"), _pearson_correlation(histogram, _rotate_profile(_MINOR_PROFILE, root))))
    scores.sort(key=lambda item: (-float(item[1]), str(item[0])))
    return scores


def _score_lookup(score_by_key: Sequence[tuple[str, float]]) -> dict[str, float]:
    lookup = {str(key_name): float(score) for key_name, score in score_by_key}
    # 后续 HMM 解码希望所有状态都有显式分数，未出现的 key 统一补成 -inf，
    # 这样既不需要到处判空，也不会让缺失状态误参与比较。
    for key_name in _ALL_KEY_NAMES:
        lookup.setdefault(key_name, float("-inf"))
    return lookup


def _global_ranked_scores(parsed: _ParsedTokenSequence, config: KeyAnalysisConfig) -> tuple[tuple[str, float], ...]:
    # 全曲 histogram 不参与直接定位切换点，而是作为整首的主调先验，
    # 用来在 HMM 中轻微偏向“全曲上更合理”的 key。
    histogram = _weighted_pitch_class_histogram(
        parsed,
        start_unit=0,
        end_unit=max(1, int(parsed.total_units)),
        config=config,
    )
    return tuple(_rank_key_scores(histogram))


def _build_raw_frames(parsed: _ParsedTokenSequence, config: KeyAnalysisConfig) -> tuple[_RawFrame, ...]:
    if parsed.bar_count <= 0 or parsed.total_units <= 0:
        return tuple()

    positions_per_bar = max(1, config.positions_per_bar)
    window_units = max(1, int(round(float(config.window_bars) * positions_per_bar)))
    hop_units = max(1, int(round(float(config.hop_bars) * positions_per_bar)))
    total_units = max(window_units, int(parsed.total_units))
    last_start = max(0, total_units - window_units)

    # 这里按固定 hop 扫描全曲，并强制补上最后一个起点，
    # 避免尾部不足一个 hop 时直接丢失最后一段音乐。
    starts = list(range(0, last_start + 1, hop_units))
    if not starts:
        starts = [0]
    if starts[-1] != last_start:
        starts.append(last_start)

    frames: list[_RawFrame] = []
    for start_unit in starts:
        end_unit = start_unit + window_units
        histogram = _weighted_pitch_class_histogram(parsed, start_unit=start_unit, end_unit=end_unit, config=config)
        ranked = _rank_key_scores(histogram)
        best_key, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
        margin = float(best_score - second_score) if math.isfinite(float(second_score)) else 0.0
        total_weight = sum(float(value) for value in histogram)
        # uncertain 只表示“这个局部窗口不够可信”，不是独立 HMM 状态。
        # 真正的路径解码仍在 24 个大/小调状态上完成，uncertain 主要用于后续段落发布时过滤抖动。
        is_uncertain = (
            total_weight <= 0.0
            or float(best_score) < float(config.min_best_score)
            or float(margin) < float(config.min_score_margin)
        )
        frames.append(
            _RawFrame(
                start_unit=start_unit,
                end_unit=end_unit,
                raw_key=best_key,
                best_score=float(best_score),
                margin_to_second=max(0.0, float(margin)),
                is_uncertain=bool(is_uncertain),
                score_by_key=tuple((key_name, float(score)) for key_name, score in ranked),
            )
        )
    return tuple(frames)


def _hmm_emission_score(
    frame_scores: dict[str, float],
    *,
    key_name: str,
    global_scores: dict[str, float],
    config: KeyAnalysisConfig,
) -> float:
    # 发射分数 = 局部窗口对该 key 的相似度 + 全曲主调先验。
    # 先验只做轻微偏置，避免局部证据足够强时被全曲主调完全压住。
    return float(frame_scores[key_name]) + (float(config.global_key_bias) * max(0.0, float(global_scores[key_name])))


def _decode_hmm_key_path(
    raw_frames: Sequence[_RawFrame],
    *,
    global_scores: dict[str, float],
    config: KeyAnalysisConfig,
) -> tuple[str, ...]:
    if not raw_frames:
        return tuple()

    # 先把每帧的候选分数字典补齐到 24 个调，后面 Viterbi 就能直接按状态矩阵做 DP。
    frame_score_maps = [_score_lookup(frame.score_by_key) for frame in raw_frames]
    state_scores: dict[str, float] = {}
    backpointers: list[dict[str, str | None]] = []

    first_backpointer: dict[str, str | None] = {}
    for key_name in _ALL_KEY_NAMES:
        emission = _hmm_emission_score(
            frame_score_maps[0],
            key_name=key_name,
            global_scores=global_scores,
            config=config,
        )
        state_scores[key_name] = emission
        first_backpointer[key_name] = None
    backpointers.append(first_backpointer)

    # Viterbi 主循环：
    # 每一帧都在“保持当前 key”与“切到其他 key”之间找全局最优，
    # 通过统一的 key_change_penalty 惩罚过于频繁的换调。
    for frame_index in range(1, len(raw_frames)):
        next_scores: dict[str, float] = {}
        frame_backpointer: dict[str, str | None] = {}
        for key_name in _ALL_KEY_NAMES:
            emission = _hmm_emission_score(
                frame_score_maps[frame_index],
                key_name=key_name,
                global_scores=global_scores,
                config=config,
            )
            best_prev_key: str | None = None
            best_prev_score = float("-inf")
            for prev_key, prev_score in state_scores.items():
                transition_penalty = 0.0 if prev_key == key_name else float(config.key_change_penalty)
                candidate_score = float(prev_score) + float(emission) - transition_penalty
                if candidate_score > best_prev_score:
                    best_prev_score = candidate_score
                    best_prev_key = prev_key
            next_scores[key_name] = best_prev_score
            frame_backpointer[key_name] = best_prev_key
        state_scores = next_scores
        backpointers.append(frame_backpointer)

    # 从最后一帧分数最高的状态开始回溯，恢复整条最优调性路径。
    final_key = max(_ALL_KEY_NAMES, key=lambda key_name: float(state_scores[key_name]))
    decoded = [final_key]
    for frame_index in range(len(raw_frames) - 1, 0, -1):
        prev_key = backpointers[frame_index][decoded[-1]]
        decoded.append(final_key if prev_key is None else prev_key)
    decoded.reverse()
    return tuple(decoded)


def _smooth_frames(
    raw_frames: Sequence[_RawFrame],
    config: KeyAnalysisConfig,
    *,
    global_scores: dict[str, float],
) -> tuple[KeyFrame, ...]:
    if not raw_frames:
        return tuple()

    # 先用 HMM 给出离散的全局最优 key 路径，再单独计算一个更平滑的支持度，
    # 这样一方面路径稳定，另一方面调试时仍能看到“这一帧为什么偏向这个 key”。
    hmm_path = _decode_hmm_key_path(raw_frames, global_scores=global_scores, config=config)
    smoothed_frames: list[KeyFrame] = []
    radius = max(0, int(config.neighborhood_radius_frames))
    positions_per_bar = max(1, config.positions_per_bar)

    for frame_index, raw_frame in enumerate(raw_frames):
        support_by_key: dict[str, float] = defaultdict(float)
        for neighbor_index in range(max(0, frame_index - radius), min(len(raw_frames), frame_index + radius + 1)):
            neighbor = raw_frames[neighbor_index]
            distance = abs(neighbor_index - frame_index)
            decay = float(config.neighborhood_decay) ** distance
            # 邻域投票不决定最终路径，只负责产出一个更容易解释的局部支持度。
            for key_name, score in neighbor.score_by_key:
                support_by_key[key_name] += max(0.0, float(score)) * decay

        best_key = hmm_path[frame_index] if frame_index < len(hmm_path) else raw_frame.raw_key
        smoothed_support = max(
            0.0,
            float(support_by_key.get(best_key, raw_frame.best_score))
            + (float(config.global_key_bias) * max(0.0, float(global_scores.get(best_key, 0.0)))),
        )

        start_bar, start_pos = _unit_to_bar_pos(raw_frame.start_unit, positions_per_bar)
        end_bar, end_pos = _unit_to_bar_pos(raw_frame.end_unit, positions_per_bar)
        smoothed_frames.append(
            KeyFrame(
                start_bar=start_bar,
                start_pos=start_pos,
                end_bar=end_bar,
                end_pos=end_pos,
                best_key=best_key,
                best_score=float(raw_frame.best_score),
                margin_to_second=float(raw_frame.margin_to_second),
                is_uncertain=bool(raw_frame.is_uncertain),
                raw_key=raw_frame.raw_key,
                smoothed_support=float(smoothed_support),
            )
        )
    return tuple(smoothed_frames)


def _frame_label(frame: KeyFrame) -> str:
    return _UNCERTAIN_KEY if frame.is_uncertain else str(frame.best_key)


def _segment_mean_score(frames: Sequence[KeyFrame], key_name: str) -> float:
    matching_scores = [float(frame.best_score) for frame in frames if not frame.is_uncertain and frame.best_key == key_name]
    if not matching_scores:
        return 0.0
    return float(sum(matching_scores) / float(len(matching_scores)))


def _build_segments(
    frames: Sequence[KeyFrame],
    *,
    total_units: int,
    config: KeyAnalysisConfig,
) -> tuple[KeySegment, ...]:
    if not frames:
        return tuple()

    labels = [_frame_label(frame) for frame in frames]
    first_stable_index = next((index for index, label in enumerate(labels) if label != _UNCERTAIN_KEY), None)
    if first_stable_index is None:
        return tuple()

    positions_per_bar = max(1, config.positions_per_bar)
    confirmation_frames = max(1, int(config.modulation_confirmation_frames))
    segments: list[tuple[str, int, int]] = []

    current_key = labels[first_stable_index]
    segment_start = first_stable_index
    index = first_stable_index + 1

    # HMM 路径已经抑制了大部分抖动，这里再做一层“发布级”的保守确认：
    # 只有新 key 连续持续足够多帧，才真正切段并发布 modulation。
    while index < len(frames):
        label = labels[index]
        if label == _UNCERTAIN_KEY or label == current_key:
            index += 1
            continue

        run_end = index
        while run_end < len(frames) and labels[run_end] == label:
            run_end += 1

        if (run_end - index) >= confirmation_frames:
            segments.append((current_key, segment_start, index))
            current_key = label
            segment_start = index
            index = run_end
            continue

        index = run_end

    segments.append((current_key, segment_start, len(frames)))

    published_segments: list[KeySegment] = []
    for key_name, start_index, end_index in segments:
        start_frame = frames[start_index]
        # 末段直接截到整曲结尾，中间段截到下一段起点，保持与上层“在新段起点插 token”的使用方式一致。
        end_unit = total_units if end_index >= len(frames) else (
            (frames[end_index].start_bar * positions_per_bar) + frames[end_index].start_pos
        )
        end_bar, end_pos = _unit_to_bar_pos(end_unit, positions_per_bar)
        published_segments.append(
            KeySegment(
                key=key_name,
                start_bar=int(start_frame.start_bar),
                start_pos=int(start_frame.start_pos),
                end_bar=int(end_bar),
                end_pos=int(end_pos),
                mean_score=_segment_mean_score(frames[start_index:end_index], key_name),
            )
        )
    return tuple(published_segments)


def _modulation_support(
    frames: Sequence[KeyFrame],
    segment: KeySegment,
    config: KeyAnalysisConfig,
) -> float:
    # support 取新段开头若干帧的平均支持度，强调“切换刚发生时是不是站得住”，
    # 而不是被整段后半截的稳定性冲淡。
    support_values = [
        float(frame.smoothed_support)
        for frame in frames
        if (
            frame.start_bar > segment.start_bar
            or (frame.start_bar == segment.start_bar and frame.start_pos >= segment.start_pos)
        )
        and not frame.is_uncertain
        and frame.best_key == segment.key
    ]
    if not support_values:
        return 0.0
    take_count = min(len(support_values), max(1, int(config.modulation_confirmation_frames)))
    return float(sum(support_values[:take_count]) / float(take_count))


def _build_modulation_points(
    frames: Sequence[KeyFrame],
    segments: Sequence[KeySegment],
    config: KeyAnalysisConfig,
) -> tuple[ModulationPoint, ...]:
    if len(segments) < 2:
        return tuple()

    points: list[ModulationPoint] = []
    for left_segment, right_segment in zip(segments[:-1], segments[1:], strict=True):
        if left_segment.key == right_segment.key:
            continue
        # 转调点直接发布为“新稳定段的起点”，
        # 这是当前脚本层最稳的插入位置，不追求音乐学上精确到瞬时的真实转调拍点。
        points.append(
            ModulationPoint(
                bar_index=int(right_segment.start_bar),
                pos_in_bar=int(right_segment.start_pos),
                from_key=left_segment.key,
                to_key=right_segment.key,
                support=_modulation_support(frames, right_segment, config),
            )
        )
    return tuple(points)


def analyze_key_timeline(
    tokens: Sequence[str],
    config: KeyAnalysisConfig | None = None,
) -> KeyTimelineAnalysis:
    """分析单条 token 序列的稳定调性时间线与稀疏转调点。"""
    config = KeyAnalysisConfig() if config is None else config
    parsed = _parse_token_events(tokens, config)
    raw_frames = _build_raw_frames(parsed, config)
    global_scores = _score_lookup(_global_ranked_scores(parsed, config))
    frames = _smooth_frames(raw_frames, config, global_scores=global_scores)
    segments = _build_segments(frames, total_units=parsed.total_units, config=config)
    modulation_points = _build_modulation_points(frames, segments, config)
    initial_key = segments[0].key if segments else _UNCERTAIN_KEY
    return KeyTimelineAnalysis(
        frames=frames,
        segments=segments,
        modulation_points=modulation_points,
        initial_key=initial_key,
    )
