"""MIDI 标注检查工具的数据整理与页面生成辅助模块。"""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import mido

from src.music_analysis import (
    BoundaryFeature,
    HierarchicalBoundaryScore,
    KeyAnalysisConfig,
    KeyFrame,
    KeySegment,
    KeyTimelineAnalysis,
    ModulationPoint,
    NoteInfo,
    PhraseAnalysis,
    PhraseAnalysisConfig,
    PhraseBoundary,
    PhraseSpan,
    analyze_key_timeline,
    analyze_phrase_candidates,
)
from src.tokenizer import TokenizerConfig, tokenize_midi
from src.tokenizer.common import load_jsonl
from src.utils.config_io import load_json_file


@dataclass(frozen=True)
class ReviewBuildConfig:
    """检查页面构建所需的统一配置。"""

    tokenizer_config: TokenizerConfig
    key_config: KeyAnalysisConfig
    phrase_config: PhraseAnalysisConfig
    low_margin_threshold: float


def _parse_prefixed_int(token: str, prefix: str) -> int | None:
    """解析带前缀的整数 token。"""
    if not str(token).startswith(prefix):
        return None
    try:
        return int(str(token)[len(prefix) :])
    except ValueError:
        return None


def _unit_to_bar_pos(unit: int, positions_per_bar: int) -> tuple[int, int]:
    """把线性位置换算成小节索引与小节内位置。"""
    safe_positions = max(1, int(positions_per_bar))
    clamped = max(0, int(unit))
    return (clamped // safe_positions, clamped % safe_positions)


def normalize_review_tokens(tokens: Sequence[str]) -> list[str]:
    """归一化 review 工具使用的 token 序列。"""
    values = [str(token).strip() for token in tokens if str(token).strip()]
    normalized: list[str] = []
    saw_bos = False
    saw_eos = False

    for token in values:
        if token in {"FIM_HOLE", "FIM_MID"} or token.startswith("TASK_"):
            continue
        if token == "BOS":
            if not saw_bos:
                normalized.append(token)
                saw_bos = True
            continue
        if token == "EOS":
            saw_eos = True
            normalized.append(token)
            break
        normalized.append(token)

    if not saw_bos:
        normalized.insert(0, "BOS")
    if not saw_eos:
        normalized.append("EOS")
    return normalized


def tokens_to_note_payloads(tokens: Sequence[str], positions_per_bar: int) -> list[dict[str, Any]]:
    """把 token 序列解析成前端可画钢琴卷帘的音符列表。"""
    normalized = normalize_review_tokens(tokens)
    if not normalized or normalized[0] != "BOS":
        return []

    notes: list[dict[str, Any]] = []
    effective_end = len(normalized) - 1 if normalized[-1] == "EOS" else len(normalized)
    idx = 1
    current_bar = -1

    if idx < effective_end and normalized[idx].startswith("TEMPO_"):
        idx += 1
    if idx < effective_end and normalized[idx].startswith("KEY_"):
        idx += 1

    while idx < effective_end:
        token = normalized[idx]
        if token == "BAR":
            current_bar += 1
            idx += 1
            if idx < effective_end and normalized[idx].startswith("TEMPO_"):
                idx += 1
            if idx < effective_end and normalized[idx].startswith("KEY_"):
                idx += 1
            continue
        if token == "PHRASE":
            idx += 1
            continue
        if token.startswith("POS_") and current_bar >= 0 and idx + 4 < effective_end:
            pos_value = _parse_prefixed_int(token, "POS_")
            inst_token = normalized[idx + 1]
            pitch_value = _parse_prefixed_int(normalized[idx + 2], "PITCH_")
            dur_value = _parse_prefixed_int(normalized[idx + 3], "DUR_")
            vel_value = _parse_prefixed_int(normalized[idx + 4], "VEL_")
            if (
                pos_value is None
                or pitch_value is None
                or dur_value is None
                or vel_value is None
                or not inst_token.startswith("INST_")
            ):
                idx += 1
                continue
            safe_pos = min(max(0, int(pos_value)), max(0, int(positions_per_bar) - 1))
            safe_dur = max(1, int(dur_value))
            start_unit = (current_bar * int(positions_per_bar)) + safe_pos
            end_unit = start_unit + safe_dur
            end_bar, end_pos = _unit_to_bar_pos(end_unit, int(positions_per_bar))
            notes.append(
                {
                    "start_bar": int(current_bar),
                    "start_pos": int(safe_pos),
                    "end_bar": int(end_bar),
                    "end_pos": int(end_pos),
                    "start_unit": int(start_unit),
                    "end_unit": int(end_unit),
                    "pitch": int(pitch_value),
                    "velocity_bin": int(vel_value),
                    "inst": str(inst_token),
                }
            )
            idx += 5
            continue
        idx += 1
    return notes


def _serialize_key_frame(frame: KeyFrame, positions_per_bar: int) -> dict[str, Any]:
    """序列化单个调性分析帧。"""
    start_unit = (int(frame.start_bar) * int(positions_per_bar)) + int(frame.start_pos)
    end_unit = (int(frame.end_bar) * int(positions_per_bar)) + int(frame.end_pos)
    return {
        "start_bar": int(frame.start_bar),
        "start_pos": int(frame.start_pos),
        "end_bar": int(frame.end_bar),
        "end_pos": int(frame.end_pos),
        "start_unit": int(start_unit),
        "end_unit": int(end_unit),
        "best_key": str(frame.best_key),
        "best_score": float(frame.best_score),
        "margin_to_second": float(frame.margin_to_second),
        "is_uncertain": bool(frame.is_uncertain),
        "raw_key": str(frame.raw_key),
        "smoothed_support": float(frame.smoothed_support),
    }


def _serialize_key_segment(segment: KeySegment, positions_per_bar: int) -> dict[str, Any]:
    """序列化稳定调性段。"""
    start_unit = (int(segment.start_bar) * int(positions_per_bar)) + int(segment.start_pos)
    end_unit = (int(segment.end_bar) * int(positions_per_bar)) + int(segment.end_pos)
    return {
        "key": str(segment.key),
        "start_bar": int(segment.start_bar),
        "start_pos": int(segment.start_pos),
        "end_bar": int(segment.end_bar),
        "end_pos": int(segment.end_pos),
        "start_unit": int(start_unit),
        "end_unit": int(end_unit),
        "length_bars": float((end_unit - start_unit) / float(max(1, positions_per_bar))),
        "mean_score": float(segment.mean_score),
    }


def _serialize_modulation_point(point: ModulationPoint, positions_per_bar: int) -> dict[str, Any]:
    """序列化转调点。"""
    unit = (int(point.bar_index) * int(positions_per_bar)) + int(point.pos_in_bar)
    return {
        "bar_index": int(point.bar_index),
        "pos_in_bar": int(point.pos_in_bar),
        "unit": int(unit),
        "from_key": str(point.from_key),
        "to_key": str(point.to_key),
        "support": float(point.support),
    }


def _summarize_dominant_key(segments: Sequence[dict[str, Any]]) -> tuple[str, float]:
    """根据稳定调性段估计主导调性与覆盖率。"""
    totals: dict[str, float] = {}
    covered = 0.0
    for segment in segments:
        length_bars = float(segment.get("length_bars", 0.0))
        if length_bars <= 0.0:
            continue
        key_name = str(segment.get("key", "uncertain"))
        totals[key_name] = float(totals.get(key_name, 0.0)) + length_bars
        covered += length_bars
    if not totals or covered <= 0.0:
        return ("uncertain", 0.0)
    best_key = min(totals, key=lambda key_name: (-float(totals[key_name]), str(key_name)))
    return (best_key, float(totals[best_key] / covered))


def serialize_key_analysis(analysis: KeyTimelineAnalysis, positions_per_bar: int) -> dict[str, Any]:
    """把调性分析结果转换成页面使用的数据格式。"""
    frames = [_serialize_key_frame(frame, positions_per_bar) for frame in analysis.frames]
    segments = [_serialize_key_segment(segment, positions_per_bar) for segment in analysis.segments]
    modulation_points = [
        _serialize_modulation_point(point, positions_per_bar)
        for point in analysis.modulation_points
    ]
    dominant_key, dominant_coverage = _summarize_dominant_key(segments)
    timeline_summary = " -> ".join(segment["key"] for segment in segments) if segments else "uncertain"
    return {
        "initial_key": str(analysis.initial_key),
        "dominant_key": str(dominant_key),
        "dominant_key_coverage": float(dominant_coverage),
        "timeline_summary": str(timeline_summary),
        "frames": frames,
        "segments": segments,
        "modulation_points": modulation_points,
    }


def _serialize_phrase_boundary(boundary: PhraseBoundary, positions_per_bar: int) -> dict[str, Any]:
    """序列化乐句边界。"""
    return {
        "bar_index": int(boundary.bar_index),
        "anchor_pos": int(boundary.anchor_pos),
        "unit": int((int(boundary.bar_index) * int(positions_per_bar)) + int(boundary.anchor_pos)),
        "anchor_kind": "mid_bar" if int(boundary.anchor_pos) > 0 else "bar_aligned",
    }


def _serialize_phrase_span(span: PhraseSpan, positions_per_bar: int) -> dict[str, Any]:
    """序列化乐句区间。"""
    start_unit = int(span.start_bar) * int(positions_per_bar)
    end_unit = int(span.end_bar) * int(positions_per_bar)
    return {
        "start_bar": int(span.start_bar),
        "end_bar": int(span.end_bar),
        "start_token": int(span.start_token),
        "end_token": int(span.end_token),
        "start_unit": int(start_unit),
        "end_unit": int(end_unit),
        "length_bars": float(max(0, int(span.end_bar) - int(span.start_bar))),
        "tempo_token": span.tempo_token,
        "key_token": span.key_token,
        "source_kind": str(span.source_kind),
    }


def _serialize_phrase_note(note: NoteInfo, positions_per_bar: int) -> dict[str, Any]:
    """序列化乐句分析使用的 note-level 音符信息。"""
    return {
        "note_index": int(note.note_index),
        "start_unit": int(note.start_unit),
        "end_unit": int(note.end_unit),
        "duration": int(note.duration),
        "pitch": int(note.pitch),
        "bar_index": int(note.bar_index),
        "pos_in_bar": int(note.pos_in_bar),
        "start_bar": int(note.start_unit // int(positions_per_bar)),
        "start_pos": int(note.start_unit % int(positions_per_bar)),
        "end_bar": int(note.end_unit // int(positions_per_bar)),
        "end_pos": int(note.end_unit % int(positions_per_bar)),
        "effective_key_token": note.effective_key_token,
    }


def _serialize_boundary_feature(feature: BoundaryFeature) -> dict[str, Any]:
    """序列化相邻音符之间的 note-level 边界特征。"""
    return {
        "note_index": int(feature.note_index),
        "left_end_unit": int(feature.left_end_unit),
        "right_start_unit": int(feature.right_start_unit),
        "bar_index": int(feature.bar_index),
        "anchor_pos": int(feature.anchor_pos),
        "gap": int(feature.gap),
        "local_gap_mean": float(feature.local_gap_mean),
        "local_duration_mean": float(feature.local_duration_mean),
        "gap_break_score": float(feature.gap_break_score),
        "duration_release_score": float(feature.duration_release_score),
        "cadence_score": float(feature.cadence_score),
        "motive_end_score": float(feature.motive_end_score),
        "repeat_start_score": float(feature.repeat_start_score),
        "repeat_end_score": float(feature.repeat_end_score),
        "sequence_stop_score": float(feature.sequence_stop_score),
        "continuity_penalty": float(feature.continuity_penalty),
        "bar_hint_score": float(feature.bar_hint_score),
        "sequence_role": str(feature.sequence_role),
        "reasons": list(feature.reasons),
    }


def _boundary_display_score(score: HierarchicalBoundaryScore) -> float:
    """为向后兼容的 review 展示提供单一 score 入口。"""
    if str(score.boundary_type) == "phrase":
        return float(score.phrase_score)
    if str(score.boundary_type) == "subphrase":
        return float(score.subphrase_score)
    if str(score.boundary_type) == "motif":
        return float(score.motif_score)
    return float(max(score.motif_score, score.subphrase_score, score.phrase_score))


def _serialize_boundary_score(score: HierarchicalBoundaryScore) -> dict[str, Any]:
    """序列化 note-level 三层边界评分，并保留向后兼容字段。"""
    return {
        "note_index": int(score.note_index),
        "unit": int(score.unit),
        "bar_index": int(score.bar_index),
        "anchor_pos": int(score.anchor_pos),
        "motif_score": float(score.motif_score),
        "subphrase_score": float(score.subphrase_score),
        "phrase_score": float(score.phrase_score),
        "boundary_type": str(score.boundary_type),
        "sequence_role": str(score.sequence_role),
        "reasons": list(score.reasons),
        "reason_labels": _phrase_reason_labels(score.reasons, anchor_pos=int(score.anchor_pos)),
        "score": float(_boundary_display_score(score)),
    }


def _prefer_boundary_score_row(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    """同一锚点存在多条评分时，优先保留真正命中的那一条。"""

    if current is None:
        return dict(candidate)

    current_type = str(current.get("boundary_type", "none"))
    candidate_type = str(candidate.get("boundary_type", "none"))
    current_rank = 0 if current_type == "none" else 1
    candidate_rank = 0 if candidate_type == "none" else 1
    if candidate_rank != current_rank:
        return dict(candidate) if candidate_rank > current_rank else dict(current)

    current_score = float(current.get("score", 0.0))
    candidate_score = float(candidate.get("score", 0.0))
    if candidate_score != current_score:
        return dict(candidate) if candidate_score > current_score else dict(current)

    current_reason_count = len(list(current.get("reasons", [])))
    candidate_reason_count = len(list(candidate.get("reasons", [])))
    if candidate_reason_count != current_reason_count:
        return dict(candidate) if candidate_reason_count > current_reason_count else dict(current)
    return dict(current)


def serialize_phrase_analysis(analysis: PhraseAnalysis, positions_per_bar: int) -> dict[str, Any]:
    """把乐句分析结果转换成页面使用的数据格式。"""
    bars = []
    for bar_index, bar in enumerate(analysis.bars):
        bars.append(
            {
                "bar_index": int(bar_index),
                "start_token": int(bar.start_token),
                "end_token": int(bar.end_token),
                "start_unit": int(bar_index * positions_per_bar),
                "end_unit": int((bar_index + 1) * positions_per_bar),
                "note_count": int(bar.note_count),
                "onset_count": int(bar.onset_count),
                "rest_ratio": float(bar.rest_ratio),
                "pitch_span": int(bar.pitch_span),
                "mean_duration": float(bar.mean_duration),
                "effective_tempo_token": bar.effective_tempo_token,
                "effective_key_token": bar.effective_key_token,
                "onset_positions": list(bar.onset_positions),
            }
        )
    notes = [_serialize_phrase_note(note, positions_per_bar) for note in analysis.notes]
    boundary_features = [
        _serialize_boundary_feature(feature)
        for feature in analysis.boundary_features
    ]
    boundary_scores = [
        _serialize_boundary_score(score)
        for score in analysis.boundary_scores
    ]
    boundary_scores_by_anchor: dict[tuple[int, int], dict[str, Any]] = {}
    for score in boundary_scores:
        anchor_key = (int(score["bar_index"]), int(score["anchor_pos"]))
        boundary_scores_by_anchor[anchor_key] = _prefer_boundary_score_row(
            boundary_scores_by_anchor.get(anchor_key),
            score,
        )
    first_content_bar = next(
        (int(index) for index, bar in enumerate(analysis.bars) if int(bar.note_count) > 0),
        None,
    )
    boundaries = []
    for boundary in analysis.boundaries:
        serialized_boundary = _serialize_phrase_boundary(boundary, positions_per_bar)
        serialized_boundary.update(
            _describe_phrase_boundary(
                serialized_boundary,
                boundary_score=boundary_scores_by_anchor.get(
                    (int(boundary.bar_index), int(boundary.anchor_pos))
                ),
                first_content_bar=first_content_bar,
            )
        )
        boundaries.append(serialized_boundary)
    phrase_spans = [
        _serialize_phrase_span(span, positions_per_bar)
        for span in analysis.phrase_spans
    ]
    mean_phrase_bars = (
        float(sum(float(span["length_bars"]) for span in phrase_spans) / len(phrase_spans))
        if phrase_spans
        else 0.0
    )
    return {
        "bars": bars,
        "notes": notes,
        "boundary_features": boundary_features,
        "boundary_scores": boundary_scores,
        "boundaries": boundaries,
        "phrase_spans": phrase_spans,
        "mean_phrase_bars": float(mean_phrase_bars),
    }


def _candidate_boundary_bars(boundary_scores: Sequence[dict[str, Any]]) -> set[int]:
    """复原启发式候选边界条目，便于 review 时标注来源规则。"""
    if not boundary_scores:
        return set()
    ordered_scores = [dict(item) for item in boundary_scores]
    threshold = _quantile_threshold(
        [float(item.get("score", 0.0)) for item in ordered_scores],
        0.75,
    )
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(ordered_scores):
        score = float(item.get("score", 0.0))
        prev_score = (
            float(ordered_scores[index - 1].get("score", float("-inf")))
            if index > 0
            else float("-inf")
        )
        next_score = (
            float(ordered_scores[index + 1].get("score", float("-inf")))
            if index + 1 < len(ordered_scores)
            else float("-inf")
        )
        if score >= threshold and score >= prev_score and score >= next_score:
            candidates.append(item)

    filtered: list[int] = []
    for item in candidates:
        bar_index = int(item.get("bar_index", -1))
        if bar_index < 0:
            continue
        if not filtered or (bar_index - filtered[-1]) >= 2:
            filtered.append(bar_index)
            continue
        current_score = float(item.get("score", 0.0))
        previous_score = float(
            next(
                (
                    candidate.get("score", 0.0)
                    for candidate in ordered_scores
                    if int(candidate.get("bar_index", -1)) == filtered[-1]
                ),
                0.0,
            )
        )
        if current_score > previous_score:
            filtered[-1] = bar_index
    return set(filtered)


def _quantile_threshold(values: Sequence[float], ratio: float) -> float:
    """计算 review 侧复原候选边界时使用的分位数阈值。"""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return float(ordered[index])


def _phrase_reason_labels(reasons: Sequence[str], *, anchor_pos: int) -> list[str]:
    """把内部原因代码映射成面向 review 的中文标签。"""
    label_map = {
        "adjacent_repeated_bar_span": "长跨度重复",
        "gap_break": "大空挡",
        "duration_release": "长时值收束",
        "cadence": "终止感",
        "sequence_stop": "模进停止",
        "sequence_inside": "模进内部",
        "motive_end": "动机收束",
        "repeat_start": "重复起点",
        "repeat_end": "重复结束",
        "bar_hint": "跨小节提示",
        "continuity_penalty": "连续性延续",
        "rest_gap": "休止/空拍",
        "note_density_delta": "音符密度变化",
        "onset_density_delta": "起音密度变化",
        "pitch_span_delta": "音域跨度变化",
        "duration_delta": "时值变化",
    }
    priority = (
        "adjacent_repeated_bar_span",
        "gap_break",
        "duration_release",
        "cadence",
        "sequence_stop",
        "repeat_end",
        "motive_end",
        "repeat_start",
        "bar_hint",
        "rest_gap",
        "note_density_delta",
        "onset_density_delta",
        "pitch_span_delta",
        "duration_delta",
        "sequence_inside",
        "continuity_penalty",
    )
    normalized = [str(reason) for reason in reasons if str(reason)]
    ordered_codes: list[str] = []
    for reason_code in priority:
        if reason_code in normalized and reason_code not in ordered_codes:
            ordered_codes.append(reason_code)
    for reason_code in normalized:
        if reason_code not in ordered_codes:
            ordered_codes.append(reason_code)

    labels: list[str] = []
    for reason in ordered_codes:
        label = label_map.get(str(reason), str(reason))
        if label not in labels:
            labels.append(label)
    if int(anchor_pos) > 0 and "前置留白锚点" not in labels:
        labels.append("前置留白锚点")
    return labels


def _select_primary_phrase_reason(reasons: Sequence[str]) -> str | None:
    """从真实原因列表中挑出最适合作为主规则名的原因代码。"""
    priority = (
        "adjacent_repeated_bar_span",
        "gap_break",
        "duration_release",
        "cadence",
        "sequence_stop",
        "repeat_end",
        "motive_end",
        "repeat_start",
        "bar_hint",
        "rest_gap",
        "note_density_delta",
        "onset_density_delta",
        "pitch_span_delta",
        "duration_delta",
        "sequence_inside",
        "continuity_penalty",
    )
    normalized = [str(reason) for reason in reasons if str(reason)]
    if not normalized:
        return None
    for reason_code in priority:
        if reason_code in normalized:
            return str(reason_code)
    return normalized[0]


def _compact_boundary_label(reason_labels: Sequence[str], *, fallback: str) -> str:
    """生成显示在线下方的短标签，避免文字过长。"""
    if not reason_labels:
        return str(fallback)
    if len(reason_labels) == 1:
        return str(reason_labels[0])
    return f"{reason_labels[0]}+"


def _describe_phrase_boundary(
    boundary: dict[str, Any],
    *,
    boundary_score: dict[str, Any] | None,
    first_content_bar: int | None,
) -> dict[str, Any]:
    """给乐句边界补充真实来源规则、命中原因和短标签。"""
    bar_index = int(boundary.get("bar_index", -1))
    anchor_pos = int(boundary.get("anchor_pos", 0))
    if first_content_bar is not None and bar_index == int(first_content_bar):
        rule_name = "首句强制"
        reason_labels = ["首句强制"]
        short_label = "首句强制"
    else:
        raw_reasons = list(boundary_score.get("reasons", [])) if boundary_score is not None else []
        reason_labels = _phrase_reason_labels(raw_reasons, anchor_pos=anchor_pos)
        primary_reason = _select_primary_phrase_reason(raw_reasons)
        fallback_rule = str(boundary_score.get("boundary_type", "边界命中")) if boundary_score is not None else "边界命中"
        rule_name = _phrase_reason_labels([str(primary_reason)], anchor_pos=0)[0] if primary_reason is not None else fallback_rule
        short_label = _compact_boundary_label(reason_labels, fallback=rule_name)
    return {
        "source_rule": str(rule_name),
        "source_label": str(short_label),
        "source_reasons": list(reason_labels),
        "score": None if boundary_score is None else float(boundary_score.get("score", 0.0)),
    }


def build_debug_flags(
    *,
    key_analysis: dict[str, Any],
    phrase_analysis: dict[str, Any],
    low_margin_threshold: float,
    min_phrase_bars: int,
    max_phrase_bars: int,
) -> dict[str, Any]:
    """根据固定规则生成可疑样本标记。"""
    uncertain_frames = [
        index
        for index, frame in enumerate(key_analysis.get("frames", []))
        if bool(frame.get("is_uncertain"))
    ]
    low_margin_frames = [
        index
        for index, frame in enumerate(key_analysis.get("frames", []))
        if float(frame.get("margin_to_second", 0.0)) < float(low_margin_threshold)
    ]
    short_key_segments = [
        {
            "index": index,
            "key": str(segment.get("key", "uncertain")),
            "length_bars": float(segment.get("length_bars", 0.0)),
        }
        for index, segment in enumerate(key_analysis.get("segments", []))
        if float(segment.get("length_bars", 0.0)) < 2.0
    ]
    boundaries = list(phrase_analysis.get("boundaries", []))
    dense_phrase_boundaries = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        bar_gap = int(right.get("bar_index", 0)) - int(left.get("bar_index", 0))
        if bar_gap < int(min_phrase_bars):
            dense_phrase_boundaries.append(
                {
                    "left_bar": int(left.get("bar_index", 0)),
                    "right_bar": int(right.get("bar_index", 0)),
                    "bar_gap": int(bar_gap),
                }
            )
    long_phrase_spans = [
        {
            "index": index,
            "start_bar": int(span.get("start_bar", 0)),
            "end_bar": int(span.get("end_bar", 0)),
            "length_bars": float(span.get("length_bars", 0.0)),
        }
        for index, span in enumerate(phrase_analysis.get("phrase_spans", []))
        if float(span.get("length_bars", 0.0)) > float(max_phrase_bars)
    ]
    empty_anchor_boundaries = []
    bars_by_index = {
        int(bar.get("bar_index", 0)): dict(bar) for bar in phrase_analysis.get("bars", [])
    }
    for boundary in boundaries:
        bar_info = bars_by_index.get(int(boundary.get("bar_index", -1)), {})
        if int(boundary.get("anchor_pos", 0)) > 0 and float(bar_info.get("rest_ratio", 0.0)) < 0.20:
            empty_anchor_boundaries.append(
                {
                    "bar_index": int(boundary.get("bar_index", 0)),
                    "anchor_pos": int(boundary.get("anchor_pos", 0)),
                    "rest_ratio": float(bar_info.get("rest_ratio", 0.0)),
                }
            )

    flag_names: list[str] = []
    if uncertain_frames:
        flag_names.append("存在 uncertain 调性帧")
    if low_margin_frames:
        flag_names.append("存在低置信调性帧")
    if short_key_segments:
        flag_names.append("存在短调性段")
    if dense_phrase_boundaries:
        flag_names.append("存在密集乐句边界")
    if long_phrase_spans:
        flag_names.append("存在超长乐句")
    if empty_anchor_boundaries:
        flag_names.append("存在可疑 mid-bar 乐句边界")

    return {
        "is_suspicious": bool(flag_names),
        "flag_names": flag_names,
        "uncertain_frame_indices": uncertain_frames,
        "low_margin_frame_indices": low_margin_frames,
        "short_key_segments": short_key_segments,
        "dense_phrase_boundaries": dense_phrase_boundaries,
        "long_phrase_spans": long_phrase_spans,
        "empty_anchor_boundaries": empty_anchor_boundaries,
    }


def build_review_case(
    *,
    case_id: str,
    source_kind: str,
    title: str,
    subtitle: str,
    source_path: str,
    meta: dict[str, Any],
    tokens: Sequence[str],
    config: ReviewBuildConfig,
) -> dict[str, Any]:
    """从 token 序列构建单条 review case。"""
    normalized_tokens = normalize_review_tokens(tokens)
    positions_per_bar = int(config.tokenizer_config.positions_per_bar)
    notes = tokens_to_note_payloads(normalized_tokens, positions_per_bar)
    key_result = analyze_key_timeline(normalized_tokens, config=config.key_config)
    phrase_result = analyze_phrase_candidates(normalized_tokens, config=config.phrase_config)
    serialized_key = serialize_key_analysis(key_result, positions_per_bar)
    serialized_phrase = serialize_phrase_analysis(phrase_result, positions_per_bar)
    debug_flags = build_debug_flags(
        key_analysis=serialized_key,
        phrase_analysis=serialized_phrase,
        low_margin_threshold=float(config.low_margin_threshold),
        min_phrase_bars=int(config.phrase_config.min_phrase_bars),
        max_phrase_bars=int(config.phrase_config.max_phrase_bars),
    )
    return {
        "case_id": str(case_id),
        "source_kind": str(source_kind),
        "title": str(title),
        "subtitle": str(subtitle),
        "source_path": str(source_path),
        "meta": dict(meta),
        "tokens": normalized_tokens,
        "notes": notes,
        "bars": list(serialized_phrase.get("bars", [])),
        "key_analysis": serialized_key,
        "phrase_analysis": serialized_phrase,
        "debug_flags": debug_flags,
    }


def _discover_midi_files(midi_root: Path) -> list[Path]:
    """递归发现目录中的 MIDI 文件。"""
    midi_files = [
        path.resolve()
        for pattern in ("*.mid", "*.midi")
        for path in midi_root.rglob(pattern)
        if path.is_file()
    ]
    deduped = {str(path).lower(): path for path in midi_files}
    return sorted(deduped.values(), key=lambda path: str(path).lower())


def _resolve_raw_midi_entries(
    *,
    midi_root: Path | None,
    midi_list_jsonl: Path | None,
) -> list[tuple[Path, dict[str, Any]]]:
    """统一整理原始 MIDI 输入列表。"""
    if midi_list_jsonl is not None:
        rows = load_jsonl(midi_list_jsonl)
        if not rows:
            return []
        resolved: list[tuple[Path, dict[str, Any]]] = []
        base_dir = midi_list_jsonl.parent.resolve()
        for row in rows:
            raw_path = row.get("midi_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                if midi_root is not None:
                    path = (midi_root / path).resolve()
                else:
                    path = (base_dir / path).resolve()
            resolved.append((path, dict(row)))
        return resolved
    if midi_root is None:
        return []
    return [(path, {}) for path in _discover_midi_files(midi_root)]


def load_raw_midi_cases(
    *,
    midi_root: Path | None,
    midi_list_jsonl: Path | None,
    config: ReviewBuildConfig,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """从原始 MIDI 文件构建 review case 列表。"""
    entries = _resolve_raw_midi_entries(midi_root=midi_root, midi_list_jsonl=midi_list_jsonl)
    if limit is not None and limit > 0:
        entries = entries[: int(limit)]

    cases: list[dict[str, Any]] = []
    for index, (midi_path, row_meta) in enumerate(entries):
        if not midi_path.exists():
            continue
        midi = mido.MidiFile(str(midi_path))
        tokens = tokenize_midi(midi, config.tokenizer_config)
        title = str(row_meta.get("title") or midi_path.stem)
        artist = str(row_meta.get("artist") or "").strip()
        subtitle = artist if artist else midi_path.name
        source_path = str(midi_path)
        meta = {
            "relative_path": str(midi_path.name if midi_root is None else midi_path.relative_to(midi_root))
            if midi_root is not None and midi_path.is_relative_to(midi_root)
            else midi_path.name,
        }
        meta.update(row_meta)
        cases.append(
            build_review_case(
                case_id=f"raw-{index}",
                source_kind="raw_midi",
                title=title,
                subtitle=subtitle,
                source_path=source_path,
                meta=meta,
                tokens=tokens,
                config=config,
            )
        )
    return cases


def _infer_benchmark_source_kind(task: str) -> str:
    """根据 benchmark task 推导来源类型。"""
    task_value = str(task).strip().lower()
    if task_value == "continuation":
        return "benchmark_continuation"
    if task_value == "infilling":
        return "benchmark_infilling"
    return f"benchmark_{task_value or 'unknown'}"


def _resolve_benchmark_tokens(case_payload: dict[str, Any]) -> tuple[list[str], str]:
    """解析 benchmark case 中最适合 review 的 token 序列。"""
    preferred_fields = (
        "raw_reconstructed_tokens",
        "fsm_reconstructed_tokens",
        "reference_full_tokens",
        "target_tokens",
        "prompt_tokens",
    )
    for field_name in preferred_fields:
        values = case_payload.get(field_name)
        if isinstance(values, list) and values:
            return ([str(token) for token in values], str(field_name))
    return (["BOS", "EOS"], "empty_fallback")


def load_benchmark_cases(
    benchmark_json_path: Path,
    config: ReviewBuildConfig,
) -> list[dict[str, Any]]:
    """从 benchmark sample JSON 构建 review case 列表。"""
    payload = load_json_file(benchmark_json_path, "benchmark sample")
    task = str(payload.get("task", "unknown"))
    source_kind = _infer_benchmark_source_kind(task)
    cases_payload = payload.get("cases", [])
    if not isinstance(cases_payload, list):
        raise ValueError("benchmark sample 缺少 cases 列表")

    cases: list[dict[str, Any]] = []
    for index, case_payload in enumerate(cases_payload):
        if not isinstance(case_payload, dict):
            continue
        meta = dict(case_payload.get("meta", {})) if isinstance(case_payload.get("meta"), dict) else {}
        tokens, token_origin = _resolve_benchmark_tokens(case_payload)
        row_id = case_payload.get("row_id", index)
        title = str(meta.get("title") or f"case_{row_id}")
        artist = str(meta.get("artist") or "").strip()
        bucket = str(case_payload.get("bucket") or "").strip()
        subtitle_parts = [part for part in (artist, bucket) if part]
        subtitle = " / ".join(subtitle_parts) if subtitle_parts else f"row_id={row_id}"
        meta.update(
            {
                "row_id": row_id,
                "bucket": case_payload.get("bucket"),
                "task": task,
                "checkpoint_name": payload.get("checkpoint_name"),
                "sample_group": payload.get("sample_group"),
                "token_origin": token_origin,
            }
        )
        source_path = str(meta.get("midi_path") or benchmark_json_path)
        cases.append(
            build_review_case(
                case_id=f"{task}-{row_id}-{index}",
                source_kind=source_kind,
                title=title,
                subtitle=subtitle,
                source_path=source_path,
                meta=meta,
                tokens=tokens,
                config=config,
            )
        )
    return cases


def build_review_payload(
    *,
    cases: Sequence[dict[str, Any]],
    positions_per_bar: int,
    only_suspicious: bool,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    """构建整体 review 数据文件。"""
    final_cases = [dict(case) for case in cases]
    if only_suspicious:
        final_cases = [case for case in final_cases if bool(case.get("debug_flags", {}).get("is_suspicious"))]
    suspicious_count = sum(
        1 for case in final_cases if bool(case.get("debug_flags", {}).get("is_suspicious"))
    )
    return {
        "meta": {
            "positions_per_bar": int(positions_per_bar),
            "only_suspicious": bool(only_suspicious),
            "case_count": int(len(final_cases)),
            "suspicious_case_count": int(suspicious_count),
            "source_summary": dict(source_summary),
        },
        "cases": final_cases,
    }


def build_review_html(payload: dict[str, Any]) -> str:
    """生成离线 review 页面 HTML。"""
    data_json = json.dumps(payload, ensure_ascii=False)
    embedded_data_json = data_json.replace("</", "<\\/")
    title = html.escape("TuneFlow MIDI 标注检查工具")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --panel-strong: #f0e6d7;
      --line: #d1c2ac;
      --text: #2e261d;
      --muted: #726456;
      --accent: #af6f2e;
      --danger: #ad3d2f;
      --note: #5b6c7c;
      --note-border: #35414d;
      --phrase-bar: #234f6d;
      --phrase-mid: #4f8a6b;
      --uncertain: #8d8d8d;
      --shadow: 0 14px 38px rgba(46, 38, 29, 0.10);
      --radius: 18px;
      --mono: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      --sans: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(175, 111, 46, 0.12), transparent 28%),
        linear-gradient(180deg, #f7f4ed 0%, #f2ede4 100%);
    }}
    .app {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 16px;
      padding: 20px;
    }}
    .toolbar, .footer-panel, .info-panel, .timeline-panel {{
      background: var(--panel);
      border: 1px solid rgba(81, 63, 43, 0.12);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: auto auto 1fr auto auto;
      gap: 12px;
      align-items: center;
      padding: 14px 16px;
    }}
    .toolbar .cluster {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .btn {{
      appearance: none;
      border: none;
      border-radius: 999px;
      background: var(--accent);
      color: #fff8f0;
      padding: 10px 16px;
      font-size: 14px;
      cursor: pointer;
      transition: transform 0.12s ease, opacity 0.12s ease;
    }}
    .btn.secondary {{
      background: #d8c5a7;
      color: var(--text);
    }}
    .btn:disabled {{
      cursor: not-allowed;
      opacity: 0.45;
    }}
    .btn:not(:disabled):hover {{
      transform: translateY(-1px);
    }}
    .toolbar input[type="text"] {{
      width: min(320px, 100%);
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fffaf3;
      color: var(--text);
      font-size: 14px;
    }}
    .toolbar label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    .main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      min-height: 0;
    }}
    .timeline-panel {{
      padding: 16px;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 12px;
      overflow: hidden;
    }}
    .timeline-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    .title-block h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .title-block p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .timeline-wrapper {{
      overflow: auto;
      border-radius: 14px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.86), rgba(247, 242, 232, 0.95));
      border: 1px solid rgba(81, 63, 43, 0.10);
      padding: 10px;
    }}
    .info-panel {{
      padding: 16px;
      overflow: auto;
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(249,244,236,0.95));
      border: 1px solid rgba(81, 63, 43, 0.09);
      border-radius: 14px;
      padding: 14px;
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 8px 10px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .kv .label {{
      color: var(--muted);
    }}
    .pill-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(175, 111, 46, 0.10);
      color: var(--text);
      font-size: 12px;
      border: 1px solid rgba(175, 111, 46, 0.18);
    }}
    .pill.danger {{
      background: rgba(173, 61, 47, 0.10);
      border-color: rgba(173, 61, 47, 0.18);
      color: var(--danger);
    }}
    .footer-panel {{
      padding: 16px;
      display: grid;
      gap: 12px;
    }}
    details {{
      border: 1px solid rgba(81, 63, 43, 0.10);
      border-radius: 14px;
      background: rgba(255, 253, 248, 0.85);
      padding: 10px 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 12px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid rgba(81, 63, 43, 0.08);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
    }}
    code {{
      font-family: var(--mono);
      font-size: 12px;
      color: #6b3d18;
    }}
    .status {{
      font-size: 14px;
      color: var(--muted);
    }}
    .empty {{
      padding: 28px;
      text-align: center;
      color: var(--muted);
    }}
    .tooltip {{
      position: fixed;
      display: none;
      pointer-events: none;
      max-width: 320px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(39, 31, 23, 0.94);
      color: #fff8ef;
      font-size: 12px;
      line-height: 1.5;
      box-shadow: 0 10px 30px rgba(0,0,0,0.22);
      z-index: 1000;
    }}
    @media (max-width: 1200px) {{
      .main {{
        grid-template-columns: 1fr;
      }}
      .info-panel {{
        max-height: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <section class="toolbar">
      <div class="cluster">
        <button id="prevBtn" class="btn secondary" type="button">上一条</button>
        <button id="nextBtn" class="btn" type="button">下一条</button>
      </div>
      <div id="statusText" class="status">准备加载...</div>
      <div class="cluster">
        <input id="searchInput" type="text" placeholder="按 row_id、文件名或标题跳转" />
        <button id="searchBtn" class="btn secondary" type="button">跳转</button>
      </div>
      <label>
        <input id="suspiciousOnlyToggle" type="checkbox" />
        只看可疑样本
      </label>
      <div id="metaSummary" class="status"></div>
    </section>
    <section class="main">
      <section class="timeline-panel">
        <div class="timeline-header">
          <div class="title-block">
            <h1 id="caseTitle">未选择样本</h1>
            <p id="caseSubtitle"></p>
          </div>
          <div id="caseMeta" class="status"></div>
        </div>
        <div class="timeline-wrapper">
          <div id="timelineContainer" class="empty">没有可显示的数据</div>
        </div>
      </section>
      <aside class="info-panel">
        <div class="card">
          <h2>基本信息</h2>
          <div id="basicInfo" class="kv"></div>
        </div>
        <div class="card">
          <h2>调性摘要</h2>
          <div id="keyInfo" class="kv"></div>
        </div>
        <div class="card">
          <h2>乐句摘要</h2>
          <div id="phraseInfo" class="kv"></div>
        </div>
        <div class="card">
          <h2>可疑标记</h2>
          <div id="flagList" class="pill-list"></div>
        </div>
      </aside>
    </section>
    <section class="footer-panel">
      <details open>
        <summary>调性帧表</summary>
        <div id="frameTable"></div>
      </details>
      <details>
        <summary>乐句边界表</summary>
        <div id="boundaryTable"></div>
      </details>
    </section>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script id="review-data" type="application/json">{embedded_data_json}</script>
  <script>
    const REVIEW_DATA = JSON.parse(document.getElementById("review-data").textContent);
    const state = {{
      allCases: Array.isArray(REVIEW_DATA.cases) ? REVIEW_DATA.cases : [],
      filteredIndices: [],
      caseIndex: 0,
      suspiciousOnly: false,
    }};

    const elements = {{
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      statusText: document.getElementById("statusText"),
      metaSummary: document.getElementById("metaSummary"),
      searchInput: document.getElementById("searchInput"),
      searchBtn: document.getElementById("searchBtn"),
      suspiciousOnlyToggle: document.getElementById("suspiciousOnlyToggle"),
      caseTitle: document.getElementById("caseTitle"),
      caseSubtitle: document.getElementById("caseSubtitle"),
      caseMeta: document.getElementById("caseMeta"),
      timelineContainer: document.getElementById("timelineContainer"),
      basicInfo: document.getElementById("basicInfo"),
      keyInfo: document.getElementById("keyInfo"),
      phraseInfo: document.getElementById("phraseInfo"),
      flagList: document.getElementById("flagList"),
      frameTable: document.getElementById("frameTable"),
      boundaryTable: document.getElementById("boundaryTable"),
      tooltip: document.getElementById("tooltip"),
    }};

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function keyColor(keyName) {{
      if (!keyName || keyName === "uncertain") {{
        return "rgba(141, 141, 141, 0.78)";
      }}
      const normalized = String(keyName).replace("KEY_", "").replaceAll("_SHARP", "#");
      const rootOrder = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
      let root = normalized;
      let isMinor = false;
      if (normalized.includes(":")) {{
        const parts = normalized.split(":");
        root = parts[0];
        isMinor = parts[1] === "min";
      }} else if (normalized.endsWith("_MAJ")) {{
        root = normalized.slice(0, -4).replaceAll("_", "");
      }} else if (normalized.endsWith("_MIN")) {{
        root = normalized.slice(0, -4).replaceAll("_", "");
        isMinor = true;
      }}
      const rootIndex = rootOrder.indexOf(root);
      const hue = rootIndex >= 0 ? rootIndex * 30 : 0;
      const lightness = isMinor ? 44 : 58;
      return `hsla(${{hue}}, 62%, ${{lightness}}%, 0.80)`;
    }}

    function supportOpacity(value) {{
      const safe = Math.max(0, Number(value) || 0);
      return Math.min(0.95, Math.max(0.20, safe / 2.5));
    }}

    function buildFilteredIndices() {{
      state.filteredIndices = [];
      state.allCases.forEach((item, index) => {{
        const suspicious = Boolean(item?.debug_flags?.is_suspicious);
        if (!state.suspiciousOnly || suspicious) {{
          state.filteredIndices.push(index);
        }}
      }});
      if (state.caseIndex >= state.filteredIndices.length) {{
        state.caseIndex = Math.max(0, state.filteredIndices.length - 1);
      }}
    }}

    function currentCase() {{
      if (!state.filteredIndices.length) {{
        return null;
      }}
      return state.allCases[state.filteredIndices[state.caseIndex]] ?? null;
    }}

    function moveCase(delta) {{
      if (!state.filteredIndices.length) {{
        return;
      }}
      const nextIndex = Math.min(
        Math.max(0, state.caseIndex + delta),
        state.filteredIndices.length - 1,
      );
      state.caseIndex = nextIndex;
      render();
    }}

    function attachTooltip(node, htmlContent) {{
      node.addEventListener("mouseenter", () => {{
        elements.tooltip.style.display = "block";
        elements.tooltip.innerHTML = htmlContent;
      }});
      node.addEventListener("mousemove", (event) => {{
        elements.tooltip.style.left = `${{event.clientX + 14}}px`;
        elements.tooltip.style.top = `${{event.clientY + 14}}px`;
      }});
      node.addEventListener("mouseleave", () => {{
        elements.tooltip.style.display = "none";
      }});
    }}

    function createKvRows(rows) {{
      return rows
        .map(([label, value]) => `<div class="label">${{escapeHtml(label)}}</div><div>${{value}}</div>`)
        .join("");
    }}

    function renderTimeline(caseData) {{
      const notes = Array.isArray(caseData.notes) ? caseData.notes : [];
      const keyFrames = Array.isArray(caseData.key_analysis?.frames) ? caseData.key_analysis.frames : [];
      const keySegments = Array.isArray(caseData.key_analysis?.segments) ? caseData.key_analysis.segments : [];
      const phraseBoundaries = Array.isArray(caseData.phrase_analysis?.boundaries) ? caseData.phrase_analysis.boundaries : [];
      const bars = Array.isArray(caseData.bars) ? caseData.bars : [];
      const barCount = Math.max(1, bars.length || Math.max(...notes.map((item) => Number(item.end_bar) || 0), 0));
      const positionsPerBar = Number(REVIEW_DATA?.meta?.positions_per_bar || 32);
      const totalUnits = Math.max(
        barCount * positionsPerBar,
        ...notes.map((item) => Number(item.end_unit) || 0),
        ...keySegments.map((item) => Number(item.end_unit) || 0),
        ...keyFrames.map((item) => Number(item.end_unit) || 0),
      );
      const width = Math.max(1100, totalUnits * 8);
      const height = 560;
      const paddingLeft = 60;
      const paddingRight = 24;
      const usableWidth = width - paddingLeft - paddingRight;
      const frameBandTop = 10;
      const frameBandHeight = 28;
      const segmentBandTop = 46;
      const segmentBandHeight = 28;
      const rollTop = 92;
      const rollHeight = 420;
      const minPitch = notes.length ? Math.min(...notes.map((item) => Number(item.pitch) || 0)) : 48;
      const maxPitch = notes.length ? Math.max(...notes.map((item) => Number(item.pitch) || 0)) : 72;
      const pitchSpan = Math.max(1, maxPitch - minPitch + 1);

      function xForUnit(unit) {{
        return paddingLeft + ((Number(unit) || 0) / Math.max(1, totalUnits)) * usableWidth;
      }}

      function yForPitch(pitch) {{
        const relative = (maxPitch - (Number(pitch) || 0)) / pitchSpan;
        return rollTop + (relative * (rollHeight - 14));
      }}

      let svg = `<svg width="${{width}}" height="${{height}}" viewBox="0 0 ${{width}} ${{height}}" xmlns="http://www.w3.org/2000/svg">`;
      svg += `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="rgba(255,255,255,0.65)" rx="18" />`;

      for (let barIndex = 0; barIndex <= barCount; barIndex += 1) {{
        const unit = barIndex * positionsPerBar;
        const x = xForUnit(unit);
        svg += `<line x1="${{x}}" y1="${{frameBandTop}}" x2="${{x}}" y2="${{rollTop + rollHeight}}" stroke="rgba(80, 63, 43, 0.18)" stroke-width="${{barIndex === 0 ? 1.6 : 1}}" />`;
        if (barIndex < barCount) {{
          svg += `<text x="${{x + 4}}" y="${{rollTop - 8}}" fill="rgba(80,63,43,0.72)" font-size="11">B${{barIndex}}</text>`;
        }}
      }}

      for (let pitch = minPitch; pitch <= maxPitch; pitch += 1) {{
        const y = yForPitch(pitch);
        svg += `<line x1="${{paddingLeft}}" y1="${{y + 7}}" x2="${{width - paddingRight}}" y2="${{y + 7}}" stroke="rgba(91, 108, 124, 0.08)" stroke-width="1" />`;
      }}

      keyFrames.forEach((frame, index) => {{
        const x = xForUnit(frame.start_unit);
        const frameWidth = Math.max(2, xForUnit(frame.end_unit) - x);
        const fill = frame.is_uncertain ? "rgba(141,141,141,0.72)" : keyColor(frame.best_key);
        const opacity = supportOpacity(frame.smoothed_support);
        svg += `<rect class="frame-cell" data-frame-index="${{index}}" x="${{x}}" y="${{frameBandTop}}" width="${{frameWidth}}" height="${{frameBandHeight}}" rx="5" fill="${{fill}}" fill-opacity="${{opacity}}" />`;
      }});

      keySegments.forEach((segment, index) => {{
        const x = xForUnit(segment.start_unit);
        const segmentWidth = Math.max(2, xForUnit(segment.end_unit) - x);
        const fill = keyColor(segment.key);
        svg += `<rect class="segment-cell" data-segment-index="${{index}}" x="${{x}}" y="${{segmentBandTop}}" width="${{segmentWidth}}" height="${{segmentBandHeight}}" rx="6" fill="${{fill}}" />`;
        svg += `<text x="${{x + 6}}" y="${{segmentBandTop + 18}}" fill="rgba(255,255,255,0.96)" font-size="12">${{escapeHtml(segment.key)}}</text>`;
      }});

      notes.forEach((note, index) => {{
        const x = xForUnit(note.start_unit);
        const noteWidth = Math.max(3, xForUnit(note.end_unit) - x);
        const y = yForPitch(note.pitch);
        svg += `<rect class="note-cell" data-note-index="${{index}}" x="${{x}}" y="${{y}}" width="${{noteWidth}}" height="12" rx="3" fill="rgba(91,108,124,0.84)" stroke="rgba(53,65,77,0.92)" stroke-width="1" />`;
      }});

      phraseBoundaries.forEach((boundary, index) => {{
        const x = xForUnit(boundary.unit);
        const stroke = boundary.anchor_kind === "mid_bar" ? "rgba(79,138,107,0.96)" : "rgba(35,79,109,0.96)";
        const dash = boundary.anchor_kind === "mid_bar" ? "5 5" : "";
        svg += `<line class="phrase-boundary" data-boundary-index="${{index}}" x1="${{x}}" y1="${{segmentBandTop}}" x2="${{x}}" y2="${{rollTop + rollHeight}}" stroke="${{stroke}}" stroke-width="2.4" stroke-dasharray="${{dash}}" />`;
      }});

      svg += `<text x="12" y="${{frameBandTop + 18}}" fill="rgba(80,63,43,0.72)" font-size="12">调性帧</text>`;
      svg += `<text x="12" y="${{segmentBandTop + 18}}" fill="rgba(80,63,43,0.72)" font-size="12">稳定调性</text>`;
      svg += `<text x="12" y="${{rollTop + 18}}" fill="rgba(80,63,43,0.72)" font-size="12">音符</text>`;
      svg += `</svg>`;

      elements.timelineContainer.innerHTML = svg;
      const svgRoot = elements.timelineContainer.querySelector("svg");
      if (!svgRoot) {{
        return;
      }}
      svgRoot.querySelectorAll(".note-cell").forEach((node) => {{
        const note = notes[Number(node.getAttribute("data-note-index"))];
        attachTooltip(
          node,
          `<strong>音符</strong><br/>pitch: <code>${{escapeHtml(note.pitch)}}</code><br/>起点: <code>${{escapeHtml(note.start_bar)}}:${{escapeHtml(note.start_pos)}}</code><br/>终点: <code>${{escapeHtml(note.end_bar)}}:${{escapeHtml(note.end_pos)}}</code><br/>力度桶: <code>${{escapeHtml(note.velocity_bin)}}</code>`,
        );
      }});
      svgRoot.querySelectorAll(".frame-cell").forEach((node) => {{
        const frame = keyFrames[Number(node.getAttribute("data-frame-index"))];
        attachTooltip(
          node,
          `<strong>调性帧</strong><br/>区间: <code>${{escapeHtml(frame.start_bar)}}:${{escapeHtml(frame.start_pos)}} → ${{escapeHtml(frame.end_bar)}}:${{escapeHtml(frame.end_pos)}}</code><br/>best: <code>${{escapeHtml(frame.best_key)}}</code><br/>raw: <code>${{escapeHtml(frame.raw_key)}}</code><br/>score: <code>${{escapeHtml(frame.best_score.toFixed(3))}}</code><br/>margin: <code>${{escapeHtml(frame.margin_to_second.toFixed(3))}}</code>`,
        );
      }});
      svgRoot.querySelectorAll(".segment-cell").forEach((node) => {{
        const segment = keySegments[Number(node.getAttribute("data-segment-index"))];
        attachTooltip(
          node,
          `<strong>稳定调性段</strong><br/>key: <code>${{escapeHtml(segment.key)}}</code><br/>区间: <code>${{escapeHtml(segment.start_bar)}}:${{escapeHtml(segment.start_pos)}} → ${{escapeHtml(segment.end_bar)}}:${{escapeHtml(segment.end_pos)}}</code><br/>长度: <code>${{escapeHtml(segment.length_bars.toFixed(2))}} bars</code><br/>mean_score: <code>${{escapeHtml(segment.mean_score.toFixed(3))}}</code>`,
        );
      }});
      svgRoot.querySelectorAll(".phrase-boundary").forEach((node) => {{
        const boundary = phraseBoundaries[Number(node.getAttribute("data-boundary-index"))];
        attachTooltip(
          node,
          `<strong>乐句边界</strong><br/>bar: <code>${{escapeHtml(boundary.bar_index)}}</code><br/>anchor_pos: <code>${{escapeHtml(boundary.anchor_pos)}}</code><br/>类型: <code>${{escapeHtml(boundary.anchor_kind)}}</code>`,
        );
      }});
    }}

    function renderTables(caseData) {{
      const frameRows = (caseData.key_analysis?.frames || []).map((frame) => `
        <tr>
          <td><code>${{escapeHtml(frame.start_bar)}}:${{escapeHtml(frame.start_pos)}}</code></td>
          <td><code>${{escapeHtml(frame.end_bar)}}:${{escapeHtml(frame.end_pos)}}</code></td>
          <td><code>${{escapeHtml(frame.best_key)}}</code></td>
          <td><code>${{escapeHtml(frame.raw_key)}}</code></td>
          <td>${{escapeHtml(Number(frame.best_score).toFixed(3))}}</td>
          <td>${{escapeHtml(Number(frame.margin_to_second).toFixed(3))}}</td>
          <td>${{escapeHtml(Number(frame.smoothed_support).toFixed(3))}}</td>
        </tr>
      `).join("");
      elements.frameTable.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>起点</th>
              <th>终点</th>
              <th>best_key</th>
              <th>raw_key</th>
              <th>best_score</th>
              <th>margin</th>
              <th>support</th>
            </tr>
          </thead>
          <tbody>${{frameRows || '<tr><td colspan="7">无调性帧</td></tr>'}}</tbody>
        </table>
      `;

      const scoreByBar = new Map((caseData.phrase_analysis?.boundary_scores || []).map((item) => [Number(item.bar_index), item]));
      const boundaryRows = (caseData.phrase_analysis?.boundaries || []).map((boundary) => {{
        const scoreItem = scoreByBar.get(Number(boundary.bar_index));
        return `
          <tr>
            <td><code>${{escapeHtml(boundary.bar_index)}}</code></td>
            <td><code>${{escapeHtml(boundary.anchor_pos)}}</code></td>
            <td>${{scoreItem ? escapeHtml(Number(scoreItem.score).toFixed(3)) : "-"}}</td>
            <td>${{scoreItem ? escapeHtml((scoreItem.reasons || []).join(", ")) : "-"}}</td>
          </tr>
        `;
      }}).join("");
      elements.boundaryTable.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>边界 bar</th>
              <th>anchor_pos</th>
              <th>score</th>
              <th>reasons</th>
            </tr>
          </thead>
          <tbody>${{boundaryRows || '<tr><td colspan="4">无乐句边界</td></tr>'}}</tbody>
        </table>
      `;
    }}

    function renderPanels(caseData) {{
      const flags = caseData.debug_flags || {{}};
      const bars = Array.isArray(caseData.bars) ? caseData.bars : [];
      const basicRows = [
        ["来源类型", `<code>${{escapeHtml(caseData.source_kind)}}</code>`],
        ["源文件", `<code>${{escapeHtml(caseData.source_path)}}</code>`],
        ["case_id", `<code>${{escapeHtml(caseData.case_id)}}</code>`],
        ["小节数", escapeHtml(bars.length)],
        ["音符数", escapeHtml((caseData.notes || []).length)],
        ["token 数", escapeHtml((caseData.tokens || []).length)],
      ];
      const keyRows = [
        ["initial_key", `<code>${{escapeHtml(caseData.key_analysis?.initial_key)}}</code>`],
        ["dominant_key", `<code>${{escapeHtml(caseData.key_analysis?.dominant_key)}}</code>`],
        ["主调覆盖率", `${{escapeHtml(((Number(caseData.key_analysis?.dominant_key_coverage) || 0) * 100).toFixed(1))}}%`],
        ["转调次数", escapeHtml((caseData.key_analysis?.modulation_points || []).length)],
        ["时间线摘要", `<code>${{escapeHtml(caseData.key_analysis?.timeline_summary)}}</code>`],
      ];
      const phraseRows = [
        ["边界数", escapeHtml((caseData.phrase_analysis?.boundaries || []).length)],
        ["乐句段数", escapeHtml((caseData.phrase_analysis?.phrase_spans || []).length)],
        ["平均句长", `${{escapeHtml(Number(caseData.phrase_analysis?.mean_phrase_bars || 0).toFixed(2))}} bars`],
        ["超长句", escapeHtml((flags.long_phrase_spans || []).length)],
        ["密集边界", escapeHtml((flags.dense_phrase_boundaries || []).length)],
      ];
      elements.basicInfo.innerHTML = createKvRows(basicRows);
      elements.keyInfo.innerHTML = createKvRows(keyRows);
      elements.phraseInfo.innerHTML = createKvRows(phraseRows);
      const flagNames = Array.isArray(flags.flag_names) ? flags.flag_names : [];
      elements.flagList.innerHTML = flagNames.length
        ? flagNames.map((item) => `<span class="pill danger">${{escapeHtml(item)}}</span>`).join("")
        : '<span class="pill">未发现可疑标记</span>';
    }}

    function renderCaseMeta(caseData) {{
      const meta = caseData.meta || {{}};
      const metaParts = [];
      if (meta.row_id !== undefined && meta.row_id !== null) {{
        metaParts.push(`row_id=${{meta.row_id}}`);
      }}
      if (meta.bucket) {{
        metaParts.push(`bucket=${{meta.bucket}}`);
      }}
      if (meta.token_origin) {{
        metaParts.push(`token_origin=${{meta.token_origin}}`);
      }}
      elements.caseMeta.textContent = metaParts.join(" | ");
    }}

    function render() {{
      buildFilteredIndices();
      const totalAll = state.allCases.length;
      const totalFiltered = state.filteredIndices.length;
      elements.metaSummary.textContent = `样本 ${{totalFiltered}} / ${{totalAll}}`;
      elements.prevBtn.disabled = !totalFiltered || state.caseIndex <= 0;
      elements.nextBtn.disabled = !totalFiltered || state.caseIndex >= totalFiltered - 1;

      const caseData = currentCase();
      if (!caseData) {{
        elements.statusText.textContent = "没有符合条件的样本";
        elements.caseTitle.textContent = "没有符合条件的样本";
        elements.caseSubtitle.textContent = "可以取消“只看可疑样本”，或重新生成 review 数据。";
        elements.caseMeta.textContent = "";
        elements.timelineContainer.innerHTML = '<div class="empty">没有可显示的数据</div>';
        elements.basicInfo.innerHTML = "";
        elements.keyInfo.innerHTML = "";
        elements.phraseInfo.innerHTML = "";
        elements.flagList.innerHTML = '<span class="pill">空</span>';
        elements.frameTable.innerHTML = "";
        elements.boundaryTable.innerHTML = "";
        return;
      }}

      const absoluteIndex = state.filteredIndices[state.caseIndex] + 1;
      elements.statusText.textContent = `当前 ${{state.caseIndex + 1}} / ${{totalFiltered}}（总 ${{totalAll}}，原始索引 ${{absoluteIndex}}）`;
      elements.caseTitle.textContent = caseData.title || "未命名样本";
      elements.caseSubtitle.textContent = caseData.subtitle || "";
      renderCaseMeta(caseData);
      renderTimeline(caseData);
      renderPanels(caseData);
      renderTables(caseData);
    }}

    function jumpToSearch() {{
      const query = String(elements.searchInput.value || "").trim().toLowerCase();
      if (!query) {{
        return;
      }}
      const matchIndex = state.allCases.findIndex((item) => {{
        const meta = item.meta || {{}};
        return [
          item.case_id,
          item.title,
          item.subtitle,
          item.source_path,
          meta.row_id,
        ].some((field) => String(field ?? "").toLowerCase().includes(query));
      }});
      if (matchIndex < 0) {{
        elements.statusText.textContent = `未找到：${{query}}`;
        return;
      }}
      if (state.suspiciousOnly && !state.filteredIndices.includes(matchIndex)) {{
        state.suspiciousOnly = false;
        elements.suspiciousOnlyToggle.checked = false;
      }}
      buildFilteredIndices();
      const filteredPos = state.filteredIndices.indexOf(matchIndex);
      if (filteredPos >= 0) {{
        state.caseIndex = filteredPos;
        render();
      }}
    }}

    elements.prevBtn.addEventListener("click", () => moveCase(-1));
    elements.nextBtn.addEventListener("click", () => moveCase(1));
    elements.searchBtn.addEventListener("click", jumpToSearch);
    elements.searchInput.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") {{
        jumpToSearch();
      }}
    }});
    elements.suspiciousOnlyToggle.addEventListener("change", () => {{
      state.suspiciousOnly = Boolean(elements.suspiciousOnlyToggle.checked);
      state.caseIndex = 0;
      render();
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.target instanceof HTMLInputElement) {{
        if (event.key === "Escape") {{
          event.target.blur();
        }}
        return;
      }}
      if (event.key === "ArrowLeft") {{
        moveCase(-1);
      }} else if (event.key === "ArrowRight") {{
        moveCase(1);
      }} else if (event.key.toLowerCase() === "f") {{
        event.preventDefault();
        elements.searchInput.focus();
        elements.searchInput.select();
      }}
    }});

    state.suspiciousOnly = Boolean(REVIEW_DATA?.meta?.only_suspicious);
    elements.suspiciousOnlyToggle.checked = state.suspiciousOnly;
    buildFilteredIndices();
    render();
  </script>
</body>
</html>
"""


def summarize_cases(cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """汇总 case 级别的简单统计。"""
    case_list = list(cases)
    suspicious_count = sum(
        1 for case in case_list if bool(case.get("debug_flags", {}).get("is_suspicious"))
    )
    return {
        "case_count": int(len(case_list)),
        "suspicious_case_count": int(suspicious_count),
        "source_kinds": sorted({str(case.get("source_kind", "")) for case in case_list if case.get("source_kind")}),
    }


def _safe_case_stem(case_id: str, used_stems: set[str]) -> str:
    """把 case_id 规范化成稳定可写入文件系统的名字。"""
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", str(case_id)).strip("._-")
    if not base:
        base = "case"
    candidate = base
    suffix = 1
    while candidate.lower() in used_stems:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used_stems.add(candidate.lower())
    return candidate


def _note_name(pitch: int) -> str:
    """把 MIDI 音高转换为便于人读的音名。"""
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    safe_pitch = int(pitch)
    octave = (safe_pitch // 12) - 1
    return f"{names[safe_pitch % 12]}{octave}"


def _build_case_summary(case: dict[str, Any], detail_path: str, positions_per_bar: int) -> dict[str, Any]:
    """从完整 case 中提取轻量索引信息。"""
    notes = list(case.get("notes", []))
    bars = list(case.get("bars", []))
    key_analysis = dict(case.get("key_analysis", {}))
    phrase_analysis = dict(case.get("phrase_analysis", {}))
    flags = dict(case.get("debug_flags", {}))
    note_count = len(notes)
    bar_count = len(bars)
    max_pitch = max((int(note.get("pitch", 0)) for note in notes), default=0)
    min_pitch = min((int(note.get("pitch", 127)) for note in notes), default=127)
    return {
        "case_id": str(case.get("case_id", "")),
        "source_kind": str(case.get("source_kind", "")),
        "title": str(case.get("title", "")),
        "subtitle": str(case.get("subtitle", "")),
        "source_path": str(case.get("source_path", "")),
        "meta": dict(case.get("meta", {})),
        "detail_path": str(detail_path),
        "stats": {
            "bar_count": int(bar_count),
            "note_count": int(note_count),
            "token_count": int(len(case.get("tokens", []))),
            "min_pitch": None if note_count == 0 else int(min_pitch),
            "max_pitch": None if note_count == 0 else int(max_pitch),
            "pitch_span": 0 if note_count == 0 else int(max_pitch - min_pitch),
            "positions_per_bar": int(positions_per_bar),
        },
        "key_summary": {
            "initial_key": str(key_analysis.get("initial_key", "uncertain")),
            "dominant_key": str(key_analysis.get("dominant_key", "uncertain")),
            "dominant_key_coverage": float(key_analysis.get("dominant_key_coverage", 0.0)),
            "timeline_summary": str(key_analysis.get("timeline_summary", "uncertain")),
            "modulation_count": int(len(key_analysis.get("modulation_points", []))),
            "segment_count": int(len(key_analysis.get("segments", []))),
            "frame_count": int(len(key_analysis.get("frames", []))),
        },
        "phrase_summary": {
            "boundary_count": int(len(phrase_analysis.get("boundaries", []))),
            "phrase_span_count": int(len(phrase_analysis.get("phrase_spans", []))),
            "mean_phrase_bars": float(phrase_analysis.get("mean_phrase_bars", 0.0)),
        },
        "debug_flags": flags,
    }


def _build_case_detail(case: dict[str, Any], *, include_tokens: bool) -> dict[str, Any]:
    """构建懒加载的 case 详情文件。"""
    detail = {
        "case_id": str(case.get("case_id", "")),
        "source_kind": str(case.get("source_kind", "")),
        "title": str(case.get("title", "")),
        "subtitle": str(case.get("subtitle", "")),
        "source_path": str(case.get("source_path", "")),
        "meta": dict(case.get("meta", {})),
        "notes": list(case.get("notes", [])),
        "bars": list(case.get("bars", [])),
        "key_analysis": dict(case.get("key_analysis", {})),
        "phrase_analysis": dict(case.get("phrase_analysis", {})),
        "debug_flags": dict(case.get("debug_flags", {})),
    }
    if include_tokens:
        detail["tokens"] = list(case.get("tokens", []))
    return detail


def write_review_bundle(
    *,
    output_dir: Path,
    cases: Sequence[dict[str, Any]],
    positions_per_bar: int,
    only_suspicious: bool,
    source_summary: dict[str, Any],
    include_tokens: bool = False,
) -> dict[str, Any]:
    """把 review 结果写成目录化、可懒加载的数据包。"""
    final_cases = [dict(case) for case in cases]
    if only_suspicious:
        final_cases = [
            case for case in final_cases if bool(case.get("debug_flags", {}).get("is_suspicious"))
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    used_stems: set[str] = set()
    for case in final_cases:
        stem = _safe_case_stem(str(case.get("case_id", "")), used_stems)
        detail_filename = f"{stem}.json"
        detail_path = cases_dir / detail_filename
        detail_payload = _build_case_detail(case, include_tokens=include_tokens)
        detail_path.write_text(
            json.dumps(detail_payload, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        summaries.append(
            _build_case_summary(
                case,
                detail_path=f"cases/{detail_filename}",
                positions_per_bar=positions_per_bar,
            )
        )

    suspicious_count = sum(
        1 for case in final_cases if bool(case.get("debug_flags", {}).get("is_suspicious"))
    )
    index_payload = {
        "meta": {
            "bundle_version": 2,
            "positions_per_bar": int(positions_per_bar),
            "only_suspicious": bool(only_suspicious),
            "case_count": int(len(final_cases)),
            "suspicious_case_count": int(suspicious_count),
            "include_tokens": bool(include_tokens),
            "source_summary": dict(source_summary),
        },
        "cases": summaries,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return index_payload
