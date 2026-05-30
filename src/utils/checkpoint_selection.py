"""Checkpoint 排名与推荐辅助工具。"""

from __future__ import annotations

import math
from typing import Any


_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "continuation": {
        "primary_score_key": "task_capability_score",
        "primary_score_coverage_key": "task_capability_score_coverage",
        "primary_score_comparison_digits": 2,
        "selection_version": "task_capability_v1",
        "display_name": "Benchmark 续写任务排序",
        "notes": [
            "continuation scope 已切到任务型主排序，主分使用 task_capability_score，stop 与结构合法性继续只做 gate。",
            "续写场景下的任务型拆解重点观察结构控制、局部发展和长程连贯，旧 balanced_score 仅保留为兼容字段。",
        ],
        "gates": [
            {"key": "continuation_stop_success_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_budget_stop_rate", "goal": "min", "threshold": 0.75},
            {"key": "continuation_time_order_validity_rate", "goal": "max", "threshold": 0.85},
        ],
        "metrics": [
            {"key": "continuation_structural_validity_rate", "weight": 0.02, "goal": "max"},
            {"key": "continuation_time_order_validity_rate", "weight": 0.015, "goal": "max"},
            {"key": "continuation_empty_bar_rate", "weight": 0.01, "goal": "min"},
            {"key": "continuation_stop_success_rate", "weight": 0.005, "goal": "max"},
            {"key": "continuation_budget_stop_rate", "weight": 0.005, "goal": "min"},
            {"key": "continuation_syntax_invalid_rate", "weight": 0.005, "goal": "min"},
            {"key": "valid_loss_from_training", "weight": 0.02, "goal": "min"},
            {"key": "phrase_coherence_score", "weight": 0.23, "goal": "max"},
            {"key": "long_context_stability_score", "weight": 0.11, "goal": "max"},
            {"key": "continuation_first_event_hit_rate", "weight": 0.10, "goal": "max"},
            {"key": "duration_bin_l1_distance", "weight": 0.08, "goal": "min"},
            {"key": "continuation_pitch_diversity_score", "weight": 0.105, "goal": "max"},
            {"key": "continuation_rhythm_diversity_score", "weight": 0.105, "goal": "max"},
            {"key": "continuation_same_pitch_overlap_rate", "weight": 0.07, "goal": "min"},
            {"key": "continuation_event_ngram_repeat_ratio", "weight": 0.07, "goal": "min"},
            {"key": "continuation_rhythm_ngram_repeat_ratio", "weight": 0.05, "goal": "min"},
        ],
        "tie_breakers": [
            ("vs_baseline_win_rate", "max"),
            ("task_realization_score", "max"),
            ("long_context_coherence_score", "max"),
            ("local_development_score", "max"),
            ("structure_control_score", "max"),
            ("step", "max"),
        ],
    },
    "infilling": {
        "primary_score_key": "task_capability_score",
        "primary_score_coverage_key": "task_capability_score_coverage",
        "primary_score_comparison_digits": 2,
        "selection_version": "task_capability_v1",
        "display_name": "Benchmark 补全任务排序",
        "notes": [
            "infilling scope 已切到任务型主排序，主分使用 task_capability_score，结构合法性继续只做 gate。",
            "补全场景下的任务型拆解重点观察补全一致性，旧 balanced_score 仅保留为兼容字段。",
        ],
        "gates": [
            {"key": "infilling_structural_validity_rate", "goal": "max", "threshold": 0.60},
        ],
        "metrics": [
            {"key": "infilling_structural_validity_rate", "weight": 0.32, "goal": "max"},
            {"key": "infilling_time_order_validity_rate", "weight": 0.08, "goal": "max"},
            {"key": "infilling_syntax_invalid_rate", "weight": 0.05, "goal": "min"},
            {"key": "valid_loss_from_training", "weight": 0.05, "goal": "min"},
            {"key": "infilling_pitch_diversity_score", "weight": 0.16, "goal": "max"},
            {"key": "infilling_rhythm_diversity_score", "weight": 0.16, "goal": "max"},
            {"key": "infilling_same_pitch_overlap_rate", "weight": 0.07, "goal": "min"},
            {"key": "infilling_event_ngram_repeat_ratio", "weight": 0.06, "goal": "min"},
            {"key": "infilling_rhythm_ngram_repeat_ratio", "weight": 0.03, "goal": "min"},
            {"key": "infilling_onset_position_l1_distance", "weight": 0.02, "goal": "min"},
        ],
        "tie_breakers": [
            ("vs_baseline_win_rate", "max"),
            ("task_realization_score", "max"),
            ("infilling_consistency_score", "max"),
            ("step", "max"),
        ],
    },
    "overall": {
        "primary_score_key": "balanced_score",
        "selection_version": "balanced_v7_expression_first",
        "display_name": "Benchmark 综合排序",
        "notes": [
            "结构主要作为门槛存在，进入可用区间后，排序应更多由乐句连贯性与整体听感主导。",
            "这个 profile 会进一步弱化结构分，把更多权重放在节奏、音高和重复控制这些更接近音乐表达的代理指标上。",
        ],
        "gates": [
            {"key": "continuation_stop_success_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_budget_stop_rate", "goal": "min", "threshold": 0.75},
            {"key": "continuation_time_order_validity_rate", "goal": "max", "threshold": 0.85},
            {"key": "infilling_structural_validity_rate", "goal": "max", "threshold": 0.60},
        ],
        "metrics": [
            {"key": "infilling_structural_validity_rate", "weight": 0.015, "goal": "max"},
            {"key": "continuation_structural_validity_rate", "weight": 0.015, "goal": "max"},
            {"key": "continuation_time_order_validity_rate", "weight": 0.015, "goal": "max"},
            {"key": "infilling_time_order_validity_rate", "weight": 0.01, "goal": "max"},
            {"key": "continuation_empty_bar_rate", "weight": 0.005, "goal": "min"},
            {"key": "continuation_stop_success_rate", "weight": 0.005, "goal": "max"},
            {"key": "continuation_budget_stop_rate", "weight": 0.005, "goal": "min"},
            {"key": "continuation_syntax_invalid_rate", "weight": 0.005, "goal": "min"},
            {"key": "valid_loss_from_training", "weight": 0.02, "goal": "min"},
            {"key": "phrase_coherence_score", "weight": 0.25, "goal": "max"},
            {"key": "long_context_stability_score", "weight": 0.14, "goal": "max"},
            {"key": "overall_pitch_diversity_score", "weight": 0.105, "goal": "max"},
            {"key": "overall_rhythm_diversity_score", "weight": 0.105, "goal": "max"},
            {"key": "continuation_first_event_hit_rate", "weight": 0.08, "goal": "max"},
            {"key": "duration_bin_l1_distance", "weight": 0.05, "goal": "min"},
            {"key": "overall_same_pitch_overlap_rate", "weight": 0.045, "goal": "min"},
            {"key": "continuation_event_ngram_repeat_ratio", "weight": 0.045, "goal": "min"},
            {"key": "continuation_rhythm_ngram_repeat_ratio", "weight": 0.03, "goal": "min"},
            {"key": "overall_event_ngram_repeat_ratio", "weight": 0.03, "goal": "min"},
            {"key": "overall_rhythm_ngram_repeat_ratio", "weight": 0.025, "goal": "min"},
        ],
        "tie_breakers": [
            ("phrase_coherence_score", "max"),
            ("overall_rhythm_diversity_score", "max"),
            ("overall_pitch_diversity_score", "max"),
            ("long_context_stability_score", "max"),
            ("continuation_first_event_hit_rate", "max"),
            ("duration_bin_l1_distance", "min"),
            ("continuation_event_ngram_repeat_ratio", "min"),
            ("overall_event_ngram_repeat_ratio", "min"),
            ("overall_same_pitch_overlap_rate", "min"),
            ("continuation_structural_validity_rate", "max"),
            ("infilling_structural_validity_rate", "max"),
            ("valid_loss_from_training", "min"),
            ("step", "max"),
        ],
    },
    "benchmark_overall": {
        "primary_score_key": "task_capability_score",
        "primary_score_coverage_key": "task_capability_score_coverage",
        "primary_score_comparison_digits": 2,
        "selection_version": "task_capability_v1",
        "display_name": "Benchmark 综合排序",
        "notes": [
            "结构主要作为门槛存在，进入可用区间后，排序应更多由乐句连贯性与整体听感主导。",
            "这个 profile 会进一步弱化结构分，把更多权重放在节奏、音高和重复控制这些更接近音乐表达的代理指标上。",
        ],
        "gates": [
            {"key": "continuation_stop_success_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_budget_stop_rate", "goal": "min", "threshold": 0.75},
            {"key": "continuation_time_order_validity_rate", "goal": "max", "threshold": 0.85},
            {"key": "infilling_structural_validity_rate", "goal": "max", "threshold": 0.60},
        ],
        "metrics": [
            {"key": "infilling_structural_validity_rate", "weight": 0.015, "goal": "max"},
            {"key": "continuation_structural_validity_rate", "weight": 0.015, "goal": "max"},
            {"key": "continuation_time_order_validity_rate", "weight": 0.015, "goal": "max"},
            {"key": "infilling_time_order_validity_rate", "weight": 0.01, "goal": "max"},
            {"key": "continuation_empty_bar_rate", "weight": 0.005, "goal": "min"},
            {"key": "continuation_stop_success_rate", "weight": 0.005, "goal": "max"},
            {"key": "continuation_budget_stop_rate", "weight": 0.005, "goal": "min"},
            {"key": "continuation_syntax_invalid_rate", "weight": 0.005, "goal": "min"},
            {"key": "valid_loss_from_training", "weight": 0.02, "goal": "min"},
            {"key": "phrase_coherence_score", "weight": 0.25, "goal": "max"},
            {"key": "long_context_stability_score", "weight": 0.14, "goal": "max"},
            {"key": "overall_pitch_diversity_score", "weight": 0.105, "goal": "max"},
            {"key": "overall_rhythm_diversity_score", "weight": 0.105, "goal": "max"},
            {"key": "continuation_first_event_hit_rate", "weight": 0.08, "goal": "max"},
            {"key": "duration_bin_l1_distance", "weight": 0.05, "goal": "min"},
            {"key": "overall_same_pitch_overlap_rate", "weight": 0.045, "goal": "min"},
            {"key": "continuation_event_ngram_repeat_ratio", "weight": 0.045, "goal": "min"},
            {"key": "continuation_rhythm_ngram_repeat_ratio", "weight": 0.03, "goal": "min"},
            {"key": "overall_event_ngram_repeat_ratio", "weight": 0.03, "goal": "min"},
            {"key": "overall_rhythm_ngram_repeat_ratio", "weight": 0.025, "goal": "min"},
        ],
        "tie_breakers": [
            ("vs_baseline_win_rate", "max"),
            ("task_realization_score", "max"),
            ("structure_control_score", "max"),
            ("local_development_score", "max"),
            ("long_context_coherence_score", "max"),
            ("infilling_consistency_score", "max"),
            ("step", "max"),
        ],
    },
}

_PROFILE_SPECS["benchmark_overall"]["notes"] = [
    "benchmark_overall 已改为 task-capability 主排序，task_capability_score 是主分，结构指标继续作为门槛。",
    "task_capability_score_coverage 会参与排序，主分接近时优先选择 coverage 更完整的结果，缺失主分的结果不参与推荐。",
]

_COMMON_SUMMARY_KEYS = (
    "absolute_score_version",
    "absolute_score",
    "absolute_score_coverage",
    "absolute_score_proxy_dimension_count",
    "absolute_score_proxy_dimensions",
    "absolute_score_missing_dimensions",
    "absolute_score_breakdown",
    "continuation_closure_score",
    "continuation_structure_score",
    "infilling_integrity_score",
    "phrase_coherence_score",
    "musical_expression_score",
    "long_context_stability_score",
    "training_health_score",
    "valid_loss",
    "ppl",
    "valid_loss_from_training",
    "train_loss_ema",
    "best_valid_loss_so_far",
    "overfit_gap",
    "structural_validity_rate",
    "eos_reached_rate",
    "budget_stop_rate",
    "first_token_accuracy",
    "fsm_structural_validity_rate",
    "fsm_first_token_accuracy",
    "infilling_structural_validity_rate",
    "infilling_time_order_validity_rate",
    "infilling_internal_time_order_validity_rate",
    "infilling_boundary_time_order_validity_rate",
    "continuation_structural_validity_rate",
    "continuation_stop_success_rate",
    "continuation_budget_stop_rate",
    "continuation_time_order_validity_rate",
    "continuation_empty_bar_rate",
    "continuation_first_event_hit_rate",
    "continuation_missing_eos_rate",
    "continuation_syntax_invalid_rate",
    "append_eos_recoverable_rate",
    "infilling_syntax_invalid_rate",
    "continuation_most_common_pitch_ratio",
    "continuation_longest_same_pitch_run_ratio",
    "continuation_pitch_diversity_score",
    "continuation_pitch_collapse_coverage",
    "continuation_event_ngram_repeat_ratio",
    "continuation_rhythm_ngram_repeat_ratio",
    "continuation_repetition_metric_coverage",
    "infilling_most_common_pitch_ratio",
    "infilling_longest_same_pitch_run_ratio",
    "infilling_pitch_diversity_score",
    "infilling_pitch_collapse_coverage",
    "infilling_event_ngram_repeat_ratio",
    "infilling_rhythm_ngram_repeat_ratio",
    "infilling_repetition_metric_coverage",
    "overall_most_common_pitch_ratio",
    "overall_longest_same_pitch_run_ratio",
    "overall_pitch_diversity_score",
    "overall_same_pitch_overlap_rate",
    "overall_pitch_collapse_coverage",
    "overall_duration_diversity_score",
    "overall_rhythm_diversity_score",
    "overall_event_ngram_repeat_ratio",
    "overall_rhythm_ngram_repeat_ratio",
    "overall_repetition_metric_coverage",
    "low_density_bar_rate",
    "multi_empty_bar_run_rate",
    "generated_bar_delta_mean",
    "generated_event_delta_mean",
    "pitch_span_delta_mean",
    "duration_bin_l1_distance",
    "fsm_illegal_top1_rate",
    "fsm_mask_intervention_rate",
    "fsm_dead_end_count",
    "fsm_legal_mass_mean",
    "task_capability_score",
    "task_capability_score_coverage",
    "task_control_score",
    "task_realization_score",
    "structure_control_score",
    "local_development_score",
    "long_context_coherence_score",
    "infilling_consistency_score",
    "vs_baseline_win_rate",
)

_BENCHMARK_OVERALL_PRIMARY_FIELDS = (
    "task_capability_score",
    "task_control_score",
    "task_realization_score",
    "structure_control_score",
    "local_development_score",
    "long_context_coherence_score",
    "infilling_consistency_score",
    "vs_baseline_win_rate",
)

_BENCHMARK_OVERALL_LEGACY_FLAT_KEYS = {
    "balanced_score",
    "balanced_score_coverage",
    "balanced_score_breakdown",
    "absolute_score_version",
    "absolute_score",
    "absolute_score_coverage",
    "absolute_score_proxy_dimension_count",
    "absolute_score_proxy_dimensions",
    "absolute_score_missing_dimensions",
    "absolute_score_breakdown",
}


def _to_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _rank_scores(values: list[tuple[int, float]], *, goal: str) -> dict[int, float]:
    if not values:
        return {}
    if len(values) == 1:
        only_index, _ = values[0]
        return {only_index: 1.0}

    reverse = goal == "max"
    ordered = sorted(values, key=lambda item: item[1], reverse=reverse)
    total = len(ordered)
    scores: dict[int, float] = {}
    pos = 0
    while pos < total:
        end = pos
        current = ordered[pos][1]
        while end + 1 < total and ordered[end + 1][1] == current:
            end += 1
        avg_rank = (pos + end) / 2.0
        score = 1.0 - (avg_rank / float(total - 1))
        for inner in range(pos, end + 1):
            scores[ordered[inner][0]] = score
        pos = end + 1
    return scores


def _transform_for_sort(value: Any, goal: str) -> float:
    numeric = _to_finite_float(value)
    if numeric is None:
        return float("-inf")
    return numeric if goal == "max" else -numeric


def _passes_gate(result: dict[str, Any], gate_specs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any], list[str]]:
    gate_details: dict[str, Any] = {}
    failed_reasons: list[str] = []
    passed = True

    for gate_spec in gate_specs:
        metric_key = str(gate_spec["key"])
        goal = str(gate_spec["goal"])
        threshold = float(gate_spec["threshold"])
        value = _to_finite_float(result.get(metric_key))
        passed_this_gate = False
        if value is not None:
            passed_this_gate = value >= threshold if goal == "max" else value <= threshold
        gate_details[metric_key] = {
            "goal": goal,
            "threshold": threshold,
            "value": result.get(metric_key),
            "passed": passed_this_gate,
        }
        if not passed_this_gate:
            passed = False
            if value is None:
                failed_reasons.append(f"{metric_key}=NA")
            elif goal == "max":
                failed_reasons.append(f"{metric_key}<{threshold:.4f}")
            else:
                failed_reasons.append(f"{metric_key}>{threshold:.4f}")

    return passed, gate_details, failed_reasons


def _leaderboard_metric_keys(profile_spec: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for spec in profile_spec.get("metrics", []):
        metric_key = str(spec["key"])
        if metric_key not in keys:
            keys.append(metric_key)
    for metric_key, _goal in profile_spec.get("tie_breakers", []):
        metric_key = str(metric_key)
        if metric_key not in keys:
            keys.append(metric_key)
    for metric_key in _COMMON_SUMMARY_KEYS:
        if metric_key not in keys:
            keys.append(metric_key)
    return keys


def _primary_score_breakdown_for_result(
    result: dict[str, Any],
    *,
    profile: str,
    primary_score_key: str,
) -> Any:
    """返回主分拆解；任务型 profile 优先使用 task_capability_score 拆解。"""
    if primary_score_key == "task_capability_score":
        return result.get(f"{primary_score_key}_breakdown")
    return result.get("balanced_score_breakdown")


def _selection_breakdown_for_result(
    result: dict[str, Any],
    *,
    profile: str,
    primary_score_key: str,
    primary_score_coverage_key: str | None,
    primary_score_comparison_digits: int | None,
    tie_breakers: list[tuple[str, str]],
) -> Any:
    """返回主排序解释；任务型 profile 不再把 balanced_score_breakdown 当主入口。"""
    if primary_score_key != "task_capability_score":
        return result.get("balanced_score_breakdown")

    tie_breaker_rows: list[dict[str, Any]] = []
    for metric_key, goal in tie_breakers:
        tie_breaker_rows.append(
            {
                "metric_key": metric_key,
                "goal": goal,
                "value": result.get(metric_key),
            }
        )

    return {
        "ranking_basis": primary_score_key,
        "primary_score": result.get("primary_score"),
        "primary_score_coverage_key": primary_score_coverage_key,
        "primary_score_coverage": result.get("primary_score_coverage"),
        "primary_score_comparison_digits": primary_score_comparison_digits,
        "tie_breakers": tie_breaker_rows,
        "gate_passed": result.get("gate_passed"),
        "gate_failed_reasons": result.get("gate_failed_reasons"),
    }


def _legacy_score_fields_for_result(result: dict[str, Any], *, profile: str) -> dict[str, Any]:
    """返回旧分数字段兼容容器，避免 benchmark_overall 继续把旧字段当主骨架。"""
    legacy_fields = {
        "balanced_score": result.get("balanced_score"),
        "balanced_rank": result.get("balanced_rank"),
        "balanced_score_coverage": result.get("balanced_score_coverage"),
        "balanced_score_breakdown": result.get("balanced_score_breakdown"),
        "absolute_score_version": result.get("absolute_score_version"),
        "absolute_score": result.get("absolute_score"),
        "absolute_score_coverage": result.get("absolute_score_coverage"),
        "absolute_score_proxy_dimension_count": result.get("absolute_score_proxy_dimension_count"),
        "absolute_score_proxy_dimensions": result.get("absolute_score_proxy_dimensions"),
        "absolute_score_missing_dimensions": result.get("absolute_score_missing_dimensions"),
        "absolute_score_breakdown": result.get("absolute_score_breakdown"),
    }
    if profile != "benchmark_overall":
        return {}
    return {
        key: value
        for key, value in legacy_fields.items()
        if value is not None
    }


def _primary_score_sort_value(result: dict[str, Any], primary_score_key: str) -> float:
    """返回主排序分数字段对应的可排序值。"""
    return _transform_for_sort(result.get(primary_score_key), "max")


def _sortable_primary_score(
    result: dict[str, Any],
    primary_score_key: str,
    primary_score_coverage_key: str | None,
) -> tuple[bool, float, float]:
    """返回结果是否具备可排序主分，以及主分和 coverage 数值。"""
    primary_score = _to_finite_float(result.get(primary_score_key))
    if primary_score is None:
        return False, float("nan"), float("nan")

    coverage = 1.0
    if primary_score_coverage_key:
        coverage_value = _to_finite_float(result.get(primary_score_coverage_key))
        if coverage_value is not None:
            coverage = coverage_value
    return True, primary_score, coverage


def _normalized_primary_score_for_comparison(primary_score: float) -> float:
    """把主分规整到可比较尺度，兼容 0-1 与 0-100 两种输入。"""
    if abs(primary_score) > 1.0:
        return primary_score / 100.0
    return primary_score


def _sort_key_for_result(
    result: dict[str, Any],
    *,
    profile: str,
    primary_score_key: str,
    primary_score_coverage_key: str | None,
    primary_score_comparison_digits: int | None,
    tie_breakers: list[tuple[str, str]],
) -> tuple[Any, ...]:
    """生成排序键，benchmark_overall 会显式把 coverage 纳入主排序语义。"""
    _has_primary_score, primary_score, primary_score_coverage = _sortable_primary_score(
        result,
        primary_score_key,
        primary_score_coverage_key,
    )
    if profile == "benchmark_overall":
        comparable_primary = float("-inf")
        if math.isfinite(primary_score):
            digits = 3 if primary_score_comparison_digits is None else primary_score_comparison_digits
            comparable_primary = round(_normalized_primary_score_for_comparison(primary_score), digits)
        return (
            comparable_primary,
            primary_score_coverage,
            primary_score,
            *[_transform_for_sort(result.get(metric_key), goal) for metric_key, goal in tie_breakers],
        )

    if primary_score_coverage_key is not None:
        comparable_primary = float("-inf")
        if math.isfinite(primary_score):
            digits = 3 if primary_score_comparison_digits is None else primary_score_comparison_digits
            comparable_primary = round(_normalized_primary_score_for_comparison(primary_score), digits)
        return (
            comparable_primary,
            primary_score_coverage,
            primary_score,
            *[_transform_for_sort(result.get(metric_key), goal) for metric_key, goal in tie_breakers],
        )

    return (
        _transform_for_sort(result.get(primary_score_key), "max"),
        *[_transform_for_sort(result.get(metric_key), goal) for metric_key, goal in tie_breakers],
    )


def score_checkpoint_results(
    results: list[dict[str, Any]],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按指定评分配置对 checkpoint 结果行进行排序。"""
    if profile not in _PROFILE_SPECS:
        raise ValueError(f"Unsupported checkpoint selection profile: {profile}")

    profile_spec = _PROFILE_SPECS[profile]
    primary_score_key = str(profile_spec.get("primary_score_key", "balanced_score"))
    primary_score_coverage_key_value = profile_spec.get("primary_score_coverage_key")
    primary_score_coverage_key = (
        str(primary_score_coverage_key_value) if primary_score_coverage_key_value is not None else None
    )
    primary_score_comparison_digits = profile_spec.get("primary_score_comparison_digits")
    selection_version = str(profile_spec.get("selection_version", "balanced_v7_expression_first"))
    metric_specs = list(profile_spec["metrics"])
    gate_specs = list(profile_spec.get("gates", []))
    total_weight = sum(float(spec["weight"]) for spec in metric_specs)

    metric_rank_maps: dict[str, dict[int, float]] = {}
    for spec in metric_specs:
        metric_key = str(spec["key"])
        goal = str(spec["goal"])
        metric_values: list[tuple[int, float]] = []
        for index, result in enumerate(results):
            numeric = _to_finite_float(result.get(metric_key))
            if numeric is not None:
                metric_values.append((index, numeric))
        metric_rank_maps[metric_key] = _rank_scores(metric_values, goal=goal)

    enriched_results: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        enriched = dict(result)
        gate_passed, gate_details, failed_reasons = _passes_gate(enriched, gate_specs)
        score_sum = 0.0
        used_weight = 0.0
        breakdown: dict[str, Any] = {}
        for spec in metric_specs:
            metric_key = str(spec["key"])
            weight = float(spec["weight"])
            goal = str(spec["goal"])
            raw_value = result.get(metric_key)
            rank_score = metric_rank_maps[metric_key].get(index)
            contribution = None
            if rank_score is not None:
                contribution = rank_score * weight
                score_sum += contribution
                used_weight += weight
            breakdown[metric_key] = {
                "goal": goal,
                "weight": weight,
                "value": raw_value,
                "rank_score": rank_score,
                "weighted_contribution": contribution,
            }

        coverage = (used_weight / total_weight) if total_weight > 0 else 0.0
        balanced_score = (score_sum / used_weight) if used_weight > 0 else float("nan")
        enriched["gate_passed"] = gate_passed
        enriched["gate_details"] = gate_details
        enriched["gate_failed_reasons"] = failed_reasons
        enriched["balanced_score"] = balanced_score
        enriched["balanced_score_coverage"] = coverage
        enriched["balanced_score_breakdown"] = breakdown
        enriched_results.append(enriched)

    gated_sortable: list[dict[str, Any]] = []
    fallback_sortable: list[dict[str, Any]] = []
    for result in enriched_results:
        has_primary_score, primary_score, primary_score_coverage = _sortable_primary_score(
            result,
            primary_score_key,
            primary_score_coverage_key,
        )
        has_balanced_score = _to_finite_float(result.get("balanced_score")) is not None
        result["primary_score"] = primary_score if has_primary_score else None
        result["primary_score_coverage"] = primary_score_coverage if has_primary_score else None
        if profile == "benchmark_overall":
            if not has_primary_score:
                continue
        elif not has_primary_score and not has_balanced_score:
            continue
        fallback_sortable.append(result)
        if bool(result.get("gate_passed")):
            gated_sortable.append(result)

    sortable = gated_sortable if gated_sortable else fallback_sortable
    tie_breakers = list(profile_spec.get("tie_breakers", []))
    sortable.sort(
        key=lambda item: _sort_key_for_result(
            item,
            profile=profile,
            primary_score_key=primary_score_key,
            primary_score_coverage_key=primary_score_coverage_key,
            primary_score_comparison_digits=primary_score_comparison_digits,
            tie_breakers=tie_breakers,
        ),
        reverse=True,
    )

    for rank, result in enumerate(sortable, start=1):
        result["balanced_rank"] = rank
    for result in enriched_results:
        if "balanced_rank" not in result:
            result["balanced_rank"] = None

    leaderboard_metric_keys = _leaderboard_metric_keys(profile_spec)
    metric_weights = {str(spec["key"]): float(spec["weight"]) for spec in metric_specs}
    leaderboard = []
    for result in sortable:
        primary_score_breakdown = _primary_score_breakdown_for_result(
            result,
            profile=profile,
            primary_score_key=primary_score_key,
        )
        selection_breakdown = _selection_breakdown_for_result(
            result,
            profile=profile,
            primary_score_key=primary_score_key,
            primary_score_coverage_key=primary_score_coverage_key,
            primary_score_comparison_digits=primary_score_comparison_digits,
            tie_breakers=tie_breakers,
        )
        legacy_score_fields = _legacy_score_fields_for_result(result, profile=profile)
        row = {
            "rank": int(result["balanced_rank"]),
            "checkpoint_name": result.get("checkpoint_name"),
            "checkpoint_path": result.get("checkpoint_path"),
            "step": result.get("step"),
            "task_scope": result.get("task_scope"),
            "evaluation_tier": result.get("evaluation_tier"),
            "primary_score_key": primary_score_key,
            "primary_score": result.get("primary_score"),
            "primary_score_coverage": result.get("primary_score_coverage"),
            "primary_score_breakdown": primary_score_breakdown,
            "selection_breakdown": selection_breakdown,
            "task_capability_score": result.get("task_capability_score"),
            "task_capability_score_breakdown": result.get("task_capability_score_breakdown"),
            "task_control_score": result.get("task_control_score"),
            "task_realization_score": result.get("task_realization_score"),
            "structure_control_score": result.get("structure_control_score"),
            "local_development_score": result.get("local_development_score"),
            "long_context_coherence_score": result.get("long_context_coherence_score"),
            "infilling_consistency_score": result.get("infilling_consistency_score"),
            "vs_baseline_win_rate": result.get("vs_baseline_win_rate"),
            "gate_passed": result.get("gate_passed"),
            "gate_failed_reasons": result.get("gate_failed_reasons"),
            "gate_details": result.get("gate_details"),
        }
        for key in leaderboard_metric_keys:
            if key in row:
                continue
            if profile == "benchmark_overall" and key in _BENCHMARK_OVERALL_LEGACY_FLAT_KEYS:
                continue
            row[key] = result.get(key)
        row["legacy_score_fields"] = legacy_score_fields
        row["compatibility_only"] = bool(legacy_score_fields)
        leaderboard.append(row)

    recommended: dict[str, Any] | None = None
    if sortable:
        top = sortable[0]
        legacy_score_fields = _legacy_score_fields_for_result(top, profile=profile)
        recommended = {
            "checkpoint_name": top.get("checkpoint_name"),
            "checkpoint_path": top.get("checkpoint_path"),
            "step": top.get("step"),
            "task_scope": top.get("task_scope"),
            "evaluation_tier": top.get("evaluation_tier"),
            "primary_score_key": primary_score_key,
            "primary_score": top.get("primary_score"),
            "primary_score_coverage": top.get("primary_score_coverage"),
            "primary_score_breakdown": _primary_score_breakdown_for_result(
                top,
                profile=profile,
                primary_score_key=primary_score_key,
            ),
            "selection_breakdown": _selection_breakdown_for_result(
                top,
                profile=profile,
                primary_score_key=primary_score_key,
                primary_score_coverage_key=primary_score_coverage_key,
                primary_score_comparison_digits=primary_score_comparison_digits,
                tie_breakers=tie_breakers,
            ),
        }
        for key in _BENCHMARK_OVERALL_PRIMARY_FIELDS:
            recommended[key] = top.get(key)
        recommended["task_capability_score_breakdown"] = top.get("task_capability_score_breakdown")
        recommended["gate_passed"] = top.get("gate_passed")
        recommended["gate_details"] = top.get("gate_details")
        recommended["gate_failed_reasons"] = top.get("gate_failed_reasons")
        if profile != "benchmark_overall":
            recommended["score_breakdown"] = top.get("balanced_score_breakdown")
        for key in leaderboard_metric_keys:
            if key not in recommended:
                if profile == "benchmark_overall" and key in _BENCHMARK_OVERALL_LEGACY_FLAT_KEYS:
                    continue
                recommended[key] = top.get(key)
        recommended["legacy_score_fields"] = legacy_score_fields
        recommended["compatibility_only"] = bool(legacy_score_fields)

    selection = {
        "profile": profile,
        "display_name": profile_spec["display_name"],
        "selection_version": selection_version,
        "primary_score_key": primary_score_key,
        "gate_metrics": gate_specs,
        "gate_enabled": bool(gate_specs),
        "eligible_checkpoint_count": len(sortable),
        "gate_passed_checkpoint_count": len(gated_sortable),
        "gate_fallback_used": (not gated_sortable and bool(fallback_sortable) and bool(gate_specs)),
        "metric_weights": metric_weights,
        "notes": list(profile_spec.get("notes", [])),
        "recommended_checkpoint": recommended,
        "leaderboard": leaderboard,
    }
    return enriched_results, selection
