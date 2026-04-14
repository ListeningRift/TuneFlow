"""checkpoint 综合打分与推荐工具。"""

from __future__ import annotations

import math
from typing import Any


_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "continuation": {
        "display_name": "续写综合推荐",
        "notes": [
            "先筛掉 raw 续写结构和停止能力明显不达标的 checkpoint，再在候选集合里做综合排序。",
            "FSM 指标通常偏高，因此只作为轻权重参考，不主导最终选择。",
        ],
        "gates": [
            {"key": "structural_validity_rate", "goal": "max", "threshold": 0.20},
            {"key": "eos_reached_rate", "goal": "max", "threshold": 0.20},
            {"key": "budget_stop_rate", "goal": "min", "threshold": 0.80},
        ],
        "metrics": [
            {"key": "structural_validity_rate", "weight": 0.30, "goal": "max"},
            {"key": "eos_reached_rate", "weight": 0.20, "goal": "max"},
            {"key": "budget_stop_rate", "weight": 0.16, "goal": "min"},
            {"key": "valid_loss", "weight": 0.18, "goal": "min"},
            {"key": "first_token_accuracy", "weight": 0.10, "goal": "max"},
            {"key": "fsm_structural_validity_rate", "weight": 0.04, "goal": "max"},
            {"key": "fsm_first_token_accuracy", "weight": 0.02, "goal": "max"},
        ],
        "tie_breakers": [
            ("structural_validity_rate", "max"),
            ("eos_reached_rate", "max"),
            ("budget_stop_rate", "min"),
            ("valid_loss", "min"),
            ("step", "max"),
        ],
    },
    "infilling": {
        "display_name": "挖空综合推荐",
        "notes": [
            "优先看 raw 结构合法率，其次兼顾 valid_loss 与 FSM 结构合法率。",
        ],
        "gates": [],
        "metrics": [
            {"key": "structural_validity_rate", "weight": 0.45, "goal": "max"},
            {"key": "valid_loss", "weight": 0.35, "goal": "min"},
            {"key": "fsm_structural_validity_rate", "weight": 0.15, "goal": "max"},
            {"key": "ppl", "weight": 0.05, "goal": "min"},
        ],
        "tie_breakers": [
            ("structural_validity_rate", "max"),
            ("valid_loss", "min"),
            ("fsm_structural_validity_rate", "max"),
            ("step", "max"),
        ],
    },
    "overall": {
        "display_name": "全局综合推荐",
        "notes": [
            "先筛掉 continuation 明显停不下来的 checkpoint，再联合 infilling 与 continuation 两类结果做综合排序。",
            "续写停止能力会被单独纳入评分，避免只看 valid_loss 选到停不下来的点。",
        ],
        "gates": [
            {"key": "continuation_structural_validity_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_eos_reached_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_budget_stop_rate", "goal": "min", "threshold": 0.80},
        ],
        "metrics": [
            {"key": "continuation_structural_validity_rate", "weight": 0.24, "goal": "max"},
            {"key": "continuation_eos_reached_rate", "weight": 0.16, "goal": "max"},
            {"key": "continuation_budget_stop_rate", "weight": 0.10, "goal": "min"},
            {"key": "continuation_first_token_accuracy", "weight": 0.08, "goal": "max"},
            {"key": "infilling_structural_validity_rate", "weight": 0.20, "goal": "max"},
            {"key": "infilling_fsm_structural_validity_rate", "weight": 0.05, "goal": "max"},
            {"key": "valid_loss", "weight": 0.17, "goal": "min"},
        ],
        "tie_breakers": [
            ("continuation_structural_validity_rate", "max"),
            ("continuation_eos_reached_rate", "max"),
            ("infilling_structural_validity_rate", "max"),
            ("valid_loss", "min"),
            ("step", "max"),
        ],
    },
}


def _to_finite_float(value: Any) -> float | None:
    """把输入安全转换为有限浮点数；失败时返回 `None`。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _rank_scores(values: list[tuple[int, float]], *, goal: str) -> dict[int, float]:
    """把同一指标按排名归一化到 `[0, 1]`，分数越高越好。"""
    if not values:
        return {}
    if len(values) == 1:
        only_index, _ = values[0]
        return {only_index: 1.0}

    reverse = goal == "max"
    ordered = sorted(values, key=lambda item: item[1], reverse=reverse)
    scores: dict[int, float] = {}
    total = len(ordered)
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
    """把不同方向的指标统一变成“越大越好”的排序键。"""
    numeric = _to_finite_float(value)
    if numeric is None:
        return float("-inf")
    return numeric if goal == "max" else -numeric


def _passes_gate(result: dict[str, Any], gate_specs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any], list[str]]:
    """检查单个结果是否满足门槛，并返回详细说明。"""
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
            if goal == "max":
                passed_this_gate = value >= threshold
            else:
                passed_this_gate = value <= threshold
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


def score_checkpoint_results(
    results: list[dict[str, Any]],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    为 checkpoint 结果列表打综合分，并返回推荐结果。

    返回：
    - enriched_results: 在原结果基础上附加 `balanced_score` 等字段
    - selection: 包含推荐 checkpoint、排行榜与权重说明
    """
    if profile not in _PROFILE_SPECS:
        raise ValueError(f"Unsupported checkpoint selection profile: {profile}")

    profile_spec = _PROFILE_SPECS[profile]
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
        if _to_finite_float(result.get("balanced_score")) is None:
            continue
        fallback_sortable.append(result)
        if bool(result.get("gate_passed")):
            gated_sortable.append(result)

    sortable = gated_sortable if gated_sortable else fallback_sortable

    tie_breakers = list(profile_spec.get("tie_breakers", []))
    sortable.sort(
        key=lambda item: (
            _transform_for_sort(item.get("balanced_score"), "max"),
            *[_transform_for_sort(item.get(metric_key), goal) for metric_key, goal in tie_breakers],
        ),
        reverse=True,
    )

    for rank, result in enumerate(sortable, start=1):
        result["balanced_rank"] = rank
    for result in enriched_results:
        if "balanced_rank" not in result:
            result["balanced_rank"] = None

    metric_weights = {str(spec["key"]): float(spec["weight"]) for spec in metric_specs}
    leaderboard = [
        {
            "rank": int(result["balanced_rank"]),
            "checkpoint_name": result.get("checkpoint_name"),
            "checkpoint_path": result.get("checkpoint_path"),
            "step": result.get("step"),
            "balanced_score": result.get("balanced_score"),
            "balanced_score_coverage": result.get("balanced_score_coverage"),
            "gate_passed": result.get("gate_passed"),
            "gate_failed_reasons": result.get("gate_failed_reasons"),
            "valid_loss": result.get("valid_loss"),
            "ppl": result.get("ppl"),
            "structural_validity_rate": result.get("structural_validity_rate"),
            "eos_reached_rate": result.get("eos_reached_rate"),
            "budget_stop_rate": result.get("budget_stop_rate"),
            "first_token_accuracy": result.get("first_token_accuracy"),
            "fsm_structural_validity_rate": result.get("fsm_structural_validity_rate"),
            "fsm_first_token_accuracy": result.get("fsm_first_token_accuracy"),
            "infilling_structural_validity_rate": result.get("infilling_structural_validity_rate"),
            "continuation_structural_validity_rate": result.get("continuation_structural_validity_rate"),
            "continuation_eos_reached_rate": result.get("continuation_eos_reached_rate"),
            "continuation_budget_stop_rate": result.get("continuation_budget_stop_rate"),
            "gate_details": result.get("gate_details"),
            "balanced_score_breakdown": result.get("balanced_score_breakdown"),
        }
        for result in sortable
    ]

    recommended: dict[str, Any] | None = None
    if sortable:
        top = sortable[0]
        recommended = {
            "checkpoint_name": top.get("checkpoint_name"),
            "checkpoint_path": top.get("checkpoint_path"),
            "step": top.get("step"),
            "balanced_score": top.get("balanced_score"),
            "balanced_rank": top.get("balanced_rank"),
            "balanced_score_coverage": top.get("balanced_score_coverage"),
            "gate_passed": top.get("gate_passed"),
            "gate_details": top.get("gate_details"),
            "valid_loss": top.get("valid_loss"),
            "ppl": top.get("ppl"),
            "score_breakdown": top.get("balanced_score_breakdown"),
        }

    selection = {
        "profile": profile,
        "display_name": profile_spec["display_name"],
        "selection_version": "balanced_v2_gate",
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
