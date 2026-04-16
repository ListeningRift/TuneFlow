"""Checkpoint ranking and recommendation helpers."""

from __future__ import annotations

import math
from typing import Any


_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "continuation": {
        "display_name": "Benchmark Continuation Ranking",
        "notes": [
            "Prioritize raw musical structure and temporal validity over stop behaviour.",
            "Stop metrics stay as safety checks; FSM metrics remain diagnostic only.",
        ],
        "gates": [
            {"key": "continuation_stop_success_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_budget_stop_rate", "goal": "min", "threshold": 0.75},
            {"key": "continuation_time_order_validity_rate", "goal": "max", "threshold": 0.85},
        ],
        "metrics": [
            {"key": "continuation_structural_validity_rate", "weight": 0.30, "goal": "max"},
            {"key": "continuation_time_order_validity_rate", "weight": 0.24, "goal": "max"},
            {"key": "continuation_empty_bar_rate", "weight": 0.12, "goal": "min"},
            {"key": "continuation_stop_success_rate", "weight": 0.12, "goal": "max"},
            {"key": "continuation_budget_stop_rate", "weight": 0.08, "goal": "min"},
            {"key": "continuation_first_event_hit_rate", "weight": 0.06, "goal": "max"},
            {"key": "continuation_syntax_invalid_rate", "weight": 0.04, "goal": "min"},
            {"key": "valid_loss_from_training", "weight": 0.04, "goal": "min"},
        ],
        "tie_breakers": [
            ("continuation_structural_validity_rate", "max"),
            ("continuation_time_order_validity_rate", "max"),
            ("continuation_stop_success_rate", "max"),
            ("valid_loss_from_training", "min"),
            ("step", "max"),
        ],
    },
    "infilling": {
        "display_name": "Benchmark Infilling Ranking",
        "notes": [
            "Rank infilling checkpoints mainly by raw structural validity.",
            "Time ordering helps break ties; FSM metrics remain diagnostic.",
        ],
        "gates": [
            {"key": "infilling_structural_validity_rate", "goal": "max", "threshold": 0.60},
        ],
        "metrics": [
            {"key": "infilling_structural_validity_rate", "weight": 0.72, "goal": "max"},
            {"key": "infilling_time_order_validity_rate", "weight": 0.18, "goal": "max"},
            {"key": "infilling_syntax_invalid_rate", "weight": 0.05, "goal": "min"},
            {"key": "valid_loss_from_training", "weight": 0.05, "goal": "min"},
        ],
        "tie_breakers": [
            ("infilling_structural_validity_rate", "max"),
            ("infilling_time_order_validity_rate", "max"),
            ("valid_loss_from_training", "min"),
            ("step", "max"),
        ],
    },
    "overall": {
        "display_name": "Benchmark Overall Ranking",
        "notes": [
            "Prioritize musical structure quality, especially infilling and continuation validity.",
            "Keep stop behaviour as a gate and light-weight scoring term rather than the main driver.",
        ],
        "gates": [
            {"key": "continuation_stop_success_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_budget_stop_rate", "goal": "min", "threshold": 0.75},
            {"key": "continuation_time_order_validity_rate", "goal": "max", "threshold": 0.85},
            {"key": "infilling_structural_validity_rate", "goal": "max", "threshold": 0.60},
        ],
        "metrics": [
            {"key": "infilling_structural_validity_rate", "weight": 0.24, "goal": "max"},
            {"key": "continuation_structural_validity_rate", "weight": 0.20, "goal": "max"},
            {"key": "continuation_time_order_validity_rate", "weight": 0.16, "goal": "max"},
            {"key": "infilling_time_order_validity_rate", "weight": 0.10, "goal": "max"},
            {"key": "continuation_empty_bar_rate", "weight": 0.08, "goal": "min"},
            {"key": "continuation_stop_success_rate", "weight": 0.08, "goal": "max"},
            {"key": "continuation_budget_stop_rate", "weight": 0.05, "goal": "min"},
            {"key": "continuation_syntax_invalid_rate", "weight": 0.04, "goal": "min"},
            {"key": "valid_loss_from_training", "weight": 0.05, "goal": "min"},
        ],
        "tie_breakers": [
            ("infilling_structural_validity_rate", "max"),
            ("continuation_structural_validity_rate", "max"),
            ("continuation_time_order_validity_rate", "max"),
            ("continuation_stop_success_rate", "max"),
            ("valid_loss_from_training", "min"),
            ("step", "max"),
        ],
    },
    "benchmark_overall": {
        "display_name": "Benchmark Overall Ranking",
        "notes": [
            "Prioritize musical structure quality, especially infilling and continuation validity.",
            "Keep stop behaviour as a gate and light-weight scoring term rather than the main driver.",
        ],
        "gates": [
            {"key": "continuation_stop_success_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_budget_stop_rate", "goal": "min", "threshold": 0.75},
            {"key": "continuation_time_order_validity_rate", "goal": "max", "threshold": 0.85},
            {"key": "infilling_structural_validity_rate", "goal": "max", "threshold": 0.60},
        ],
        "metrics": [
            {"key": "infilling_structural_validity_rate", "weight": 0.24, "goal": "max"},
            {"key": "continuation_structural_validity_rate", "weight": 0.20, "goal": "max"},
            {"key": "continuation_time_order_validity_rate", "weight": 0.16, "goal": "max"},
            {"key": "infilling_time_order_validity_rate", "weight": 0.10, "goal": "max"},
            {"key": "continuation_empty_bar_rate", "weight": 0.08, "goal": "min"},
            {"key": "continuation_stop_success_rate", "weight": 0.08, "goal": "max"},
            {"key": "continuation_budget_stop_rate", "weight": 0.05, "goal": "min"},
            {"key": "continuation_syntax_invalid_rate", "weight": 0.04, "goal": "min"},
            {"key": "valid_loss_from_training", "weight": 0.05, "goal": "min"},
        ],
        "tie_breakers": [
            ("infilling_structural_validity_rate", "max"),
            ("continuation_structural_validity_rate", "max"),
            ("continuation_time_order_validity_rate", "max"),
            ("continuation_stop_success_rate", "max"),
            ("valid_loss_from_training", "min"),
            ("step", "max"),
        ],
    },
}

_COMMON_SUMMARY_KEYS = (
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
)


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


def score_checkpoint_results(
    results: list[dict[str, Any]],
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank checkpoint result rows under a named scoring profile."""
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

    leaderboard_metric_keys = _leaderboard_metric_keys(profile_spec)
    metric_weights = {str(spec["key"]): float(spec["weight"]) for spec in metric_specs}
    leaderboard = []
    for result in sortable:
        row = {
            "rank": int(result["balanced_rank"]),
            "checkpoint_name": result.get("checkpoint_name"),
            "checkpoint_path": result.get("checkpoint_path"),
            "step": result.get("step"),
            "task_scope": result.get("task_scope"),
            "evaluation_tier": result.get("evaluation_tier"),
            "balanced_score": result.get("balanced_score"),
            "balanced_score_coverage": result.get("balanced_score_coverage"),
            "gate_passed": result.get("gate_passed"),
            "gate_failed_reasons": result.get("gate_failed_reasons"),
            "gate_details": result.get("gate_details"),
            "balanced_score_breakdown": result.get("balanced_score_breakdown"),
        }
        for key in leaderboard_metric_keys:
            if key in row:
                continue
            row[key] = result.get(key)
        leaderboard.append(row)

    recommended: dict[str, Any] | None = None
    if sortable:
        top = sortable[0]
        recommended = {
            "checkpoint_name": top.get("checkpoint_name"),
            "checkpoint_path": top.get("checkpoint_path"),
            "step": top.get("step"),
            "task_scope": top.get("task_scope"),
            "evaluation_tier": top.get("evaluation_tier"),
            "balanced_score": top.get("balanced_score"),
            "balanced_rank": top.get("balanced_rank"),
            "balanced_score_coverage": top.get("balanced_score_coverage"),
            "gate_passed": top.get("gate_passed"),
            "gate_details": top.get("gate_details"),
            "score_breakdown": top.get("balanced_score_breakdown"),
        }
        for key in leaderboard_metric_keys:
            if key not in recommended:
                recommended[key] = top.get(key)

    selection = {
        "profile": profile,
        "display_name": profile_spec["display_name"],
        "selection_version": "balanced_v4_structure_first",
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
