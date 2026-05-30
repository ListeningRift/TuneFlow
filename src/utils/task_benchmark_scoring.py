"""任务型能力评分工具。"""

from __future__ import annotations

import math
from typing import Any


TASK_SCORE_VERSION = "task_v1_preparation_layer"

_TASK_SPECS: dict[str, dict[str, Any]] = {
    "structure_control_score": {
        "label": "结构控制",
        "weight": 0.30,
        "control_metrics": [
            ("structure_control_boundary_type_hit_rate", 0.50, False),
            ("structure_control_boundary_timing_hit_rate", 0.50, False),
        ],
        "music_metrics": [
            ("structure_control_post_boundary_realization_score", 1.00, False),
        ],
    },
    "local_development_score": {
        "label": "局部发展",
        "weight": 0.25,
        "control_metrics": [
            ("local_development_motif_relation_hit_rate", 1.00, False),
        ],
        "music_metrics": [
            ("local_development_quality_score", 0.60, False),
            ("local_development_copy_overuse_penalty", 0.20, True),
            ("local_development_unrelated_drift_penalty", 0.20, True),
        ],
    },
    "long_context_coherence_score": {
        "label": "长程连贯",
        "weight": 0.25,
        "control_metrics": [
            ("long_context_completion_rate", 1.00, False),
        ],
        "music_metrics": [
            ("long_context_theme_retention_score", 0.40, False),
            ("long_context_section_continuity_score", 0.40, False),
            ("long_context_degeneration_penalty", 0.20, True),
        ],
    },
    "infilling_consistency_score": {
        "label": "补全一致性",
        "weight": 0.20,
        "control_metrics": [
            ("infilling_bridge_validity_rate", 0.50, False),
            ("infilling_boundary_compatibility_hit_rate", 0.50, False),
        ],
        "music_metrics": [
            ("infilling_rhythmic_connection_score", 1.0 / 3.0, False),
            ("infilling_pitch_connection_score", 1.0 / 3.0, False),
            ("infilling_structural_fit_score", 1.0 / 3.0, False),
        ],
    },
}


def _clamp_unit(value: float) -> float:
    """将原始指标限制在 0 到 1 之间，避免异常值污染评分。"""
    return min(1.0, max(0.0, value))


def _sanitize_external_numeric(value: Any) -> float | None:
    """对外暴露的数值字段统一做清洗，非有限值返回 None。"""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _normalize_metric_value(raw_value: Any, invert: bool) -> float | None:
    """把原始指标归一到 0 到 1；惩罚项会先做反向转换。"""
    numeric_value = _sanitize_external_numeric(raw_value)
    if numeric_value is None:
        return None
    unit_value = _clamp_unit(numeric_value)
    if invert:
        return 1.0 - unit_value
    return unit_value


def _normalize_external_score(score: float | None) -> float | None:
    """把内部浮点结果整理成对外稳定值，缺失时统一返回 None。"""
    return _sanitize_external_numeric(score)


def _with_coverage_metadata(payload: dict[str, Any], coverage: float, coverage_semantics: str) -> dict[str, Any]:
    """为覆盖率字段补充兼容别名与明确语义说明。"""
    payload["coverage"] = coverage
    payload["weighted_coverage"] = coverage
    payload["coverage_semantics"] = coverage_semantics
    return payload


def _weighted_metric_average(
    result: dict[str, Any],
    metrics: list[tuple[str, float, bool]],
) -> tuple[float | None, float, list[str], dict[str, Any]]:
    """计算一组指标的加权均值，并返回覆盖率、缺失项和明细。"""
    weighted_sum = 0.0
    covered_weight = 0.0
    total_weight = 0.0
    missing_metrics: list[str] = []
    metric_breakdown: dict[str, Any] = {}

    for metric_key, weight, invert in metrics:
        total_weight += weight
        raw_value = _sanitize_external_numeric(result.get(metric_key))
        normalized_value = _normalize_metric_value(raw_value, invert)
        metric_breakdown[metric_key] = {
            "weight": weight,
            "invert": invert,
            "raw_value": raw_value,
            "normalized_value": normalized_value,
        }
        if normalized_value is None:
            missing_metrics.append(metric_key)
            continue
        weighted_sum += weight * normalized_value
        covered_weight += weight

    if covered_weight <= 0.0:
        return None, 0.0, missing_metrics, metric_breakdown
    return weighted_sum / covered_weight, covered_weight / total_weight, missing_metrics, metric_breakdown


def _aggregate_component_scores(task_payloads: dict[str, dict[str, Any]], component_key: str) -> tuple[float | None, float]:
    """聚合所有一级任务的 control 或 realization 分，供摘要层直接复用。"""
    weighted_sum = 0.0
    covered_weight = 0.0
    total_weight = 0.0

    for task_key, spec in _TASK_SPECS.items():
        task_weight = float(spec["weight"])
        total_weight += task_weight
        component_payload = task_payloads[task_key]["submetrics"][component_key]
        component_score = component_payload["score"]
        component_coverage = float(component_payload["coverage"])
        covered_weight += task_weight * component_coverage
        if component_score is None:
            continue
        weighted_sum += task_weight * component_coverage * float(component_score)

    if covered_weight <= 0.0:
        return None, 0.0
    return weighted_sum / covered_weight, covered_weight / total_weight


def _score_primary_task(result: dict[str, Any], task_key: str, spec: dict[str, Any]) -> dict[str, Any]:
    """按 control hit 与 music realization 的固定配比计算一级任务分。"""
    control_hit, control_coverage, control_missing, control_breakdown = _weighted_metric_average(
        result,
        spec["control_metrics"],
    )
    music_realization, music_coverage, music_missing, music_breakdown = _weighted_metric_average(
        result,
        spec["music_metrics"],
    )

    effective_weight = 0.0
    weighted_sum = 0.0
    if control_hit is not None:
        effective_weight += 0.40 * control_coverage
        weighted_sum += 0.40 * control_coverage * control_hit
    if music_realization is not None:
        effective_weight += 0.60 * music_coverage
        weighted_sum += 0.60 * music_coverage * music_realization

    task_score = None if effective_weight <= 0.0 else (weighted_sum / effective_weight) * 100.0
    control_score = None if control_hit is None else control_hit * 100.0
    realization_score = None if music_realization is None else music_realization * 100.0

    return {
        "task_key": task_key,
        "label": spec["label"],
        "weight": float(spec["weight"]),
        "score": task_score,
        "coverage": effective_weight,
        "missing_metrics": control_missing + music_missing,
        "control_hit": control_score,
        "music_realization": realization_score,
        "submetrics": {
            "control_hit": _with_coverage_metadata(
                {
                    "score": control_score,
                    "missing_metrics": control_missing,
                    "submetrics": control_breakdown,
                },
                control_coverage,
                "metric_weight_coverage_ratio",
            ),
            "music_realization": _with_coverage_metadata(
                {
                    "score": realization_score,
                    "missing_metrics": music_missing,
                    "submetrics": music_breakdown,
                },
                music_coverage,
                "metric_weight_coverage_ratio",
            ),
        },
    }


def score_task_capabilities(result: dict[str, Any]) -> dict[str, Any]:
    """为单条 benchmark 结果计算任务型能力分与四个一级任务分。"""
    dimension_breakdown: dict[str, Any] = {}
    task_payloads: dict[str, dict[str, Any]] = {}
    total_weight = 0.0
    covered_weight = 0.0
    weighted_score_sum = 0.0
    missing_metrics: list[str] = []

    payload: dict[str, Any] = {
        "task_capability_score_version": TASK_SCORE_VERSION,
        "task_capability_score_breakdown": {
            "version": TASK_SCORE_VERSION,
            "dimensions": dimension_breakdown,
        },
    }

    for task_key, spec in _TASK_SPECS.items():
        task_payload = _score_primary_task(result, task_key, spec)
        task_payloads[task_key] = task_payload
        payload[task_key] = _normalize_external_score(task_payload["score"])
        payload[f"{task_key}_coverage"] = task_payload["coverage"]
        payload[f"{task_key}_weighted_coverage"] = task_payload["coverage"]
        payload[f"{task_key}_coverage_semantics"] = "weighted_component_coverage_ratio"
        dimension_breakdown[task_key] = _with_coverage_metadata(
            {
                "label": task_payload["label"],
                "weight": task_payload["weight"],
                "score": _normalize_external_score(task_payload["score"]),
                "missing_metrics": sorted(set(task_payload["missing_metrics"])),
                "control_hit": task_payload["control_hit"],
                "music_realization": task_payload["music_realization"],
                "submetrics": task_payload["submetrics"],
            },
            task_payload["coverage"],
            "weighted_component_coverage_ratio",
        )
        task_weight = float(spec["weight"])
        total_weight += task_weight
        covered_weight += task_weight * float(task_payload["coverage"])
        missing_metrics.extend(task_payload["missing_metrics"])
        if task_payload["score"] is not None:
            weighted_score_sum += task_weight * float(task_payload["coverage"]) * float(task_payload["score"])

    task_capability_score = None if covered_weight <= 0.0 else (weighted_score_sum / covered_weight)
    missing_tasks = [
        task_key for task_key, task_payload in task_payloads.items() if bool(task_payload["missing_metrics"])
    ]
    unscored_tasks = [task_key for task_key in _TASK_SPECS if payload[task_key] is None]
    task_control_score, task_control_coverage = _aggregate_component_scores(task_payloads, "control_hit")
    task_realization_score, task_realization_coverage = _aggregate_component_scores(task_payloads, "music_realization")

    payload["task_capability_score"] = _normalize_external_score(task_capability_score)
    payload["task_capability_score_coverage"] = (covered_weight / total_weight) if total_weight > 0.0 else 0.0
    payload["task_capability_score_weighted_coverage"] = payload["task_capability_score_coverage"]
    payload["task_capability_score_coverage_semantics"] = "weighted_task_coverage_ratio"
    payload["task_capability_score_missing_metrics"] = sorted(set(missing_metrics))
    payload["task_capability_score_missing_tasks"] = list(missing_tasks)
    payload["task_capability_score_unscored_tasks"] = list(unscored_tasks)
    payload["task_control_score"] = _normalize_external_score(task_control_score)
    payload["task_control_score_coverage"] = task_control_coverage
    payload["task_control_score_weighted_coverage"] = task_control_coverage
    payload["task_control_score_coverage_semantics"] = "weighted_task_component_coverage_ratio"
    payload["task_realization_score"] = _normalize_external_score(task_realization_score)
    payload["task_realization_score_coverage"] = task_realization_coverage
    payload["task_realization_score_weighted_coverage"] = task_realization_coverage
    payload["task_realization_score_coverage_semantics"] = "weighted_task_component_coverage_ratio"
    return payload


def attach_task_capability_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为一组 benchmark 结果附加任务型能力评分字段。"""
    enriched_results: list[dict[str, Any]] = []
    for result in results:
        enriched = dict(result)
        enriched.update(score_task_capabilities(enriched))
        enriched_results.append(enriched)
    return enriched_results
