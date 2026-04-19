"""Benchmark 结果的绝对能力评分工具。"""

from __future__ import annotations

import math
from typing import Any


ABSOLUTE_SCORE_VERSION = "absolute_v2_tightened_capability_panel"

DIMENSION_SCORE_KEYS = {
    "continuation_closure": "absolute_continuation_closure_score",
    "continuation_structure": "absolute_continuation_structure_score",
    "infilling_integrity": "absolute_infilling_integrity_score",
    "phrase_coherence": "absolute_phrase_coherence_score",
    "long_context_stability": "absolute_long_context_stability_score",
    "training_health": "absolute_training_health_score",
}

_DIMENSION_SPECS: dict[str, dict[str, Any]] = {
    "continuation_closure": {
        "label": "续写收束",
        "weight": 0.20,
        "proxy": False,
        "description": "衡量续写样本能否自然收束，而不是缺少 EOS 或被预算截断。",
        "metrics": [
            {
                "key": "continuation_stop_success_rate",
                "label": "续写成功停机率",
                "goal": "max",
                "weight": 0.40,
                "bad": 0.05,
                "acceptable": 0.28,
                "ideal": 0.55,
            },
            {
                "key": "continuation_budget_stop_rate",
                "label": "续写预算截断率",
                "goal": "min",
                "weight": 0.25,
                "bad": 0.98,
                "acceptable": 0.60,
                "ideal": 0.18,
            },
            {
                "key": "continuation_missing_eos_rate",
                "label": "续写缺失 EOS 率",
                "goal": "min",
                "weight": 0.25,
                "bad": 0.98,
                "acceptable": 0.60,
                "ideal": 0.15,
            },
            {
                "key": "append_eos_recoverable_rate",
                "label": "补 EOS 可恢复率",
                "goal": "max",
                "weight": 0.10,
                "bad": 0.02,
                "acceptable": 0.15,
                "ideal": 0.40,
            },
        ],
    },
    "continuation_structure": {
        "label": "续写结构",
        "weight": 0.20,
        "proxy": False,
        "description": "衡量续写输出是否保持结构合法、时间顺序正确且不过于空洞。",
        "metrics": [
            {
                "key": "continuation_structural_validity_rate",
                "label": "续写结构合法率",
                "goal": "max",
                "weight": 0.35,
                "bad": 0.15,
                "acceptable": 0.55,
                "ideal": 0.82,
            },
            {
                "key": "continuation_time_order_validity_rate",
                "label": "续写时间顺序合法率",
                "goal": "max",
                "weight": 0.30,
                "bad": 0.75,
                "acceptable": 0.94,
                "ideal": 0.995,
            },
            {
                "key": "continuation_empty_bar_rate",
                "label": "续写空 BAR 率",
                "goal": "min",
                "weight": 0.20,
                "bad": 0.25,
                "acceptable": 0.05,
                "ideal": 0.01,
            },
            {
                "key": "continuation_syntax_invalid_rate",
                "label": "续写语法非法率",
                "goal": "min",
                "weight": 0.15,
                "bad": 0.95,
                "acceptable": 0.30,
                "ideal": 0.06,
            },
        ],
    },
    "infilling_integrity": {
        "label": "补全完整性",
        "weight": 0.22,
        "proxy": False,
        "description": "衡量补全输出是否保持结构合法、时间顺序正确且语法合法。",
        "metrics": [
            {
                "key": "infilling_structural_validity_rate",
                "label": "补全结构合法率",
                "goal": "max",
                "weight": 0.50,
                "bad": 0.20,
                "acceptable": 0.72,
                "ideal": 0.96,
            },
            {
                "key": "infilling_time_order_validity_rate",
                "label": "补全时间顺序合法率",
                "goal": "max",
                "weight": 0.30,
                "bad": 0.45,
                "acceptable": 0.82,
                "ideal": 0.97,
            },
            {
                "key": "infilling_syntax_invalid_rate",
                "label": "补全语法非法率",
                "goal": "min",
                "weight": 0.20,
                "bad": 0.85,
                "acceptable": 0.18,
                "ideal": 0.03,
            },
        ],
    },
    "phrase_coherence": {
        "label": "乐句连贯性",
        "weight": 0.15,
        "proxy": True,
        "description": "v1 代理维度，使用首事件对齐、时值分布漂移和音高多样性近似衡量乐句自然度。",
        "metrics": [
            {
                "key": "continuation_first_event_hit_rate",
                "label": "续写首事件命中率",
                "goal": "max",
                "weight": 0.25,
                "bad": 0.12,
                "acceptable": 0.48,
                "ideal": 0.82,
            },
            {
                "key": "duration_bin_l1_distance",
                "label": "时值分桶 L1 距离",
                "goal": "min",
                "weight": 0.25,
                "bad": 1.30,
                "acceptable": 0.55,
                "ideal": 0.15,
            },
            {
                "key": "continuation_pitch_diversity_score",
                "label": "续写音高多样性分数",
                "goal": "max",
                "weight": 0.20,
                "bad": 0.12,
                "acceptable": 0.52,
                "ideal": 0.82,
            },
            {
                "key": "infilling_pitch_diversity_score",
                "label": "补全音高多样性分数",
                "goal": "max",
                "weight": 0.20,
                "bad": 0.12,
                "acceptable": 0.52,
                "ideal": 0.82,
            },
            {
                "key": "continuation_most_common_pitch_ratio",
                "label": "续写最高频 pitch 占比",
                "goal": "min",
                "weight": 0.10,
                "bad": 0.85,
                "acceptable": 0.38,
                "ideal": 0.18,
            },
        ],
    },
    "long_context_stability": {
        "label": "长上下文稳定性",
        "weight": 0.15,
        "proxy": True,
        "description": "v1 代理维度，使用稀疏 BAR、塌缩比例和长度漂移近似衡量长程退化风险。",
        "metrics": [
            {
                "key": "low_density_bar_rate",
                "label": "低密度 BAR 率",
                "goal": "min",
                "weight": 0.15,
                "bad": 0.28,
                "acceptable": 0.08,
                "ideal": 0.02,
            },
            {
                "key": "multi_empty_bar_run_rate",
                "label": "连续空 BAR 样本率",
                "goal": "min",
                "weight": 0.20,
                "bad": 0.25,
                "acceptable": 0.05,
                "ideal": 0.01,
            },
            {
                "key": "generated_event_delta_mean",
                "label": "生成事件数偏差均值",
                "transform": "abs_min",
                "weight": 0.20,
                "bad": 40.0,
                "acceptable": 12.0,
                "ideal": 2.0,
            },
            {
                "key": "pitch_span_delta_mean",
                "label": "音高跨度偏差均值",
                "transform": "abs_min",
                "weight": 0.15,
                "bad": 22.0,
                "acceptable": 7.0,
                "ideal": 1.5,
            },
            {
                "key": "continuation_longest_same_pitch_run_ratio",
                "label": "续写最长同 pitch 连续 run 占比",
                "goal": "min",
                "weight": 0.15,
                "bad": 0.78,
                "acceptable": 0.24,
                "ideal": 0.10,
            },
            {
                "key": "continuation_most_common_pitch_ratio",
                "label": "续写最高频 pitch 占比",
                "goal": "min",
                "weight": 0.15,
                "bad": 0.85,
                "acceptable": 0.38,
                "ideal": 0.18,
            },
        ],
    },
    "training_health": {
        "label": "训练健康度",
        "weight": 0.08,
        "proxy": False,
        "description": "基于验证损失、历史最佳验证损失和过拟合间隙衡量训练阶段健康度。",
        "metrics": [
            {
                "key": "valid_loss_from_training",
                "label": "训练期验证损失",
                "goal": "min",
                "weight": 0.45,
                "bad": 2.00,
                "acceptable": 0.95,
                "ideal": 0.55,
            },
            {
                "key": "best_valid_loss_so_far",
                "label": "历史最佳验证损失",
                "goal": "min",
                "weight": 0.35,
                "bad": 1.90,
                "acceptable": 0.90,
                "ideal": 0.55,
            },
            {
                "key": "overfit_gap",
                "label": "过拟合间隙",
                "transform": "band",
                "weight": 0.20,
                "bad_low": -0.80,
                "acceptable_low": -0.35,
                "ideal_low": -0.10,
                "ideal_high": 0.03,
                "acceptable_high": 0.10,
                "bad_high": 0.25,
            },
        ],
    },
}


def _to_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _smoothstep(value: float) -> float:
    clipped = max(0.0, min(1.0, value))
    return clipped * clipped * (3.0 - (2.0 * clipped))


def _piecewise_score(value: float, *, goal: str, bad: float, acceptable: float, ideal: float) -> float:
    if goal == "max":
        if value <= bad:
            return 0.0
        if value >= ideal:
            return 100.0
        if value <= acceptable:
            ratio = (value - bad) / max(1e-12, acceptable - bad)
            return 60.0 * _smoothstep(ratio)
        ratio = (value - acceptable) / max(1e-12, ideal - acceptable)
        return 60.0 + (40.0 * _smoothstep(ratio))

    if value >= bad:
        return 0.0
    if value <= ideal:
        return 100.0
    if value >= acceptable:
        ratio = (bad - value) / max(1e-12, bad - acceptable)
        return 60.0 * _smoothstep(ratio)
    ratio = (acceptable - value) / max(1e-12, acceptable - ideal)
    return 60.0 + (40.0 * _smoothstep(ratio))


def _band_score(
    value: float,
    *,
    bad_low: float,
    acceptable_low: float,
    ideal_low: float,
    ideal_high: float,
    acceptable_high: float,
    bad_high: float,
) -> float:
    if ideal_low <= value <= ideal_high:
        return 100.0
    if acceptable_low <= value < ideal_low:
        ratio = (value - acceptable_low) / max(1e-12, ideal_low - acceptable_low)
        return 60.0 + (40.0 * _smoothstep(ratio))
    if ideal_high < value <= acceptable_high:
        ratio = (acceptable_high - value) / max(1e-12, acceptable_high - ideal_high)
        return 60.0 + (40.0 * _smoothstep(ratio))
    if bad_low <= value < acceptable_low:
        ratio = (value - bad_low) / max(1e-12, acceptable_low - bad_low)
        return 60.0 * _smoothstep(ratio)
    if acceptable_high < value <= bad_high:
        ratio = (bad_high - value) / max(1e-12, bad_high - acceptable_high)
        return 60.0 * _smoothstep(ratio)
    return 0.0


def _score_metric(result: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    raw_value = result.get(spec["key"])
    numeric = _to_finite_float(raw_value)
    transformed_value = numeric
    mapped_score: float | None = None
    missing = numeric is None

    if numeric is not None:
        transform = spec.get("transform")
        if transform == "abs_min":
            transformed_value = abs(numeric)
            mapped_score = _piecewise_score(
                transformed_value,
                goal="min",
                bad=float(spec["bad"]),
                acceptable=float(spec["acceptable"]),
                ideal=float(spec["ideal"]),
            )
        elif transform == "band":
            mapped_score = _band_score(
                numeric,
                bad_low=float(spec["bad_low"]),
                acceptable_low=float(spec["acceptable_low"]),
                ideal_low=float(spec["ideal_low"]),
                ideal_high=float(spec["ideal_high"]),
                acceptable_high=float(spec["acceptable_high"]),
                bad_high=float(spec["bad_high"]),
            )
        else:
            mapped_score = _piecewise_score(
                numeric,
                goal=str(spec["goal"]),
                bad=float(spec["bad"]),
                acceptable=float(spec["acceptable"]),
                ideal=float(spec["ideal"]),
            )

    payload = {
        "label": spec["label"],
        "metric_key": spec["key"],
        "weight": float(spec["weight"]),
        "raw_value": raw_value,
        "transformed_value": transformed_value,
        "mapped_score": mapped_score,
        "score": mapped_score,
        "missing": missing,
    }
    transform = spec.get("transform")
    if "goal" in spec:
        payload["goal"] = spec["goal"]
        payload["thresholds"] = {
            "bad": float(spec["bad"]),
            "acceptable": float(spec["acceptable"]),
            "ideal": float(spec["ideal"]),
        }
    elif transform == "band":
        payload["transform"] = transform
        payload["thresholds"] = {
            "bad_low": float(spec["bad_low"]),
            "acceptable_low": float(spec["acceptable_low"]),
            "ideal_low": float(spec["ideal_low"]),
            "ideal_high": float(spec["ideal_high"]),
            "acceptable_high": float(spec["acceptable_high"]),
            "bad_high": float(spec["bad_high"]),
        }
    else:
        payload["transform"] = transform
        payload["thresholds"] = {
            "bad": float(spec["bad"]),
            "acceptable": float(spec["acceptable"]),
            "ideal": float(spec["ideal"]),
        }
    return payload


def _score_dimension(result: dict[str, Any], dimension_key: str, spec: dict[str, Any]) -> dict[str, Any]:
    metric_breakdown: dict[str, Any] = {}
    total_weight = 0.0
    covered_weight = 0.0
    weighted_score_sum = 0.0
    missing_metrics: list[str] = []

    for metric_spec in spec["metrics"]:
        metric_payload = _score_metric(result, metric_spec)
        metric_key = str(metric_spec["key"])
        metric_breakdown[metric_key] = metric_payload
        metric_weight = float(metric_spec["weight"])
        total_weight += metric_weight
        if metric_payload["mapped_score"] is None:
            missing_metrics.append(metric_key)
            continue
        covered_weight += metric_weight
        weighted_score_sum += metric_weight * float(metric_payload["mapped_score"])

    coverage = (covered_weight / total_weight) if total_weight > 0 else 0.0
    score = (weighted_score_sum / covered_weight) if covered_weight > 0 else float("nan")
    return {
        "dimension_key": dimension_key,
        "label": spec["label"],
        "description": spec["description"],
        "weight": float(spec["weight"]),
        "proxy": bool(spec.get("proxy", False)),
        "score": score,
        "coverage": coverage,
        "missing_metrics": missing_metrics,
        "metric_breakdown": metric_breakdown,
        "submetrics": metric_breakdown,
        "weighted_contribution": (float(spec["weight"]) * coverage * score / 100.0) if math.isfinite(score) else None,
    }


def score_absolute_capabilities(result: dict[str, Any]) -> dict[str, Any]:
    """使用固定映射对单个 benchmark 结果打绝对能力分。"""
    dimensions: dict[str, Any] = {}
    total_weight = 0.0
    covered_weight = 0.0
    weighted_score_sum = 0.0
    missing_metrics: list[str] = []
    proxy_dimensions: list[str] = []

    for dimension_key, dimension_spec in _DIMENSION_SPECS.items():
        dimension_payload = _score_dimension(result, dimension_key, dimension_spec)
        dimensions[dimension_key] = dimension_payload
        dimension_weight = float(dimension_spec["weight"])
        total_weight += dimension_weight
        covered_weight += dimension_weight * float(dimension_payload["coverage"])
        if bool(dimension_payload["proxy"]):
            proxy_dimensions.append(dimension_key)
        missing_metrics.extend(dimension_payload["missing_metrics"])
        if math.isfinite(float(dimension_payload["score"])):
            weighted_score_sum += dimension_weight * float(dimension_payload["coverage"]) * float(dimension_payload["score"])

    absolute_score = (weighted_score_sum / covered_weight) if covered_weight > 0 else float("nan")
    absolute_score_coverage = (covered_weight / total_weight) if total_weight > 0 else 0.0

    payload: dict[str, Any] = {
        "absolute_score_version": ABSOLUTE_SCORE_VERSION,
        "absolute_score": absolute_score,
        "absolute_score_coverage": absolute_score_coverage,
        "absolute_score_missing_metrics": sorted(set(missing_metrics)),
        "absolute_score_missing_dimensions": [
            DIMENSION_SCORE_KEYS[dimension_key].replace("absolute_", "")
            for dimension_key, dimension_payload in dimensions.items()
            if not math.isfinite(float(dimension_payload["score"]))
        ],
        "absolute_proxy_dimensions": proxy_dimensions,
        "absolute_score_proxy_dimensions": proxy_dimensions,
        "absolute_score_proxy_dimension_count": len(proxy_dimensions),
        "absolute_score_breakdown": {
            "version": ABSOLUTE_SCORE_VERSION,
            "dimensions": {},
        },
        "absolute_proxy_dimensions": proxy_dimensions,
    }
    dimension_key_aliases = {
        "continuation_closure": "continuation_closure_score",
        "continuation_structure": "continuation_structure_score",
        "infilling_integrity": "infilling_integrity_score",
        "phrase_coherence": "phrase_coherence_score",
        "long_context_stability": "long_context_stability_score",
        "training_health": "training_health_score",
    }
    for dimension_key, flat_key in DIMENSION_SCORE_KEYS.items():
        dimension_payload = dimensions[dimension_key]
        payload[flat_key] = dimension_payload["score"]
        alias_key = dimension_key_aliases[dimension_key]
        payload[alias_key] = dimension_payload["score"]
        payload[f"{alias_key}_coverage"] = dimension_payload["coverage"]
        payload["absolute_score_breakdown"]["dimensions"][alias_key] = {
            "label": dimension_payload["label"],
            "description": dimension_payload["description"],
            "weight": dimension_payload["weight"],
            "proxy": dimension_payload["proxy"],
            "score": dimension_payload["score"],
            "coverage": dimension_payload["coverage"],
            "missing_metrics": list(dimension_payload["missing_metrics"]),
            "submetrics": dict(dimension_payload["metric_breakdown"]),
        }
    return payload


def attach_absolute_capability_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为一组 benchmark 结果行附加绝对能力评分字段。"""
    enriched_results: list[dict[str, Any]] = []
    for result in results:
        enriched = dict(result)
        enriched.update(score_absolute_capabilities(enriched))
        enriched_results.append(enriched)
    return enriched_results


def describe_absolute_scoring() -> dict[str, Any]:
    """返回可序列化的绝对评分配置说明。"""
    dimensions: dict[str, Any] = {}
    for dimension_key, spec in _DIMENSION_SPECS.items():
        dimensions[dimension_key] = {
            "label": spec["label"],
            "weight": float(spec["weight"]),
            "proxy": bool(spec.get("proxy", False)),
            "description": spec["description"],
            "flat_score_key": DIMENSION_SCORE_KEYS[dimension_key],
            "metrics": [dict(metric_spec) for metric_spec in spec["metrics"]],
        }
    return {
        "version": ABSOLUTE_SCORE_VERSION,
        "score_range": [0, 100],
        "total_score_notes": [
            "absolute_score 使用固定映射，不依赖当前候选 checkpoint 集合。",
            "absolute_score_coverage 表示已配置维度权重中有多少具备可用指标。",
            "代理维度只应解读为 v1 阶段的方向性信号，不应直接等同于最终音乐质量判断。",
        ],
        "dimensions": dimensions,
    }
