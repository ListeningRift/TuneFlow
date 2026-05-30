# Benchmark 任务型能力重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 直接用 `task_capability_score` 替换现有 `balanced_score / absolute_score` 主排序，把 benchmark 改造成 4 类任务主导、诊断面板退居辅助的能力评测体系。

**Architecture:** 先在 `src/utils/benchmarking.py` 上扩展 4 类任务 case 与任务级原始指标，再引入单独的任务评分模块负责 `task_capability_score`，最后改写 `checkpoint_selection.py` 和 `scripts/eval/benchmark_runner.py`，让排行榜、推荐逻辑、摘要和图表全部切到新的任务型字段。第一阶段复用现有 continuation / infilling decode 管线，只重构 case 定义、聚合逻辑和排序语义，不先动训练或 tokenizer。

**Tech Stack:** Python 3.11、unittest、现有 TuneFlow benchmark runner、现有 grammar/FSM decode 管线

---

### Task 1: 扩展 Benchmark Manifest 为 4 类任务 Case

**Files:**
- Modify: `src/utils/benchmarking.py`
- Test: `tests/test_benchmarking.py`

- [ ] **Step 1: 先写 manifest 结构测试，锁定 4 类任务 case 的输出字段**

```python
def test_build_benchmark_manifest_emits_four_task_cases(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        eval_jsonl_path = tmp_path / "eval.jsonl"
        eval_tok_path = tmp_path / "eval.tok"
        rows = [
            {
                "artist": "Artist",
                "title": "Title",
                "family_key": "family::0",
                "midi_path": "path/0.mid",
                "note_count": 128,
                "duration_sec": 96.0,
            }
        ]
        eval_jsonl_path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
        eval_tok_path.write_text(" ".join(_long_sequence()) + "\n", encoding="utf-8")

        manifest = build_benchmark_manifest(
            eval_jsonl_path=eval_jsonl_path,
            eval_tok_path=eval_tok_path,
            config={
                "tier": "fast",
                "seed": 42,
                "sample_count": None,
                "per_bucket_limit": None,
                "min_prefix_tokens": 8,
                "continuation_prefix_ratio_min": 0.35,
                "continuation_prefix_ratio_max": 0.70,
                "infilling_hole_ratio_min": 0.10,
                "infilling_hole_ratio_max": 0.25,
            },
            max_positions=64,
        )

        case = manifest["cases"][0]
        self.assertIn("structure_control_case", case)
        self.assertIn("local_development_case", case)
        self.assertIn("long_context_case", case)
        self.assertIn("infilling_consistency_case", case)
```

- [ ] **Step 2: 运行测试，确认当前实现失败**

Run: `pytest tests/test_benchmarking.py::BenchmarkingTests::test_build_benchmark_manifest_emits_four_task_cases -v`
Expected: FAIL，提示缺少 `structure_control_case` 等新字段

- [ ] **Step 3: 在 `src/utils/benchmarking.py` 增加 4 类 task case 构造函数，并让 manifest 直接产出新 schema**

```python
def build_structure_control_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    min_prefix_tokens: int,
    prefix_ratio_min: float,
    prefix_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    base_case = build_continuation_case(
        source_tokens,
        max_positions=max_positions,
        min_prefix_tokens=min_prefix_tokens,
        prefix_ratio_min=prefix_ratio_min,
        prefix_ratio_max=prefix_ratio_max,
        seed=seed,
    )
    if base_case is None:
        return None
    prompt_core = [token for token in base_case["prompt_tokens"] if token not in {"BOS", "EOS"}]
    boundary_label = "start_new_phrase" if prompt_core and prompt_core[-1] == "PHRASE" else "continue_inside_phrase"
    return {
        **base_case,
        "task_name": "structure_control",
        "boundary_label": boundary_label,
    }


def build_local_development_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    min_prefix_tokens: int,
    prefix_ratio_min: float,
    prefix_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    base_case = build_continuation_case(
        source_tokens,
        max_positions=max_positions,
        min_prefix_tokens=min_prefix_tokens,
        prefix_ratio_min=prefix_ratio_min,
        prefix_ratio_max=prefix_ratio_max,
        seed=seed,
    )
    if base_case is None:
        return None
    return {
        **base_case,
        "task_name": "local_development",
        "development_label": "develop",
    }


def build_long_context_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    min_prefix_tokens: int,
    prefix_ratio_min: float,
    prefix_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    base_case = build_continuation_case(
        source_tokens,
        max_positions=max_positions,
        min_prefix_tokens=max(min_prefix_tokens, 48),
        prefix_ratio_min=max(prefix_ratio_min, 0.45),
        prefix_ratio_max=max(prefix_ratio_max, 0.75),
        seed=seed,
    )
    if base_case is None:
        return None
    return {
        **base_case,
        "task_name": "long_context_coherence",
        "section_label": "continue_section",
    }


def build_infilling_consistency_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    hole_ratio_min: float,
    hole_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    base_case = build_infilling_case(
        source_tokens,
        max_positions=max_positions,
        hole_ratio_min=hole_ratio_min,
        hole_ratio_max=hole_ratio_max,
        seed=seed,
    )
    if base_case is None:
        return None
    prefix_core = [token for token in base_case["prefix_tokens"] if token not in {"BOS", "EOS"}]
    structure_label = "across_boundary" if prefix_core and prefix_core[-1] == "PHRASE" else "inside_phrase"
    return {
        **base_case,
        "task_name": "infilling_consistency",
        "structure_label": structure_label,
    }
```

- [ ] **Step 4: 在 `build_benchmark_manifest` 中替换旧的 `continuation_case / infilling_case` 输出结构**

```python
        structure_control_case = build_structure_control_case(
            tokens,
            max_positions=max_positions,
            min_prefix_tokens=int(config["min_prefix_tokens"]),
            prefix_ratio_min=float(config["continuation_prefix_ratio_min"]),
            prefix_ratio_max=float(config["continuation_prefix_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 1,
        )
        local_development_case = build_local_development_case(
            tokens,
            max_positions=max_positions,
            min_prefix_tokens=int(config["min_prefix_tokens"]),
            prefix_ratio_min=float(config["continuation_prefix_ratio_min"]),
            prefix_ratio_max=float(config["continuation_prefix_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 3,
        )
        long_context_case = build_long_context_case(
            tokens,
            max_positions=max_positions,
            min_prefix_tokens=int(config["min_prefix_tokens"]),
            prefix_ratio_min=float(config["continuation_prefix_ratio_min"]),
            prefix_ratio_max=float(config["continuation_prefix_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 5,
        )
        infilling_consistency_case = build_infilling_consistency_case(
            tokens,
            max_positions=max_positions,
            hole_ratio_min=float(config["infilling_hole_ratio_min"]),
            hole_ratio_max=float(config["infilling_hole_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 7,
        )
        if any(
            case_payload is None
            for case_payload in (
                structure_control_case,
                local_development_case,
                long_context_case,
                infilling_consistency_case,
            )
        ):
            continue

        valid_cases.append(
            {
                "row_id": int(row["row_id"]),
                "tier": str(config["tier"]),
                "bucket": bucket,
                "meta": {
                    "artist": meta.get("artist"),
                    "title": meta.get("title"),
                    "family_key": meta.get("family_key"),
                    "midi_path": meta.get("midi_path"),
                    "note_count": meta.get("note_count"),
                    "duration_sec": meta.get("duration_sec"),
                    "tok_len": len(tokens),
                },
                "structure_control_case": structure_control_case,
                "local_development_case": local_development_case,
                "long_context_case": long_context_case,
                "infilling_consistency_case": infilling_consistency_case,
            }
        )
```

- [ ] **Step 5: 跑 `tests/test_benchmarking.py`，确认 manifest 结构稳定**

Run: `pytest tests/test_benchmarking.py -v`
Expected: PASS，原有 manifest 相关测试和新 task case 测试都通过

- [ ] **Step 6: Commit**

```bash
git add src/utils/benchmarking.py tests/test_benchmarking.py
git commit -m "feat: add task benchmark manifest cases"
```

### Task 2: 为 4 类任务增加任务级原始指标和聚合字段

**Files:**
- Modify: `src/utils/benchmarking.py`
- Modify: `scripts/eval/benchmark_runner.py`
- Test: `tests/test_benchmarking.py`

- [ ] **Step 1: 先写任务级 enrich 测试，锁定每个任务的 raw metric 字段**

```python
def test_structure_control_record_contains_boundary_metrics(self) -> None:
    record = enrich_structure_control_record(
        {
            "generated_tokens": ["BAR", "PHRASE", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8"],
            "reconstructed_tokens": ["BOS", "BAR", "PHRASE", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8", "EOS"],
            "reached_eos": True,
            "is_structurally_valid": True,
        },
        target_tokens=["PHRASE", "POS_0", "INST_PIANO", "PITCH_62", "DUR_4", "VEL_8", "EOS"],
        boundary_label="start_new_phrase",
    )
    self.assertIn("boundary_type_hit", record)
    self.assertIn("boundary_timing_hit", record)
    self.assertIn("post_boundary_realization_score", record)
```

- [ ] **Step 2: 运行测试，确认新 enrich 函数尚不存在**

Run: `pytest tests/test_benchmarking.py::BenchmarkingTests::test_structure_control_record_contains_boundary_metrics -v`
Expected: FAIL，提示 `enrich_structure_control_record` 未定义

- [ ] **Step 3: 在 `src/utils/benchmarking.py` 增加 4 个任务 enrich 函数，复用现有 continuation / infilling 分析结果**

```python
def enrich_structure_control_record(
    record: dict[str, Any],
    *,
    target_tokens: Sequence[str],
    boundary_label: str,
) -> dict[str, Any]:
    enriched = enrich_continuation_record(record, target_tokens=target_tokens)
    predicted_boundary = "start_new_phrase" if _extract_first_unit(record.get("generated_tokens", [])) == ("BAR",) else "continue_inside_phrase"
    enriched.update(
        {
            "task_name": "structure_control",
            "boundary_label": boundary_label,
            "boundary_type_hit": bool(predicted_boundary == boundary_label),
            "boundary_timing_hit": bool(enriched.get("first_unit_match")),
            "post_boundary_realization_score": 1.0 if bool(enriched.get("time_order_valid")) else 0.0,
        }
    )
    return enriched


def enrich_local_development_record(
    record: dict[str, Any],
    *,
    target_tokens: Sequence[str],
    development_label: str,
) -> dict[str, Any]:
    enriched = enrich_continuation_record(record, target_tokens=target_tokens)
    enriched.update(
        {
            "task_name": "local_development",
            "development_label": development_label,
            "motif_relation_hit": bool(enriched.get("duration_bin_l1_distance", 1.0) <= 0.80),
            "copy_overuse_penalty": float(enriched.get("event_ngram_repeat_ratio") or 0.0),
            "unrelated_drift_penalty": float(enriched.get("duration_bin_l1_distance") or 0.0),
            "development_quality_score": 1.0 - min(1.0, float(enriched.get("duration_bin_l1_distance") or 1.0)),
        }
    )
    return enriched


def enrich_long_context_record(
    record: dict[str, Any],
    *,
    target_tokens: Sequence[str],
    section_label: str,
) -> dict[str, Any]:
    enriched = enrich_continuation_record(record, target_tokens=target_tokens)
    enriched.update(
        {
            "task_name": "long_context_coherence",
            "section_label": section_label,
            "long_horizon_completion": bool(enriched.get("stop_success")),
            "theme_retention_score": 1.0 - min(1.0, float(enriched.get("duration_bin_l1_distance") or 1.0)),
            "section_continuity_score": 1.0 if bool(enriched.get("time_order_valid")) else 0.0,
            "degeneration_penalty": float(enriched.get("empty_bar_rate") or 0.0),
        }
    )
    return enriched


def enrich_infilling_consistency_record(
    record: dict[str, Any],
    *,
    target_hole_tokens: Sequence[str],
    structure_label: str,
) -> dict[str, Any]:
    enriched = enrich_infilling_record(record, target_hole_tokens=target_hole_tokens)
    enriched.update(
        {
            "task_name": "infilling_consistency",
            "structure_label": structure_label,
            "bridge_validity": bool(enriched.get("boundary_time_order_valid")),
            "boundary_compatibility_hit": bool(enriched.get("boundary_time_order_valid")),
            "rhythmic_connection_score": 1.0 - min(1.0, float(enriched.get("onset_position_l1_distance") or 1.0)),
            "pitch_connection_score": 1.0 - min(1.0, float(enriched.get("duration_bin_l1_distance") or 1.0)),
            "structural_fit_score": 1.0 if bool(enriched.get("time_order_valid")) else 0.0,
        }
    )
    return enriched
```

- [ ] **Step 4: 在 `scripts/eval/benchmark_runner.py` 的聚合阶段输出任务级 raw metric 字段**

```python
    task_metrics = {
        "structure_control_boundary_type_hit_rate": _safe_rate(structure_control_boundary_hits, structure_control_attempted),
        "structure_control_boundary_timing_hit_rate": _safe_rate(structure_control_boundary_timing_hits, structure_control_attempted),
        "structure_control_post_boundary_realization_score": _safe_mean(structure_control_realization_scores),
        "local_development_motif_relation_hit_rate": _safe_rate(local_development_relation_hits, local_development_attempted),
        "local_development_copy_overuse_penalty": _safe_mean(local_development_copy_penalties),
        "local_development_unrelated_drift_penalty": _safe_mean(local_development_drift_penalties),
        "local_development_quality_score": _safe_mean(local_development_quality_scores),
        "long_context_completion_rate": _safe_rate(long_context_completions, long_context_attempted),
        "long_context_theme_retention_score": _safe_mean(long_context_theme_scores),
        "long_context_section_continuity_score": _safe_mean(long_context_section_scores),
        "long_context_degeneration_penalty": _safe_mean(long_context_degeneration_penalties),
        "infilling_bridge_validity_rate": _safe_rate(infilling_bridge_validity_hits, infilling_attempted),
        "infilling_boundary_compatibility_hit_rate": _safe_rate(infilling_boundary_hits, infilling_attempted),
        "infilling_rhythmic_connection_score": _safe_mean(infilling_rhythm_scores),
        "infilling_pitch_connection_score": _safe_mean(infilling_pitch_scores),
        "infilling_structural_fit_score": _safe_mean(infilling_structural_fit_scores),
    }
    result.update(task_metrics)
```

- [ ] **Step 5: 跑测试，确认原始任务字段被稳定产出**

Run: `pytest tests/test_benchmarking.py -v`
Expected: PASS，新的 enrich 测试通过，原有 continuation / infilling 解析测试不回归

- [ ] **Step 6: Commit**

```bash
git add src/utils/benchmarking.py scripts/eval/benchmark_runner.py tests/test_benchmarking.py
git commit -m "feat: aggregate raw task benchmark metrics"
```

### Task 3: 引入任务型能力评分模块，替换绝对能力主分

**Files:**
- Create: `src/utils/task_benchmark_scoring.py`
- Modify: `src/utils/__init__.py`
- Test: `tests/test_task_benchmark_scoring.py`

- [ ] **Step 1: 先写任务评分测试，锁定 `task_capability_score` 和 4 个一级任务分**

```python
from __future__ import annotations

import unittest

from src.utils.task_benchmark_scoring import score_task_capabilities


def _base_result() -> dict[str, float]:
    return {
        "structure_control_boundary_type_hit_rate": 0.80,
        "structure_control_boundary_timing_hit_rate": 0.72,
        "structure_control_post_boundary_realization_score": 0.70,
        "local_development_motif_relation_hit_rate": 0.62,
        "local_development_copy_overuse_penalty": 0.12,
        "local_development_unrelated_drift_penalty": 0.25,
        "local_development_quality_score": 0.68,
        "long_context_completion_rate": 0.74,
        "long_context_theme_retention_score": 0.66,
        "long_context_section_continuity_score": 0.70,
        "long_context_degeneration_penalty": 0.10,
        "infilling_bridge_validity_rate": 0.86,
        "infilling_boundary_compatibility_hit_rate": 0.78,
        "infilling_rhythmic_connection_score": 0.72,
        "infilling_pitch_connection_score": 0.69,
        "infilling_structural_fit_score": 0.76,
    }


class TaskBenchmarkScoringTests(unittest.TestCase):
    def test_task_capability_score_stays_in_0_100(self) -> None:
        scored = score_task_capabilities(_base_result())
        self.assertGreaterEqual(float(scored["task_capability_score"]), 0.0)
        self.assertLessEqual(float(scored["task_capability_score"]), 100.0)
        self.assertIn("structure_control_score", scored)
        self.assertIn("local_development_score", scored)
        self.assertIn("long_context_coherence_score", scored)
        self.assertIn("infilling_consistency_score", scored)
```

- [ ] **Step 2: 运行测试，确认新模块不存在**

Run: `pytest tests/test_task_benchmark_scoring.py -v`
Expected: FAIL，提示 `src.utils.task_benchmark_scoring` 不存在

- [ ] **Step 3: 新建 `src/utils/task_benchmark_scoring.py`，实现统一的任务评分映射**

```python
"""Benchmark 任务型能力评分工具。"""

from __future__ import annotations

import math
from typing import Any


TASK_SCORE_VERSION = "task_v1_direct_replace"


def _to_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _bounded_score(value: float, *, reverse: bool = False) -> float:
    clipped = max(0.0, min(1.0, value))
    if reverse:
        clipped = 1.0 - clipped
    return clipped * 100.0


def _task_score(control_hit_score: float, music_realization_score: float) -> float:
    return (0.40 * control_hit_score) + (0.60 * music_realization_score)


def score_task_capabilities(result: dict[str, Any]) -> dict[str, Any]:
    structure_control_control = (
        0.50 * _bounded_score(float(result["structure_control_boundary_type_hit_rate"]))
        + 0.50 * _bounded_score(float(result["structure_control_boundary_timing_hit_rate"]))
    )
    structure_control_realization = _bounded_score(float(result["structure_control_post_boundary_realization_score"]))
    structure_control_score = _task_score(structure_control_control, structure_control_realization)

    local_development_control = _bounded_score(float(result["local_development_motif_relation_hit_rate"]))
    local_development_realization = (
        0.50 * _bounded_score(float(result["local_development_quality_score"]))
        + 0.25 * _bounded_score(float(result["local_development_copy_overuse_penalty"]), reverse=True)
        + 0.25 * _bounded_score(float(result["local_development_unrelated_drift_penalty"]), reverse=True)
    )
    local_development_score = _task_score(local_development_control, local_development_realization)

    long_context_control = _bounded_score(float(result["long_context_completion_rate"]))
    long_context_realization = (
        0.45 * _bounded_score(float(result["long_context_theme_retention_score"]))
        + 0.35 * _bounded_score(float(result["long_context_section_continuity_score"]))
        + 0.20 * _bounded_score(float(result["long_context_degeneration_penalty"]), reverse=True)
    )
    long_context_coherence_score = _task_score(long_context_control, long_context_realization)

    infilling_control = (
        0.50 * _bounded_score(float(result["infilling_bridge_validity_rate"]))
        + 0.50 * _bounded_score(float(result["infilling_boundary_compatibility_hit_rate"]))
    )
    infilling_realization = (
        0.35 * _bounded_score(float(result["infilling_rhythmic_connection_score"]))
        + 0.35 * _bounded_score(float(result["infilling_pitch_connection_score"]))
        + 0.30 * _bounded_score(float(result["infilling_structural_fit_score"]))
    )
    infilling_consistency_score = _task_score(infilling_control, infilling_realization)

    task_capability_score = (
        0.30 * structure_control_score
        + 0.25 * local_development_score
        + 0.25 * long_context_coherence_score
        + 0.20 * infilling_consistency_score
    )

    return {
        "task_score_version": TASK_SCORE_VERSION,
        "structure_control_score": structure_control_score,
        "local_development_score": local_development_score,
        "long_context_coherence_score": long_context_coherence_score,
        "infilling_consistency_score": infilling_consistency_score,
        "task_control_score": (
            0.30 * structure_control_control
            + 0.25 * local_development_control
            + 0.25 * long_context_control
            + 0.20 * infilling_control
        ),
        "task_realization_score": (
            0.30 * structure_control_realization
            + 0.25 * local_development_realization
            + 0.25 * long_context_realization
            + 0.20 * infilling_realization
        ),
        "task_capability_score": task_capability_score,
    }


def attach_task_capability_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched_results: list[dict[str, Any]] = []
    for result in results:
        enriched = dict(result)
        enriched.update(score_task_capabilities(enriched))
        enriched_results.append(enriched)
    return enriched_results
```

- [ ] **Step 4: 在 `src/utils/__init__.py` 暴露新评分接口，并保留旧模块导出仅作兼容**

```python
from .task_benchmark_scoring import attach_task_capability_scores, score_task_capabilities

__all__ = [
    "attach_absolute_capability_scores",
    "attach_task_capability_scores",
    "score_absolute_capabilities",
    "score_task_capabilities",
    "analyze_token_sequence",
    "build_benchmark_manifest",
    "build_continuation_trace",
    "build_infilling_trace",
    "checkpoint_sort_key",
    "discover_checkpoints",
    "enrich_continuation_record",
    "enrich_infilling_record",
    "generate_continuation_tokens",
    "generate_middle_tokens",
    "load_benchmark_config",
    "load_vocab",
    "sample_step_checkpoints",
    "select_export_cases",
    "score_checkpoint_results",
]
```

- [ ] **Step 5: 运行测试，确认任务评分模块稳定**

Run: `pytest tests/test_task_benchmark_scoring.py -v`
Expected: PASS，`task_capability_score` 和 4 个一级任务分都落在 `0-100`

- [ ] **Step 6: Commit**

```bash
git add src/utils/task_benchmark_scoring.py src/utils/__init__.py tests/test_task_benchmark_scoring.py
git commit -m "feat: add task capability benchmark scoring"
```

### Task 4: 重写 Checkpoint Selection，直接按任务主分排序

**Files:**
- Modify: `src/utils/checkpoint_selection.py`
- Test: `tests/test_checkpoint_selection.py`

- [ ] **Step 1: 先写新的 selection 测试，锁定主排序字段切到 `task_capability_score`**

```python
def test_benchmark_overall_profile_sorts_by_task_capability_score(self) -> None:
    results = [
        {
            "checkpoint_name": "step_1.pt",
            "checkpoint_path": "outputs/checkpoints/run/step_1.pt",
            "step": 1,
            "task_capability_score": 61.0,
            "structure_control_score": 63.0,
            "local_development_score": 60.0,
            "long_context_coherence_score": 58.0,
            "infilling_consistency_score": 64.0,
            "continuation_stop_success_rate": 0.70,
            "continuation_budget_stop_rate": 0.20,
            "continuation_time_order_validity_rate": 0.95,
            "infilling_structural_validity_rate": 0.82,
        },
        {
            "checkpoint_name": "step_2.pt",
            "checkpoint_path": "outputs/checkpoints/run/step_2.pt",
            "step": 2,
            "task_capability_score": 74.0,
            "structure_control_score": 72.0,
            "local_development_score": 75.0,
            "long_context_coherence_score": 73.0,
            "infilling_consistency_score": 77.0,
            "continuation_stop_success_rate": 0.69,
            "continuation_budget_stop_rate": 0.22,
            "continuation_time_order_validity_rate": 0.94,
            "infilling_structural_validity_rate": 0.80,
        },
    ]

    _scored, selection = score_checkpoint_results(results, profile="benchmark_overall")
    self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_2.pt")
    self.assertAlmostEqual(float(selection["leaderboard"][0]["task_capability_score"]), 74.0)
```

- [ ] **Step 2: 运行测试，确认当前 selection 仍然绑在旧字段上**

Run: `pytest tests/test_checkpoint_selection.py::CheckpointSelectionTests::test_benchmark_overall_profile_sorts_by_task_capability_score -v`
Expected: FAIL，提示 leaderboard 中没有 `task_capability_score` 或推荐逻辑仍用旧分

- [ ] **Step 3: 改写 `src/utils/checkpoint_selection.py` 的 profile 和 leaderboard 字段，主排序直接改用任务分**

```python
_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "benchmark_overall": {
        "display_name": "Benchmark 任务型综合排序",
        "notes": [
            "主排序直接使用任务型能力分，不再使用 balanced_score。",
            "结构合法性、时间顺序和 stop 行为保留为 gate，不再主导任务能力排序。",
        ],
        "gates": [
            {"key": "continuation_stop_success_rate", "goal": "max", "threshold": 0.20},
            {"key": "continuation_budget_stop_rate", "goal": "min", "threshold": 0.75},
            {"key": "continuation_time_order_validity_rate", "goal": "max", "threshold": 0.85},
            {"key": "infilling_structural_validity_rate", "goal": "max", "threshold": 0.60},
        ],
        "primary_score_key": "task_capability_score",
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
```

- [ ] **Step 4: 把 `score_checkpoint_results` 中的相对 rank 聚合改成直接排序模式**

```python
    primary_score_key = str(profile_spec["primary_score_key"])
    sortable.sort(
        key=lambda item: (
            _transform_for_sort(item.get(primary_score_key), "max"),
            *[_transform_for_sort(item.get(metric_key), goal) for metric_key, goal in tie_breakers],
        ),
        reverse=True,
    )

    for result in enriched_results:
        result["task_rank"] = result.get("task_rank")
        result["selection_score"] = result.get(primary_score_key)
```

- [ ] **Step 5: 跑 `tests/test_checkpoint_selection.py`，确认推荐逻辑切换成功**

Run: `pytest tests/test_checkpoint_selection.py -v`
Expected: PASS，推荐 checkpoint、leaderboard 和 gate 都按新任务分语义工作

- [ ] **Step 6: Commit**

```bash
git add src/utils/checkpoint_selection.py tests/test_checkpoint_selection.py
git commit -m "feat: rank checkpoints by task capability score"
```

### Task 5: 改写 Runner、摘要和报告页，切到任务型字段与基线对战

**Files:**
- Modify: `scripts/eval/benchmark_runner.py`
- Modify: `docs/benchmark_metrics.md`
- Test: `tests/test_benchmark_reporting.py`

- [ ] **Step 1: 先写 reporting 测试，锁定摘要里显示任务主分和基线对战**

```python
from __future__ import annotations

import unittest

from scripts.eval.benchmark_runner import _build_summary_markdown_v3


class BenchmarkReportingTests(unittest.TestCase):
    def test_summary_markdown_prefers_task_capability_fields(self) -> None:
        summary = _build_summary_markdown_v3(
            run_id="run",
            task_scope="all",
            benchmark_root=Path("outputs/benchmark/demo"),
            recommended={
                "checkpoint_name": "step_1000.pt",
                "step": 1000,
                "evaluation_tier": "formal",
                "task_capability_score": 74.5,
                "task_control_score": 70.0,
                "task_realization_score": 77.2,
                "vs_baseline_win_rate": 0.61,
            },
            fast_results=[],
            formal_results=[],
            top_results=[],
            training_summary={},
            plot_artifacts={},
            sample_artifacts={},
            exported_samples={},
            manifest_stats={"fast_case_count": 0, "formal_case_count": 0, "candidate_count": 0},
            checkpoint_prefilter={},
            evaluation_context={"config_paths": {}, "benchmark_configs": {}, "decoding": {}},
        )
        self.assertIn("任务型能力分", summary)
        self.assertIn("基线胜率", summary)
```

- [ ] **Step 2: 运行测试，确认摘要仍然输出旧的相对分 / 绝对分**

Run: `pytest tests/test_benchmark_reporting.py -v`
Expected: FAIL，摘要文本仍然依赖 `balanced_score` 或 `absolute_score`

- [ ] **Step 3: 在 runner 主流程中切换评分模块和基线对战逻辑**

```python
    from src.utils.task_benchmark_scoring import attach_task_capability_scores
    from src.utils.checkpoint_selection import score_checkpoint_results

    fast_results = attach_task_capability_scores(fast_results)
    formal_results = attach_task_capability_scores(formal_results)

    baseline_checkpoint_name = args.baseline_checkpoint or min(
        (result["checkpoint_name"] for result in formal_results or fast_results),
        key=lambda name: int(name.split("_")[1].split(".")[0]),
    )
    _attach_baseline_win_rates(
        fast_results,
        baseline_checkpoint_name=baseline_checkpoint_name,
        primary_score_key="task_capability_score",
    )
    _attach_baseline_win_rates(
        formal_results,
        baseline_checkpoint_name=baseline_checkpoint_name,
        primary_score_key="task_capability_score",
    )
```

- [ ] **Step 4: 改写 metric labels、核心图表、摘要文案和 JSON 输出字段**

```python
_PERCENT_METRICS = {
    "task_capability_score",
    "task_control_score",
    "task_realization_score",
    "vs_baseline_win_rate",
    "continuation_stop_success_rate",
    "continuation_budget_stop_rate",
    "continuation_time_order_validity_rate",
    "infilling_structural_validity_rate",
}

_METRIC_LABELS = {
    "task_rank": "任务排名",
    "task_capability_score": "任务型能力分",
    "task_control_score": "任务控制分",
    "task_realization_score": "音乐实现分",
    "structure_control_score": "结构控制能力",
    "local_development_score": "局部发展能力",
    "long_context_coherence_score": "长程连贯能力",
    "infilling_consistency_score": "补全一致性能力",
    "vs_baseline_win_rate": "基线胜率",
}

summary_lines.extend(
    [
        f"- 任务型能力分：{_format_metric_value_v2(recommended.get('task_capability_score'), key='task_capability_score')}",
        f"- 任务控制分：{_format_metric_value_v2(recommended.get('task_control_score'), key='task_control_score')}",
        f"- 音乐实现分：{_format_metric_value_v2(recommended.get('task_realization_score'), key='task_realization_score')}",
        f"- 基线胜率：{_format_metric_value_v2(recommended.get('vs_baseline_win_rate'), key='vs_baseline_win_rate')}",
    ]
)
```

- [ ] **Step 5: 更新 `docs/benchmark_metrics.md`，删除旧的主分说明，改成任务型 benchmark 文档**

```markdown
## 1. 指标分层

TuneFlow benchmark 现在分为 3 层：

1. `task metrics`
   - 结构控制能力
   - 局部发展能力
   - 长程连贯能力
   - 补全一致性能力

2. `task capability score`
   - `task_capability_score`
   - `task_control_score`
   - `task_realization_score`
   - `vs_baseline_win_rate`

3. `diagnostics`
   - pitch / rhythm / repetition / 退化报警 / 训练健康
```

- [ ] **Step 6: 运行新旧相关测试，确认排序、摘要和 runner 字段一致**

Run: `pytest tests/test_benchmarking.py tests/test_task_benchmark_scoring.py tests/test_checkpoint_selection.py tests/test_benchmark_reporting.py -v`
Expected: PASS，任务主分、selection、summary 文案全部对齐

- [ ] **Step 7: Commit**

```bash
git add scripts/eval/benchmark_runner.py docs/benchmark_metrics.md tests/test_benchmark_reporting.py
git commit -m "feat: switch benchmark reports to task capability scoring"
```

### Task 6: 清理旧主排序依赖，保留有限兼容字段

**Files:**
- Modify: `scripts/eval/benchmark_runner.py`
- Modify: `src/utils/checkpoint_selection.py`
- Modify: `src/utils/__init__.py`
- Test: `tests/test_checkpoint_selection.py`

- [ ] **Step 1: 先写兼容性测试，保证最终 leaderboard 不再要求 `balanced_score / absolute_score` 才能工作**

```python
def test_selection_does_not_require_legacy_scores(self) -> None:
    results = [
        {
            "checkpoint_name": "step_1.pt",
            "checkpoint_path": "outputs/checkpoints/run/step_1.pt",
            "step": 1,
            "task_capability_score": 65.0,
            "structure_control_score": 66.0,
            "local_development_score": 63.0,
            "long_context_coherence_score": 64.0,
            "infilling_consistency_score": 67.0,
            "continuation_stop_success_rate": 0.72,
            "continuation_budget_stop_rate": 0.18,
            "continuation_time_order_validity_rate": 0.95,
            "infilling_structural_validity_rate": 0.84,
        }
    ]
    scored, selection = score_checkpoint_results(results, profile="benchmark_overall")
    self.assertEqual(selection["recommended_checkpoint"]["checkpoint_name"], "step_1.pt")
    self.assertNotIn("balanced_score", selection["recommended_checkpoint"])
```

- [ ] **Step 2: 运行测试，确认旧字段仍被强依赖**

Run: `pytest tests/test_checkpoint_selection.py::CheckpointSelectionTests::test_selection_does_not_require_legacy_scores -v`
Expected: FAIL，推荐结果或 leaderboard 仍强绑定旧字段

- [ ] **Step 3: 删除 runner / selection 中对 `balanced_score` 和 `absolute_score` 的主展示依赖，仅保留兼容别名**

```python
    row = {
        "rank": int(result["task_rank"]),
        "checkpoint_name": result.get("checkpoint_name"),
        "checkpoint_path": result.get("checkpoint_path"),
        "step": result.get("step"),
        "task_capability_score": result.get("task_capability_score"),
        "task_control_score": result.get("task_control_score"),
        "task_realization_score": result.get("task_realization_score"),
        "vs_baseline_win_rate": result.get("vs_baseline_win_rate"),
        "gate_passed": result.get("gate_passed"),
    }
```

- [ ] **Step 4: 跑全部 benchmark 相关测试，确认旧主分已经退出主逻辑**

Run: `pytest tests/test_benchmarking.py tests/test_task_benchmark_scoring.py tests/test_checkpoint_selection.py tests/test_benchmark_reporting.py tests/test_absolute_benchmark_scoring.py -v`
Expected: PASS；如果保留 `test_absolute_benchmark_scoring.py`，它应退化为兼容模块测试，而不是主流程测试

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/benchmark_runner.py src/utils/checkpoint_selection.py src/utils/__init__.py tests/test_checkpoint_selection.py
git commit -m "refactor: remove legacy benchmark score dependencies from main flow"
```
