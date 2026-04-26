#!/usr/bin/env python
"""供各个评估入口共用的 TuneFlow benchmark runner。"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


_TASK_LABELS = {
    "all": "benchmark",
    "infilling": "benchmark_infilling",
    "continuation": "benchmark_continuation",
}

_TASK_TITLES = {
    "all": "TuneFlow 综合评测",
    "infilling": "TuneFlow 补全评测",
    "continuation": "TuneFlow 续写评测",
}

_TASK_REPORT_NAMES = {
    "all": "benchmark_report.json",
    "infilling": "benchmark_infilling_report.json",
    "continuation": "benchmark_continuation_report.json",
}

_TASK_SUMMARY_NAMES = {
    "all": "benchmark_summary.md",
    "infilling": "benchmark_infilling_summary.md",
    "continuation": "benchmark_continuation_summary.md",
}

_TASK_PROFILE_NAMES = {
    "all": "benchmark_overall",
    "infilling": "infilling",
    "continuation": "continuation",
}


def _ensure_project_root_on_path() -> Path:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _task_names_for_scope(task_scope: str) -> tuple[str, ...]:
    if task_scope == "all":
        return ("continuation", "infilling")
    return (task_scope,)


def _task_banner(task_scope: str) -> str:
    if task_scope == "infilling":
        return "TuneFlow 补全 benchmark"
    if task_scope == "continuation":
        return "TuneFlow 续写 benchmark"
    return "TuneFlow 综合 benchmark"


def _task_examples(task_scope: str) -> str:
    script_name = {
        "all": "scripts/eval/eval_all.py",
        "infilling": "scripts/eval/eval_infilling.py",
        "continuation": "scripts/eval/eval_continuation.py",
    }[task_scope]
    return (
        "常用示例：\n"
        f"  1. 跑默认 small 评估\n"
        f"     python {script_name} --preset small\n"
        f"  2. 指定训练配置文件\n"
        f"     python {script_name} --config configs/train/train_base_run_small.yaml\n"
        f"  3. 只做一轮快速 smoke 检查\n"
        f"     python {script_name} --preset small --limit-checkpoints 2 --max-new-tokens 64\n"
        f"  4. 强制只在 CPU 上跑\n"
        f"     python {script_name} --preset small --device cpu --precision fp32\n"
        f"  5. 关闭 checkpoint 预筛，跑全部 step checkpoint\n"
        f"     python {script_name} --preset small --prefilter-top-k-by-valid-loss 0\n"
    )


def _artifact_file_names(task_scope: str) -> tuple[str, str]:
    label = _TASK_LABELS[task_scope]
    return f"{label}_fast_manifest.json", f"{label}_formal_manifest.json"


def _default_prefilter_top_k(*, preset: str | None, config_path: Path | None) -> int:
    if preset == "full":
        return 16
    if config_path is not None:
        stem = config_path.stem.lower()
        name = config_path.name.lower()
        if "full" in stem or "full" in name:
            return 16
    return 8


def _clean_benchmark_outputs(benchmark_root: Path, task_scope: str) -> None:
    """在重新生成前，清理当前 benchmark 任务的旧产物。"""
    benchmark_root.mkdir(parents=True, exist_ok=True)
    fast_manifest_name, formal_manifest_name = _artifact_file_names(task_scope)
    artifact_paths = [
        benchmark_root / _TASK_REPORT_NAMES[task_scope],
        benchmark_root / _TASK_SUMMARY_NAMES[task_scope],
        benchmark_root / fast_manifest_name,
        benchmark_root / formal_manifest_name,
        benchmark_root / f"{_TASK_LABELS[task_scope]}_core_metrics.png",
        benchmark_root / f"{_TASK_LABELS[task_scope]}_diagnostics.png",
        benchmark_root / f"{_TASK_LABELS[task_scope]}_absolute_capabilities.png",
        benchmark_root / f"{_TASK_LABELS[task_scope]}_training_health.png",
    ]
    for artifact_path in artifact_paths:
        if artifact_path.exists():
            artifact_path.unlink()

    samples_root = benchmark_root / "samples"
    if not samples_root.exists():
        return

    task_sample_names = [f"{task_name}.json" for task_name in _task_names_for_scope(task_scope)]
    for sample_path in samples_root.rglob("*.json"):
        if sample_path.name in task_sample_names:
            sample_path.unlink()
    for directory in sorted(
        [path for path in samples_root.rglob("*") if path.is_dir()],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    if not any(samples_root.iterdir()):
        samples_root.rmdir()


def _parse_args(*, task_scope: str, argv: list[str] | None = None) -> argparse.Namespace:
    # 三个评估入口共用这一套参数，避免 all/continuation/infilling 的行为不一致。
    parser = argparse.ArgumentParser(
        description=(
            f"{_task_banner(task_scope)}。\n"
            "作用：自动读取训练输出目录，先做 fast benchmark 预筛，再做 formal benchmark 复评，"
            "最后导出排行榜、推荐 checkpoint、样本与图表。"
        ),
        epilog=_task_examples(task_scope),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "训练配置文件路径。作用：从配置里读取 output_dir，自动定位要评估的 checkpoint 目录。\n"
            "不填时会改用 --preset。\n"
            "例子：--config configs/train/train_base_run_small.yaml"
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="small",
        choices=["small", "full"],
        help=(
            "内置训练预设。作用：不用手写配置路径，直接评估固定的 run。\n"
            "small -> configs/train/train_base_run_small.yaml\n"
            "full  -> configs/train/train_base_run_full.yaml\n"
            "例子：--preset small"
        ),
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=Path("data/eval/fixed_eval.jsonl"),
        help=(
            "benchmark 元数据文件。作用：提供 artist/title/note_count/duration 等样本信息。\n"
            "通常保持默认即可。\n"
            "例子：--eval-jsonl data/eval/fixed_eval.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-tok",
        type=Path,
        default=Path("data/tokenized/eval.tok"),
        help=(
            "token 化后的 benchmark 数据。作用：真正用于续写/补全评估的 token 序列来源。\n"
            "通常保持默认即可。\n"
            "例子：--eval-tok data/tokenized/eval.tok"
        ),
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("data/tokenized/tokenizer_vocab.json"),
        help=(
            "tokenizer 词表路径。作用：把 token 文本和 token id 互相映射。\n"
            "通常保持默认即可。\n"
            "例子：--vocab-path data/tokenized/tokenizer_vocab.json"
        ),
    )
    parser.add_argument(
        "--fast-config",
        type=Path,
        default=Path("configs/eval/benchmark_fast.yaml"),
        help=(
            "fast benchmark 配置。作用：控制 fast 阶段抽样多少条样本、导出多少对比样本。\n"
            "例子：--fast-config configs/eval/benchmark_fast.yaml"
        ),
    )
    parser.add_argument(
        "--formal-config",
        type=Path,
        default=Path("configs/eval/benchmark_formal.yaml"),
        help=(
            "formal benchmark 配置。作用：控制 formal 阶段使用的全量样本集。\n"
            "例子：--formal-config configs/eval/benchmark_formal.yaml"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help=(
            "评估设备。作用：指定推理跑在什么设备上。\n"
            "auto 会优先尝试 CUDA；cpu 强制只用 CPU；cuda 强制只用 GPU。\n"
            "例子：--device cpu"
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help=(
            "推理精度。作用：控制显存占用和速度。\n"
            "auto 会按设备自动选；cpu 上通常用 fp32；支持 bf16/fp16 的 GPU 可更快。\n"
            "例子：--precision bf16"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "单条样本最多允许生成多少个新 token。作用：限制续写/补全长度，防止无限生成。\n"
            "值越大越慢，但也更不容易被截断。\n"
            "例子：--max-new-tokens 384"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help=(
            "采样温度。`0` 表示贪心解码，`>0` 时启用随机采样。\n"
            "数值越大越发散，越小越保守。\n"
            "例子：--temperature 0.9"
        ),
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=0.9,
        help=(
            "top-p 采样的累计概率阈值，取值范围 `(0, 1]`。\n"
            "`1.0` 表示不截断，较小值会更保守。\n"
            "例子：--top-p 0.9"
        ),
    )
    parser.add_argument(
        "--limit-checkpoints",
        type=int,
        default=None,
        help=(
            "只评估前 N 个 checkpoint。作用：做 smoke 测试或快速检查时缩短时间。\n"
            "不填表示不额外限制。\n"
            "例子：--limit-checkpoints 2"
        ),
    )
    parser.add_argument(
        "--checkpoint-policy",
        type=str,
        default="all",
        choices=["all", "sampled"],
        help=(
            "checkpoint 扫描策略。作用：决定 fast 阶段是跑全部 step checkpoint，还是只做均匀抽样。\n"
            "all = 全跑；sampled = 从全部 step checkpoint 中均匀抽样。\n"
            "例子：--checkpoint-policy sampled"
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=6,
        help=(
            "配合 --checkpoint-policy sampled 使用。作用：指定 sampled 模式下保留多少个 step checkpoint。\n"
            "只有 sampled 模式才生效。\n"
            "例子：--checkpoint-policy sampled --sample-count 6"
        ),
    )
    parser.add_argument(
        "--include-alias-checkpoints",
        action="store_true",
        help=(
            "是否把 best.pt / last.pt / latest.pt 这类别名 checkpoint 也加入 fast 扫描。\n"
            "默认不加，只看 step_*.pt，避免重复和干扰。\n"
            "例子：--include-alias-checkpoints"
        ),
    )
    parser.add_argument(
        "--prefilter-top-k-by-valid-loss",
        type=int,
        default=None,
        help=(
            "checkpoint 预筛数量。作用：在 fast benchmark 前，先按训练期 valid_loss 只保留 top K 个 checkpoint。\n"
            "设为 0 表示关闭预筛，全部 checkpoint 都跑。\n"
            "默认值会按 run 类型自动决定：small=8，full=16。\n"
            "例子：--prefilter-top-k-by-valid-loss 16"
        ),
    )
    parser.add_argument(
        "--prefilter-preserve-earliest",
        type=int,
        default=4,
        help=(
            "预筛时额外保留最早的 K 个 eval 对齐 checkpoint。作用：防止早期生成能力强但 valid_loss 不占优的点被误删。\n"
            "只有开启 valid_loss 预筛时才有意义。\n"
            "例子：--prefilter-preserve-earliest 4"
        ),
    )
    args = parser.parse_args(argv)
    if float(args.temperature) < 0.0:
        parser.error("--temperature must be >= 0.")
    if not (0.0 < float(args.top_p) <= 1.0):
        parser.error("--top-p must be within (0, 1].")
    return args


def _load_train_mapping(config_path: Path) -> dict[str, Any]:
    from src.utils.config_io import load_yaml_mapping

    payload = load_yaml_mapping(config_path, "train run config")
    if "train" in payload:
        train_payload = payload["train"]
        if not isinstance(train_payload, dict):
            raise ValueError(f"`train` section in {config_path} must be a mapping.")
        return train_payload
    return payload


def _resolve_preset_config(project_root: Path, preset: str) -> Path:
    mapping = {
        "small": project_root / "configs" / "train" / "train_base_run_small.yaml",
        "full": project_root / "configs" / "train" / "train_base_run_full.yaml",
    }
    config_path = mapping[preset].resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Preset config not found for --preset {preset}: {config_path}")
    return config_path


def _legacy_checkpoint_aliases(checkpoint_name: str) -> list[str]:
    aliases = [checkpoint_name]
    legacy_map = {
        "base_small": "train_base_run_small",
        "base_full": "train_base_run_full",
    }
    legacy_name = legacy_map.get(checkpoint_name)
    if legacy_name is not None:
        aliases.append(legacy_name)
    return aliases


def _resolve_eval_target(project_root: Path, args: argparse.Namespace) -> tuple[Path, dict[str, Any], str]:
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else (project_root / args.config)
        config_path = config_path.resolve()
    else:
        config_path = _resolve_preset_config(project_root, args.preset)

    train_mapping = _load_train_mapping(config_path)
    output_dir_value = train_mapping.get("output_dir")
    if output_dir_value is None:
        raise ValueError(f"Missing output_dir in train config: {config_path}")
    configured_checkpoint_dir = Path(str(output_dir_value))
    if not configured_checkpoint_dir.is_absolute():
        configured_checkpoint_dir = (project_root / configured_checkpoint_dir).resolve()

    checkpoint_dir = configured_checkpoint_dir
    if checkpoint_dir.exists():
        return checkpoint_dir, train_mapping, configured_checkpoint_dir.name

    checkpoint_candidates = [configured_checkpoint_dir]
    if configured_checkpoint_dir.parent.name == "checkpoints":
        for alias_name in _legacy_checkpoint_aliases(configured_checkpoint_dir.name):
            legacy_candidate = configured_checkpoint_dir.parent / "base" / alias_name
            if legacy_candidate not in checkpoint_candidates:
                checkpoint_candidates.append(legacy_candidate)
            alias_candidate = configured_checkpoint_dir.parent / alias_name
            if alias_candidate not in checkpoint_candidates:
                checkpoint_candidates.append(alias_candidate)

    for candidate in checkpoint_candidates[1:]:
        if candidate.exists():
            print(
                "[benchmark] warning: configured checkpoint directory does not exist; "
                f"falling back to legacy path {candidate}"
            )
            return candidate, train_mapping, configured_checkpoint_dir.name

    return checkpoint_dir, train_mapping, configured_checkpoint_dir.name


def _safe_rate(numerator: int, denominator: int) -> float:
    return (float(numerator) / float(denominator)) if denominator > 0 else float("nan")


def _safe_mean(values: list[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    return sum(finite_values) / float(len(finite_values))


def _finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _append_if_finite(values: list[float], value: Any) -> bool:
    numeric = _finite_float_or_none(value)
    if numeric is None:
        return False
    values.append(numeric)
    return True


def _to_jsonable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_name": result.get("checkpoint_name"),
        "checkpoint_path": result.get("checkpoint_path"),
        "step": result.get("step"),
        "task_scope": result.get("task_scope"),
        "evaluation_tier": result.get("evaluation_tier"),
        "training_metrics": dict(result.get("training_metrics", {})),
        "selection_metrics": dict(result.get("selection_metrics", {})),
        "diagnostic_metrics": dict(result.get("diagnostic_metrics", {})),
        "failure_reason_counts": dict(result.get("failure_reason_counts", {})),
        "syntax_reason_counts": dict(result.get("syntax_reason_counts", {})),
        "continuation_most_common_pitch_ratio": result.get("continuation_most_common_pitch_ratio"),
        "continuation_longest_same_pitch_run_ratio": result.get("continuation_longest_same_pitch_run_ratio"),
        "continuation_pitch_diversity_score": result.get("continuation_pitch_diversity_score"),
        "continuation_onset_position_l1_distance": result.get("continuation_onset_position_l1_distance"),
        "continuation_onset_position_entropy": result.get("continuation_onset_position_entropy"),
        "continuation_bar_start_onset_ratio": result.get("continuation_bar_start_onset_ratio"),
        "continuation_strong_beat_onset_ratio": result.get("continuation_strong_beat_onset_ratio"),
        "continuation_duration_diversity_score": result.get("continuation_duration_diversity_score"),
        "continuation_rhythm_diversity_score": result.get("continuation_rhythm_diversity_score"),
        "continuation_rhythm_metric_coverage": result.get("continuation_rhythm_metric_coverage"),
        "continuation_event_ngram_repeat_ratio": result.get("continuation_event_ngram_repeat_ratio"),
        "continuation_rhythm_ngram_repeat_ratio": result.get("continuation_rhythm_ngram_repeat_ratio"),
        "continuation_repetition_metric_coverage": result.get("continuation_repetition_metric_coverage"),
        "infilling_most_common_pitch_ratio": result.get("infilling_most_common_pitch_ratio"),
        "infilling_longest_same_pitch_run_ratio": result.get("infilling_longest_same_pitch_run_ratio"),
        "infilling_pitch_diversity_score": result.get("infilling_pitch_diversity_score"),
        "infilling_onset_position_l1_distance": result.get("infilling_onset_position_l1_distance"),
        "infilling_onset_position_entropy": result.get("infilling_onset_position_entropy"),
        "infilling_bar_start_onset_ratio": result.get("infilling_bar_start_onset_ratio"),
        "infilling_strong_beat_onset_ratio": result.get("infilling_strong_beat_onset_ratio"),
        "infilling_duration_diversity_score": result.get("infilling_duration_diversity_score"),
        "infilling_rhythm_diversity_score": result.get("infilling_rhythm_diversity_score"),
        "infilling_rhythm_metric_coverage": result.get("infilling_rhythm_metric_coverage"),
        "infilling_event_ngram_repeat_ratio": result.get("infilling_event_ngram_repeat_ratio"),
        "infilling_rhythm_ngram_repeat_ratio": result.get("infilling_rhythm_ngram_repeat_ratio"),
        "infilling_repetition_metric_coverage": result.get("infilling_repetition_metric_coverage"),
        "overall_most_common_pitch_ratio": result.get("overall_most_common_pitch_ratio"),
        "overall_longest_same_pitch_run_ratio": result.get("overall_longest_same_pitch_run_ratio"),
        "overall_pitch_diversity_score": result.get("overall_pitch_diversity_score"),
        "overall_onset_position_l1_distance": result.get("overall_onset_position_l1_distance"),
        "overall_onset_position_entropy": result.get("overall_onset_position_entropy"),
        "overall_bar_start_onset_ratio": result.get("overall_bar_start_onset_ratio"),
        "overall_strong_beat_onset_ratio": result.get("overall_strong_beat_onset_ratio"),
        "overall_duration_diversity_score": result.get("overall_duration_diversity_score"),
        "overall_rhythm_diversity_score": result.get("overall_rhythm_diversity_score"),
        "overall_rhythm_metric_coverage": result.get("overall_rhythm_metric_coverage"),
        "overall_event_ngram_repeat_ratio": result.get("overall_event_ngram_repeat_ratio"),
        "overall_rhythm_ngram_repeat_ratio": result.get("overall_rhythm_ngram_repeat_ratio"),
        "overall_repetition_metric_coverage": result.get("overall_repetition_metric_coverage"),
        "balanced_score": result.get("balanced_score"),
        "balanced_rank": result.get("balanced_rank"),
        "balanced_score_coverage": result.get("balanced_score_coverage"),
        "gate_passed": result.get("gate_passed"),
        "gate_details": result.get("gate_details"),
        "gate_failed_reasons": result.get("gate_failed_reasons"),
        "absolute_score_version": result.get("absolute_score_version"),
        "absolute_score": result.get("absolute_score"),
        "absolute_score_coverage": result.get("absolute_score_coverage"),
        "absolute_score_proxy_dimension_count": result.get("absolute_score_proxy_dimension_count"),
        "absolute_score_proxy_dimensions": result.get("absolute_score_proxy_dimensions"),
        "absolute_score_missing_dimensions": result.get("absolute_score_missing_dimensions"),
        "absolute_score_breakdown": result.get("absolute_score_breakdown"),
        "continuation_closure_score": result.get("continuation_closure_score"),
        "continuation_structure_score": result.get("continuation_structure_score"),
        "infilling_integrity_score": result.get("infilling_integrity_score"),
        "phrase_coherence_score": result.get("phrase_coherence_score"),
        "long_context_stability_score": result.get("long_context_stability_score"),
        "training_health_score": result.get("training_health_score"),
    }


def _top_counter_items(counter_mapping: dict[str, int], *, limit: int = 5) -> list[tuple[str, int]]:
    counter = Counter({str(key): int(value) for key, value in counter_mapping.items()})
    return counter.most_common(limit)


def _sample_preview(tokens: list[str], limit: int = 24) -> str:
    if not tokens:
        return "(empty)"
    if len(tokens) <= limit:
        return " ".join(tokens)
    return " ".join(tokens[:limit]) + " ..."


def _case_sample_payload(case: dict[str, Any], record: dict[str, Any], fsm_record: dict[str, Any], *, task: str) -> dict[str, Any]:
    payload = {
        "row_id": case["row_id"],
        "bucket": case["bucket"],
        "meta": dict(case["meta"]),
        "prompt_tokens": list(case[f"{task}_case"]["prompt_tokens"]),
        "raw_output_tokens": list(
            record.get("generated_tokens", record.get("generated_middle_tokens", []))
        ),
        "raw_reconstructed_tokens": list(record.get("reconstructed_tokens", [])),
        "fsm_output_tokens": list(
            fsm_record.get("generated_tokens", fsm_record.get("generated_middle_tokens", []))
        ),
        "fsm_reconstructed_tokens": list(fsm_record.get("reconstructed_tokens", [])),
        "raw_failure_reason": record.get("failure_reason"),
        "raw_syntax_reason": record.get("syntax_reason"),
        "fsm_failure_reason": fsm_record.get("failure_reason"),
        "fsm_syntax_reason": fsm_record.get("syntax_reason"),
        "stop_success": record.get("stop_success"),
        "budget_stop": record.get("budget_stop"),
        "time_order_valid": record.get("time_order_valid"),
        "internal_time_order_valid": record.get("internal_time_order_valid"),
        "boundary_time_order_valid": record.get("boundary_time_order_valid"),
        "time_order_violation_count": record.get("time_order_violation_count"),
        "internal_time_order_violation_count": record.get("internal_time_order_violation_count"),
        "boundary_time_order_violation_count": record.get("boundary_time_order_violation_count"),
        "prefix_to_middle_time_order_violation_count": record.get("prefix_to_middle_time_order_violation_count"),
        "middle_to_suffix_time_order_violation_count": record.get("middle_to_suffix_time_order_violation_count"),
        "empty_bar_rate": record.get("empty_bar_rate"),
        "pitch_analysis_coverage": record.get("pitch_analysis_coverage"),
        "rhythm_analysis_coverage": record.get("rhythm_analysis_coverage"),
        "repetition_analysis_coverage": record.get("repetition_analysis_coverage"),
        "most_common_pitch_ratio": record.get("most_common_pitch_ratio"),
        "longest_same_pitch_run_ratio": record.get("longest_same_pitch_run_ratio"),
        "pitch_diversity_score": record.get("pitch_diversity_score"),
        "onset_position_entropy": record.get("onset_position_entropy"),
        "bar_start_onset_ratio": record.get("bar_start_onset_ratio"),
        "strong_beat_onset_ratio": record.get("strong_beat_onset_ratio"),
        "duration_diversity_score": record.get("duration_diversity_score"),
        "rhythm_diversity_score": record.get("rhythm_diversity_score"),
        "event_ngram_repeat_ratio": record.get("event_ngram_repeat_ratio"),
        "rhythm_ngram_repeat_ratio": record.get("rhythm_ngram_repeat_ratio"),
    }
    if task == "continuation":
        payload["target_tokens"] = list(case["continuation_case"]["target_tokens"])
    else:
        payload["target_hole_tokens"] = list(case["infilling_case"]["target_hole_tokens"])
    return payload


def _build_sample_capture_manifest(*, fast_manifest: dict[str, Any], case_count: int) -> tuple[dict[str, Any], set[int]]:
    """基于 fast manifest 选出用于样本导出的固定 case 子集。"""

    from src.utils.benchmarking import select_export_cases

    export_cases = select_export_cases(fast_manifest["cases"], count=int(case_count))
    export_row_ids = {int(case["row_id"]) for case in export_cases}
    sample_manifest = {
        **fast_manifest,
        "case_count": len(export_cases),
        "cases": export_cases,
    }
    return sample_manifest, export_row_ids


def _write_sample_group(
    *,
    samples_root: Path,
    group_name: str,
    checkpoint_paths: list[Path],
    captured_by_checkpoint: dict[str, dict[str, list[dict[str, Any]]]],
    run_id: str,
    task_scope: str,
    log_prefix: str,
    extra_payload_fields: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, list[dict[str, Any]]]]]:
    """把指定 checkpoint 分组的样本写入磁盘，并返回产物索引。"""

    from src.utils.config_io import dump_json_file

    group_root = samples_root / group_name
    sample_tasks = _task_names_for_scope(task_scope)
    group_artifacts: dict[str, dict[str, str]] = {}
    group_exports: dict[str, dict[str, list[dict[str, Any]]]] = {}
    payload_extras = dict(extra_payload_fields or {})

    for ckpt_path in checkpoint_paths:
        checkpoint_name = ckpt_path.name
        print(f"[{log_prefix}][samples:{group_name}] checkpoint={checkpoint_name}")
        captured = captured_by_checkpoint.get(
            str(ckpt_path),
            {"continuation": [], "infilling": []},
        )
        checkpoint_sample_dir = group_root / ckpt_path.stem
        checkpoint_sample_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_artifacts: dict[str, str] = {}
        checkpoint_exports: dict[str, list[dict[str, Any]]] = {}
        for task_name in sample_tasks:
            sample_path = checkpoint_sample_dir / f"{task_name}.json"
            payload = {
                "run_id": run_id,
                "sample_group": group_name,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(ckpt_path),
                "task": task_name,
                **payload_extras,
                "case_count": len(captured[task_name]),
                "cases": captured[task_name],
            }
            dump_json_file(sample_path, payload, ensure_ascii=False, indent=2)
            checkpoint_artifacts[task_name] = str(sample_path)
            checkpoint_exports[task_name] = captured[task_name]
        group_artifacts[checkpoint_name] = checkpoint_artifacts
        group_exports[checkpoint_name] = checkpoint_exports

    return group_artifacts, group_exports


def _evaluate_checkpoint_on_manifest(
    *,
    ckpt_path: Path,
    manifest: dict[str, Any],
    capture_row_ids: set[int] | None,
    task_scope: str,
    token_to_id: dict[str, int],
    id_to_token: list[str],
    grammar_fsm,
    training_metrics_payload: dict[str, Any],
    args: argparse.Namespace,
    fallback_model_config_path: Path,
    torch,
    DecoderConfig,
    DecoderForCausalLM,
    load_checkpoint_fn,
    autocast_context_fn,
    resolve_precision_fn,
    resolve_torch_device_fn,
    generate_continuation_tokens_fn,
    build_continuation_trace_fn,
    generate_middle_tokens_fn,
    build_infilling_trace_fn,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    # 单个 checkpoint 的完整评估单元：raw/fsm 与 continuation/infilling 都在这里统一执行。
    from src.utils.benchmarking import enrich_continuation_record, enrich_infilling_record
    from src.utils.training_metrics import training_metrics_for_step

    ckpt_payload = load_checkpoint_fn(torch, ckpt_path)
    ckpt_model_cfg = ckpt_payload.get("model_config")
    if isinstance(ckpt_model_cfg, dict):
        config = DecoderConfig.from_dict(ckpt_model_cfg)
    else:
        config = DecoderConfig.from_yaml(fallback_model_config_path.resolve())

    device = resolve_torch_device_fn(torch, args.device)
    precision_name, use_amp, amp_dtype, _ = resolve_precision_fn(torch_mod=torch, requested=args.precision, device=device)
    model = DecoderForCausalLM(config).to(device)
    model.load_state_dict(ckpt_payload["model_state_dict"])
    model.eval()

    step = int(ckpt_payload.get("step", -1))
    training_metrics = training_metrics_for_step(training_metrics_payload, step)
    checkpoint_name = ckpt_path.name
    capture_samples = {"continuation": [], "infilling": []}
    run_continuation = task_scope in ("all", "continuation")
    run_infilling = task_scope in ("all", "infilling")

    continuation_attempted = 0
    continuation_stop_success = 0
    continuation_structural_valid = 0
    continuation_time_order_valid = 0
    continuation_budget_stop = 0
    continuation_append_eos_recoverable = 0
    continuation_first_event_hits = 0
    continuation_first_event_total = 0
    continuation_empty_bar_rates: list[float] = []
    continuation_low_density_rates: list[float] = []
    continuation_multi_empty_runs = 0
    continuation_missing_eos = 0
    continuation_syntax_invalid = 0
    continuation_bar_deltas: list[float] = []
    continuation_event_deltas: list[float] = []
    continuation_pitch_span_deltas: list[float] = []
    continuation_onset_position_l1: list[float] = []
    continuation_duration_l1: list[float] = []
    continuation_most_common_pitch_ratios: list[float] = []
    continuation_longest_same_pitch_run_ratios: list[float] = []
    continuation_pitch_diversity_scores: list[float] = []
    continuation_pitch_collapse_valid = 0
    continuation_onset_position_entropies: list[float] = []
    continuation_bar_start_onset_ratios: list[float] = []
    continuation_strong_beat_onset_ratios: list[float] = []
    continuation_duration_diversity_scores: list[float] = []
    continuation_rhythm_diversity_scores: list[float] = []
    continuation_rhythm_metric_valid = 0
    continuation_event_ngram_repeat_ratios: list[float] = []
    continuation_rhythm_ngram_repeat_ratios: list[float] = []
    continuation_repetition_metric_valid = 0
    continuation_failure_reason_counts: Counter[str] = Counter()
    continuation_syntax_reason_counts: Counter[str] = Counter()
    fsm_cont_structural_valid = 0
    fsm_cont_time_order_valid = 0
    fsm_illegal_top1_count = 0
    fsm_mask_intervention_count = 0
    fsm_decoding_step_count = 0
    fsm_dead_end_count = 0
    fsm_legal_mass_sum = 0.0

    infilling_attempted = 0
    infilling_structural_valid = 0
    infilling_time_order_valid = 0
    infilling_internal_time_order_valid = 0
    infilling_boundary_time_order_valid = 0
    infilling_syntax_invalid = 0
    infilling_onset_position_l1: list[float] = []
    infilling_most_common_pitch_ratios: list[float] = []
    infilling_longest_same_pitch_run_ratios: list[float] = []
    infilling_pitch_diversity_scores: list[float] = []
    infilling_pitch_collapse_valid = 0
    infilling_onset_position_entropies: list[float] = []
    infilling_bar_start_onset_ratios: list[float] = []
    infilling_strong_beat_onset_ratios: list[float] = []
    infilling_duration_diversity_scores: list[float] = []
    infilling_rhythm_diversity_scores: list[float] = []
    infilling_rhythm_metric_valid = 0
    infilling_event_ngram_repeat_ratios: list[float] = []
    infilling_rhythm_ngram_repeat_ratios: list[float] = []
    infilling_repetition_metric_valid = 0
    infilling_failure_reason_counts: Counter[str] = Counter()
    infilling_syntax_reason_counts: Counter[str] = Counter()
    fsm_infill_structural_valid = 0
    fsm_infill_time_order_valid = 0

    try:
        for case in manifest["cases"]:
            if run_continuation:
                continuation_case = dict(case["continuation_case"])
                prompt_tokens = list(continuation_case["prompt_tokens"])
                target_tokens = list(continuation_case["target_tokens"])
                if len(prompt_tokens) >= int(config.max_position_embeddings):
                    raise ValueError(
                        f"Benchmark prompt exceeds model context for {checkpoint_name}: row_id={case['row_id']}"
                    )

                dyn_max_new = min(int(args.max_new_tokens), int(config.max_position_embeddings) - len(prompt_tokens))
                raw_generated_tokens, raw_reached_eos, _ = generate_continuation_tokens_fn(
                    model=model,
                    torch_mod=torch,
                    prompt_tokens=prompt_tokens,
                    token_to_id=token_to_id,
                    id_to_token=id_to_token,
                    grammar_fsm=None,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    autocast_context_fn=autocast_context_fn,
                    max_positions=int(config.max_position_embeddings),
                    max_new_tokens=dyn_max_new,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                )
                fsm_generated_tokens, fsm_reached_eos, fsm_stats = generate_continuation_tokens_fn(
                    model=model,
                    torch_mod=torch,
                    prompt_tokens=prompt_tokens,
                    token_to_id=token_to_id,
                    id_to_token=id_to_token,
                    grammar_fsm=grammar_fsm,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    autocast_context_fn=autocast_context_fn,
                    max_positions=int(config.max_position_embeddings),
                    max_new_tokens=dyn_max_new,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                )
                raw_record = build_continuation_trace_fn(
                    prompt_tokens=prompt_tokens,
                    target_tokens=target_tokens,
                    generated_tokens=raw_generated_tokens,
                    reached_eos=raw_reached_eos,
                    source_tokens=continuation_case["window_tokens"],
                    grammar_fsm=grammar_fsm,
                    extra_fields={
                        "budget_stop": (not raw_reached_eos and len(raw_generated_tokens) >= dyn_max_new),
                        "auto_closed_with_eos": False,
                    },
                )
                fsm_record = build_continuation_trace_fn(
                    prompt_tokens=prompt_tokens,
                    target_tokens=target_tokens,
                    generated_tokens=fsm_generated_tokens,
                    reached_eos=fsm_reached_eos,
                    source_tokens=continuation_case["window_tokens"],
                    grammar_fsm=grammar_fsm,
                    extra_fields={
                        "decoding_step_count": int(fsm_stats["step_count"]),
                        "illegal_top1_count": int(fsm_stats["illegal_top1_count"]),
                        "mask_intervention_count": int(fsm_stats["mask_intervention_count"]),
                        "legal_mass_mean": (
                            float(fsm_stats["legal_mass_sum"]) / float(fsm_stats["step_count"])
                            if int(fsm_stats["step_count"]) > 0
                            else float("nan")
                        ),
                        "dead_end_count": int(fsm_stats["dead_end_count"]),
                        "budget_stop": (not fsm_reached_eos and len(fsm_generated_tokens) >= dyn_max_new),
                        "auto_closed_with_eos": bool(int(fsm_stats.get("auto_close_count", 0)) > 0),
                    },
                )
                raw_record = enrich_continuation_record(raw_record, target_tokens=target_tokens)
                fsm_record = enrich_continuation_record(fsm_record, target_tokens=target_tokens)

                continuation_attempted += 1
                continuation_stop_success += int(bool(raw_record["stop_success"]))
                continuation_structural_valid += int(bool(raw_record["structural_match_without_eos"]))
                continuation_time_order_valid += int(bool(raw_record["time_order_valid"]))
                continuation_budget_stop += int(bool(raw_record["budget_stop"]))
                continuation_append_eos_recoverable += int(
                    (not bool(raw_record["reached_eos"])) and bool(raw_record["append_eos_would_validate"])
                )
                continuation_missing_eos += int(not bool(raw_record["reached_eos"]))
                continuation_syntax_invalid += int(not bool(raw_record["append_eos_would_validate"]))
                if raw_record.get("first_unit_match") is not None:
                    continuation_first_event_total += 1
                    continuation_first_event_hits += int(bool(raw_record["first_unit_match"]))
                continuation_empty_bar_rates.append(float(raw_record["empty_bar_rate"]))
                continuation_low_density_rates.append(float(raw_record["low_density_bar_rate"]))
                continuation_multi_empty_runs += int(bool(raw_record["has_multi_empty_bar_run"]))
                continuation_bar_deltas.append(float(raw_record["generated_bar_delta"]))
                continuation_event_deltas.append(float(raw_record["generated_event_delta"]))
                continuation_pitch_span_deltas.append(float(raw_record["pitch_span_delta"]))
                continuation_onset_position_l1.append(float(raw_record["onset_position_l1_distance"]))
                continuation_duration_l1.append(float(raw_record["duration_bin_l1_distance"]))
                pitch_metric_present = False
                pitch_metric_present = _append_if_finite(
                    continuation_most_common_pitch_ratios,
                    raw_record.get("most_common_pitch_ratio"),
                ) or pitch_metric_present
                pitch_metric_present = _append_if_finite(
                    continuation_longest_same_pitch_run_ratios,
                    raw_record.get("longest_same_pitch_run_ratio"),
                ) or pitch_metric_present
                pitch_metric_present = _append_if_finite(
                    continuation_pitch_diversity_scores,
                    raw_record.get("pitch_diversity_score"),
                ) or pitch_metric_present
                continuation_pitch_collapse_valid += int(pitch_metric_present)
                rhythm_metric_present = False
                rhythm_metric_present = _append_if_finite(
                    continuation_onset_position_entropies,
                    raw_record.get("onset_position_entropy"),
                ) or rhythm_metric_present
                rhythm_metric_present = _append_if_finite(
                    continuation_bar_start_onset_ratios,
                    raw_record.get("bar_start_onset_ratio"),
                ) or rhythm_metric_present
                rhythm_metric_present = _append_if_finite(
                    continuation_strong_beat_onset_ratios,
                    raw_record.get("strong_beat_onset_ratio"),
                ) or rhythm_metric_present
                rhythm_metric_present = _append_if_finite(
                    continuation_duration_diversity_scores,
                    raw_record.get("duration_diversity_score"),
                ) or rhythm_metric_present
                rhythm_metric_present = _append_if_finite(
                    continuation_rhythm_diversity_scores,
                    raw_record.get("rhythm_diversity_score"),
                ) or rhythm_metric_present
                continuation_rhythm_metric_valid += int(rhythm_metric_present)
                repetition_metric_present = False
                repetition_metric_present = _append_if_finite(
                    continuation_event_ngram_repeat_ratios,
                    raw_record.get("event_ngram_repeat_ratio"),
                ) or repetition_metric_present
                repetition_metric_present = _append_if_finite(
                    continuation_rhythm_ngram_repeat_ratios,
                    raw_record.get("rhythm_ngram_repeat_ratio"),
                ) or repetition_metric_present
                continuation_repetition_metric_valid += int(repetition_metric_present)
                continuation_failure_reason_counts[str(raw_record["failure_reason"])] += 1
                continuation_syntax_reason_counts[str(raw_record["syntax_reason"])] += 1

                fsm_cont_structural_valid += int(bool(fsm_record["structural_match_without_eos"]))
                fsm_cont_time_order_valid += int(bool(fsm_record["time_order_valid"]))
                fsm_illegal_top1_count += int(fsm_stats["illegal_top1_count"])
                fsm_mask_intervention_count += int(fsm_stats["mask_intervention_count"])
                fsm_decoding_step_count += int(fsm_stats["step_count"])
                fsm_dead_end_count += int(fsm_stats["dead_end_count"])
                fsm_legal_mass_sum += float(fsm_stats["legal_mass_sum"])

                if capture_row_ids is not None and int(case["row_id"]) in capture_row_ids:
                    # 只抓固定 case 作为样本产物，避免为了导出样本再多跑一轮 benchmark。
                    capture_samples["continuation"].append(
                        _case_sample_payload(case, raw_record, fsm_record, task="continuation")
                    )

            if run_infilling:
                infilling_case = dict(case["infilling_case"])
                prompt_tokens = list(infilling_case["prompt_tokens"])
                prefix_tokens = list(infilling_case["prefix_tokens"])
                suffix_tokens = list(infilling_case["suffix_tokens"])
                target_hole_tokens = list(infilling_case["target_hole_tokens"])
                if len(prompt_tokens) >= int(config.max_position_embeddings):
                    raise ValueError(
                        f"Benchmark infilling prompt exceeds model context for {checkpoint_name}: row_id={case['row_id']}"
                    )

                dyn_max_new = min(int(args.max_new_tokens), int(config.max_position_embeddings) - len(prompt_tokens))
                raw_middle_tokens, raw_reached_eos, _ = generate_middle_tokens_fn(
                    model=model,
                    torch_mod=torch,
                    prompt_tokens=prompt_tokens,
                    token_to_id=token_to_id,
                    id_to_token=id_to_token,
                    grammar_fsm=None,
                    prefix_tokens=prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    autocast_context_fn=autocast_context_fn,
                    max_positions=int(config.max_position_embeddings),
                    max_new_tokens=dyn_max_new,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                )
                fsm_middle_tokens, fsm_reached_eos, fsm_infill_stats = generate_middle_tokens_fn(
                    model=model,
                    torch_mod=torch,
                    prompt_tokens=prompt_tokens,
                    token_to_id=token_to_id,
                    id_to_token=id_to_token,
                    grammar_fsm=grammar_fsm,
                    prefix_tokens=prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    autocast_context_fn=autocast_context_fn,
                    max_positions=int(config.max_position_embeddings),
                    max_new_tokens=dyn_max_new,
                    temperature=float(getattr(args, "temperature", 0.0)),
                    top_p=float(getattr(args, "top_p", 1.0)),
                )
                raw_infill_record = build_infilling_trace_fn(
                    prefix_tokens=prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    generated_middle_tokens=raw_middle_tokens,
                    reached_eos=raw_reached_eos,
                    prompt_tokens=prompt_tokens,
                    source_tokens=infilling_case["window_tokens"],
                    grammar_fsm=grammar_fsm,
                )
                fsm_infill_record = build_infilling_trace_fn(
                    prefix_tokens=prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    generated_middle_tokens=fsm_middle_tokens,
                    reached_eos=fsm_reached_eos,
                    prompt_tokens=prompt_tokens,
                    source_tokens=infilling_case["window_tokens"],
                    grammar_fsm=grammar_fsm,
                    extra_fields={
                        "decoding_step_count": int(fsm_infill_stats["step_count"]),
                        "illegal_top1_count": int(fsm_infill_stats["illegal_top1_count"]),
                        "mask_intervention_count": int(fsm_infill_stats["mask_intervention_count"]),
                        "legal_mass_mean": (
                            float(fsm_infill_stats["legal_mass_sum"]) / float(fsm_infill_stats["step_count"])
                            if int(fsm_infill_stats["step_count"]) > 0
                            else float("nan")
                        ),
                        "dead_end_count": int(fsm_infill_stats["dead_end_count"]),
                    },
                )
                raw_infill_record = enrich_infilling_record(raw_infill_record, target_hole_tokens=target_hole_tokens)
                fsm_infill_record = enrich_infilling_record(fsm_infill_record, target_hole_tokens=target_hole_tokens)

                infilling_attempted += 1
                infilling_structural_valid += int(bool(raw_infill_record["is_structurally_valid"]))
                infilling_time_order_valid += int(bool(raw_infill_record["time_order_valid"]))
                infilling_internal_time_order_valid += int(bool(raw_infill_record["internal_time_order_valid"]))
                infilling_boundary_time_order_valid += int(bool(raw_infill_record["boundary_time_order_valid"]))
                infilling_syntax_invalid += int(not bool(raw_infill_record["is_structurally_valid"]))
                infilling_onset_position_l1.append(float(raw_infill_record["onset_position_l1_distance"]))
                pitch_metric_present = False
                pitch_metric_present = _append_if_finite(
                    infilling_most_common_pitch_ratios,
                    raw_infill_record.get("most_common_pitch_ratio"),
                ) or pitch_metric_present
                pitch_metric_present = _append_if_finite(
                    infilling_longest_same_pitch_run_ratios,
                    raw_infill_record.get("longest_same_pitch_run_ratio"),
                ) or pitch_metric_present
                pitch_metric_present = _append_if_finite(
                    infilling_pitch_diversity_scores,
                    raw_infill_record.get("pitch_diversity_score"),
                ) or pitch_metric_present
                infilling_pitch_collapse_valid += int(pitch_metric_present)
                rhythm_metric_present = False
                rhythm_metric_present = _append_if_finite(
                    infilling_onset_position_entropies,
                    raw_infill_record.get("onset_position_entropy"),
                ) or rhythm_metric_present
                rhythm_metric_present = _append_if_finite(
                    infilling_bar_start_onset_ratios,
                    raw_infill_record.get("bar_start_onset_ratio"),
                ) or rhythm_metric_present
                rhythm_metric_present = _append_if_finite(
                    infilling_strong_beat_onset_ratios,
                    raw_infill_record.get("strong_beat_onset_ratio"),
                ) or rhythm_metric_present
                rhythm_metric_present = _append_if_finite(
                    infilling_duration_diversity_scores,
                    raw_infill_record.get("duration_diversity_score"),
                ) or rhythm_metric_present
                rhythm_metric_present = _append_if_finite(
                    infilling_rhythm_diversity_scores,
                    raw_infill_record.get("rhythm_diversity_score"),
                ) or rhythm_metric_present
                infilling_rhythm_metric_valid += int(rhythm_metric_present)
                repetition_metric_present = False
                repetition_metric_present = _append_if_finite(
                    infilling_event_ngram_repeat_ratios,
                    raw_infill_record.get("event_ngram_repeat_ratio"),
                ) or repetition_metric_present
                repetition_metric_present = _append_if_finite(
                    infilling_rhythm_ngram_repeat_ratios,
                    raw_infill_record.get("rhythm_ngram_repeat_ratio"),
                ) or repetition_metric_present
                infilling_repetition_metric_valid += int(repetition_metric_present)
                infilling_failure_reason_counts[str(raw_infill_record["failure_reason"])] += 1
                infilling_syntax_reason_counts[str(raw_infill_record["syntax_reason"])] += 1
                fsm_infill_structural_valid += int(bool(fsm_infill_record["is_structurally_valid"]))
                fsm_infill_time_order_valid += int(bool(fsm_infill_record["time_order_valid"]))
                fsm_illegal_top1_count += int(fsm_infill_stats["illegal_top1_count"])
                fsm_mask_intervention_count += int(fsm_infill_stats["mask_intervention_count"])
                fsm_decoding_step_count += int(fsm_infill_stats["step_count"])
                fsm_dead_end_count += int(fsm_infill_stats["dead_end_count"])
                fsm_legal_mass_sum += float(fsm_infill_stats["legal_mass_sum"])

                if capture_row_ids is not None and int(case["row_id"]) in capture_row_ids:
                    # infilling 样本同样直接在主评估过程中顺手采集。
                    capture_samples["infilling"].append(
                        _case_sample_payload(case, raw_infill_record, fsm_infill_record, task="infilling")
                    )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selection_metrics = {
        "continuation_stop_success_rate": _safe_rate(continuation_stop_success, continuation_attempted),
        "continuation_structural_validity_rate": _safe_rate(continuation_structural_valid, continuation_attempted),
        "continuation_budget_stop_rate": _safe_rate(continuation_budget_stop, continuation_attempted),
        "continuation_time_order_validity_rate": _safe_rate(continuation_time_order_valid, continuation_attempted),
        "continuation_empty_bar_rate": _safe_mean(continuation_empty_bar_rates),
        "infilling_structural_validity_rate": _safe_rate(infilling_structural_valid, infilling_attempted),
        "infilling_time_order_validity_rate": _safe_rate(infilling_time_order_valid, infilling_attempted),
        "valid_loss_from_training": training_metrics.get("valid_loss_from_training"),
    }
    overall_pitch_metric_count = (
        len(continuation_most_common_pitch_ratios) + len(infilling_most_common_pitch_ratios)
    )
    diagnostic_metrics = {
        "continuation_first_event_hit_rate": _safe_rate(continuation_first_event_hits, continuation_first_event_total),
        "continuation_missing_eos_rate": _safe_rate(continuation_missing_eos, continuation_attempted),
        "continuation_syntax_invalid_rate": _safe_rate(continuation_syntax_invalid, continuation_attempted),
        "infilling_syntax_invalid_rate": _safe_rate(infilling_syntax_invalid, infilling_attempted),
        "infilling_internal_time_order_validity_rate": _safe_rate(
            infilling_internal_time_order_valid,
            infilling_attempted,
        ),
        "infilling_boundary_time_order_validity_rate": _safe_rate(
            infilling_boundary_time_order_valid,
            infilling_attempted,
        ),
        "append_eos_recoverable_rate": _safe_rate(continuation_append_eos_recoverable, continuation_attempted),
        "low_density_bar_rate": _safe_mean(continuation_low_density_rates),
        "multi_empty_bar_run_rate": _safe_rate(continuation_multi_empty_runs, continuation_attempted),
        "generated_bar_delta_mean": _safe_mean(continuation_bar_deltas),
        "generated_event_delta_mean": _safe_mean(continuation_event_deltas),
        "pitch_span_delta_mean": _safe_mean(continuation_pitch_span_deltas),
        "continuation_onset_position_l1_distance": _safe_mean(continuation_onset_position_l1),
        "infilling_onset_position_l1_distance": _safe_mean(infilling_onset_position_l1),
        "overall_onset_position_l1_distance": _safe_mean(continuation_onset_position_l1 + infilling_onset_position_l1),
        "duration_bin_l1_distance": _safe_mean(continuation_duration_l1),
        "continuation_most_common_pitch_ratio": _safe_mean(continuation_most_common_pitch_ratios),
        "continuation_longest_same_pitch_run_ratio": _safe_mean(continuation_longest_same_pitch_run_ratios),
        "continuation_pitch_diversity_score": _safe_mean(continuation_pitch_diversity_scores),
        "continuation_pitch_collapse_coverage": _safe_rate(continuation_pitch_collapse_valid, continuation_attempted),
        "continuation_onset_position_entropy": _safe_mean(continuation_onset_position_entropies),
        "continuation_bar_start_onset_ratio": _safe_mean(continuation_bar_start_onset_ratios),
        "continuation_strong_beat_onset_ratio": _safe_mean(continuation_strong_beat_onset_ratios),
        "continuation_duration_diversity_score": _safe_mean(continuation_duration_diversity_scores),
        "continuation_rhythm_diversity_score": _safe_mean(continuation_rhythm_diversity_scores),
        "continuation_rhythm_metric_coverage": _safe_rate(continuation_rhythm_metric_valid, continuation_attempted),
        "continuation_event_ngram_repeat_ratio": _safe_mean(continuation_event_ngram_repeat_ratios),
        "continuation_rhythm_ngram_repeat_ratio": _safe_mean(continuation_rhythm_ngram_repeat_ratios),
        "continuation_repetition_metric_coverage": _safe_rate(
            continuation_repetition_metric_valid,
            continuation_attempted,
        ),
        "infilling_most_common_pitch_ratio": _safe_mean(infilling_most_common_pitch_ratios),
        "infilling_longest_same_pitch_run_ratio": _safe_mean(infilling_longest_same_pitch_run_ratios),
        "infilling_pitch_diversity_score": _safe_mean(infilling_pitch_diversity_scores),
        "infilling_pitch_collapse_coverage": _safe_rate(infilling_pitch_collapse_valid, infilling_attempted),
        "infilling_onset_position_entropy": _safe_mean(infilling_onset_position_entropies),
        "infilling_bar_start_onset_ratio": _safe_mean(infilling_bar_start_onset_ratios),
        "infilling_strong_beat_onset_ratio": _safe_mean(infilling_strong_beat_onset_ratios),
        "infilling_duration_diversity_score": _safe_mean(infilling_duration_diversity_scores),
        "infilling_rhythm_diversity_score": _safe_mean(infilling_rhythm_diversity_scores),
        "infilling_rhythm_metric_coverage": _safe_rate(infilling_rhythm_metric_valid, infilling_attempted),
        "infilling_event_ngram_repeat_ratio": _safe_mean(infilling_event_ngram_repeat_ratios),
        "infilling_rhythm_ngram_repeat_ratio": _safe_mean(infilling_rhythm_ngram_repeat_ratios),
        "infilling_repetition_metric_coverage": _safe_rate(
            infilling_repetition_metric_valid,
            infilling_attempted,
        ),
        "overall_most_common_pitch_ratio": _safe_mean(
            continuation_most_common_pitch_ratios + infilling_most_common_pitch_ratios
        ),
        "overall_longest_same_pitch_run_ratio": _safe_mean(
            continuation_longest_same_pitch_run_ratios + infilling_longest_same_pitch_run_ratios
        ),
        "overall_pitch_diversity_score": _safe_mean(
            continuation_pitch_diversity_scores + infilling_pitch_diversity_scores
        ),
        "overall_onset_position_entropy": _safe_mean(
            continuation_onset_position_entropies + infilling_onset_position_entropies
        ),
        "overall_bar_start_onset_ratio": _safe_mean(
            continuation_bar_start_onset_ratios + infilling_bar_start_onset_ratios
        ),
        "overall_strong_beat_onset_ratio": _safe_mean(
            continuation_strong_beat_onset_ratios + infilling_strong_beat_onset_ratios
        ),
        "overall_duration_diversity_score": _safe_mean(
            continuation_duration_diversity_scores + infilling_duration_diversity_scores
        ),
        "overall_rhythm_diversity_score": _safe_mean(
            continuation_rhythm_diversity_scores + infilling_rhythm_diversity_scores
        ),
        "overall_rhythm_metric_coverage": (
            (
                float(continuation_rhythm_metric_valid + infilling_rhythm_metric_valid)
                / float(continuation_attempted + infilling_attempted)
            )
            if (continuation_attempted + infilling_attempted) > 0
            else float("nan")
        ),
        "overall_event_ngram_repeat_ratio": _safe_mean(
            continuation_event_ngram_repeat_ratios + infilling_event_ngram_repeat_ratios
        ),
        "overall_rhythm_ngram_repeat_ratio": _safe_mean(
            continuation_rhythm_ngram_repeat_ratios + infilling_rhythm_ngram_repeat_ratios
        ),
        "overall_repetition_metric_coverage": (
            (
                float(continuation_repetition_metric_valid + infilling_repetition_metric_valid)
                / float(continuation_attempted + infilling_attempted)
            )
            if (continuation_attempted + infilling_attempted) > 0
            else float("nan")
        ),
        "overall_pitch_collapse_coverage": (
            (
                float(continuation_pitch_collapse_valid + infilling_pitch_collapse_valid)
                / float(continuation_attempted + infilling_attempted)
            )
            if (continuation_attempted + infilling_attempted) > 0
            else float("nan")
        ),
        "overall_pitch_metric_count": overall_pitch_metric_count,
        "fsm_structural_validity_rate": _safe_rate(
            fsm_cont_structural_valid + fsm_infill_structural_valid,
            continuation_attempted + infilling_attempted,
        ),
        "fsm_time_order_validity_rate": _safe_rate(
            fsm_cont_time_order_valid + fsm_infill_time_order_valid,
            continuation_attempted + infilling_attempted,
        ),
        "fsm_illegal_top1_rate": _safe_rate(fsm_illegal_top1_count, fsm_decoding_step_count),
        "fsm_mask_intervention_rate": _safe_rate(fsm_mask_intervention_count, fsm_decoding_step_count),
        "fsm_dead_end_count": fsm_dead_end_count,
        "fsm_legal_mass_mean": (
            (fsm_legal_mass_sum / float(fsm_decoding_step_count))
            if fsm_decoding_step_count > 0
            else float("nan")
        ),
        "precision": precision_name,
    }

    result = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(ckpt_path),
        "step": step,
        "task_scope": task_scope,
        "evaluation_tier": manifest["tier"],
        "training_metrics": {
            "valid_loss_from_training": training_metrics.get("valid_loss_from_training"),
            "train_loss_ema": training_metrics.get("train_loss_ema"),
            "best_valid_loss_so_far": training_metrics.get("best_valid_loss_so_far"),
            "overfit_gap": training_metrics.get("overfit_gap"),
            "tokens_seen": training_metrics.get("tokens_seen"),
        },
        "selection_metrics": selection_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "failure_reason_counts": {
            "continuation": dict(continuation_failure_reason_counts),
            "infilling": dict(infilling_failure_reason_counts),
        },
        "syntax_reason_counts": {
            "continuation": dict(continuation_syntax_reason_counts),
            "infilling": dict(infilling_syntax_reason_counts),
        },
    }
    result.update(result["training_metrics"])
    result.update(result["selection_metrics"])
    result.update(result["diagnostic_metrics"])
    return result, capture_samples


def _scoped_failure_counts(result: dict[str, Any], *, task_scope: str) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for task_name in _task_names_for_scope(task_scope):
        merged.update(result.get("failure_reason_counts", {}).get(task_name, {}))
    return dict(merged)


def _scoped_syntax_counts(result: dict[str, Any], *, task_scope: str) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for task_name in _task_names_for_scope(task_scope):
        merged.update(result.get("syntax_reason_counts", {}).get(task_name, {}))
    return dict(merged)


_PERCENT_METRICS = {
    "balanced_score",
    "balanced_score_coverage",
    "continuation_stop_success_rate",
    "continuation_budget_stop_rate",
    "continuation_structural_validity_rate",
    "continuation_time_order_validity_rate",
    "continuation_empty_bar_rate",
    "continuation_first_event_hit_rate",
    "continuation_missing_eos_rate",
    "continuation_syntax_invalid_rate",
    "infilling_structural_validity_rate",
    "infilling_time_order_validity_rate",
    "infilling_internal_time_order_validity_rate",
    "infilling_boundary_time_order_validity_rate",
    "infilling_syntax_invalid_rate",
    "append_eos_recoverable_rate",
    "low_density_bar_rate",
    "multi_empty_bar_run_rate",
    "fsm_structural_validity_rate",
    "fsm_time_order_validity_rate",
    "fsm_illegal_top1_rate",
    "fsm_mask_intervention_rate",
    "fim_ratio_mean",
    "fim_ratio_std",
}

_INTEGER_METRICS = {
    "step",
    "balanced_rank",
    "tokens_seen",
    "tokens_seen_last",
    "best_valid_step",
    "best_valid_tokens_seen",
    "plateau_eval_streak",
    "train_event_count",
    "eval_event_count",
    "fsm_dead_end_count",
    "last_train_step",
    "last_eval_step",
}

_METRIC_LABELS = {
    "checkpoint_name": "Checkpoint",
    "evaluation_tier": "评估层级",
    "balanced_rank": "排名",
    "balanced_score": "综合得分",
    "step": "step",
    "continuation_stop_success_rate": "续写成功停机率",
    "continuation_budget_stop_rate": "续写预算截断率",
    "continuation_structural_validity_rate": "续写结构合法率",
    "continuation_time_order_validity_rate": "续写时间顺序合法率",
    "continuation_empty_bar_rate": "续写空 BAR 率",
    "continuation_first_event_hit_rate": "续写首事件命中率",
    "continuation_missing_eos_rate": "续写缺失 EOS 率",
    "continuation_syntax_invalid_rate": "续写语法非法率",
    "infilling_structural_validity_rate": "补全结构合法率",
    "infilling_time_order_validity_rate": "补全时间顺序合法率",
    "infilling_internal_time_order_validity_rate": "补全内部时间顺序合法率",
    "infilling_boundary_time_order_validity_rate": "补全边界时间顺序合法率",
    "infilling_syntax_invalid_rate": "补全语法非法率",
    "valid_loss_from_training": "训练期验证 loss",
    "train_loss_ema": "训练 loss EMA",
    "best_valid_loss_so_far": "历史最优验证 loss",
    "overfit_gap": "过拟合间隙",
    "tokens_seen": "已见 token 数",
    "append_eos_recoverable_rate": "补 EOS 可恢复率",
    "low_density_bar_rate": "低密度 BAR 率",
    "multi_empty_bar_run_rate": "连续空 BAR 样本率",
    "generated_bar_delta_mean": "BAR 数偏差均值",
    "generated_event_delta_mean": "事件数偏差均值",
    "pitch_span_delta_mean": "音高跨度偏差均值",
    "duration_bin_l1_distance": "时值分布 L1 距离",
    "fsm_structural_validity_rate": "FSM 结构合法率",
    "fsm_time_order_validity_rate": "FSM 时间顺序合法率",
    "fsm_illegal_top1_rate": "FSM 非法 top1 率",
    "fsm_mask_intervention_rate": "FSM 掩码干预率",
    "fsm_dead_end_count": "FSM 死路次数",
    "fsm_legal_mass_mean": "FSM 合法质量均值",
    "latest_train_loss": "最近训练 loss",
    "latest_train_loss_ema": "最近训练 EMA",
    "latest_valid_loss": "最近验证 loss",
    "best_valid_loss": "最佳验证 loss",
    "best_valid_step": "最佳验证 step",
    "latest_overfit_gap": "最近过拟合间隙",
    "latest_valid_loss_delta": "最近两次验证 loss 变化",
    "latest_train_loss_ema_delta": "最近 EMA 变化",
    "plateau_eval_streak": "连续未刷新最佳 eval 次数",
    "tok_per_sec_mean": "吞吐均值",
    "tok_per_sec_median": "吞吐中位数",
    "fim_ratio_mean": "FIM 比例均值",
    "fim_ratio_std": "FIM 比例波动",
    "train_loss_min": "训练 loss 最小值",
    "train_loss_ema_min": "训练 EMA 最小值",
}


def _core_metric_specs(task_scope: str) -> list[tuple[str, str]]:
    if task_scope == "continuation":
        return [
            ("balanced_rank", "排名"),
            ("checkpoint_name", "Checkpoint"),
            ("evaluation_tier", "评估层级"),
            ("balanced_score", "综合得分"),
            ("continuation_stop_success_rate", "续写成功停机率"),
            ("continuation_budget_stop_rate", "续写预算截断率"),
            ("continuation_structural_validity_rate", "续写结构合法率"),
            ("continuation_time_order_validity_rate", "续写时间顺序合法率"),
            ("continuation_empty_bar_rate", "续写空 BAR 率"),
            ("valid_loss_from_training", "训练期验证 loss"),
        ]
    if task_scope == "infilling":
        return [
            ("balanced_rank", "排名"),
            ("checkpoint_name", "Checkpoint"),
            ("evaluation_tier", "评估层级"),
            ("balanced_score", "综合得分"),
            ("infilling_structural_validity_rate", "补全结构合法率"),
            ("infilling_time_order_validity_rate", "补全时间顺序合法率"),
            ("fsm_structural_validity_rate", "FSM 结构合法率"),
            ("valid_loss_from_training", "训练期验证 loss"),
        ]
    return [
        ("balanced_rank", "排名"),
        ("checkpoint_name", "Checkpoint"),
        ("evaluation_tier", "评估层级"),
        ("balanced_score", "综合得分"),
        ("continuation_stop_success_rate", "续写成功停机率"),
        ("continuation_budget_stop_rate", "续写预算截断率"),
        ("continuation_time_order_validity_rate", "续写时间顺序合法率"),
        ("infilling_structural_validity_rate", "补全结构合法率"),
        ("valid_loss_from_training", "训练期验证 loss"),
    ]


def _diagnostic_metric_specs(task_scope: str) -> list[tuple[str, str]]:
    common = [
        ("append_eos_recoverable_rate", "补 EOS 可恢复率"),
        ("low_density_bar_rate", "低密度 BAR 率"),
        ("multi_empty_bar_run_rate", "连续空 BAR 样本率"),
        ("generated_bar_delta_mean", "BAR 数偏差均值"),
        ("generated_event_delta_mean", "事件数偏差均值"),
        ("pitch_span_delta_mean", "音高跨度偏差均值"),
        ("duration_bin_l1_distance", "时值分布 L1 距离"),
        ("fsm_illegal_top1_rate", "FSM 非法 top1 率"),
        ("fsm_mask_intervention_rate", "FSM 掩码干预率"),
        ("fsm_dead_end_count", "FSM 死路次数"),
    ]
    if task_scope == "continuation":
        return [
            ("continuation_first_event_hit_rate", "续写首事件命中率"),
            ("continuation_missing_eos_rate", "续写缺失 EOS 率"),
            ("continuation_syntax_invalid_rate", "续写语法非法率"),
            *common,
        ]
    if task_scope == "infilling":
        return [
            ("infilling_syntax_invalid_rate", "补全语法非法率"),
            ("infilling_internal_time_order_validity_rate", "补全内部时间顺序合法率"),
            ("infilling_boundary_time_order_validity_rate", "补全边界时间顺序合法率"),
            ("fsm_structural_validity_rate", "FSM 结构合法率"),
            ("fsm_time_order_validity_rate", "FSM 时间顺序合法率"),
            ("fsm_illegal_top1_rate", "FSM 非法 top1 率"),
            ("fsm_mask_intervention_rate", "FSM 掩码干预率"),
            ("fsm_dead_end_count", "FSM 死路次数"),
        ]
    return [
        ("continuation_first_event_hit_rate", "续写首事件命中率"),
        ("continuation_missing_eos_rate", "续写缺失 EOS 率"),
        ("continuation_syntax_invalid_rate", "续写语法非法率"),
        ("infilling_syntax_invalid_rate", "补全语法非法率"),
        ("infilling_internal_time_order_validity_rate", "补全内部时间顺序合法率"),
        ("infilling_boundary_time_order_validity_rate", "补全边界时间顺序合法率"),
        *common,
    ]


def _training_metric_specs() -> list[tuple[str, str]]:
    return [
        ("step", "step"),
        ("valid_loss_from_training", "训练期验证 loss"),
        ("train_loss_ema", "训练 loss EMA"),
        ("best_valid_loss_so_far", "历史最优验证 loss"),
        ("overfit_gap", "过拟合间隙"),
        ("tokens_seen", "已见 token 数"),
    ]


def _plot_metric_specs(task_scope: str, *, diagnostics: bool) -> list[dict[str, Any]]:
    if diagnostics:
        if task_scope == "infilling":
            return [
                {"key": "infilling_syntax_invalid_rate", "label": "补全语法非法率", "percent": True, "goal": "min", "color": "#dc2626"},
                {"key": "fsm_structural_validity_rate", "label": "FSM 结构合法率", "percent": True, "goal": "max", "color": "#16a34a"},
                {"key": "fsm_time_order_validity_rate", "label": "FSM 时间顺序合法率", "percent": True, "goal": "max", "color": "#0f766e"},
                {"key": "fsm_illegal_top1_rate", "label": "FSM 非法 top1 率", "percent": True, "goal": "min", "color": "#7c3aed"},
                {"key": "fsm_mask_intervention_rate", "label": "FSM 掩码干预率", "percent": True, "goal": "min", "color": "#ea580c"},
            ]
        return [
            {"key": "continuation_first_event_hit_rate", "label": "续写首事件命中率", "percent": True, "goal": "max", "color": "#2563eb"},
            {"key": "continuation_missing_eos_rate", "label": "续写缺失 EOS 率", "percent": True, "goal": "min", "color": "#dc2626"},
            {"key": "low_density_bar_rate", "label": "低密度 BAR 率", "percent": True, "goal": "min", "color": "#0891b2"},
            {"key": "multi_empty_bar_run_rate", "label": "连续空 BAR 样本率", "percent": True, "goal": "min", "color": "#7c3aed"},
            {"key": "duration_bin_l1_distance", "label": "时值分布 L1 距离", "goal": "min", "color": "#ea580c"},
            {"key": "fsm_illegal_top1_rate", "label": "FSM 非法 top1 率", "percent": True, "goal": "min", "color": "#16a34a"},
        ]
    if task_scope == "continuation":
        return [
            {"key": "balanced_score", "label": "综合得分", "goal": "max", "color": "#111827"},
            {"key": "continuation_stop_success_rate", "label": "续写成功停机率", "percent": True, "goal": "max", "color": "#2563eb"},
            {"key": "continuation_budget_stop_rate", "label": "续写预算截断率", "percent": True, "goal": "min", "color": "#dc2626"},
            {"key": "continuation_structural_validity_rate", "label": "续写结构合法率", "percent": True, "goal": "max", "color": "#16a34a"},
            {"key": "continuation_time_order_validity_rate", "label": "续写时间顺序合法率", "percent": True, "goal": "max", "color": "#0891b2"},
            {"key": "valid_loss_from_training", "label": "训练期验证 loss", "goal": "min", "color": "#7c3aed"},
        ]
    if task_scope == "infilling":
        return [
            {"key": "balanced_score", "label": "综合得分", "goal": "max", "color": "#111827"},
            {"key": "infilling_structural_validity_rate", "label": "补全结构合法率", "percent": True, "goal": "max", "color": "#2563eb"},
            {"key": "infilling_time_order_validity_rate", "label": "补全时间顺序合法率", "percent": True, "goal": "max", "color": "#16a34a"},
            {"key": "fsm_structural_validity_rate", "label": "FSM 结构合法率", "percent": True, "goal": "max", "color": "#0891b2"},
            {"key": "valid_loss_from_training", "label": "训练期验证 loss", "goal": "min", "color": "#7c3aed"},
        ]
    return [
        {"key": "balanced_score", "label": "综合得分", "goal": "max", "color": "#111827"},
        {"key": "continuation_stop_success_rate", "label": "续写成功停机率", "percent": True, "goal": "max", "color": "#2563eb"},
        {"key": "continuation_budget_stop_rate", "label": "续写预算截断率", "percent": True, "goal": "min", "color": "#dc2626"},
        {"key": "continuation_time_order_validity_rate", "label": "续写时间顺序合法率", "percent": True, "goal": "max", "color": "#16a34a"},
        {"key": "infilling_structural_validity_rate", "label": "补全结构合法率", "percent": True, "goal": "max", "color": "#0891b2"},
        {"key": "valid_loss_from_training", "label": "训练期验证 loss", "goal": "min", "color": "#7c3aed"},
    ]


def _format_metric_value(value: Any, *, key: str | None = None) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    if key in _INTEGER_METRICS:
        return f"{int(round(numeric))}"
    if key in _PERCENT_METRICS:
        return f"{numeric * 100:.2f}%"
    return f"{numeric:.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_无数据_", ""]
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [
        "| " + " | ".join(headers) + " |",
        separator,
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _result_table_rows(results: list[dict[str, Any]], specs: list[tuple[str, str]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for result in results:
        row: list[str] = []
        for key, _label in specs:
            row.append(_format_metric_value(result.get(key), key=key))
        rows.append(row)
    return rows


def _counter_table_rows(counter_mapping: dict[str, int], *, limit: int = 5) -> list[list[str]]:
    items = _top_counter_items(counter_mapping, limit=limit)
    if not items:
        return [["无", "0"]]
    return [[f"`{key}`", str(value)] for key, value in items]


def _relative_artifact_path(root: Path, artifact_path: str) -> str:
    try:
        return os.path.relpath(artifact_path, start=root)
    except ValueError:
        return artifact_path


def _summary_plot_section(root: Path, plot_artifacts: dict[str, str]) -> list[str]:
    lines = ["## 图表总览", ""]
    if not plot_artifacts:
        return [*lines, "_未生成图表_", ""]
    for label, artifact_path in plot_artifacts.items():
        relative_path = _relative_artifact_path(root, artifact_path)
        lines.append(f"### {label}")
        lines.append(f"![{label}]({relative_path})")
        lines.append("")
    return lines


def _sample_success_key(task_name: str) -> str:
    if task_name == "continuation":
        return "stop_success"
    return "time_order_valid"


def _sample_group_artifacts(
    group_mapping: dict[str, dict[str, dict[str, str]]],
    group_name: str,
) -> dict[str, dict[str, str]]:
    return dict(group_mapping.get(group_name, {}))


def _sample_group_exports(
    group_mapping: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    group_name: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return dict(group_mapping.get(group_name, {}))


def _build_summary_markdown(
    *,
    run_id: str,
    task_scope: str,
    benchmark_root: Path,
    recommended: dict[str, Any] | None,
    top_results: list[dict[str, Any]],
    training_summary: dict[str, Any],
    plot_artifacts: dict[str, str],
    sample_artifacts: dict[str, dict[str, dict[str, str]]],
    exported_samples: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    manifest_stats: dict[str, Any],
    checkpoint_prefilter: dict[str, Any],
) -> str:
    final_sample_artifacts = _sample_group_artifacts(sample_artifacts, "final_top3")
    final_exported_samples = _sample_group_exports(exported_samples, "final_top3")
    formal_candidate_artifacts = _sample_group_artifacts(sample_artifacts, "formal_candidates")
    lines = [f"# {_TASK_TITLES[task_scope]}: {run_id}", ""]
    if recommended is None:
        lines.extend(["没有 checkpoint 通过当前 benchmark 流程。", ""])
    else:
        lines.extend(
            [
                "## 最终推荐",
                f"- 推荐 checkpoint：`{recommended.get('checkpoint_name')}`",
                f"- step：{_format_metric_value(recommended.get('step'), key='step')}",
                f"- 综合得分：{_format_metric_value(recommended.get('balanced_score'), key='balanced_score')}",
                f"- 评估层级：{recommended.get('evaluation_tier')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Benchmark 概览",
            f"- fast 集样本数：{manifest_stats.get('fast_case_count')}",
            f"- formal 集样本数：{manifest_stats.get('formal_case_count')}",
            f"- 复评 checkpoint 数：{manifest_stats.get('candidate_count')}",
            (
                f"- checkpoint 预筛：训练期 valid_loss top "
                f"{checkpoint_prefilter.get('requested_top_k', 0)}，"
                f"另保留最早 {checkpoint_prefilter.get('preserve_earliest', 0)} 个 eval 点，"
                f"实际保留 {checkpoint_prefilter.get('selected_count', 0)} / {checkpoint_prefilter.get('original_count', 0)}"
                if checkpoint_prefilter.get("enabled")
                else "- checkpoint 预筛：关闭"
            ),
            "- samples 默认导出到 `samples/final_top3/`，与最终 Top 3 保持一致",
            (
                "- `samples/formal_candidates/` 保留进入 formal 复评的候选 checkpoint，便于排查"
                if formal_candidate_artifacts
                else "- 未导出 formal_candidates 样本"
            ),
            "",
            "## Top 3 排行",
        ]
    )
    lines.extend(_markdown_table([label for _key, label in _core_metric_specs(task_scope)], _result_table_rows(top_results, _core_metric_specs(task_scope))))

    lines.extend(_summary_plot_section(benchmark_root, plot_artifacts))

    lines.extend(["## 训练健康度摘要", ""])
    training_summary_specs = [
        ("last_train_step", "最后训练 step"),
        ("last_eval_step", "最后验证 step"),
        ("tokens_seen_last", "已见 token 数"),
        ("latest_train_loss_ema", "最近训练 EMA"),
        ("latest_valid_loss", "最近验证 loss"),
        ("best_valid_loss", "最佳验证 loss"),
        ("best_valid_step", "最佳验证 step"),
        ("latest_overfit_gap", "最近过拟合间隙"),
        ("latest_valid_loss_delta", "最近两次验证 loss 变化"),
        ("plateau_eval_streak", "连续未刷新最佳 eval 次数"),
        ("tok_per_sec_median", "吞吐中位数"),
        ("fim_ratio_mean", "FIM 比例均值"),
    ]
    lines.extend(
        _markdown_table(
            ["指标", "值"],
            [[label, _format_metric_value(training_summary.get(key), key=key)] for key, label in training_summary_specs],
        )
    )

    if recommended is not None:
        gate_rows = []
        for metric_key, payload in sorted(recommended.get("gate_details", {}).items()):
            gate_rows.append(
                [
                    _METRIC_LABELS.get(metric_key, metric_key),
                    str(payload.get("goal", "")),
                    _format_metric_value(payload.get("threshold"), key=metric_key),
                    _format_metric_value(payload.get("value"), key=metric_key),
                    _format_metric_value(payload.get("passed")),
                ]
            )
        lines.extend(["## 最佳 Checkpoint 门槛检查", ""])
        lines.extend(_markdown_table(["指标", "方向", "阈值", "实际值", "是否通过"], gate_rows))

        score_rows = []
        for metric_key, payload in recommended.get("score_breakdown", {}).items():
            score_rows.append(
                [
                    _METRIC_LABELS.get(metric_key, metric_key),
                    str(payload.get("goal", "")),
                    _format_metric_value(payload.get("weight")),
                    _format_metric_value(payload.get("value"), key=metric_key),
                    _format_metric_value(payload.get("rank_score")),
                    _format_metric_value(payload.get("weighted_contribution")),
                ]
            )
        lines.extend(["## 最佳 Checkpoint 计分拆解", ""])
        lines.extend(_markdown_table(["指标", "方向", "权重", "实际值", "排序分", "加权贡献"], score_rows))

    lines.extend(["## Top 3 核心指标详表", ""])
    lines.extend(_markdown_table([label for _key, label in _core_metric_specs(task_scope)], _result_table_rows(top_results, _core_metric_specs(task_scope))))
    lines.extend(["## Top 3 诊断指标详表", ""])
    lines.extend(
        _markdown_table(
            [label for _key, label in _diagnostic_metric_specs(task_scope)],
            _result_table_rows(top_results, _diagnostic_metric_specs(task_scope)),
        )
    )
    lines.extend(["## Top 3 训练期指标详表", ""])
    lines.extend(_markdown_table([label for _key, label in _training_metric_specs()], _result_table_rows(top_results, _training_metric_specs())))

    for result in top_results:
        checkpoint_name = str(result.get("checkpoint_name"))
        lines.append(f"## {checkpoint_name}")
        failure_items = _top_counter_items(_scoped_failure_counts(result, task_scope=task_scope))
        syntax_items = _top_counter_items(_scoped_syntax_counts(result, task_scope=task_scope))
        lines.append("### 高频失败模式")
        lines.extend(_markdown_table(["失败原因", "次数"], _counter_table_rows(_scoped_failure_counts(result, task_scope=task_scope))))
        lines.append("### 高频语法原因")
        lines.extend(_markdown_table(["语法原因", "次数"], _counter_table_rows(_scoped_syntax_counts(result, task_scope=task_scope))))
        artifact_paths = final_sample_artifacts.get(checkpoint_name, {})
        lines.append("### 样本产物")
        for task_name in _task_names_for_scope(task_scope):
            if task_name in artifact_paths:
                relative_path = _relative_artifact_path(benchmark_root, artifact_paths.get(task_name, ""))
                lines.append(f"- `{task_name}` 样本文件：`{relative_path}`")

        checkpoint_samples = final_exported_samples.get(checkpoint_name, {})
        lines.append("### 代表样本")
        for task_name in _task_names_for_scope(task_scope):
            task_samples = checkpoint_samples.get(task_name, [])
            success_key = _sample_success_key(task_name)
            success = next((item for item in task_samples if bool(item.get(success_key))), None)
            failure = next((item for item in task_samples if not bool(item.get(success_key))), None)
            if success is not None:
                lines.append(
                    f"- `{task_name}` 成功样本：row_id={success.get('row_id')} "
                    f"{success.get('meta', {}).get('artist')} - {success.get('meta', {}).get('title')} | "
                    f"输出预览={_sample_preview(success.get('raw_output_tokens', []))}"
                )
            if failure is not None:
                lines.append(
                    f"- `{task_name}` 失败样本：row_id={failure.get('row_id')} "
                    f"{failure.get('meta', {}).get('artist')} - {failure.get('meta', {}).get('title')} | "
                    f"原因={failure.get('raw_failure_reason')} / {failure.get('raw_syntax_reason')} | "
                    f"输出预览={_sample_preview(failure.get('raw_output_tokens', []))}"
                )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


_PERCENT_METRICS_V2 = {
    "balanced_score",
    "balanced_score_coverage",
    "continuation_stop_success_rate",
    "continuation_budget_stop_rate",
    "continuation_structural_validity_rate",
    "continuation_time_order_validity_rate",
    "continuation_empty_bar_rate",
    "continuation_first_event_hit_rate",
    "continuation_missing_eos_rate",
    "continuation_syntax_invalid_rate",
    "infilling_structural_validity_rate",
    "infilling_time_order_validity_rate",
    "infilling_internal_time_order_validity_rate",
    "infilling_boundary_time_order_validity_rate",
    "infilling_syntax_invalid_rate",
    "append_eos_recoverable_rate",
    "low_density_bar_rate",
    "multi_empty_bar_run_rate",
    "fsm_structural_validity_rate",
    "fsm_time_order_validity_rate",
    "fsm_illegal_top1_rate",
    "fsm_mask_intervention_rate",
    "continuation_bar_start_onset_ratio",
    "continuation_strong_beat_onset_ratio",
    "continuation_rhythm_metric_coverage",
    "continuation_event_ngram_repeat_ratio",
    "continuation_rhythm_ngram_repeat_ratio",
    "continuation_repetition_metric_coverage",
    "infilling_bar_start_onset_ratio",
    "infilling_strong_beat_onset_ratio",
    "infilling_rhythm_metric_coverage",
    "infilling_event_ngram_repeat_ratio",
    "infilling_rhythm_ngram_repeat_ratio",
    "infilling_repetition_metric_coverage",
    "overall_bar_start_onset_ratio",
    "overall_strong_beat_onset_ratio",
    "overall_rhythm_metric_coverage",
    "overall_event_ngram_repeat_ratio",
    "overall_rhythm_ngram_repeat_ratio",
    "overall_repetition_metric_coverage",
    "continuation_pitch_collapse_coverage",
    "infilling_pitch_collapse_coverage",
    "overall_pitch_collapse_coverage",
    "fim_ratio_mean",
    "fim_ratio_std",
}

_INTEGER_METRICS_V2 = {
    "step",
    "balanced_rank",
    "tokens_seen",
    "tokens_seen_last",
    "best_valid_step",
    "best_valid_tokens_seen",
    "plateau_eval_streak",
    "train_event_count",
    "eval_event_count",
    "fsm_dead_end_count",
    "last_train_step",
    "last_eval_step",
    "overall_pitch_metric_count",
}

_METRIC_LABELS_V2 = {
    "checkpoint_name": "Checkpoint",
    "evaluation_tier": "评估层级",
    "balanced_rank": "相对排名",
    "balanced_score": "相对分",
    "absolute_score": "绝对分",
    "absolute_score_coverage": "绝对分覆盖率",
    "continuation_closure_score": "续写收束",
    "continuation_structure_score": "续写结构",
    "infilling_integrity_score": "补全完整性",
    "phrase_coherence_score": "乐句连贯性",
    "long_context_stability_score": "长上下文稳定性",
    "training_health_score": "训练健康度",
    "continuation_stop_success_rate": "续写成功停机率",
    "continuation_budget_stop_rate": "续写预算截断率",
    "continuation_missing_eos_rate": "续写缺失 EOS 率",
    "append_eos_recoverable_rate": "补 EOS 可恢复率",
    "continuation_structural_validity_rate": "续写结构合法率",
    "continuation_time_order_validity_rate": "续写时间顺序合法率",
    "continuation_empty_bar_rate": "续写空 BAR 率",
    "continuation_syntax_invalid_rate": "续写语法非法率",
    "infilling_structural_validity_rate": "补全结构合法率",
    "infilling_time_order_validity_rate": "补全时间顺序合法率",
    "infilling_internal_time_order_validity_rate": "补全内部时间顺序合法率",
    "infilling_boundary_time_order_validity_rate": "补全边界时间顺序合法率",
    "infilling_syntax_invalid_rate": "补全语法非法率",
    "continuation_first_event_hit_rate": "续写首事件命中率",
    "continuation_onset_position_l1_distance": "续写起拍位置分布 L1 距离",
    "infilling_onset_position_l1_distance": "补全起拍位置分布 L1 距离",
    "overall_onset_position_l1_distance": "总体起拍位置分布 L1 距离",
    "duration_bin_l1_distance": "时值分桶 L1 距离",
    "low_density_bar_rate": "低密度 BAR 率",
    "multi_empty_bar_run_rate": "连续空 BAR 样本率",
    "generated_bar_delta_mean": "生成 BAR 数偏差均值",
    "generated_event_delta_mean": "生成事件数偏差均值",
    "pitch_span_delta_mean": "音高跨度偏差均值",
    "valid_loss_from_training": "训练期验证损失",
    "best_valid_loss_so_far": "历史最佳验证损失",
    "train_loss_ema": "训练损失 EMA",
    "overfit_gap": "过拟合间隙",
    "tokens_seen": "已见 token 数",
    "continuation_most_common_pitch_ratio": "续写最高频 pitch 占比",
    "continuation_longest_same_pitch_run_ratio": "续写最长同 pitch 连续 run 占比",
    "continuation_pitch_diversity_score": "续写音高多样性分数",
    "continuation_pitch_collapse_coverage": "续写 pitch 指标覆盖率",
    "continuation_onset_position_entropy": "续写起拍位置熵",
    "continuation_bar_start_onset_ratio": "续写小节起点占比",
    "continuation_strong_beat_onset_ratio": "续写强拍占比",
    "continuation_duration_diversity_score": "续写时值多样性分数",
    "continuation_rhythm_diversity_score": "续写节奏多样性分数",
    "continuation_rhythm_metric_coverage": "续写节奏指标覆盖率",
    "continuation_event_ngram_repeat_ratio": "续写事件 n-gram 重复占比",
    "continuation_rhythm_ngram_repeat_ratio": "续写节奏 n-gram 重复占比",
    "continuation_repetition_metric_coverage": "续写重复指标覆盖率",
    "infilling_most_common_pitch_ratio": "补全最高频 pitch 占比",
    "infilling_longest_same_pitch_run_ratio": "补全最长同 pitch 连续 run 占比",
    "infilling_pitch_diversity_score": "补全音高多样性分数",
    "infilling_pitch_collapse_coverage": "补全 pitch 指标覆盖率",
    "infilling_onset_position_entropy": "补全起拍位置熵",
    "infilling_bar_start_onset_ratio": "补全小节起点占比",
    "infilling_strong_beat_onset_ratio": "补全强拍占比",
    "infilling_duration_diversity_score": "补全时值多样性分数",
    "infilling_rhythm_diversity_score": "补全节奏多样性分数",
    "infilling_rhythm_metric_coverage": "补全节奏指标覆盖率",
    "infilling_event_ngram_repeat_ratio": "补全事件 n-gram 重复占比",
    "infilling_rhythm_ngram_repeat_ratio": "补全节奏 n-gram 重复占比",
    "infilling_repetition_metric_coverage": "补全重复指标覆盖率",
    "overall_most_common_pitch_ratio": "总体最高频 pitch 占比",
    "overall_longest_same_pitch_run_ratio": "总体最长同 pitch 连续 run 占比",
    "overall_pitch_diversity_score": "总体音高多样性分数",
    "overall_pitch_collapse_coverage": "总体 pitch 指标覆盖率",
    "overall_onset_position_entropy": "总体起拍位置熵",
    "overall_bar_start_onset_ratio": "总体小节起点占比",
    "overall_strong_beat_onset_ratio": "总体强拍占比",
    "overall_duration_diversity_score": "总体时值多样性分数",
    "overall_rhythm_diversity_score": "总体节奏多样性分数",
    "overall_rhythm_metric_coverage": "总体节奏指标覆盖率",
    "overall_event_ngram_repeat_ratio": "总体事件 n-gram 重复占比",
    "overall_rhythm_ngram_repeat_ratio": "总体节奏 n-gram 重复占比",
    "overall_repetition_metric_coverage": "总体重复指标覆盖率",
    "fsm_structural_validity_rate": "FSM 结构合法率",
    "fsm_time_order_validity_rate": "FSM 时间顺序合法率",
    "fsm_illegal_top1_rate": "FSM 非法 top1 率",
    "fsm_mask_intervention_rate": "FSM 掩码干预率",
    "fsm_dead_end_count": "FSM 死路次数",
}


def _format_metric_value_v2(value: Any, *, key: str | None = None) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "NA"
    if isinstance(value, str):
        if key == "evaluation_tier":
            return _format_eval_tier_v2(value)
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    if key in _INTEGER_METRICS_V2:
        return f"{int(round(numeric))}"
    if key in _PERCENT_METRICS_V2:
        return f"{numeric * 100:.2f}%"
    return f"{numeric:.2f}"


def _markdown_table_v2(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["_暂无数据_", ""]
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", separator]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _result_table_rows_v2(results: list[dict[str, Any]], specs: list[tuple[str, str]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for result in results:
        rows.append([_format_metric_value_v2(result.get(key), key=key) for key, _label in specs])
    return rows


def _counter_table_rows_v2(counter_mapping: dict[str, int], *, limit: int = 5) -> list[list[str]]:
    items = _top_counter_items(counter_mapping, limit=limit)
    if not items:
        return [["（无）", "0"]]
    return [[f"`{key}`", str(value)] for key, value in items]


def _format_goal_v2(goal: Any) -> str:
    if goal == "max":
        return "越高越好"
    if goal == "min":
        return "越低越好"
    return str(goal)


def _format_capability_type_v2(is_proxy: bool) -> str:
    return "代理" if is_proxy else "直接"


def _format_eval_tier_v2(value: Any) -> str:
    if value == "fast":
        return "fast"
    if value == "formal":
        return "formal"
    return str(value)


def _core_metric_specs_v2(task_scope: str) -> list[tuple[str, str]]:
    base = [
        ("balanced_rank", "排名"),
        ("checkpoint_name", "Checkpoint"),
        ("evaluation_tier", "评估层级"),
        ("balanced_score", "相对分"),
        ("absolute_score", "绝对分"),
    ]
    if task_scope == "continuation":
        return [
            *base,
            ("continuation_closure_score", "续写收束"),
            ("continuation_structure_score", "续写结构"),
            ("long_context_stability_score", "长上下文稳定性"),
            ("training_health_score", "训练健康度"),
        ]
    if task_scope == "infilling":
        return [
            *base,
            ("infilling_integrity_score", "补全完整性"),
            ("phrase_coherence_score", "乐句连贯性"),
            ("long_context_stability_score", "长上下文稳定性"),
            ("training_health_score", "训练健康度"),
        ]
    return [
        *base,
        ("continuation_closure_score", "续写收束"),
        ("continuation_structure_score", "续写结构"),
        ("infilling_integrity_score", "补全完整性"),
        ("phrase_coherence_score", "乐句连贯性"),
        ("long_context_stability_score", "长上下文稳定性"),
        ("training_health_score", "训练健康度"),
    ]


def _diagnostic_metric_specs_v2(task_scope: str) -> list[tuple[str, str]]:
    common = [
        ("append_eos_recoverable_rate", "补 EOS 可恢复率"),
        ("low_density_bar_rate", "低密度 BAR 率"),
        ("multi_empty_bar_run_rate", "连续空 BAR 样本率"),
        ("generated_bar_delta_mean", "生成 BAR 数偏差均值"),
        ("generated_event_delta_mean", "生成事件数偏差均值"),
        ("pitch_span_delta_mean", "音高跨度偏差均值"),
        ("duration_bin_l1_distance", "时值分桶 L1 距离"),
    ]
    if task_scope == "continuation":
        return [
            ("continuation_first_event_hit_rate", "续写首事件命中率"),
            ("continuation_missing_eos_rate", "续写缺失 EOS 率"),
            ("continuation_syntax_invalid_rate", "续写语法非法率"),
            *common,
        ]
    if task_scope == "infilling":
        return [
            ("infilling_syntax_invalid_rate", "补全语法非法率"),
            ("infilling_internal_time_order_validity_rate", "补全内部时间顺序合法率"),
            ("infilling_boundary_time_order_validity_rate", "补全边界时间顺序合法率"),
            ("fsm_structural_validity_rate", "FSM 结构合法率"),
            ("fsm_time_order_validity_rate", "FSM 时间顺序合法率"),
            ("fsm_illegal_top1_rate", "FSM 非法 top1 率"),
            ("fsm_mask_intervention_rate", "FSM 掩码干预率"),
            ("fsm_dead_end_count", "FSM 死路次数"),
        ]
    return [
        ("continuation_first_event_hit_rate", "续写首事件命中率"),
        ("continuation_missing_eos_rate", "续写缺失 EOS 率"),
        ("continuation_syntax_invalid_rate", "续写语法非法率"),
        ("infilling_syntax_invalid_rate", "补全语法非法率"),
        ("infilling_internal_time_order_validity_rate", "补全内部时间顺序合法率"),
        ("infilling_boundary_time_order_validity_rate", "补全边界时间顺序合法率"),
        *common,
    ]


def _pitch_metric_specs_v2(task_scope: str) -> list[tuple[str, str]]:
    if task_scope == "continuation":
        return [
            ("continuation_most_common_pitch_ratio", "续写最高频 pitch 占比"),
            ("continuation_longest_same_pitch_run_ratio", "续写最长同 pitch 连续 run 占比"),
            ("continuation_pitch_diversity_score", "续写音高多样性分数"),
            ("continuation_pitch_collapse_coverage", "续写 pitch 指标覆盖率"),
        ]
    if task_scope == "infilling":
        return [
            ("infilling_most_common_pitch_ratio", "补全最高频 pitch 占比"),
            ("infilling_longest_same_pitch_run_ratio", "补全最长同 pitch 连续 run 占比"),
            ("infilling_pitch_diversity_score", "补全音高多样性分数"),
            ("infilling_pitch_collapse_coverage", "补全 pitch 指标覆盖率"),
        ]
    return [
        ("overall_most_common_pitch_ratio", "总体最高频 pitch 占比"),
        ("overall_longest_same_pitch_run_ratio", "总体最长同 pitch 连续 run 占比"),
        ("overall_pitch_diversity_score", "总体音高多样性分数"),
        ("overall_pitch_collapse_coverage", "总体 pitch 指标覆盖率"),
    ]


def _rhythm_metric_specs_v2(task_scope: str) -> list[tuple[str, str]]:
    if task_scope == "continuation":
        return [
            ("continuation_onset_position_l1_distance", "续写起拍位置分布 L1 距离"),
            ("continuation_onset_position_entropy", "续写起拍位置熵"),
            ("continuation_bar_start_onset_ratio", "续写小节起点占比"),
            ("continuation_strong_beat_onset_ratio", "续写强拍占比"),
            ("continuation_duration_diversity_score", "续写时值多样性分数"),
            ("continuation_rhythm_diversity_score", "续写节奏多样性分数"),
            ("continuation_rhythm_metric_coverage", "续写节奏指标覆盖率"),
        ]
    if task_scope == "infilling":
        return [
            ("infilling_onset_position_l1_distance", "补全起拍位置分布 L1 距离"),
            ("infilling_onset_position_entropy", "补全起拍位置熵"),
            ("infilling_bar_start_onset_ratio", "补全小节起点占比"),
            ("infilling_strong_beat_onset_ratio", "补全强拍占比"),
            ("infilling_duration_diversity_score", "补全时值多样性分数"),
            ("infilling_rhythm_diversity_score", "补全节奏多样性分数"),
            ("infilling_rhythm_metric_coverage", "补全节奏指标覆盖率"),
        ]
    return [
        ("overall_onset_position_l1_distance", "总体起拍位置分布 L1 距离"),
        ("overall_onset_position_entropy", "总体起拍位置熵"),
        ("overall_bar_start_onset_ratio", "总体小节起点占比"),
        ("overall_strong_beat_onset_ratio", "总体强拍占比"),
        ("overall_duration_diversity_score", "总体时值多样性分数"),
        ("overall_rhythm_diversity_score", "总体节奏多样性分数"),
        ("overall_rhythm_metric_coverage", "总体节奏指标覆盖率"),
    ]


def _repetition_metric_specs_v2(task_scope: str) -> list[tuple[str, str]]:
    if task_scope == "continuation":
        return [
            ("continuation_event_ngram_repeat_ratio", "续写事件 n-gram 重复占比"),
            ("continuation_rhythm_ngram_repeat_ratio", "续写节奏 n-gram 重复占比"),
            ("continuation_repetition_metric_coverage", "续写重复指标覆盖率"),
        ]
    if task_scope == "infilling":
        return [
            ("infilling_event_ngram_repeat_ratio", "补全事件 n-gram 重复占比"),
            ("infilling_rhythm_ngram_repeat_ratio", "补全节奏 n-gram 重复占比"),
            ("infilling_repetition_metric_coverage", "补全重复指标覆盖率"),
        ]
    return [
        ("overall_event_ngram_repeat_ratio", "总体事件 n-gram 重复占比"),
        ("overall_rhythm_ngram_repeat_ratio", "总体节奏 n-gram 重复占比"),
        ("overall_repetition_metric_coverage", "总体重复指标覆盖率"),
    ]


def _training_metric_specs_v2() -> list[tuple[str, str]]:
    return [
        ("step", "Step"),
        ("valid_loss_from_training", "训练期验证损失"),
        ("train_loss_ema", "训练损失 EMA"),
        ("best_valid_loss_so_far", "历史最佳验证损失"),
        ("overfit_gap", "过拟合间隙"),
        ("tokens_seen", "已见 token 数"),
    ]


def _plot_metric_specs_v2(task_scope: str, *, diagnostics: bool) -> list[dict[str, Any]]:
    if diagnostics:
        if task_scope == "continuation":
            return [
                {"key": "continuation_missing_eos_rate", "label": "续写缺失 EOS 率", "percent": True, "goal": "min", "color": "#dc2626"},
                {"key": "continuation_most_common_pitch_ratio", "label": "最高频 pitch 占比", "goal": "min", "color": "#2563eb"},
                {"key": "continuation_rhythm_diversity_score", "label": "节奏多样性分数", "goal": "max", "color": "#16a34a"},
                {"key": "continuation_bar_start_onset_ratio", "label": "小节起点占比", "percent": True, "goal": "min", "color": "#7c3aed"},
                {"key": "continuation_event_ngram_repeat_ratio", "label": "事件 n-gram 重复占比", "percent": True, "goal": "min", "color": "#0891b2"},
                {"key": "multi_empty_bar_run_rate", "label": "连续空 BAR 样本率", "percent": True, "goal": "min", "color": "#ea580c"},
            ]
        if task_scope == "infilling":
            return [
                {"key": "infilling_syntax_invalid_rate", "label": "补全语法非法率", "percent": True, "goal": "min", "color": "#dc2626"},
                {"key": "infilling_internal_time_order_validity_rate", "label": "内部时间顺序合法率", "percent": True, "goal": "max", "color": "#2563eb"},
                {"key": "infilling_boundary_time_order_validity_rate", "label": "边界时间顺序合法率", "percent": True, "goal": "max", "color": "#16a34a"},
                {"key": "infilling_most_common_pitch_ratio", "label": "最高频 pitch 占比", "goal": "min", "color": "#2563eb"},
                {"key": "infilling_rhythm_diversity_score", "label": "节奏多样性分数", "goal": "max", "color": "#7c3aed"},
                {"key": "infilling_bar_start_onset_ratio", "label": "小节起点占比", "percent": True, "goal": "min", "color": "#0891b2"},
                {"key": "infilling_event_ngram_repeat_ratio", "label": "事件 n-gram 重复占比", "percent": True, "goal": "min", "color": "#ea580c"},
            ]
        return [
            {"key": "continuation_missing_eos_rate", "label": "续写缺失 EOS 率", "percent": True, "goal": "min", "color": "#dc2626"},
            {"key": "overall_most_common_pitch_ratio", "label": "总体最高频 pitch 占比", "goal": "min", "color": "#2563eb"},
            {"key": "overall_rhythm_diversity_score", "label": "总体节奏多样性分数", "goal": "max", "color": "#16a34a"},
            {"key": "overall_bar_start_onset_ratio", "label": "总体小节起点占比", "percent": True, "goal": "min", "color": "#7c3aed"},
            {"key": "overall_event_ngram_repeat_ratio", "label": "总体事件 n-gram 重复占比", "percent": True, "goal": "min", "color": "#0891b2"},
            {"key": "overall_rhythm_metric_coverage", "label": "总体节奏指标覆盖率", "percent": True, "goal": "max", "color": "#ea580c"},
        ]
    if task_scope == "continuation":
        return [
            {"key": "balanced_score", "label": "相对分", "goal": "max", "color": "#111827"},
            {"key": "absolute_score", "label": "绝对分", "goal": "max", "color": "#1d4ed8"},
            {"key": "continuation_closure_score", "label": "续写收束", "goal": "max", "color": "#2563eb"},
            {"key": "continuation_structure_score", "label": "续写结构", "goal": "max", "color": "#16a34a"},
            {"key": "long_context_stability_score", "label": "长上下文稳定性", "goal": "max", "color": "#7c3aed"},
        ]
    if task_scope == "infilling":
        return [
            {"key": "balanced_score", "label": "相对分", "goal": "max", "color": "#111827"},
            {"key": "absolute_score", "label": "绝对分", "goal": "max", "color": "#1d4ed8"},
            {"key": "infilling_integrity_score", "label": "补全完整性", "goal": "max", "color": "#2563eb"},
            {"key": "phrase_coherence_score", "label": "乐句连贯性", "goal": "max", "color": "#16a34a"},
            {"key": "training_health_score", "label": "训练健康度", "goal": "max", "color": "#7c3aed"},
        ]
    return [
        {"key": "balanced_score", "label": "相对分", "goal": "max", "color": "#111827"},
        {"key": "absolute_score", "label": "绝对分", "goal": "max", "color": "#1d4ed8"},
        {"key": "continuation_closure_score", "label": "续写收束", "goal": "max", "color": "#2563eb"},
        {"key": "continuation_structure_score", "label": "续写结构", "goal": "max", "color": "#16a34a"},
        {"key": "infilling_integrity_score", "label": "补全完整性", "goal": "max", "color": "#0891b2"},
        {"key": "phrase_coherence_score", "label": "乐句连贯性", "goal": "max", "color": "#ea580c"},
        {"key": "long_context_stability_score", "label": "长上下文稳定性", "goal": "max", "color": "#7c3aed"},
    ]


def _absolute_plot_metric_specs_v2(task_scope: str) -> list[dict[str, Any]]:
    if task_scope == "continuation":
        return [
            {"key": "absolute_score", "label": "绝对分", "goal": "max", "color": "#1d4ed8"},
            {"key": "continuation_closure_score", "label": "续写收束", "goal": "max", "color": "#2563eb"},
            {"key": "continuation_structure_score", "label": "续写结构", "goal": "max", "color": "#16a34a"},
            {"key": "long_context_stability_score", "label": "长上下文稳定性", "goal": "max", "color": "#7c3aed"},
            {"key": "training_health_score", "label": "训练健康度", "goal": "max", "color": "#ea580c"},
        ]
    if task_scope == "infilling":
        return [
            {"key": "absolute_score", "label": "绝对分", "goal": "max", "color": "#1d4ed8"},
            {"key": "infilling_integrity_score", "label": "补全完整性", "goal": "max", "color": "#2563eb"},
            {"key": "phrase_coherence_score", "label": "乐句连贯性", "goal": "max", "color": "#16a34a"},
            {"key": "long_context_stability_score", "label": "长上下文稳定性", "goal": "max", "color": "#7c3aed"},
            {"key": "training_health_score", "label": "训练健康度", "goal": "max", "color": "#ea580c"},
        ]
    return [
        {"key": "absolute_score", "label": "绝对分", "goal": "max", "color": "#1d4ed8"},
        {"key": "continuation_closure_score", "label": "续写收束", "goal": "max", "color": "#2563eb"},
        {"key": "continuation_structure_score", "label": "续写结构", "goal": "max", "color": "#16a34a"},
        {"key": "infilling_integrity_score", "label": "补全完整性", "goal": "max", "color": "#0891b2"},
        {"key": "phrase_coherence_score", "label": "乐句连贯性", "goal": "max", "color": "#ea580c"},
        {"key": "long_context_stability_score", "label": "长上下文稳定性", "goal": "max", "color": "#7c3aed"},
        {"key": "training_health_score", "label": "训练健康度", "goal": "max", "color": "#b45309"},
    ]


def _build_summary_markdown_v2(
    *,
    run_id: str,
    task_scope: str,
    benchmark_root: Path,
    recommended: dict[str, Any] | None,
    top_results: list[dict[str, Any]],
    training_summary: dict[str, Any],
    plot_artifacts: dict[str, str],
    sample_artifacts: dict[str, dict[str, dict[str, str]]],
    exported_samples: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    manifest_stats: dict[str, Any],
    checkpoint_prefilter: dict[str, Any],
    evaluation_context: dict[str, Any],
) -> str:
    final_sample_artifacts = dict(sample_artifacts.get("final_top3", {}))
    final_exported_samples = dict(exported_samples.get("final_top3", {}))
    formal_candidate_artifacts = dict(sample_artifacts.get("formal_candidates", {}))
    decoding = dict(evaluation_context.get("decoding", {}))
    benchmark_configs = dict(evaluation_context.get("benchmark_configs", {}))
    config_paths = dict(evaluation_context.get("config_paths", {}))
    train_run = dict(training_summary.get("run", {})) if isinstance(training_summary.get("run"), dict) else {}
    lines = [f"# {_TASK_TITLES[task_scope]}: {run_id}", ""]
    lines.extend(
        [
            "## 分数说明",
            "- `balanced_score` 是相对分，只用于当前候选 checkpoint 集合内的排序。",
            "- `absolute_score` 是固定量纲的 0-100 绝对能力分，用于跨 run、跨阶段、跨轮次长期追踪模型能力变化。",
            "- 当前推荐逻辑仍沿用既有的 `balanced_score` 加 gate 机制；`absolute_score` 作为稳定能力面板提供，不直接替代现有主推荐逻辑。",
            "",
        ]
    )

    if recommended is None:
        lines.extend(["## 推荐结果", "", "_当前 benchmark 流程没有给出推荐 checkpoint。_", ""])
    else:
        lines.extend(
            [
                "## 推荐结果",
                "",
                f"- Checkpoint：`{recommended.get('checkpoint_name')}`",
                f"- Step：{_format_metric_value_v2(recommended.get('step'), key='step')}",
                f"- 评估层级：{_format_eval_tier_v2(recommended.get('evaluation_tier'))}",
                f"- 相对分：{_format_metric_value_v2(recommended.get('balanced_score'), key='balanced_score')}",
                f"- 绝对分：{_format_metric_value_v2(recommended.get('absolute_score'), key='absolute_score')}",
                f"- 绝对分覆盖率：{_format_metric_value_v2(recommended.get('absolute_score_coverage'), key='absolute_score_coverage')}",
                "",
            ]
        )

    eval_context_rows = [
        ["task_scope", str(task_scope)],
        ["train_config", str(config_paths.get("train_config", ""))],
        ["fast_config", str(config_paths.get("fast_config", ""))],
        ["formal_config", str(config_paths.get("formal_config", ""))],
        ["vocab_path", str(config_paths.get("vocab_path", ""))],
    ]
    decoding_rows = [
        ["max_new_tokens", _format_metric_value_v2(decoding.get("max_new_tokens"))],
        ["temperature", _format_metric_value_v2(decoding.get("temperature"))],
        ["top_p", _format_metric_value_v2(decoding.get("top_p"))],
    ]
    benchmark_param_rows = [
        [
            "fast",
            _format_metric_value_v2(benchmark_configs.get("fast", {}).get("sample_count")),
            _format_metric_value_v2(benchmark_configs.get("fast", {}).get("per_bucket_limit")),
            _format_metric_value_v2(benchmark_configs.get("fast", {}).get("min_prefix_tokens")),
            (
                f"{_format_metric_value_v2(benchmark_configs.get('fast', {}).get('continuation_prefix_ratio_min'))} - "
                f"{_format_metric_value_v2(benchmark_configs.get('fast', {}).get('continuation_prefix_ratio_max'))}"
            ),
            (
                f"{_format_metric_value_v2(benchmark_configs.get('fast', {}).get('infilling_hole_ratio_min'))} - "
                f"{_format_metric_value_v2(benchmark_configs.get('fast', {}).get('infilling_hole_ratio_max'))}"
            ),
        ],
        [
            "formal",
            _format_metric_value_v2(benchmark_configs.get("formal", {}).get("sample_count")),
            _format_metric_value_v2(benchmark_configs.get("formal", {}).get("per_bucket_limit")),
            _format_metric_value_v2(benchmark_configs.get("formal", {}).get("min_prefix_tokens")),
            (
                f"{_format_metric_value_v2(benchmark_configs.get('formal', {}).get('continuation_prefix_ratio_min'))} - "
                f"{_format_metric_value_v2(benchmark_configs.get('formal', {}).get('continuation_prefix_ratio_max'))}"
            ),
            (
                f"{_format_metric_value_v2(benchmark_configs.get('formal', {}).get('infilling_hole_ratio_min'))} - "
                f"{_format_metric_value_v2(benchmark_configs.get('formal', {}).get('infilling_hole_ratio_max'))}"
            ),
        ],
    ]
    train_sampling_rows = [
        ["seq_len", _format_metric_value_v2(train_run.get("seq_len"), key="seq_len")],
        ["fim_ratio", _format_metric_value_v2(train_run.get("fim_ratio"))],
        ["fim_eos_ratio", _format_metric_value_v2(train_run.get("fim_eos_ratio"))],
        ["single_phrase_sample_ratio", _format_metric_value_v2(train_run.get("single_phrase_sample_ratio"))],
        ["cross_phrase_sample_ratio", _format_metric_value_v2(train_run.get("cross_phrase_sample_ratio"))],
        ["long_context_sample_ratio", _format_metric_value_v2(train_run.get("long_context_sample_ratio"))],
    ]
    lines.extend(["## 测评参数", ""])
    lines.extend(["### 运行与配置路径", ""])
    lines.extend(_markdown_table_v2(["参数", "值"], eval_context_rows))
    lines.extend(["### 解码参数", ""])
    lines.extend(_markdown_table_v2(["参数", "值"], decoding_rows))
    lines.extend(["### Benchmark 切分参数", ""])
    lines.extend(
        _markdown_table_v2(
            ["阶段", "sample_count", "per_bucket_limit", "min_prefix_tokens", "continuation_prefix_ratio", "infilling_hole_ratio"],
            benchmark_param_rows,
        )
    )
    lines.extend(["### 训练可比性提示", ""])
    lines.extend(_markdown_table_v2(["参数", "值"], train_sampling_rows))

    prefilter_line = (
        f"已启用，保留 {checkpoint_prefilter.get('selected_count', 0)} / {checkpoint_prefilter.get('original_count', 0)} "
        f"(top_k={checkpoint_prefilter.get('requested_top_k', 0)}, preserve_earliest={checkpoint_prefilter.get('preserve_earliest', 0)})"
        if checkpoint_prefilter.get("enabled")
        else "未启用"
    )
    lines.extend(
        [
            "## 评测概览",
            "",
            f"- Fast manifest 样本数：{manifest_stats.get('fast_case_count')}",
            f"- Formal manifest 样本数：{manifest_stats.get('formal_case_count')}",
            f"- Formal 候选 checkpoint 数：{manifest_stats.get('candidate_count')}",
            f"- Checkpoint 预筛：{prefilter_line}",
            (
                "- Formal 候选样本已导出到 `samples/formal_candidates/`。"
                if formal_candidate_artifacts
                else "- 本次没有生成 `formal_candidates` 样本导出。"
            ),
            "- 最终样本包导出在 `samples/final_top3/`。",
            "",
            "## Top 3 排行",
            "",
        ]
    )
    lines.extend(
        _markdown_table_v2(
            [label for _key, label in _core_metric_specs_v2(task_scope)],
            _result_table_rows_v2(top_results, _core_metric_specs_v2(task_scope)),
        )
    )

    if recommended is not None:
        recommended_dimensions = (
            dict(recommended.get("absolute_score_breakdown", {}).get("dimensions", {}))
            if isinstance(recommended.get("absolute_score_breakdown"), dict)
            else {}
        )
        capability_rows: list[list[str]] = []
        for dimension_key, payload in recommended_dimensions.items():
            capability_rows.append(
                [
                    str(payload.get("label", dimension_key)),
                    _format_metric_value_v2(payload.get("score"), key=dimension_key),
                    _format_metric_value_v2(payload.get("coverage"), key="absolute_score_coverage"),
                    _format_metric_value_v2(payload.get("weight")),
                    _format_capability_type_v2(bool(payload.get("proxy"))),
                ]
            )
        lines.extend(["## 推荐 Checkpoint 能力面板", ""])
        lines.extend(_markdown_table_v2(["维度", "分数", "覆盖率", "权重", "类型"], capability_rows))

        gate_rows = []
        for metric_key, payload in sorted(recommended.get("gate_details", {}).items()):
            gate_rows.append(
                [
                    _METRIC_LABELS_V2.get(metric_key, metric_key),
                    _format_goal_v2(payload.get("goal", "")),
                    _format_metric_value_v2(payload.get("threshold"), key=metric_key),
                    _format_metric_value_v2(payload.get("value"), key=metric_key),
                    _format_metric_value_v2(payload.get("passed")),
                ]
            )
        lines.extend(["## 推荐 Checkpoint 门槛检查", ""])
        lines.extend(_markdown_table_v2(["指标", "方向", "阈值", "实际值", "是否通过"], gate_rows))

        relative_rows = []
        for metric_key, payload in recommended.get("score_breakdown", {}).items():
            relative_rows.append(
                [
                    _METRIC_LABELS_V2.get(metric_key, metric_key),
                    _format_goal_v2(payload.get("goal", "")),
                    _format_metric_value_v2(payload.get("weight")),
                    _format_metric_value_v2(payload.get("value"), key=metric_key),
                    _format_metric_value_v2(payload.get("rank_score")),
                    _format_metric_value_v2(payload.get("weighted_contribution")),
                ]
            )
        lines.extend(["## 推荐 Checkpoint 相对分拆解", ""])
        lines.extend(
            _markdown_table_v2(
                ["指标", "方向", "权重", "原始值", "排序分", "加权贡献"],
                relative_rows,
            )
        )

        absolute_rows: list[list[str]] = []
        for dimension_key, payload in recommended_dimensions.items():
            for metric_key, metric_payload in dict(payload.get("submetrics", {})).items():
                absolute_rows.append(
                    [
                        str(payload.get("label", dimension_key)),
                        str(metric_payload.get("label", _METRIC_LABELS_V2.get(metric_key, metric_key))),
                        _format_metric_value_v2(metric_payload.get("raw_value"), key=metric_key),
                        _format_metric_value_v2(metric_payload.get("score")),
                        _format_metric_value_v2(metric_payload.get("weight")),
                        str(metric_payload.get("status", "")),
                    ]
                )
        lines.extend(["## 推荐 Checkpoint 绝对分拆解", ""])
        lines.extend(
            _markdown_table_v2(
                ["维度", "指标", "原始值", "映射分", "权重", "状态"],
                absolute_rows,
            )
        )

    lines.extend(["## 图表", ""])
    if plot_artifacts:
        for label, artifact_path in plot_artifacts.items():
            relative_path = _relative_artifact_path(benchmark_root, artifact_path)
            lines.append(f"### {label}")
            lines.append(f"![{label}]({relative_path})")
            lines.append("")
    else:
        lines.extend(["_本次未生成图表_", ""])

    training_summary_specs = [
        ("last_train_step", "最近训练 step"),
        ("last_eval_step", "最近验证 step"),
        ("tokens_seen_last", "已见 token 数"),
        ("latest_train_loss_ema", "最近训练 EMA"),
        ("latest_valid_loss", "最近验证损失"),
        ("best_valid_loss", "最佳验证损失"),
        ("best_valid_step", "最佳验证 step"),
        ("latest_overfit_gap", "最近过拟合间隙"),
        ("latest_valid_loss_delta", "最近验证损失变化"),
        ("plateau_eval_streak", "平台期连续 eval 次数"),
        ("tok_per_sec_median", "吞吐中位数"),
        ("fim_ratio_mean", "FIM 比例均值"),
    ]
    lines.extend(["## 训练健康度摘要", ""])
    lines.extend(
        _markdown_table_v2(
            ["指标", "值"],
            [[label, _format_metric_value_v2(training_summary.get(key), key=key)] for key, label in training_summary_specs],
        )
    )

    lines.extend(["## Top 3 诊断指标", ""])
    lines.extend(
        _markdown_table_v2(
            [label for _key, label in _diagnostic_metric_specs_v2(task_scope)],
            _result_table_rows_v2(top_results, _diagnostic_metric_specs_v2(task_scope)),
        )
    )

    lines.extend(["## Top 3 音高塌缩摘要", ""])
    lines.extend(
        _markdown_table_v2(
            [label for _key, label in _pitch_metric_specs_v2(task_scope)],
            _result_table_rows_v2(top_results, _pitch_metric_specs_v2(task_scope)),
        )
    )

    lines.extend(["## Top 3 节奏丰富性摘要", ""])
    lines.extend(
        _markdown_table_v2(
            [label for _key, label in _rhythm_metric_specs_v2(task_scope)],
            _result_table_rows_v2(top_results, _rhythm_metric_specs_v2(task_scope)),
        )
    )

    lines.extend(["## Top 3 重复度摘要", ""])
    lines.extend(
        _markdown_table_v2(
            [label for _key, label in _repetition_metric_specs_v2(task_scope)],
            _result_table_rows_v2(top_results, _repetition_metric_specs_v2(task_scope)),
        )
    )

    lines.extend(["## Top 3 训练指标", ""])
    lines.extend(
        _markdown_table_v2(
            [label for _key, label in _training_metric_specs_v2()],
            _result_table_rows_v2(top_results, _training_metric_specs_v2()),
        )
    )

    for result in top_results:
        checkpoint_name = str(result.get("checkpoint_name"))
        lines.append(f"## {checkpoint_name}")
        lines.append("")
        lines.append("### 高频失败原因")
        lines.extend(
            _markdown_table_v2(
                ["失败原因", "次数"],
                _counter_table_rows_v2(_scoped_failure_counts(result, task_scope=task_scope)),
            )
        )
        lines.append("### 高频语法原因")
        lines.extend(
            _markdown_table_v2(
                ["语法原因", "次数"],
                _counter_table_rows_v2(_scoped_syntax_counts(result, task_scope=task_scope)),
            )
        )
        artifact_paths = final_sample_artifacts.get(checkpoint_name, {})
        lines.append("### 样本产物")
        for task_name in _task_names_for_scope(task_scope):
            if task_name in artifact_paths:
                relative_path = _relative_artifact_path(benchmark_root, artifact_paths.get(task_name, ""))
                lines.append(f"- `{task_name}` 样本文件：`{relative_path}`")

        checkpoint_samples = final_exported_samples.get(checkpoint_name, {})
        lines.append("### 代表样本")
        for task_name in _task_names_for_scope(task_scope):
            task_samples = checkpoint_samples.get(task_name, [])
            success_key = _sample_success_key(task_name)
            success = next((item for item in task_samples if bool(item.get(success_key))), None)
            failure = next((item for item in task_samples if not bool(item.get(success_key))), None)
            if success is not None:
                lines.append(
                    f"- `{task_name}` 成功样本：row_id={success.get('row_id')} "
                    f"{success.get('meta', {}).get('artist')} - {success.get('meta', {}).get('title')} | "
                    f"预览={_sample_preview(success.get('raw_output_tokens', []))}"
                )
            if failure is not None:
                lines.append(
                    f"- `{task_name}` 失败样本：row_id={failure.get('row_id')} "
                    f"{failure.get('meta', {}).get('artist')} - {failure.get('meta', {}).get('title')} | "
                    f"原因={failure.get('raw_failure_reason')} / {failure.get('raw_syntax_reason')} | "
                    f"预览={_sample_preview(failure.get('raw_output_tokens', []))}"
                )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main(*, task_scope: str = "all", argv: list[str] | None = None) -> None:
    project_root = _ensure_project_root_on_path()
    os.chdir(project_root)
    args = _parse_args(task_scope=task_scope, argv=argv)

    checkpoint_dir, train_mapping, run_id = _resolve_eval_target(project_root, args)
    config_path = None
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else (project_root / args.config)
        config_path = config_path.resolve()
    elif args.preset is not None:
        config_path = _resolve_preset_config(project_root, args.preset)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    benchmark_root = project_root / "outputs" / "benchmark" / run_id
    report_path = benchmark_root / _TASK_REPORT_NAMES[task_scope]
    summary_path = benchmark_root / _TASK_SUMMARY_NAMES[task_scope]
    _clean_benchmark_outputs(benchmark_root, task_scope)

    from src.decoding import TuneFlowGrammarFSM
    from src.model.configuration import DecoderConfig
    from src.model.modeling import DecoderForCausalLM
    from src.training.train_base import _autocast_context, _load_checkpoint, _resolve_precision
    from src.utils.benchmark_decode import (
        build_continuation_trace,
        build_infilling_trace,
        discover_checkpoints,
        generate_continuation_tokens,
        generate_middle_tokens,
        load_vocab,
    )
    from src.utils.absolute_benchmark_scoring import attach_absolute_capability_scores
    from src.utils.benchmarking import build_benchmark_manifest, load_benchmark_config
    from src.utils.checkpoint_selection import score_checkpoint_results
    from src.utils.config_io import dump_json_file
    from src.utils.report_plots import write_eval_report_plot, write_training_metrics_dashboard
    from src.utils.torch_utils import lazy_import_torch, resolve_torch_device
    from src.utils.training_metrics import (
        load_training_metrics,
        prefilter_checkpoints_by_valid_loss,
        resolve_metrics_path,
        summarize_training_metrics,
    )

    torch = lazy_import_torch()

    checkpoints = discover_checkpoints(
        checkpoint_dir=checkpoint_dir,
        limit=args.limit_checkpoints,
        policy=args.checkpoint_policy,
        sample_count=args.sample_count,
        include_aliases=bool(args.include_alias_checkpoints),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No *.pt checkpoints found under: {checkpoint_dir}")

    first_ckpt_payload = _load_checkpoint(torch, checkpoints[0])
    ckpt_model_cfg = first_ckpt_payload.get("model_config")
    if isinstance(ckpt_model_cfg, dict):
        reference_config = DecoderConfig.from_dict(ckpt_model_cfg)
        fallback_model_config = Path(str(train_mapping.get("model_config", "configs/train/model_base.yaml")))
        if not fallback_model_config.is_absolute():
            fallback_model_config = (project_root / fallback_model_config).resolve()
    else:
        fallback_model_config = Path(str(train_mapping.get("model_config", "configs/train/model_base.yaml")))
        if not fallback_model_config.is_absolute():
            fallback_model_config = (project_root / fallback_model_config).resolve()
        reference_config = DecoderConfig.from_yaml(fallback_model_config.resolve())

    vocab_path = args.vocab_path if args.vocab_path.is_absolute() else (project_root / args.vocab_path)
    token_to_id, id_to_token = load_vocab(vocab_path.resolve())
    grammar_fsm = TuneFlowGrammarFSM.from_vocab(token_to_id)

    metrics_path = resolve_metrics_path(checkpoint_dir, None)
    training_metrics_payload = load_training_metrics(metrics_path)
    training_summary = summarize_training_metrics(training_metrics_payload)
    prefilter_top_k = (
        int(args.prefilter_top_k_by_valid_loss)
        if args.prefilter_top_k_by_valid_loss is not None
        else _default_prefilter_top_k(preset=args.preset, config_path=config_path)
    )
    # fast 前先做训练期 valid_loss 预筛，尽量减少需要真正推理的 checkpoint 数量。
    checkpoints, checkpoint_prefilter = prefilter_checkpoints_by_valid_loss(
        checkpoints,
        training_metrics_payload,
        top_k=prefilter_top_k,
        preserve_earliest=int(args.prefilter_preserve_earliest),
    )

    fast_config_path = args.fast_config if args.fast_config.is_absolute() else (project_root / args.fast_config)
    formal_config_path = args.formal_config if args.formal_config.is_absolute() else (project_root / args.formal_config)
    fast_config = load_benchmark_config(fast_config_path.resolve())
    formal_config = load_benchmark_config(formal_config_path.resolve())

    eval_jsonl_path = args.eval_jsonl if args.eval_jsonl.is_absolute() else (project_root / args.eval_jsonl)
    eval_tok_path = args.eval_tok if args.eval_tok.is_absolute() else (project_root / args.eval_tok)
    fast_manifest = build_benchmark_manifest(
        eval_jsonl_path=eval_jsonl_path.resolve(),
        eval_tok_path=eval_tok_path.resolve(),
        config=fast_config,
        max_positions=int(reference_config.max_position_embeddings),
    )
    formal_manifest = build_benchmark_manifest(
        eval_jsonl_path=eval_jsonl_path.resolve(),
        eval_tok_path=eval_tok_path.resolve(),
        config=formal_config,
        max_positions=int(reference_config.max_position_embeddings),
    )
    fast_manifest_name, formal_manifest_name = _artifact_file_names(task_scope)
    fast_manifest_path = benchmark_root / fast_manifest_name
    formal_manifest_path = benchmark_root / formal_manifest_name
    dump_json_file(fast_manifest_path, fast_manifest, ensure_ascii=False, indent=2)
    dump_json_file(formal_manifest_path, formal_manifest, ensure_ascii=False, indent=2)

    print(
        f"[{_TASK_LABELS[task_scope]}] run_id={run_id} checkpoints={len(checkpoints)} "
        f"fast_cases={fast_manifest['case_count']} formal_cases={formal_manifest['case_count']}"
    )
    if checkpoint_prefilter.get("enabled"):
        print(
            f"[{_TASK_LABELS[task_scope]}] checkpoint prefilter by valid_loss -> "
            f"kept {checkpoint_prefilter.get('selected_count')}/{checkpoint_prefilter.get('original_count')}"
        )

    sample_manifest, export_row_ids = _build_sample_capture_manifest(
        fast_manifest=fast_manifest,
        case_count=int(fast_config["sample_export_case_count"]),
    )

    fast_results: list[dict[str, Any]] = []
    fast_samples_by_checkpoint: dict[str, dict[str, list[dict[str, Any]]]] = {}
    # fast: 小样本扫点，主要目标是缩小 formal 复评范围。
    for index, ckpt_path in enumerate(checkpoints, start=1):
        print(f"[{_TASK_LABELS[task_scope]}][fast] ({index}/{len(checkpoints)}) checkpoint={ckpt_path.name}")
        result, captured = _evaluate_checkpoint_on_manifest(
            ckpt_path=ckpt_path,
            manifest=sample_manifest,
            capture_row_ids=export_row_ids,
            task_scope=task_scope,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            grammar_fsm=grammar_fsm,
            training_metrics_payload=training_metrics_payload,
            args=args,
            fallback_model_config_path=fallback_model_config,
            torch=torch,
            DecoderConfig=DecoderConfig,
            DecoderForCausalLM=DecoderForCausalLM,
            load_checkpoint_fn=_load_checkpoint,
            autocast_context_fn=_autocast_context,
            resolve_precision_fn=_resolve_precision,
            resolve_torch_device_fn=resolve_torch_device,
            generate_continuation_tokens_fn=generate_continuation_tokens,
            build_continuation_trace_fn=build_continuation_trace,
            generate_middle_tokens_fn=generate_middle_tokens,
            build_infilling_trace_fn=build_infilling_trace,
        )
        fast_results.append(result)
        fast_samples_by_checkpoint[str(ckpt_path)] = captured

    selection_profile = _TASK_PROFILE_NAMES[task_scope]
    fast_results = attach_absolute_capability_scores(fast_results)
    fast_results, fast_selection = score_checkpoint_results(fast_results, profile=selection_profile)
    candidate_rows = fast_selection["leaderboard"][: max(1, int(fast_config["sample_export_top_k"]))]
    candidate_paths = [Path(str(row["checkpoint_path"])) for row in candidate_rows]

    formal_results: list[dict[str, Any]] = []
    # formal: 只复评 fast 前几名，用全量样本稳定最终推荐。
    for index, ckpt_path in enumerate(candidate_paths, start=1):
        print(f"[{_TASK_LABELS[task_scope]}][formal] ({index}/{len(candidate_paths)}) checkpoint={ckpt_path.name}")
        result, _ = _evaluate_checkpoint_on_manifest(
            ckpt_path=ckpt_path,
            manifest=formal_manifest,
            capture_row_ids=None,
            task_scope=task_scope,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            grammar_fsm=grammar_fsm,
            training_metrics_payload=training_metrics_payload,
            args=args,
            fallback_model_config_path=fallback_model_config,
            torch=torch,
            DecoderConfig=DecoderConfig,
            DecoderForCausalLM=DecoderForCausalLM,
            load_checkpoint_fn=_load_checkpoint,
            autocast_context_fn=_autocast_context,
            resolve_precision_fn=_resolve_precision,
            resolve_torch_device_fn=resolve_torch_device,
            generate_continuation_tokens_fn=generate_continuation_tokens,
            build_continuation_trace_fn=build_continuation_trace,
            generate_middle_tokens_fn=generate_middle_tokens,
            build_infilling_trace_fn=build_infilling_trace,
        )
        formal_results.append(result)

    formal_results = attach_absolute_capability_scores(formal_results)
    formal_results, formal_selection = score_checkpoint_results(formal_results, profile=selection_profile)
    formal_by_path = {str(item["checkpoint_path"]): item for item in formal_results}

    combined_results: list[dict[str, Any]] = []
    for fast_result in fast_results:
        checkpoint_path = str(fast_result["checkpoint_path"])
        if checkpoint_path in formal_by_path:
            merged = dict(fast_result)
            merged.update(formal_by_path[checkpoint_path])
            merged["fast_pass_balanced_score"] = fast_result.get("balanced_score")
            merged["fast_pass_balanced_rank"] = fast_result.get("balanced_rank")
            merged["evaluation_tier"] = "formal"
            combined_results.append(merged)
        else:
            combined_results.append(dict(fast_result))

    combined_results = attach_absolute_capability_scores(combined_results)
    combined_results, combined_selection = score_checkpoint_results(combined_results, profile=selection_profile)
    recommended = combined_selection.get("recommended_checkpoint")
    if recommended is None:
        recommended = formal_selection.get("recommended_checkpoint")

    samples_root = benchmark_root / "samples"
    sample_artifacts: dict[str, dict[str, dict[str, str]]] = {
        "final_top3": {},
        "formal_candidates": {},
    }
    exported_samples: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        "final_top3": {},
        "formal_candidates": {},
    }
    top_results = sorted(
        [item for item in combined_results if item.get("balanced_rank") is not None],
        key=lambda item: int(item["balanced_rank"]),
    )[:3]

    group_artifacts, group_exports = _write_sample_group(
        samples_root=samples_root,
        group_name="formal_candidates",
        checkpoint_paths=candidate_paths,
        captured_by_checkpoint=fast_samples_by_checkpoint,
        run_id=run_id,
        task_scope=task_scope,
        log_prefix=_TASK_LABELS[task_scope],
    )
    sample_artifacts["formal_candidates"] = group_artifacts
    exported_samples["formal_candidates"] = group_exports
    final_top_paths = [Path(str(result["checkpoint_path"])) for result in top_results]
    group_artifacts, group_exports = _write_sample_group(
        samples_root=samples_root,
        group_name="final_top3",
        checkpoint_paths=final_top_paths,
        captured_by_checkpoint=fast_samples_by_checkpoint,
        run_id=run_id,
        task_scope=task_scope,
        log_prefix=_TASK_LABELS[task_scope],
    )
    sample_artifacts["final_top3"] = group_artifacts
    exported_samples["final_top3"] = group_exports
    plot_results = sorted(combined_results, key=lambda item: int(item.get("step", -1)))
    plot_artifacts: dict[str, str] = {}
    if plot_results:
        core_plot_path = benchmark_root / f"{_TASK_LABELS[task_scope]}_core_metrics.png"
        diagnostic_plot_path = benchmark_root / f"{_TASK_LABELS[task_scope]}_diagnostics.png"
        absolute_plot_path = benchmark_root / f"{_TASK_LABELS[task_scope]}_absolute_capabilities.png"
        write_eval_report_plot(
            report_path=report_path,
            report={"run_id": run_id, "results": plot_results},
            title=f"{_TASK_TITLES[task_scope]} - 核心指标",
            metric_specs=_plot_metric_specs_v2(task_scope, diagnostics=False),
            chart_path=core_plot_path,
        )
        write_eval_report_plot(
            report_path=report_path,
            report={"run_id": run_id, "results": plot_results},
            title=f"{_TASK_TITLES[task_scope]} - 诊断指标",
            metric_specs=_plot_metric_specs_v2(task_scope, diagnostics=True),
            chart_path=diagnostic_plot_path,
        )
        write_eval_report_plot(
            report_path=report_path,
            report={"run_id": run_id, "results": plot_results},
            title=f"{_TASK_TITLES[task_scope]} - 绝对能力面板",
            metric_specs=_absolute_plot_metric_specs_v2(task_scope),
            chart_path=absolute_plot_path,
        )
        plot_artifacts["核心指标"] = str(core_plot_path)
        plot_artifacts["诊断指标"] = str(diagnostic_plot_path)
        plot_artifacts["绝对能力面板"] = str(absolute_plot_path)
    if training_metrics_payload.get("train_by_step") or training_metrics_payload.get("eval_by_step"):
        training_plot_path = benchmark_root / f"{_TASK_LABELS[task_scope]}_training_health.png"
        write_training_metrics_dashboard(
            chart_path=training_plot_path,
            metrics_payload=training_metrics_payload,
            run_id=run_id,
        )
        plot_artifacts["训练健康度"] = str(training_plot_path)
    config_paths_payload = {
        "train_config": None if config_path is None else str(config_path),
        "fast_config": str(fast_config_path.resolve()),
        "formal_config": str(formal_config_path.resolve()),
        "vocab_path": str(vocab_path.resolve()),
    }
    decoding_payload = {
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
    }
    report = {
        "run_id": run_id,
        "task_scope": task_scope,
        "created_at": time.time(),
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_path": None if metrics_path is None else str(metrics_path),
        "config_paths": config_paths_payload,
        "benchmark_configs": {
            "fast": fast_config,
            "formal": formal_config,
        },
        "decoding": decoding_payload,
        "manifests": {
            "fast_path": str(fast_manifest_path),
            "formal_path": str(formal_manifest_path),
            "fast_case_count": fast_manifest["case_count"],
            "formal_case_count": formal_manifest["case_count"],
        },
        "checkpoint_prefilter": checkpoint_prefilter,
        "training_health": {
            "summary": training_summary,
        },
        "plot_artifacts": plot_artifacts,
        "summary": {
            "recommended_checkpoint": recommended,
            "top_k_candidates": [str(path.name) for path in candidate_paths],
            "final_top3_checkpoints": [str(path.name) for path in final_top_paths],
            "sample_artifacts": sample_artifacts,
        },
        "fast_pass": {
            "selection": fast_selection,
            "results": [_to_jsonable_result(result) for result in fast_results],
        },
        "formal_pass": {
            "selection": formal_selection,
            "results": [_to_jsonable_result(result) for result in formal_results],
        },
        "final_selection": {
            "selection": combined_selection,
            "recommended_checkpoint": recommended,
            "leaderboard": combined_selection["leaderboard"],
        },
        "sample_artifacts": sample_artifacts,
    }
    dump_json_file(report_path, report, ensure_ascii=False, indent=2)

    summary_text = _build_summary_markdown_v2(
        run_id=run_id,
        task_scope=task_scope,
        benchmark_root=benchmark_root,
        recommended=recommended,
        top_results=top_results,
        training_summary=training_summary,
        plot_artifacts=plot_artifacts,
        sample_artifacts=sample_artifacts,
        exported_samples=exported_samples,
        manifest_stats={
            "fast_case_count": fast_manifest["case_count"],
            "formal_case_count": formal_manifest["case_count"],
            "candidate_count": len(candidate_paths),
        },
        checkpoint_prefilter=checkpoint_prefilter,
        evaluation_context={
            "config_paths": config_paths_payload,
            "benchmark_configs": {
                "fast": fast_config,
                "formal": formal_config,
            },
            "decoding": decoding_payload,
        },
    )
    summary_path.write_text(summary_text, encoding="utf-8")

    if isinstance(recommended, dict):
        print(
            f"[{_TASK_LABELS[task_scope]}] recommended checkpoint -> "
            f"{recommended.get('checkpoint_name')} "
            f"(step={recommended.get('step')}, score={float(recommended.get('balanced_score', float('nan'))):.4f})"
        )
    print(f"[{_TASK_LABELS[task_scope]}] report -> {report_path}")
    print(f"[{_TASK_LABELS[task_scope]}] summary -> {summary_path}")


if __name__ == "__main__":
    main()
