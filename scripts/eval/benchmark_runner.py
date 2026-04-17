#!/usr/bin/env python
"""Shared TuneFlow benchmark runner used by all eval entrypoints."""

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
    "all": "TuneFlow 综合 Benchmark",
    "infilling": "TuneFlow 补全 Benchmark",
    "continuation": "TuneFlow 续写 Benchmark",
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


def _clean_benchmark_outputs(benchmark_root: Path, task_scope: str) -> None:
    """Remove stale artifacts for the current benchmark task before regenerating them."""
    benchmark_root.mkdir(parents=True, exist_ok=True)
    fast_manifest_name, formal_manifest_name = _artifact_file_names(task_scope)
    artifact_paths = [
        benchmark_root / _TASK_REPORT_NAMES[task_scope],
        benchmark_root / _TASK_SUMMARY_NAMES[task_scope],
        benchmark_root / fast_manifest_name,
        benchmark_root / formal_manifest_name,
        benchmark_root / f"{_TASK_LABELS[task_scope]}_core_metrics.png",
        benchmark_root / f"{_TASK_LABELS[task_scope]}_diagnostics.png",
        benchmark_root / f"{_TASK_LABELS[task_scope]}_training_health.png",
    ]
    for artifact_path in artifact_paths:
        if artifact_path.exists():
            artifact_path.unlink()

    samples_root = benchmark_root / "samples"
    if not samples_root.exists():
        return

    task_sample_names = [f"{task_name}.json" for task_name in _task_names_for_scope(task_scope)]
    for checkpoint_dir in samples_root.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        for sample_name in task_sample_names:
            sample_path = checkpoint_dir / sample_name
            if sample_path.exists():
                sample_path.unlink()
        if not any(checkpoint_dir.iterdir()):
            checkpoint_dir.rmdir()
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
        default=128,
        help=(
            "单条样本最多允许生成多少个新 token。作用：限制续写/补全长度，防止无限生成。\n"
            "值越大越慢，但也更不容易被截断。\n"
            "例子：--max-new-tokens 96"
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
        default=8,
        help=(
            "checkpoint 预筛数量。作用：在 fast benchmark 前，先按训练期 valid_loss 只保留 top K 个 checkpoint。\n"
            "设为 0 表示关闭预筛，全部 checkpoint 都跑。\n"
            "例子：--prefilter-top-k-by-valid-loss 8"
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
    return parser.parse_args(argv)


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
        "balanced_score": result.get("balanced_score"),
        "balanced_rank": result.get("balanced_rank"),
        "balanced_score_coverage": result.get("balanced_score_coverage"),
        "gate_passed": result.get("gate_passed"),
        "gate_details": result.get("gate_details"),
        "gate_failed_reasons": result.get("gate_failed_reasons"),
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
        "empty_bar_rate": record.get("empty_bar_rate"),
    }
    if task == "continuation":
        payload["target_tokens"] = list(case["continuation_case"]["target_tokens"])
    else:
        payload["target_hole_tokens"] = list(case["infilling_case"]["target_hole_tokens"])
    return payload


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
    continuation_duration_l1: list[float] = []
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
    infilling_syntax_invalid = 0
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
                continuation_duration_l1.append(float(raw_record["duration_bin_l1_distance"]))
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
                infilling_syntax_invalid += int(not bool(raw_infill_record["is_structurally_valid"]))
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
    diagnostic_metrics = {
        "continuation_first_event_hit_rate": _safe_rate(continuation_first_event_hits, continuation_first_event_total),
        "continuation_missing_eos_rate": _safe_rate(continuation_missing_eos, continuation_attempted),
        "continuation_syntax_invalid_rate": _safe_rate(continuation_syntax_invalid, continuation_attempted),
        "infilling_syntax_invalid_rate": _safe_rate(infilling_syntax_invalid, infilling_attempted),
        "append_eos_recoverable_rate": _safe_rate(continuation_append_eos_recoverable, continuation_attempted),
        "low_density_bar_rate": _safe_mean(continuation_low_density_rates),
        "multi_empty_bar_run_rate": _safe_rate(continuation_multi_empty_runs, continuation_attempted),
        "generated_bar_delta_mean": _safe_mean(continuation_bar_deltas),
        "generated_event_delta_mean": _safe_mean(continuation_event_deltas),
        "pitch_span_delta_mean": _safe_mean(continuation_pitch_span_deltas),
        "duration_bin_l1_distance": _safe_mean(continuation_duration_l1),
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


def _build_summary_markdown(
    *,
    run_id: str,
    task_scope: str,
    benchmark_root: Path,
    recommended: dict[str, Any] | None,
    top_results: list[dict[str, Any]],
    training_summary: dict[str, Any],
    plot_artifacts: dict[str, str],
    sample_artifacts: dict[str, dict[str, str]],
    exported_samples: dict[str, dict[str, list[dict[str, Any]]]],
    manifest_stats: dict[str, Any],
    checkpoint_prefilter: dict[str, Any],
) -> str:
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
        artifact_paths = sample_artifacts.get(checkpoint_name, {})
        lines.append("### 样本产物")
        for task_name in _task_names_for_scope(task_scope):
            if task_name in artifact_paths:
                relative_path = _relative_artifact_path(benchmark_root, artifact_paths.get(task_name, ""))
                lines.append(f"- `{task_name}` 样本文件：`{relative_path}`")

        checkpoint_samples = exported_samples.get(checkpoint_name, {})
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


def main(*, task_scope: str = "all", argv: list[str] | None = None) -> None:
    project_root = _ensure_project_root_on_path()
    os.chdir(project_root)
    args = _parse_args(task_scope=task_scope, argv=argv)

    checkpoint_dir, train_mapping, run_id = _resolve_eval_target(project_root, args)
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
    from src.utils.benchmarking import build_benchmark_manifest, load_benchmark_config, select_export_cases
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
    # fast 前先做训练期 valid_loss 预筛，尽量减少需要真正推理的 checkpoint 数量。
    checkpoints, checkpoint_prefilter = prefilter_checkpoints_by_valid_loss(
        checkpoints,
        training_metrics_payload,
        top_k=int(args.prefilter_top_k_by_valid_loss),
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

    export_cases = select_export_cases(
        fast_manifest["cases"],
        count=int(fast_config["sample_export_case_count"]),
    )
    export_row_ids = {int(case["row_id"]) for case in export_cases}

    fast_results: list[dict[str, Any]] = []
    fast_samples_by_checkpoint: dict[str, dict[str, list[dict[str, Any]]]] = {}
    # fast: 小样本扫点，主要目标是缩小 formal 复评范围。
    for index, ckpt_path in enumerate(checkpoints, start=1):
        print(f"[{_TASK_LABELS[task_scope]}][fast] ({index}/{len(checkpoints)}) checkpoint={ckpt_path.name}")
        result, captured = _evaluate_checkpoint_on_manifest(
            ckpt_path=ckpt_path,
            manifest=fast_manifest,
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

    combined_results, combined_selection = score_checkpoint_results(combined_results, profile=selection_profile)
    recommended = combined_selection.get("recommended_checkpoint")
    if recommended is None:
        recommended = formal_selection.get("recommended_checkpoint")

    samples_root = benchmark_root / "samples"
    sample_artifacts: dict[str, dict[str, str]] = {}
    exported_samples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    sample_tasks = _task_names_for_scope(task_scope)
    # 样本导出直接复用 fast 阶段抓到的结果，不再重复推理。
    for ckpt_path in candidate_paths:
        checkpoint_name = ckpt_path.name
        print(f"[{_TASK_LABELS[task_scope]}][samples] checkpoint={checkpoint_name}")
        captured = fast_samples_by_checkpoint.get(
            str(ckpt_path),
            {"continuation": [], "infilling": []},
        )
        checkpoint_sample_dir = samples_root / ckpt_path.stem
        checkpoint_sample_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_artifacts: dict[str, str] = {}
        checkpoint_exports: dict[str, list[dict[str, Any]]] = {}
        for task_name in sample_tasks:
            sample_path = checkpoint_sample_dir / f"{task_name}.json"
            payload = {
                "run_id": run_id,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(ckpt_path),
                "task": task_name,
                "cases": captured[task_name],
            }
            dump_json_file(sample_path, payload, ensure_ascii=False, indent=2)
            checkpoint_artifacts[task_name] = str(sample_path)
            checkpoint_exports[task_name] = captured[task_name]
        sample_artifacts[checkpoint_name] = checkpoint_artifacts
        exported_samples[checkpoint_name] = checkpoint_exports

    top_results = sorted(
        [item for item in combined_results if item.get("balanced_rank") is not None],
        key=lambda item: int(item["balanced_rank"]),
    )[:3]
    plot_results = sorted(combined_results, key=lambda item: int(item.get("step", -1)))
    plot_artifacts: dict[str, str] = {}
    if plot_results:
        core_plot_path = benchmark_root / f"{_TASK_LABELS[task_scope]}_core_metrics.png"
        diagnostic_plot_path = benchmark_root / f"{_TASK_LABELS[task_scope]}_diagnostics.png"
        write_eval_report_plot(
            report_path=report_path,
            report={"run_id": run_id, "results": plot_results},
            title=f"{_TASK_TITLES[task_scope]} - 核心指标",
            metric_specs=_plot_metric_specs(task_scope, diagnostics=False),
            chart_path=core_plot_path,
        )
        write_eval_report_plot(
            report_path=report_path,
            report={"run_id": run_id, "results": plot_results},
            title=f"{_TASK_TITLES[task_scope]} - 诊断指标",
            metric_specs=_plot_metric_specs(task_scope, diagnostics=True),
            chart_path=diagnostic_plot_path,
        )
        plot_artifacts["核心指标对比图"] = str(core_plot_path)
        plot_artifacts["诊断指标对比图"] = str(diagnostic_plot_path)
    if training_metrics_payload.get("train_by_step") or training_metrics_payload.get("eval_by_step"):
        training_plot_path = benchmark_root / f"{_TASK_LABELS[task_scope]}_training_health.png"
        write_training_metrics_dashboard(
            chart_path=training_plot_path,
            metrics_payload=training_metrics_payload,
            run_id=run_id,
        )
        plot_artifacts["训练健康度图"] = str(training_plot_path)
    report = {
        "run_id": run_id,
        "task_scope": task_scope,
        "created_at": time.time(),
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_path": None if metrics_path is None else str(metrics_path),
        "benchmark_configs": {
            "fast": fast_config,
            "formal": formal_config,
        },
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

    summary_text = _build_summary_markdown(
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
