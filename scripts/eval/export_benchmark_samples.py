#!/usr/bin/env python
"""只重导出指定 checkpoint 的 benchmark samples，不重跑完整排名。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "只重放指定 checkpoint 的少量样本用例，重新导出 benchmark samples，"
            "跳过 fast/formal 排名和图表生成。"
        )
    )
    parser.add_argument(
        "--task-scope",
        type=str,
        default="all",
        choices=["all", "infilling", "continuation"],
        help="使用哪一套 benchmark report 命名空间。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="训练 run 配置路径；不传时使用 --preset。",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="small",
        choices=["small", "full"],
        help="内置训练预设；传入 --config 时忽略。",
    )
    parser.add_argument(
        "--sample-group",
        type=str,
        default="final_top3",
        choices=["final_top3", "formal_candidates"],
        help="从已有 benchmark report 中选择哪个 checkpoint 分组来重导出。",
    )
    parser.add_argument(
        "--checkpoint-names",
        type=str,
        default=None,
        help="可选，逗号分隔的 checkpoint 名称；传入后覆盖 report 里的默认分组。",
    )
    parser.add_argument(
        "--case-count",
        type=int,
        default=None,
        help="可选，重导出的 case 数量；默认使用 fast benchmark 配置值。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备。",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="推理精度。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="每个样本用例最多生成多少 token。",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("data/tokenized/tokenizer_vocab.json"),
        help="tokenizer 词表路径。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度。`0` 表示贪心解码，`>0` 时启用随机采样。",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="top-p 采样阈值，取值范围 `(0, 1]`。",
    )
    args = parser.parse_args()
    if float(args.temperature) < 0.0:
        parser.error("--temperature must be >= 0.")
    if not (0.0 < float(args.top_p) <= 1.0):
        parser.error("--top-p must be within (0, 1].")
    return args


def _parse_checkpoint_names(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_group_checkpoint_names(report_payload: dict[str, Any], sample_group: str) -> list[str]:
    summary = dict(report_payload.get("summary", {}))
    if sample_group == "final_top3":
        checkpoint_names = summary.get("final_top3_checkpoints")
    else:
        checkpoint_names = summary.get("top_k_candidates")
    if not isinstance(checkpoint_names, list) or not checkpoint_names:
        raise ValueError(f"Checkpoint group `{sample_group}` is missing in benchmark report.")
    return [str(item) for item in checkpoint_names]


def _task_report_name(task_scope: str) -> str:
    mapping = {
        "all": "benchmark_report.json",
        "infilling": "benchmark_infilling_report.json",
        "continuation": "benchmark_continuation_report.json",
    }
    return mapping[task_scope]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    from scripts.eval import benchmark_runner as runner

    project_root = runner._ensure_project_root_on_path()
    os.chdir(project_root)
    args = _parse_args()

    checkpoint_dir, train_mapping, run_id = runner._resolve_eval_target(project_root, args)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    benchmark_root = project_root / "outputs" / "benchmark" / run_id
    report_path = benchmark_root / _task_report_name(args.task_scope)
    if not report_path.exists():
        raise FileNotFoundError(f"benchmark report not found: {report_path}")

    from src.decoding import TuneFlowGrammarFSM
    from src.model.configuration import DecoderConfig
    from src.model.modeling import DecoderForCausalLM
    from src.training.train_base import _autocast_context, _load_checkpoint, _resolve_precision
    from src.utils.benchmark_decode import (
        build_continuation_trace,
        build_infilling_trace,
        generate_continuation_tokens,
        generate_middle_tokens,
        load_vocab,
    )
    from src.utils.benchmarking import load_benchmark_config
    from src.utils.config_io import load_json_file
    from src.utils.torch_utils import lazy_import_torch, resolve_torch_device
    from src.utils.training_metrics import load_training_metrics, resolve_metrics_path

    report_payload = load_json_file(report_path, "benchmark report")
    requested_checkpoint_names = _parse_checkpoint_names(args.checkpoint_names)
    checkpoint_names = (
        requested_checkpoint_names
        if requested_checkpoint_names
        else _resolve_group_checkpoint_names(report_payload, args.sample_group)
    )

    fast_manifest_path = Path(str(dict(report_payload.get("manifests", {})).get("fast_path", "")))
    if not fast_manifest_path.is_absolute():
        fast_manifest_path = (project_root / fast_manifest_path).resolve()
    if not fast_manifest_path.exists():
        raise FileNotFoundError(f"fast manifest not found: {fast_manifest_path}")
    fast_manifest = load_json_file(fast_manifest_path, "fast benchmark manifest")

    config_path = None
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else (project_root / args.config)
        config_path = config_path.resolve()
    elif args.preset is not None:
        config_path = runner._resolve_preset_config(project_root, args.preset)

    prefilter_top_k = runner._default_prefilter_top_k(preset=args.preset, config_path=config_path)
    fast_config_path = project_root / "configs" / "eval" / "benchmark_fast.yaml"
    fast_config = load_benchmark_config(fast_config_path.resolve())
    export_case_count = int(args.case_count) if args.case_count is not None else int(fast_config["sample_export_case_count"])

    sample_manifest, export_row_ids = runner._build_sample_capture_manifest(
        fast_manifest=fast_manifest,
        case_count=export_case_count,
    )

    vocab_path = args.vocab_path if args.vocab_path.is_absolute() else (project_root / args.vocab_path)
    token_to_id, id_to_token = load_vocab(vocab_path.resolve())
    grammar_fsm = TuneFlowGrammarFSM.from_vocab(token_to_id)

    torch = lazy_import_torch()
    metrics_path = resolve_metrics_path(checkpoint_dir, None)
    training_metrics_payload = load_training_metrics(metrics_path)

    checkpoint_paths: list[Path] = []
    for checkpoint_name in checkpoint_names:
        ckpt_path = checkpoint_dir / checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        checkpoint_paths.append(ckpt_path)

    fallback_model_config = Path(str(train_mapping.get("model_config", "configs/train/model_base.yaml")))
    if not fallback_model_config.is_absolute():
        fallback_model_config = (project_root / fallback_model_config).resolve()
    samples_root = benchmark_root / "samples"

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[sample_export] run_id={run_id} task_scope={args.task_scope} sample_group={args.sample_group} "
        f"checkpoints={len(checkpoint_paths)} cases={sample_manifest['case_count']} started_at={started_at}"
    )
    print(f"[sample_export] checkpoints={', '.join(path.name for path in checkpoint_paths)}")

    eval_args = argparse.Namespace(
        device=args.device,
        precision=args.precision,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        prefilter_top_k_by_valid_loss=prefilter_top_k,
        prefilter_preserve_earliest=4,
    )

    captured_by_checkpoint: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for index, ckpt_path in enumerate(checkpoint_paths, start=1):
        print(f"[sample_export] ({index}/{len(checkpoint_paths)}) checkpoint={ckpt_path.name}")
        _result, captured = runner._evaluate_checkpoint_on_manifest(
            ckpt_path=ckpt_path,
            manifest=sample_manifest,
            capture_row_ids=export_row_ids,
            task_scope=args.task_scope,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            grammar_fsm=grammar_fsm,
            training_metrics_payload=training_metrics_payload,
            args=eval_args,
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
        captured_by_checkpoint[str(ckpt_path)] = captured

    runner._write_sample_group(
        samples_root=samples_root,
        group_name=args.sample_group,
        checkpoint_paths=checkpoint_paths,
        captured_by_checkpoint=captured_by_checkpoint,
        run_id=run_id,
        task_scope=args.task_scope,
        log_prefix="sample_export",
        extra_payload_fields={
            "export_mode": "samples_only",
            "exported_at": started_at,
            "fast_manifest_path": str(fast_manifest_path),
        },
    )

    print("[sample_export] done")


if __name__ == "__main__":
    main()
