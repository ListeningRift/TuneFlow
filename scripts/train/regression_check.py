#!/usr/bin/env python
"""Smoke regression for train/save/resume/benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _ensure_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a smoke regression across train/save/resume/benchmark."
    )
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="Python executable used for subprocess commands.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. Defaults to regression_smoke_<timestamp>.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device forwarded to training and benchmark scripts.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="Precision forwarded to training and benchmark scripts.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length used by the smoke training run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used by the smoke training run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the smoke run.",
    )
    return parser.parse_args()


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[regression_check] cmd:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _check_runtime_deps(python_exec: str, cwd: Path) -> None:
    probe = subprocess.run(
        [python_exec, "-c", "import torch"],  # noqa: S603,S607
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        stderr = (probe.stderr or "").strip()
        stdout = (probe.stdout or "").strip()
        detail = stderr if stderr else stdout
        raise SystemExit(
            "Regression check requires torch in the target Python environment. "
            f"python_exec={python_exec}. "
            f"Import error: {detail}"
        )


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _assert_exists(path: Path, msg: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{msg}: {path}")


def _build_train_payload(
    output_dir: Path,
    device: str,
    precision: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    steps: int,
    resume_from: Path | None,
) -> dict:
    return {
        "train": {
            "model_config": "configs/train/model_base.yaml",
            "train_idx": "data/tokenized/train.idx.json",
            "train_bin": None,
            "valid_idx": "data/tokenized/valid.idx.json",
            "valid_bin": None,
            "resume_from": None if resume_from is None else str(resume_from),
            "device": device,
            "precision": precision,
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
            "grad_accum_steps": 1,
            "seq_len": seq_len,
            "lr": 0.0003,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "fim_ratio": 1.0,
            "fim_min_span": 4,
            "fim_max_span": 16,
            "scheduler": "none",
            "warmup_steps": 0,
            "min_lr_scale": 0.1,
            "log_every": 1,
            "eval_every": 1,
            "eval_batches": 1,
            "save_every": 1,
            "save_best": True,
            "no_restore_rng": False,
            "output_dir": str(output_dir),
            "metrics_path": None,
        }
    }


def _build_benchmark_payload(
    *,
    tier: str,
    sample_count: int,
    per_bucket_limit: int,
    sample_export_case_count: int,
    sample_export_top_k: int,
) -> dict:
    return {
        "tier": tier,
        "seed": 42,
        "sample_count": sample_count,
        "per_bucket_limit": per_bucket_limit,
        "min_prefix_tokens": 32,
        "continuation_prefix_ratio_min": 0.35,
        "continuation_prefix_ratio_max": 0.70,
        "infilling_hole_ratio_min": 0.10,
        "infilling_hole_ratio_max": 0.25,
        "sample_export_case_count": sample_export_case_count,
        "sample_export_top_k": sample_export_top_k,
    }


def main() -> None:
    args = _parse_args()
    project_root = _ensure_project_root()

    run_id = args.run_id or f"regression_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = (project_root / "outputs" / "checkpoints" / run_id).resolve()
    benchmark_root = (project_root / "outputs" / "benchmark" / run_id).resolve()
    benchmark_report_path = benchmark_root / "benchmark_report.json"
    benchmark_summary_path = benchmark_root / "benchmark_summary.md"

    _check_runtime_deps(args.python_exec, project_root)

    required = [
        project_root / "data" / "tokenized" / "train.idx.json",
        project_root / "data" / "tokenized" / "valid.idx.json",
        project_root / "data" / "tokenized" / "eval.tok",
        project_root / "data" / "tokenized" / "tokenizer_vocab.json",
        project_root / "data" / "eval" / "fixed_eval.jsonl",
    ]
    for path in required:
        _assert_exists(path, "required input not found")

    cfg_dir = project_root / "outputs" / "tmp" / "regression_check" / run_id
    cfg_phase1 = cfg_dir / "train_phase1.yaml"
    cfg_phase2 = cfg_dir / "train_phase2_resume.yaml"
    fast_benchmark_cfg = cfg_dir / "benchmark_fast_smoke.yaml"
    formal_benchmark_cfg = cfg_dir / "benchmark_formal_smoke.yaml"

    payload1 = _build_train_payload(
        output_dir=output_dir,
        device=args.device,
        precision=args.precision,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        steps=1,
        resume_from=None,
    )
    _write_yaml(cfg_phase1, payload1)
    _run_cmd(
        [
            args.python_exec,
            "scripts/train/train_base_from_config.py",
            "--config",
            str(cfg_phase1),
        ],
        cwd=project_root,
    )

    latest_pt = output_dir / "latest.pt"
    step1_pt = output_dir / "step_1.pt"
    _assert_exists(latest_pt, "phase1 latest checkpoint missing")
    _assert_exists(step1_pt, "phase1 step_1 checkpoint missing")

    payload2 = _build_train_payload(
        output_dir=output_dir,
        device=args.device,
        precision=args.precision,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        steps=2,
        resume_from=latest_pt,
    )
    _write_yaml(cfg_phase2, payload2)
    _run_cmd(
        [
            args.python_exec,
            "scripts/train/train_base_from_config.py",
            "--config",
            str(cfg_phase2),
        ],
        cwd=project_root,
    )

    step2_pt = output_dir / "step_2.pt"
    last_pt = output_dir / "last.pt"
    metrics_jsonl = output_dir / "metrics.jsonl"
    _assert_exists(step2_pt, "phase2 step_2 checkpoint missing")
    _assert_exists(last_pt, "phase2 last checkpoint missing")
    _assert_exists(metrics_jsonl, "training metrics.jsonl missing")

    _write_yaml(
        fast_benchmark_cfg,
        _build_benchmark_payload(
            tier="fast",
            sample_count=8,
            per_bucket_limit=2,
            sample_export_case_count=4,
            sample_export_top_k=1,
        ),
    )
    _write_yaml(
        formal_benchmark_cfg,
        _build_benchmark_payload(
            tier="formal",
            sample_count=8,
            per_bucket_limit=2,
            sample_export_case_count=4,
            sample_export_top_k=1,
        ),
    )

    _run_cmd(
        [
            args.python_exec,
            "scripts/eval/eval_all.py",
            "--config",
            str(cfg_phase2),
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--limit-checkpoints",
            "1",
            "--checkpoint-policy",
            "sampled",
            "--sample-count",
            "1",
            "--max-new-tokens",
            "32",
            "--fast-config",
            str(fast_benchmark_cfg),
            "--formal-config",
            str(formal_benchmark_cfg),
        ],
        cwd=project_root,
    )

    _assert_exists(benchmark_report_path, "benchmark report missing")
    _assert_exists(benchmark_summary_path, "benchmark summary missing")

    benchmark_report = json.loads(benchmark_report_path.read_text(encoding="utf-8"))
    final_selection = benchmark_report.get("final_selection", {})
    leaderboard = final_selection.get("leaderboard")
    if not isinstance(leaderboard, list) or not leaderboard:
        raise ValueError(f"benchmark leaderboard is empty: {benchmark_report_path}")

    benchmark_required_keys = {
        "continuation_stop_success_rate",
        "continuation_time_order_validity_rate",
        "infilling_structural_validity_rate",
        "valid_loss_from_training",
    }
    missing = [key for key in benchmark_required_keys if key not in leaderboard[0]]
    if missing:
        raise ValueError(
            f"benchmark leaderboard missing keys {missing}: {benchmark_report_path}"
        )

    sample_artifacts = benchmark_report.get("sample_artifacts")
    if not isinstance(sample_artifacts, dict) or not sample_artifacts:
        raise ValueError(f"benchmark sample artifacts missing: {benchmark_report_path}")

    for checkpoint_name, artifact_paths in sample_artifacts.items():
        continuation_path = artifact_paths.get("continuation")
        infilling_path = artifact_paths.get("infilling")
        if not continuation_path or not infilling_path:
            raise ValueError(f"sample artifact paths incomplete for {checkpoint_name}: {benchmark_report_path}")
        _assert_exists(Path(continuation_path), "continuation sample artifact missing")
        _assert_exists(Path(infilling_path), "infilling sample artifact missing")

    print(f"[regression_check] PASS run_id={run_id}", flush=True)
    print(f"[regression_check] checkpoints={output_dir}", flush=True)
    print(f"[regression_check] benchmark_report={benchmark_report_path}", flush=True)
    print(f"[regression_check] benchmark_summary={benchmark_summary_path}", flush=True)


if __name__ == "__main__":
    main()
