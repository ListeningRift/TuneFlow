#!/usr/bin/env python
"""一键回归冒烟检查：覆盖 train/eval/save/resume 最小链路。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _ensure_project_root() -> Path:
    """确保仓库根目录在 `sys.path` 中。"""
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="执行 train/eval/save/resume 回归冒烟检查。")
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="用于执行子进程命令的 Python 解释器路径。",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="可选 run_id；默认是 regression_smoke_<timestamp>。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="传给训练和评估脚本的设备参数。",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="传给训练和评估脚本的精度参数。",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="冒烟测试使用的序列长度。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="冒烟测试使用的 batch size。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    return parser.parse_args()


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    """执行子命令并在失败时抛出异常。"""
    print("[regression_check] cmd:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _check_runtime_deps(python_exec: str, cwd: Path) -> None:
    """检查目标 Python 环境是否安装 torch，避免跑到中途才失败。"""
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


def _write_train_cfg(path: Path, payload: dict) -> None:
    """把阶段性训练配置写入临时 YAML 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _assert_exists(path: Path, msg: str) -> None:
    """断言文件存在。"""
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
    """构建 train_base_from_config 需要的 YAML 载荷。"""
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
            # 回归测试中固定启用 FIM，确保该分支被覆盖到。
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


def main() -> None:
    """
    回归检查主流程：
    1) 训练 1 步并保存 checkpoint。
    2) 从 latest.pt 恢复训练到第 2 步。
    3) 对该 run 的所有 checkpoint 执行 eval_infilling 与 eval_continuation。
    4) 校验评估报告包含关键指标字段。
    """
    args = _parse_args()
    project_root = _ensure_project_root()

    run_id = args.run_id or f"regression_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = (project_root / "outputs" / "checkpoints" / "base" / run_id).resolve()
    infilling_report_path = (project_root / "outputs" / "reports" / "eval_infilling" / f"{run_id}.json").resolve()
    infilling_plot_path = infilling_report_path.with_suffix(".png")
    continuation_report_path = (
        project_root / "outputs" / "reports" / "eval_continuation" / f"{run_id}.json"
    ).resolve()
    continuation_plot_path = continuation_report_path.with_suffix(".png")
    _check_runtime_deps(args.python_exec, project_root)

    required = [
        project_root / "data" / "tokenized" / "train.idx.json",
        project_root / "data" / "tokenized" / "valid.idx.json",
        project_root / "data" / "tokenized" / "eval.tok",
        project_root / "data" / "tokenized" / "tokenizer_vocab.json",
    ]
    for p in required:
        # 提前校验关键输入，避免进入训练后才报路径错误。
        _assert_exists(p, "required input not found")

    cfg_dir = project_root / "outputs" / "tmp" / "regression_check" / run_id
    cfg_phase1 = cfg_dir / "train_phase1.yaml"
    cfg_phase2 = cfg_dir / "train_phase2_resume.yaml"

    # 阶段 1：训练到 1 步，强制执行评估与保存，验证“训练+保存”链路。
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
    _write_train_cfg(cfg_phase1, payload1)
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

    # 阶段 2：从 latest.pt 恢复，训练到第 2 步，验证“恢复+继续训练”链路。
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
    _write_train_cfg(cfg_phase2, payload2)
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

    # 阶段 3：对当前 run 的全部 checkpoint 执行评估，验证“checkpoint->评估->报告”链路。
    _run_cmd(
        [
            args.python_exec,
            "scripts/eval/eval_infilling.py",
            "--checkpoint-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--seq-len",
            str(args.seq_len),
            "--batch-size",
            str(args.batch_size),
            "--eval-batches",
            "1",
            "--num-infilling-samples",
            "2",
            "--max-new-tokens",
            "32",
        ],
        cwd=project_root,
    )
    _run_cmd(
        [
            args.python_exec,
            "scripts/eval/eval_continuation.py",
            "--checkpoint-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--seq-len",
            str(args.seq_len),
            "--batch-size",
            str(args.batch_size),
            "--eval-batches",
            "1",
            "--num-continuation-samples",
            "2",
            "--max-new-tokens",
            "32",
        ],
        cwd=project_root,
    )

    _assert_exists(infilling_report_path, "infilling eval report missing")
    infilling_report = json.loads(infilling_report_path.read_text(encoding="utf-8"))
    infilling_results = infilling_report.get("results")
    if not isinstance(infilling_results, list) or not infilling_results:
        raise ValueError(f"infilling eval report has empty results: {infilling_report_path}")

    infilling_required_keys = {"valid_loss", "ppl", "structural_validity_rate"}
    infilling_missing = [k for k in infilling_required_keys if k not in infilling_results[0]]
    if infilling_missing:
        raise ValueError(
            f"infilling eval result missing keys {infilling_missing}: {infilling_report_path}"
        )
    _assert_exists(infilling_plot_path, "infilling eval plot missing")

    _assert_exists(continuation_report_path, "continuation eval report missing")
    continuation_report = json.loads(continuation_report_path.read_text(encoding="utf-8"))
    continuation_results = continuation_report.get("results")
    if not isinstance(continuation_results, list) or not continuation_results:
        raise ValueError(f"continuation eval report has empty results: {continuation_report_path}")

    continuation_required_keys = {"valid_loss", "ppl", "structural_validity_rate", "first_token_accuracy"}
    continuation_missing = [k for k in continuation_required_keys if k not in continuation_results[0]]
    if continuation_missing:
        raise ValueError(
            f"continuation eval result missing keys {continuation_missing}: {continuation_report_path}"
        )
    _assert_exists(continuation_plot_path, "continuation eval plot missing")

    print(f"[regression_check] PASS run_id={run_id}", flush=True)
    print(f"[regression_check] checkpoints={output_dir}", flush=True)
    print(f"[regression_check] infilling_eval_report={infilling_report_path}", flush=True)
    print(f"[regression_check] infilling_eval_plot={infilling_plot_path}", flush=True)
    print(f"[regression_check] continuation_eval_report={continuation_report_path}", flush=True)
    print(f"[regression_check] continuation_eval_plot={continuation_plot_path}", flush=True)


if __name__ == "__main__":
    main()
