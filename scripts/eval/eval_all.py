#!/usr/bin/env python
"""一条命令串行执行 infilling 与 continuation 两类评估。"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> Path:
    """确保仓库根目录可导入，并返回该路径。"""
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args() -> argparse.Namespace:
    """解析统一评估入口的命令行参数。"""
    parser = argparse.ArgumentParser(description="顺序执行 eval_infilling.py 与 eval_continuation.py。")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="某次训练 run 的 checkpoint 目录（包含 *.pt）。",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="报告文件名中的 run_id；默认使用 checkpoint 目录名。",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/train/model_base.yaml"),
        help="当 checkpoint 不含 model_config 时使用的兜底模型配置。",
    )
    parser.add_argument(
        "--valid-idx",
        type=Path,
        default=Path("data/tokenized/valid.idx.json"),
        help="用于计算 valid_loss/ppl 的验证集 idx 路径。",
    )
    parser.add_argument(
        "--valid-bin",
        type=Path,
        default=None,
        help="可选：覆盖验证集 bin 路径。",
    )
    parser.add_argument(
        "--eval-tok",
        type=Path,
        default=Path("data/tokenized/eval.tok"),
        help="用于构造 infilling/continuation prompt 的评估 token 文件。",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("data/tokenized/tokenizer_vocab.json"),
        help="tokenizer 词表 JSON 路径。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="评估设备。",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="数值精度模式。",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="计算 valid_loss 采样时使用的序列长度。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="验证 micro-batch 大小。",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=10,
        help="每个 checkpoint 评估的验证 batch 数。",
    )
    parser.add_argument(
        "--num-infilling-samples",
        type=int,
        default=32,
        help="每个 checkpoint 用于 infilling 结构评估的样本数。",
    )
    parser.add_argument(
        "--num-continuation-samples",
        type=int,
        default=32,
        help="每个 checkpoint 用于 continuation 评估的样本数。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="每条 prompt 最大生成 token 数。",
    )
    parser.add_argument(
        "--min-prefix-tokens",
        type=int,
        default=16,
        help="continuation 评估时保留的最小 prefix token 数。",
    )
    parser.add_argument(
        "--limit-checkpoints",
        type=int,
        default=None,
        help="可选：限制评估的 checkpoint 数量（用于快速冒烟）。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="用于执行子评估脚本的 Python 解释器路径。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将执行的命令，不真正启动评估。",
    )
    return parser.parse_args()


def _append_optional_arg(cmd: list[str], option: str, value: object | None) -> None:
    """仅在参数非空时把 `--option value` 追加到命令列表。"""
    if value is None:
        return
    cmd.extend([option, str(value)])


def _build_shared_args(args: argparse.Namespace, project_root: Path) -> list[str]:
    """构建两个评估脚本共用的参数片段。"""
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else (project_root / args.checkpoint_dir)
    shared = [
        "--checkpoint-dir",
        str(checkpoint_dir.resolve()),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_size),
        "--eval-batches",
        str(args.eval_batches),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
    ]
    _append_optional_arg(shared, "--run-id", args.run_id)
    _append_optional_arg(
        shared,
        "--model-config",
        args.model_config if args.model_config.is_absolute() else (project_root / args.model_config),
    )
    _append_optional_arg(
        shared,
        "--valid-idx",
        args.valid_idx if args.valid_idx.is_absolute() else (project_root / args.valid_idx),
    )
    _append_optional_arg(
        shared,
        "--valid-bin",
        None if args.valid_bin is None else (args.valid_bin if args.valid_bin.is_absolute() else (project_root / args.valid_bin)),
    )
    _append_optional_arg(
        shared,
        "--eval-tok",
        args.eval_tok if args.eval_tok.is_absolute() else (project_root / args.eval_tok),
    )
    _append_optional_arg(
        shared,
        "--vocab-path",
        args.vocab_path if args.vocab_path.is_absolute() else (project_root / args.vocab_path),
    )
    _append_optional_arg(shared, "--limit-checkpoints", args.limit_checkpoints)
    return shared


def _run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    """打印命令，并在非 dry-run 模式下执行。"""
    print("[eval_all] cmd:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    """
    程序入口。

    执行顺序：
    1. 运行 infilling 评估；
    2. 运行 continuation 评估；
    3. 打印两类报告所在目录，方便后续查看 JSON/PNG。
    """
    project_root = _ensure_project_root_on_path()
    os.chdir(project_root)
    args = _parse_args()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else (project_root / args.checkpoint_dir)
    checkpoint_dir = checkpoint_dir.resolve()
    run_id = args.run_id or checkpoint_dir.name
    shared_args = _build_shared_args(args, project_root)

    infilling_cmd = [
        args.python_exec,
        "scripts/eval/eval_infilling.py",
        *shared_args,
        "--num-infilling-samples",
        str(args.num_infilling_samples),
    ]
    continuation_cmd = [
        args.python_exec,
        "scripts/eval/eval_continuation.py",
        *shared_args,
        "--num-continuation-samples",
        str(args.num_continuation_samples),
        "--min-prefix-tokens",
        str(args.min_prefix_tokens),
    ]

    _run_cmd(infilling_cmd, cwd=project_root, dry_run=args.dry_run)
    _run_cmd(continuation_cmd, cwd=project_root, dry_run=args.dry_run)

    print(f"[eval_all] infilling report dir={project_root / 'outputs' / 'reports' / 'eval'}", flush=True)
    print(f"[eval_all] continuation report dir={project_root / 'outputs' / 'reports' / 'eval_continuation'}", flush=True)
    print(f"[eval_all] run_id={run_id}", flush=True)


if __name__ == "__main__":
    main()
