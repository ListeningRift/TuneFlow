#!/usr/bin/env python
"""一条命令顺序执行 TuneFlow 的训练与完整评估。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _ensure_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一条命令顺序执行 TuneFlow 的训练与完整 benchmark 评估。"
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        choices=["small", "full"],
        help="内置训练/评估预设：small=小规模正式流程，full=完整规模正式流程。",
    )
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="用于启动子进程的 Python 解释器，默认使用当前解释器。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的训练与评估命令，不真正启动。",
    )
    return parser.parse_args()


def _run_cmd(*, cmd: list[str], cwd: Path, stage: str, dry_run: bool) -> None:
    print(f"[train_and_eval] {stage}: " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    args = _parse_args()
    project_root = _ensure_project_root()

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[train_and_eval] started_at={started_at} preset={args.preset} python_exec={args.python_exec}",
        flush=True,
    )

    train_cmd = [
        args.python_exec,
        "scripts/train/train_base_from_config.py",
        "--preset",
        args.preset,
    ]
    eval_cmd = [
        args.python_exec,
        "scripts/eval/eval_all.py",
        "--preset",
        args.preset,
    ]

    _run_cmd(cmd=train_cmd, cwd=project_root, stage="train", dry_run=bool(args.dry_run))
    _run_cmd(cmd=eval_cmd, cwd=project_root, stage="eval", dry_run=bool(args.dry_run))

    print(
        f"[train_and_eval] done preset={args.preset} "
        f"checkpoints=outputs/checkpoints/base_{args.preset} "
        f"benchmark=outputs/benchmark/base_{args.preset}",
        flush=True,
    )


if __name__ == "__main__":
    main()
