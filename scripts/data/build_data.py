#!/usr/bin/env python
"""TuneFlow 一键数据构建脚本。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


STEP_ORDER = ["clean", "split", "tokenize", "build", "validate"]


@dataclass
class PipelineArgs:
    """数据构建流水线参数。"""

    python_exec: str
    clean_config: Path
    split_config: Path
    tokenizer_config: Path
    build_config: Path
    validate_report_path: Path
    start_from: str
    stop_after: str
    clean_limit: Optional[int]
    split_limit: Optional[int]
    tokenize_limit_per_split: Optional[int]


def parse_args() -> PipelineArgs:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="一键执行 clean/split/tokenize/build 数据流程。")
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="用于执行子脚本的 Python 可执行文件路径。",
    )
    parser.add_argument(
        "--clean-config",
        type=Path,
        default=Path("configs/data/cleaning.yaml"),
        help="clean 阶段配置文件。",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/data/split.yaml"),
        help="split 阶段配置文件。",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        default=Path("configs/tokenizer/tokenizer.yaml"),
        help="tokenize 阶段配置文件。",
    )
    parser.add_argument(
        "--build-config",
        type=Path,
        default=Path("configs/data/build_training.yaml"),
        help="build_training 阶段配置文件。",
    )
    parser.add_argument(
        "--validate-report-path",
        type=Path,
        default=Path("outputs/reports/data/validate_data_report.json"),
        help="validate 阶段报告输出路径。",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        choices=STEP_ORDER,
        default="clean",
        help="从哪个阶段开始执行。",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        choices=STEP_ORDER,
        default="validate",
        help="执行到哪个阶段结束（包含该阶段）。",
    )
    parser.add_argument(
        "--clean-limit",
        type=int,
        default=None,
        help="clean 阶段限制文件数量（用于冒烟测试）。",
    )
    parser.add_argument(
        "--split-limit",
        type=int,
        default=None,
        help="split 阶段限制文件数量（用于冒烟测试）。",
    )
    parser.add_argument(
        "--tokenize-limit-per-split",
        type=int,
        default=None,
        help="tokenize 阶段每个 split 的样本上限（用于冒烟测试）。",
    )
    ns = parser.parse_args()
    start_idx = STEP_ORDER.index(ns.start_from)
    stop_idx = STEP_ORDER.index(ns.stop_after)
    if start_idx > stop_idx:
        raise SystemExit("--start-from 不能晚于 --stop-after")
    return PipelineArgs(
        python_exec=ns.python_exec,
        clean_config=ns.clean_config,
        split_config=ns.split_config,
        tokenizer_config=ns.tokenizer_config,
        build_config=ns.build_config,
        validate_report_path=ns.validate_report_path,
        start_from=ns.start_from,
        stop_after=ns.stop_after,
        clean_limit=ns.clean_limit,
        split_limit=ns.split_limit,
        tokenize_limit_per_split=ns.tokenize_limit_per_split,
    )


def should_run(step: str, start_from: str, stop_after: str) -> bool:
    """判断某个阶段是否需要执行。"""
    idx = STEP_ORDER.index(step)
    start_idx = STEP_ORDER.index(start_from)
    stop_idx = STEP_ORDER.index(stop_after)
    return start_idx <= idx <= stop_idx


def build_commands(args: PipelineArgs) -> Dict[str, List[str]]:
    """构建各阶段对应的命令行。"""
    commands: Dict[str, List[str]] = {}

    clean_cmd = [
        args.python_exec,
        "scripts/data/clean_dataset.py",
        "--config",
        str(args.clean_config),
    ]
    if args.clean_limit is not None:
        clean_cmd += ["--limit", str(args.clean_limit)]
    commands["clean"] = clean_cmd

    split_cmd = [
        args.python_exec,
        "scripts/data/split_dataset.py",
        "--config",
        str(args.split_config),
    ]
    if args.split_limit is not None:
        split_cmd += ["--limit", str(args.split_limit)]
    commands["split"] = split_cmd

    tokenize_cmd = [
        args.python_exec,
        "scripts/data/tokenize_dataset.py",
        "--config",
        str(args.tokenizer_config),
    ]
    if args.tokenize_limit_per_split is not None:
        tokenize_cmd += ["--limit-per-split", str(args.tokenize_limit_per_split)]
    commands["tokenize"] = tokenize_cmd

    commands["build"] = [
        args.python_exec,
        "scripts/data/build_training_data.py",
        "--config",
        str(args.build_config),
    ]
    commands["validate"] = [
        args.python_exec,
        "scripts/data/validate_data_outputs.py",
        "--report-path",
        str(args.validate_report_path),
    ]
    return commands


def run_step(step: str, cmd: List[str], repo_root: Path) -> None:
    """执行单个阶段。"""
    pretty = " ".join(cmd)
    print(f"[data-build] step={step} cmd={pretty}", flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)
    print(f"[data-build] step={step} done", flush=True)


def main() -> None:
    """程序入口。"""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    commands = build_commands(args)

    for step in STEP_ORDER:
        if should_run(step, args.start_from, args.stop_after):
            run_step(step=step, cmd=commands[step], repo_root=repo_root)

    print("[data-build] all requested steps completed")


if __name__ == "__main__":
    main()
