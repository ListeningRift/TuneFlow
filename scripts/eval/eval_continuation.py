#!/usr/bin/env python
"""Continuation-only benchmark entrypoint for TuneFlow."""

from __future__ import annotations

try:
    from .benchmark_runner import main as _run_benchmark
except ImportError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from scripts.eval.benchmark_runner import main as _run_benchmark


def main() -> None:
    # 单任务入口：只按 continuation 指标做评估与选点。
    _run_benchmark(task_scope="continuation")


if __name__ == "__main__":
    main()
