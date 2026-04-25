#!/usr/bin/env python
"""Unified benchmark entrypoint for TuneFlow."""

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
    # 综合 benchmark：同时评估 continuation 与 infilling。
    _run_benchmark(task_scope="all")


if __name__ == "__main__":
    main()
