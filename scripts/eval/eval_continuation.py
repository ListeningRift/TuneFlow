#!/usr/bin/env python
"""Continuation-only benchmark entrypoint for TuneFlow."""

from __future__ import annotations

from benchmark_runner import main as _run_benchmark


def main() -> None:
    # 单任务入口：只按 continuation 指标做评估与选点。
    _run_benchmark(task_scope="continuation")


if __name__ == "__main__":
    main()
