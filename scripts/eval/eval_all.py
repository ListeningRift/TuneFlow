#!/usr/bin/env python
"""Unified benchmark entrypoint for TuneFlow."""

from __future__ import annotations

from benchmark_runner import main as _run_benchmark


def main() -> None:
    # 综合 benchmark：同时评估 continuation 与 infilling。
    _run_benchmark(task_scope="all")


if __name__ == "__main__":
    main()
