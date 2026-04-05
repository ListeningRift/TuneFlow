#!/usr/bin/env python
"""Base 训练脚本入口（薄封装，核心实现位于 src/training/train_base.py）。"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    """确保脚本运行时可以导入仓库根目录下的 `src` 包。"""
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def main() -> None:
    """转发到训练核心实现。"""
    _ensure_project_root_on_path()
    from src.training.train_base import main as core_main

    core_main()


if __name__ == "__main__":
    main()
