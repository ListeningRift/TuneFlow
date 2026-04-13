"""Tokenizer 核心模块。"""

from .velocity import (
    VelocityConfig,
    bin_to_velocity,
    build_velocity_table,
    velocity_to_bin,
)


def tokenize_main() -> None:
    """懒加载调用分词主入口，避免导入时强依赖 mido。"""
    from .tokenize_dataset import main as _main

    _main()


__all__ = [
    "VelocityConfig",
    "velocity_to_bin",
    "bin_to_velocity",
    "build_velocity_table",
    "tokenize_main",
]
