"""Tokenizer 核心模块。"""

from .midi_codec import (
    TokenizerConfig,
    build_vocab,
    load_config,
    tokenize_midi,
    tokens_to_midi,
    validate_token_order,
)
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
    "TokenizerConfig",
    "VelocityConfig",
    "build_vocab",
    "load_config",
    "tokenize_midi",
    "tokens_to_midi",
    "validate_token_order",
    "velocity_to_bin",
    "bin_to_velocity",
    "build_velocity_table",
    "tokenize_main",
]
