"""Tokenizer 核心模块。"""


def tokenize_main() -> None:
    """懒加载调用分词主入口，避免导入时强依赖 mido。"""
    from .tokenize_dataset import main as _main

    _main()


__all__ = ["tokenize_main"]
