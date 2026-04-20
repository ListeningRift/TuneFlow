"""训练相关的 PyTorch 通用工具。"""

from __future__ import annotations


def lazy_import_torch():
    """按需导入 torch，便于在未安装环境中仍可做非训练操作。"""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖：torch。请先在你当前环境中执行 `uv sync --active`。"
        ) from exc
    return torch


def resolve_torch_device(torch_mod, choice: str):
    """
    解析设备参数。

    支持:
    - cpu
    - cuda
    - auto（优先 cuda）
    """
    if choice == "cpu":
        return torch_mod.device("cpu")
    if choice == "cuda":
        if not torch_mod.cuda.is_available():
            raise SystemExit("指定了 `--device cuda`，但当前环境不可用。")
        return torch_mod.device("cuda")
    return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


def count_parameters(model) -> int:
    """统计模型总参数量。"""
    return sum(p.numel() for p in model.parameters())
