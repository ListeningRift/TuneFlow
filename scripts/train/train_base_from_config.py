#!/usr/bin/env python
"""从 YAML 配置启动 base 训练。"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Any


def _ensure_project_root_on_path() -> Path:
    """确保仓库根目录在 `sys.path` 中，便于导入 `src.*` 模块。"""
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="从 YAML 配置运行 src/training/train_base.py。")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="训练 YAML 路径（支持顶层字段，或嵌套在 `train:` 下）；传入后优先级高于 --preset。",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="small",
        choices=["small", "full"],
        help="内置训练配置档位：small=小规模正式训练，full=完整规模训练。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印即将转发给 train_base 的参数，不真正启动训练。",
    )
    return parser.parse_args()


def _load_train_mapping(config_path: Path) -> dict[str, Any]:
    """读取训练配置并兼容两种结构：顶层字段 / `train:` 子字段。"""
    from src.utils.config_io import load_yaml_mapping

    payload = load_yaml_mapping(config_path, "train run config")
    if "train" in payload:
        train_payload = payload["train"]
        if not isinstance(train_payload, dict):
            raise ValueError(f"`train` section in {config_path} must be a mapping.")
        return train_payload
    return payload


def _resolve_preset_config(project_root: Path, preset: str) -> Path:
    """
    把 `--preset` 映射到仓库内的具体 YAML 路径。

    约定：
    - `small`：小规模正式训练，用于先看训练趋势与评估曲线。
    - `full`：完整规模训练模板，用于长时间正式训练。
    """
    mapping = {
        "small": project_root / "configs" / "train" / "train_base_run_small.yaml",
        "full": project_root / "configs" / "train" / "train_base_run_full.yaml",
    }
    config_path = mapping[preset].resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Preset config not found for --preset {preset}: {config_path}")
    return config_path


def _option_maps(parser: argparse.ArgumentParser) -> tuple[dict[str, argparse.Action], dict[str, str]]:
    """从 argparse parser 提取 `dest -> action/--option` 的映射表。"""
    action_by_dest: dict[str, argparse.Action] = {}
    option_by_dest: dict[str, str] = {}
    for action in parser._actions:  # pylint: disable=protected-access
        if action.dest == "help":
            continue
        long_options = [opt for opt in action.option_strings if opt.startswith("--")]
        if not long_options:
            continue
        action_by_dest[action.dest] = action
        option_by_dest[action.dest] = long_options[0]
    return action_by_dest, option_by_dest


def _resolve_steps_alias(config_mapping: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """
    在把配置转发给 train_base 前，先解析符号化的 `steps` 别名。

    当前支持：
    - `one_pass`：按当前数据集估算覆盖一整遍训练 token 所需步数
    """
    resolved = dict(config_mapping)
    steps_value = resolved.get("steps")
    if not isinstance(steps_value, str):
        return resolved

    alias = steps_value.strip().lower()
    if alias != "one_pass":
        return resolved

    train_idx_value = resolved.get("train_idx", "data/tokenized/train.idx.json")
    train_idx_path = Path(str(train_idx_value))
    if not train_idx_path.is_absolute():
        train_idx_path = (project_root / train_idx_path).resolve()
    if not train_idx_path.exists():
        raise FileNotFoundError(f"Cannot resolve steps=one_pass because train idx is missing: {train_idx_path}")

    idx_payload = json.loads(train_idx_path.read_text(encoding="utf-8"))
    train_num_tokens = int(idx_payload.get("num_tokens", 0))
    if train_num_tokens <= 0:
        raise ValueError(f"Cannot resolve steps=one_pass because num_tokens is invalid in {train_idx_path}")

    batch_size = int(resolved.get("batch_size", 2))
    grad_accum_steps = int(resolved.get("grad_accum_steps", 1))
    seq_len = int(resolved.get("seq_len", 256))
    tokens_per_step = batch_size * grad_accum_steps * seq_len
    if tokens_per_step <= 0:
        raise ValueError("Cannot resolve steps=one_pass because tokens_per_step <= 0.")

    resolved["steps"] = int(math.ceil(train_num_tokens / tokens_per_step))
    return resolved


def _to_train_argv(config_mapping: dict[str, Any], parser: argparse.ArgumentParser) -> list[str]:
    """
    把 YAML 映射转换为 train_base 可直接消费的 argv 列表。

    说明：
    - 严格校验未知字段，避免配置写错后静默忽略。
    - `store_true` 参数要求 bool，且仅在 True 时写入开关。
    - `None` 表示“不传该参数”，由 train_base 使用默认值。
    """
    action_by_dest, option_by_dest = _option_maps(parser)
    known_keys = set(action_by_dest.keys())
    unknown_keys = sorted(set(config_mapping.keys()) - known_keys)
    if unknown_keys:
        known_str = ", ".join(sorted(known_keys))
        raise ValueError(
            f"Unknown keys in train config: {unknown_keys}. "
            f"Supported keys: {known_str}."
        )

    argv: list[str] = []
    for key, value in config_mapping.items():
        action = action_by_dest[key]
        option = option_by_dest[key]

        # 布尔开关：True -> 仅追加 `--flag`，False -> 不追加。
        if isinstance(action, argparse._StoreTrueAction):  # pylint: disable=protected-access
            if not isinstance(value, bool):
                raise TypeError(f"`{key}` must be bool for store_true option {option}.")
            if value:
                argv.append(option)
            continue

        if value is None:
            continue

        # 其余参数必须是可字符串化的标量，不支持嵌套结构。
        if isinstance(value, bool):
            raise TypeError(f"`{key}` should not be bool. Use a concrete scalar/path value.")
        if isinstance(value, (dict, list, tuple)):
            raise TypeError(f"`{key}` does not support nested values: got {type(value).__name__}.")

        argv.extend([option, str(value)])
    return argv


def _warn_if_output_dir_is_dirty(config_mapping: dict[str, Any], project_root: Path) -> None:
    """
    当本次是全新训练、但输出目录里已经存在 checkpoint 时给出提示。

    这样可以避免后续评估时把旧的 `step_*.pt` 和新产物混在一起。
    """
    resume_from = config_mapping.get("resume_from")
    if resume_from is not None:
        return

    output_dir_value = config_mapping.get("output_dir")
    if output_dir_value is None:
        return

    output_dir = Path(str(output_dir_value))
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    if not output_dir.exists():
        return

    checkpoint_files = sorted(output_dir.glob("*.pt"))
    if checkpoint_files:
        names = ", ".join(path.name for path in checkpoint_files[:5])
        suffix = "" if len(checkpoint_files) <= 5 else ", ..."
        print(
            "[train_base_cfg] warning: output_dir already contains checkpoints while resume_from is null. "
            "A later eval may mix old and new artifacts from the same directory.\n"
            f"[train_base_cfg] existing checkpoints in {output_dir}: {names}{suffix}"
        )


def main() -> None:
    """程序入口：读取 YAML、转换参数并调用训练主函数。"""
    project_root = _ensure_project_root_on_path()
    os.chdir(project_root)
    args = _parse_args()

    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else (project_root / args.config)
        config_path = config_path.resolve()
    else:
        config_path = _resolve_preset_config(project_root, args.preset)

    from src.training.train_base import build_arg_parser, main as train_base_main

    train_mapping = _resolve_steps_alias(_load_train_mapping(config_path), project_root)
    # 基于 train_base 的真实 parser 做映射，保证参数名称与行为始终一致。
    train_argv = _to_train_argv(train_mapping, build_arg_parser())

    print(f"[train_base_cfg] config={config_path}")
    print("[train_base_cfg] argv=" + " ".join(shlex.quote(token) for token in train_argv))
    if args.dry_run:
        return

    train_base_main(train_argv)


if __name__ == "__main__":
    main()
