"""配置与结构化文件读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: Path, name: str) -> dict[str, Any]:
    """
    读取 YAML 并保证返回 mapping。

    参数:
    - path: YAML 文件路径
    - name: 资源名称，用于错误信息（例如 "tokenizer 配置"）
    """
    if not path.exists():
        raise FileNotFoundError(f"{name}不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name}必须是字典（mapping）")
    return raw


def load_json_file(path: Path, name: str) -> dict[str, Any]:
    """读取 JSON 并保证返回 dict。"""
    if not path.exists():
        raise FileNotFoundError(f"{name}不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name}格式错误：预期 object")
    return payload


def dump_json_file(path: Path, payload: dict[str, Any], ensure_ascii: bool = False, indent: int = 2) -> None:
    """写入 JSON 文件（自动创建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent),
        encoding="utf-8",
    )

