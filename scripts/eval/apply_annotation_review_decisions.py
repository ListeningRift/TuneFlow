#!/usr/bin/env python
"""应用标注检查 viewer 导出的保留/剔除决策。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _ensure_project_root_on_path() -> Path:
    """把项目根目录加入导入路径。"""
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="根据 annotation review viewer 导出的决策文件过滤 split，或把坏 MIDI 移入隔离目录。"
    )
    parser.add_argument("--decisions-json", type=Path, required=True, help="viewer 导出的决策 JSON。")
    parser.add_argument("--midi-root", type=Path, default=None, help="原始 MIDI 根目录，用于解析相对路径。")
    parser.add_argument("--split-jsonl", type=Path, default=None, help="需要过滤的 split JSONL。")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="过滤后的输出 JSONL。")
    parser.add_argument("--move-rejected-to", type=Path, default=None, help="把判定为剔除的 MIDI 移到隔离目录。")
    args = parser.parse_args(argv)
    if (args.split_jsonl is None) != (args.output_jsonl is None):
        parser.error("--split-jsonl 和 --output-jsonl 必须同时提供。")
    if args.move_rejected_to is not None and args.midi_root is None:
        parser.error("使用 --move-rejected-to 时必须提供 --midi-root。")
    if args.split_jsonl is None and args.move_rejected_to is None:
        parser.error("至少需要执行一种动作：过滤 split 或移动被剔除的 MIDI。")
    return args


def _load_decisions(path: Path) -> list[dict[str, Any]]:
    """读取决策文件。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("决策文件格式错误：缺少 decisions 列表。")
    return [dict(item) for item in decisions if isinstance(item, dict)]


def _resolve_path(raw_path: str, base_dir: Path | None) -> Path:
    """把相对路径解析为绝对路径。"""
    path = Path(str(raw_path))
    if path.is_absolute() or base_dir is None:
        return path.resolve()
    return (base_dir / path).resolve()


def _collect_rejected_sets(
    decisions: list[dict[str, Any]],
    *,
    midi_root: Path | None,
) -> tuple[set[str], set[str]]:
    """收集被剔除样本的绝对路径集合与原始路径键集合。"""
    absolute_paths: set[str] = set()
    raw_keys: set[str] = set()
    for item in decisions:
        if str(item.get("decision", "")) != "drop":
            continue
        source_path = str(item.get("source_path", "")).strip()
        if source_path:
            absolute_paths.add(str(_resolve_path(source_path, None)))
        for key in ("midi_path", "relative_path"):
            raw_value = str(item.get(key, "")).strip()
            if raw_value:
                raw_keys.add(raw_value.replace("\\", "/"))
                if midi_root is not None:
                    absolute_paths.add(str(_resolve_path(raw_value, midi_root)))
    return absolute_paths, raw_keys


def _filter_split_jsonl(
    *,
    split_jsonl: Path,
    output_jsonl: Path,
    midi_root: Path | None,
    rejected_abs: set[str],
    rejected_keys: set[str],
) -> tuple[int, int]:
    """过滤 split JSONL 中被判定为剔除的样本。"""
    kept_rows: list[str] = []
    removed_count = 0
    total_count = 0
    base_dir = midi_root if midi_root is not None else split_jsonl.parent
    for line in split_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total_count += 1
        row = json.loads(line)
        midi_path = str(row.get("midi_path", "")).strip()
        normalized_key = midi_path.replace("\\", "/")
        absolute_path = str(_resolve_path(midi_path, base_dir)) if midi_path else ""
        if normalized_key in rejected_keys or absolute_path in rejected_abs:
            removed_count += 1
            continue
        kept_rows.append(json.dumps(row, ensure_ascii=False))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(kept_rows) + ("\n" if kept_rows else ""), encoding="utf-8")
    return total_count, removed_count


def _move_rejected_midis(
    *,
    midi_root: Path,
    rejected_abs: set[str],
    target_root: Path,
) -> int:
    """把被剔除的 MIDI 移动到隔离目录，保留相对目录结构。"""
    moved_count = 0
    safe_root = midi_root.resolve()
    target_root = target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    for raw_path in sorted(rejected_abs):
        source_path = Path(raw_path)
        if not source_path.exists():
            continue
        resolved = source_path.resolve()
        if not resolved.is_relative_to(safe_root):
            continue
        relative_path = resolved.relative_to(safe_root)
        target_path = target_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(target_path))
        moved_count += 1
    return moved_count


def main(argv: list[str] | None = None) -> None:
    """脚本主入口。"""
    project_root = _ensure_project_root_on_path()
    args = _parse_args(argv)

    decisions_json = args.decisions_json if args.decisions_json.is_absolute() else (project_root / args.decisions_json).resolve()
    midi_root = args.midi_root if args.midi_root is None or args.midi_root.is_absolute() else (project_root / args.midi_root).resolve()
    split_jsonl = args.split_jsonl if args.split_jsonl is None or args.split_jsonl.is_absolute() else (project_root / args.split_jsonl).resolve()
    output_jsonl = args.output_jsonl if args.output_jsonl is None or args.output_jsonl.is_absolute() else (project_root / args.output_jsonl).resolve()
    move_rejected_to = args.move_rejected_to if args.move_rejected_to is None or args.move_rejected_to.is_absolute() else (project_root / args.move_rejected_to).resolve()

    decisions = _load_decisions(decisions_json)
    rejected_abs, rejected_keys = _collect_rejected_sets(decisions, midi_root=midi_root)

    if split_jsonl is not None and output_jsonl is not None:
        total_count, removed_count = _filter_split_jsonl(
            split_jsonl=split_jsonl,
            output_jsonl=output_jsonl,
            midi_root=midi_root,
            rejected_abs=rejected_abs,
            rejected_keys=rejected_keys,
        )
        print(f"[annotation-apply] split_total={total_count} removed={removed_count} kept={total_count - removed_count}")
        print(f"[annotation-apply] output_jsonl -> {output_jsonl}")

    if move_rejected_to is not None and midi_root is not None:
        moved_count = _move_rejected_midis(
            midi_root=midi_root,
            rejected_abs=rejected_abs,
            target_root=move_rejected_to,
        )
        print(f"[annotation-apply] moved_rejected={moved_count}")
        print(f"[annotation-apply] move_target -> {move_rejected_to}")


if __name__ == "__main__":
    main()
