#!/usr/bin/env python
"""构建供固定 viewer 使用的 MIDI 标注检查数据包。"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


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
        description="构建一个供固定 viewer 懒加载的 MIDI 调性与乐句标注检查数据包。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/tokenizer.yaml"),
        help="tokenizer 配置文件路径。",
    )
    parser.add_argument(
        "--midi-root",
        type=Path,
        default=None,
        help="原始 MIDI 根目录；与 --benchmark-json 二选一。",
    )
    parser.add_argument(
        "--midi-list-jsonl",
        type=Path,
        default=None,
        help="原始 MIDI 清单 JSONL；可与 --midi-root 配合用于解析相对路径。",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=None,
        help="benchmark sample JSON 路径；与原始 MIDI 输入二选一。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/debug/annotation_review"),
        help="输出目录。",
    )
    parser.add_argument(
        "--copy-midi",
        action="store_true",
        help="原始 MIDI 模式下把样本复制到输出目录。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理多少条样本。",
    )
    parser.add_argument(
        "--window-bars",
        type=float,
        default=1.0,
        help="调性分析窗口大小（单位：bar）。",
    )
    parser.add_argument(
        "--hop-bars",
        type=float,
        default=0.5,
        help="调性分析滑动步长（单位：bar）。",
    )
    parser.add_argument(
        "--min-best-score",
        type=float,
        default=0.30,
        help="调性帧稳定判定的最小得分。",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=0.10,
        help="调性帧第一名与第二名的最小差值。",
    )
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        default=2,
        help="新调性需要连续多少帧才发布为转调。",
    )
    parser.add_argument(
        "--only-suspicious",
        action="store_true",
        help="输出时只保留命中可疑规则的样本。",
    )
    parser.add_argument(
        "--include-tokens",
        action="store_true",
        help="把完整 token 序列写入每条 case 详情。默认关闭以减小产物、加快打开速度。",
    )
    args = parser.parse_args(argv)
    has_raw_input = args.midi_root is not None or args.midi_list_jsonl is not None
    has_benchmark_input = args.benchmark_json is not None
    if has_raw_input == has_benchmark_input:
        parser.error("请在原始 MIDI 输入和 --benchmark-json 之间二选一。")
    if args.limit is not None and int(args.limit) <= 0:
        parser.error("--limit 必须大于 0。")
    return args


def _copy_raw_midis(cases: list[dict[str, object]], output_dir: Path) -> None:
    """复制原始 MIDI 样本，便于人工定位。"""
    copied_root = output_dir / "copied_midi"
    for case in cases:
        if str(case.get("source_kind")) != "raw_midi":
            continue
        source_path = Path(str(case.get("source_path", "")))
        if not source_path.exists():
            continue
        target_path = copied_root / source_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def main(argv: list[str] | None = None) -> None:
    """脚本主入口。"""
    project_root = _ensure_project_root_on_path()
    args = _parse_args(argv)

    from src.music_analysis import KeyAnalysisConfig, PhraseAnalysisConfig
    from src.tokenizer import load_config
    from src.utils.annotation_review import (
        ReviewBuildConfig,
        load_benchmark_cases,
        load_raw_midi_cases,
        summarize_cases,
        write_review_bundle,
    )

    config_path = args.config if args.config.is_absolute() else (project_root / args.config).resolve()
    tokenizer_config = load_config(config_path)
    key_config = KeyAnalysisConfig(
        positions_per_bar=int(tokenizer_config.positions_per_bar),
        window_bars=float(args.window_bars),
        hop_bars=float(args.hop_bars),
        min_best_score=float(args.min_best_score),
        min_score_margin=float(args.min_score_margin),
        modulation_confirmation_frames=int(args.confirmation_frames),
    )
    phrase_config = PhraseAnalysisConfig(positions_per_bar=int(tokenizer_config.positions_per_bar))
    review_config = ReviewBuildConfig(
        tokenizer_config=tokenizer_config,
        key_config=key_config,
        phrase_config=phrase_config,
        low_margin_threshold=float(args.min_score_margin),
    )

    if args.benchmark_json is not None:
        benchmark_json_path = (
            args.benchmark_json
            if args.benchmark_json.is_absolute()
            else (project_root / args.benchmark_json).resolve()
        )
        cases = load_benchmark_cases(benchmark_json_path, review_config)
        if args.limit is not None:
            cases = cases[: int(args.limit)]
        source_summary = {
            "mode": "benchmark",
            "benchmark_json": str(benchmark_json_path),
        }
    else:
        midi_root = None
        if args.midi_root is not None:
            midi_root = args.midi_root if args.midi_root.is_absolute() else (project_root / args.midi_root).resolve()
        midi_list_jsonl = None
        if args.midi_list_jsonl is not None:
            midi_list_jsonl = (
                args.midi_list_jsonl
                if args.midi_list_jsonl.is_absolute()
                else (project_root / args.midi_list_jsonl).resolve()
            )
        cases = load_raw_midi_cases(
            midi_root=midi_root,
            midi_list_jsonl=midi_list_jsonl,
            config=review_config,
            limit=args.limit,
        )
        source_summary = {
            "mode": "raw_midi",
            "midi_root": str(midi_root) if midi_root is not None else None,
            "midi_list_jsonl": str(midi_list_jsonl) if midi_list_jsonl is not None else None,
        }

    output_dir = args.output_dir if args.output_dir.is_absolute() else (project_root / args.output_dir).resolve()
    index_payload = write_review_bundle(
        output_dir=output_dir,
        cases=cases,
        positions_per_bar=int(tokenizer_config.positions_per_bar),
        only_suspicious=bool(args.only_suspicious),
        source_summary={
            **source_summary,
            **summarize_cases(cases),
        },
        include_tokens=bool(args.include_tokens),
    )

    if args.copy_midi:
        _copy_raw_midis(cases, output_dir)

    viewer_path = project_root / "tools" / "annotation_review_viewer.html"
    print(f"[annotation-review] cases={len(index_payload.get('cases', []))}")
    print(f"[annotation-review] index -> {output_dir / 'index.json'}")
    print(f"[annotation-review] cases_dir -> {output_dir / 'cases'}")
    print(f"[annotation-review] viewer -> {viewer_path}")
    if args.copy_midi:
        print(f"[annotation-review] copied_midi -> {output_dir / 'copied_midi'}")


if __name__ == "__main__":
    main()
