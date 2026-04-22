#!/usr/bin/env python
"""随机抽样 MIDI，导出便于人工复核的调性分析结果。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample MIDI files, run key analysis, and export results for manual review."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/tokenizer.yaml"),
        help="Tokenizer config used to tokenize sampled MIDI files.",
    )
    parser.add_argument(
        "--midi-root",
        type=Path,
        default=None,
        help="Optional MIDI root directory. Defaults to tokenizer midi_root_dir, with clean_midi fallback.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=12,
        help="How many MIDI files to sample for manual review.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/debug/key_review"),
        help="Directory for exported review artifacts.",
    )
    parser.add_argument(
        "--copy-midi",
        action="store_true",
        help="Copy sampled MIDI files into the output directory for easier manual listening.",
    )
    parser.add_argument(
        "--window-bars",
        type=float,
        default=4.0,
        help="Key-analysis window size in bars for review mode. Default is a conservative 4.0.",
    )
    parser.add_argument(
        "--hop-bars",
        type=float,
        default=2.0,
        help="Key-analysis hop size in bars for review mode. Default is 2.0.",
    )
    parser.add_argument(
        "--min-best-score",
        type=float,
        default=0.40,
        help="Minimum local K-S score to accept a frame as stable in review mode.",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=0.12,
        help="Minimum margin to the second-best key to accept a frame as stable in review mode.",
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=int,
        default=3,
        help="Neighborhood radius for frame voting in review mode.",
    )
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        default=3,
        help="How many consecutive hop frames a new key needs before it becomes a published modulation.",
    )
    return parser.parse_args(argv)


def _resolve_midi_root(project_root: Path, args: argparse.Namespace):
    from src.tokenizer import load_config

    config_path = args.config if args.config is not None else Path("configs/tokenizer/tokenizer.yaml")
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    tokenizer_config = load_config(config_path)

    if args.midi_root is not None:
        midi_root = args.midi_root
        if not midi_root.is_absolute():
            midi_root = (project_root / midi_root).resolve()
        return config_path, tokenizer_config, midi_root

    midi_root = Path(str(tokenizer_config.midi_root_dir))
    if not midi_root.is_absolute():
        midi_root = (project_root / midi_root).resolve()

    # 优先尊重 tokenizer 配置；如果配置根目录本身不是 MIDI 目录，
    # 再兼容当前工程里常见的 `clean_midi/` 子目录结构。
    if any(midi_root.rglob("*.mid")) or any(midi_root.rglob("*.midi")):
        return config_path, tokenizer_config, midi_root

    clean_midi_root = midi_root / "clean_midi"
    if clean_midi_root.exists():
        return config_path, tokenizer_config, clean_midi_root.resolve()
    return config_path, tokenizer_config, midi_root


def _discover_midi_files(midi_root: Path) -> list[Path]:
    midi_files = [
        path
        for pattern in ("*.mid", "*.midi")
        for path in midi_root.rglob(pattern)
        if path.is_file()
    ]
    deduped: dict[str, Path] = {}
    for path in midi_files:
        deduped[str(path.resolve())] = path.resolve()
    # 统一按绝对路径去重，避免同一文件因为大小写或相对路径差异被重复抽样。
    return sorted(deduped.values(), key=lambda path: str(path).lower())


def _sample_midi_files(midi_files: list[Path], *, sample_count: int, seed: int) -> list[Path]:
    if sample_count <= 0:
        return []
    if len(midi_files) <= sample_count:
        return list(midi_files)
    rng = random.Random(seed)
    return sorted(rng.sample(midi_files, sample_count), key=lambda path: str(path).lower())


def _segment_payload(segment) -> dict[str, Any]:
    return {
        "key": str(segment.key),
        "start_bar": int(segment.start_bar),
        "start_pos": int(segment.start_pos),
        "end_bar": int(segment.end_bar),
        "end_pos": int(segment.end_pos),
        "mean_score": float(segment.mean_score),
    }


def _modulation_payload(point) -> dict[str, Any]:
    return {
        "bar_index": int(point.bar_index),
        "pos_in_bar": int(point.pos_in_bar),
        "from_key": str(point.from_key),
        "to_key": str(point.to_key),
        "support": float(point.support),
    }


def _frame_payload(frame) -> dict[str, Any]:
    return {
        "start_bar": int(frame.start_bar),
        "start_pos": int(frame.start_pos),
        "end_bar": int(frame.end_bar),
        "end_pos": int(frame.end_pos),
        "best_key": str(frame.best_key),
        "best_score": float(frame.best_score),
        "margin_to_second": float(frame.margin_to_second),
        "is_uncertain": bool(frame.is_uncertain),
        "raw_key": str(frame.raw_key),
        "smoothed_support": float(frame.smoothed_support),
    }


def _format_position(bar_index: int, pos_in_bar: int) -> str:
    return f"{int(bar_index)}:{int(pos_in_bar)}"


def _segment_duration_units(segment, *, positions_per_bar: int) -> int:
    return max(
        0,
        ((int(segment.end_bar) * int(positions_per_bar)) + int(segment.end_pos))
        - ((int(segment.start_bar) * int(positions_per_bar)) + int(segment.start_pos)),
    )


def _dominant_key(analysis, *, positions_per_bar: int) -> tuple[str, float]:
    if not analysis.segments:
        return ("uncertain", 0.0)
    totals: dict[str, int] = {}
    overall = 0
    for segment in analysis.segments:
        span = _segment_duration_units(segment, positions_per_bar=positions_per_bar)
        totals[str(segment.key)] = int(totals.get(str(segment.key), 0)) + span
        overall += span
    if not totals or overall <= 0:
        return ("uncertain", 0.0)
    # 人工 review 最先想看的通常是“整首主调像什么”，
    # 这里直接用稳定段覆盖时长最长的 key 作为 predicted_key。
    best_key = min(totals, key=lambda key_name: (-int(totals[key_name]), str(key_name)))
    coverage = float(totals[best_key]) / float(overall)
    return best_key, coverage


def _timeline_summary(analysis) -> str:
    if not analysis.segments:
        return "uncertain"
    if len(analysis.segments) == 1:
        return str(analysis.segments[0].key)
    parts = [str(segment.key) for segment in analysis.segments]
    return " -> ".join(parts)


def _modulation_summary(analysis) -> str:
    if not analysis.modulation_points:
        return "none"
    return "; ".join(
        f"{point.from_key}->{point.to_key}@{_format_position(point.bar_index, point.pos_in_bar)}"
        for point in analysis.modulation_points
    )


def _result_payload(*, midi_path: Path, midi_root: Path, tokens: list[str], analysis) -> dict[str, Any]:
    try:
        relative_path = str(midi_path.relative_to(midi_root))
    except ValueError:
        relative_path = midi_path.name
    dominant_key, dominant_coverage = _dominant_key(analysis, positions_per_bar=32)
    return {
        "source_path": str(midi_path),
        "relative_path": relative_path,
        "predicted_key": dominant_key,
        "dominant_key_coverage": float(dominant_coverage),
        "key_summary": dominant_key,
        "timeline_summary": _timeline_summary(analysis),
        "modulation_summary": _modulation_summary(analysis),
        "token_count": len(tokens),
        "frame_count": len(analysis.frames),
        "segment_count": len(analysis.segments),
        "modulation_count": len(analysis.modulation_points),
        "initial_key": str(analysis.initial_key),
    }


def _detailed_result_payload(*, midi_path: Path, midi_root: Path, tokens: list[str], analysis) -> dict[str, Any]:
    payload = _result_payload(midi_path=midi_path, midi_root=midi_root, tokens=tokens, analysis=analysis)
    payload.update(
        {
            # 详细文件保留完整段落、转调点和帧级结果，
            # 简洁文件则只放人工快速判断最需要的字段。
            "segments": [_segment_payload(segment) for segment in analysis.segments],
            "modulation_points": [_modulation_payload(point) for point in analysis.modulation_points],
            "frames": [_frame_payload(frame) for frame in analysis.frames],
        }
    )
    return payload


def _copy_sampled_midi(sampled_files: list[Path], *, midi_root: Path, output_dir: Path) -> None:
    review_midi_dir = output_dir / "midi"
    for midi_path in sampled_files:
        try:
            relative_path = midi_path.relative_to(midi_root)
        except ValueError:
            relative_path = Path(midi_path.name)
        target_path = review_midi_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(midi_path, target_path)


def _write_results(
    output_dir: Path,
    results: list[dict[str, Any]],
    detailed_results: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "key_review.jsonl"
    detailed_jsonl_path = output_dir / "key_review_detailed.jsonl"
    markdown_path = output_dir / "key_review.md"

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with detailed_jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in detailed_results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Key Analysis Review",
        "",
        f"- reviewed_files: {len(results)}",
        f"- failed_files: {len(errors)}",
        "",
    ]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for item in errors:
            lines.append(f"- `{item['source_path']}`")
            lines.append(f"  error: {item['error']}")
        lines.append("")

    for index, row in enumerate(results, start=1):
        lines.append(f"## Sample {index}")
        lines.append("")
        lines.append(f"- source: `{row['source_path']}`")
        lines.append(f"- relative_path: `{row['relative_path']}`")
        lines.append(f"- predicted_key: `{row['predicted_key']}`")
        lines.append(f"- dominant_key_coverage: {row['dominant_key_coverage']:.3f}")
        lines.append(f"- key_summary: `{row['key_summary']}`")
        lines.append(f"- timeline_summary: `{row['timeline_summary']}`")
        lines.append(f"- modulation_summary: `{row['modulation_summary']}`")
        lines.append(f"- segments: {row['segment_count']}")
        lines.append(f"- modulation_points: {row['modulation_count']}")
        lines.append("")

    markdown_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return jsonl_path, detailed_jsonl_path, markdown_path


def main(argv: list[str] | None = None) -> None:
    project_root = _ensure_project_root_on_path()
    args = _parse_args(argv)

    import mido

    from src.music_analysis import KeyAnalysisConfig, analyze_key_timeline
    from src.tokenizer import tokenize_midi

    config_path, tokenizer_config, midi_root = _resolve_midi_root(project_root, args)
    if not midi_root.exists():
        raise FileNotFoundError(f"MIDI root not found: {midi_root}")

    midi_files = _discover_midi_files(midi_root)
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under: {midi_root}")

    sampled_files = _sample_midi_files(
        midi_files,
        sample_count=max(0, int(args.sample_count)),
        seed=int(args.seed),
    )
    if not sampled_files:
        raise ValueError("No MIDI files selected. Increase --sample-count above 0.")

    output_dir = args.output_dir if args.output_dir.is_absolute() else (project_root / args.output_dir).resolve()
    results: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    print(
        "[key-review] "
        f"config={config_path} midi_root={midi_root} available={len(midi_files)} sampled={len(sampled_files)}"
    )
    analysis_config = KeyAnalysisConfig(
        window_bars=float(args.window_bars),
        hop_bars=float(args.hop_bars),
        min_best_score=float(args.min_best_score),
        min_score_margin=float(args.min_score_margin),
        neighborhood_radius_frames=int(args.neighborhood_radius),
        modulation_confirmation_frames=int(args.confirmation_frames),
    )
    # review 脚本默认比库内默认值更保守，
    # 目标是给人工试听时一个更稳定的“主调/少量转调”概览，而不是最敏感的分析轨迹。
    print(
        "[key-review] "
        f"analysis window={analysis_config.window_bars} hop={analysis_config.hop_bars} "
        f"confirm={analysis_config.modulation_confirmation_frames}"
    )

    for index, midi_path in enumerate(sampled_files, start=1):
        try:
            midi = mido.MidiFile(str(midi_path))
            tokens = tokenize_midi(midi, tokenizer_config)
            analysis = analyze_key_timeline(tokens, config=analysis_config)
            payload = _result_payload(midi_path=midi_path, midi_root=midi_root, tokens=tokens, analysis=analysis)
            detailed_payload = _detailed_result_payload(
                midi_path=midi_path,
                midi_root=midi_root,
                tokens=tokens,
                analysis=analysis,
            )
            results.append(payload)
            detailed_results.append(detailed_payload)
            print(
                f"[key-review] ({index}/{len(sampled_files)}) "
                f"{payload['relative_path']} -> {payload['predicted_key']} "
                f"(coverage={payload['dominant_key_coverage']:.2f}, mods={payload['modulation_count']})"
            )
        except Exception as exc:
            errors.append({"source_path": str(midi_path), "error": str(exc)})
            print(f"[key-review] ({index}/{len(sampled_files)}) {midi_path.name} -> ERROR: {exc}")

    if args.copy_midi:
        _copy_sampled_midi(sampled_files, midi_root=midi_root, output_dir=output_dir)

    jsonl_path, detailed_jsonl_path, markdown_path = _write_results(output_dir, results, detailed_results, errors)
    print(f"[key-review] jsonl -> {jsonl_path}")
    print(f"[key-review] detailed -> {detailed_jsonl_path}")
    print(f"[key-review] summary -> {markdown_path}")
    if args.copy_midi:
        print(f"[key-review] copied midi -> {output_dir / 'midi'}")


if __name__ == "__main__":
    main()
