#!/usr/bin/env python
"""TuneFlow 数据分词脚本。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：mido。请先在当前环境中执行 `uv sync --active`。"
    ) from exc

from ..utils.config_io import dump_json_file
from ..utils.output_cleanup import ensure_clean_directory, remove_file_if_exists
from .common import collect_tempo_changes, get_bar_ticks, load_jsonl, summarize_lengths, write_tok_lines
from .midi_codec import (
    TokenizerConfig,
    _collect_tokenizer_notes,
    _tokenize_note_events,
    _transpose_notes,
    build_key_vocab_tokens,
    build_vocab,
    is_key_token,
    load_config,
    validate_token_order,
    velocity_to_bucket,
)
from .velocity import build_velocity_table


@dataclass
class SplitRowResult:
    """表示单条 split 记录的分词结果。"""

    midi_path: str
    token_variants: List[List[str]]
    invalid_count: int
    oov_count: int
    augmented_rows: int
    skipped_transpose_rows: int
    applied_transpose_counts: Dict[str, int]
    skipped_transpose_counts: Dict[str, int]
    error: str | None


@dataclass(frozen=True)
class ProgressSnapshot:
    """表示一次可打印的进度快照。"""

    split_name: str
    split_processed: int
    split_total: int
    total_processed: int
    total_planned: int
    elapsed_seconds: float


def _empty_key_token_stats(key_tokens: List[str]) -> Dict[str, object]:
    """初始化调性 token 统计结构。"""
    return {
        "total_key_tokens": 0,
        "counts_by_token": {token: 0 for token in key_tokens},
        "major_total": 0,
        "minor_total": 0,
        "uncertain_total": 0,
    }


def _empty_phrase_token_stats() -> Dict[str, object]:
    """初始化乐句 token 统计结构。"""
    return {
        "phrase_token_total": 0,
        "bar_aligned_phrase_total": 0,
        "mid_bar_phrase_total": 0,
        "bar_spans_sum": 0,
        "bar_spans_count": 0,
    }


def _accumulate_phrase_token_stats(stats: Dict[str, object], tokens: List[str]) -> None:
    """把单条 token 序列的乐句统计累加到聚合结构中。"""
    bar_positions: List[int] = []
    phrase_indices: List[int] = []
    for idx, token in enumerate(tokens):
        if token == "BAR":
            bar_positions.append(idx)
        elif token == "PHRASE":
            phrase_indices.append(idx)
    if not phrase_indices:
        return
    stats["phrase_token_total"] = int(stats.get("phrase_token_total", 0)) + len(phrase_indices)

    def _enclosing_bar(token_index: int) -> int:
        """返回某个 PHRASE 所属的小节索引。"""
        bar_index = -1
        for i, pos in enumerate(bar_positions):
            if pos < token_index:
                bar_index = i
            else:
                break
        return bar_index

    phrase_bar_indices: List[int] = []
    for phrase_idx in phrase_indices:
        prev = phrase_idx - 1
        while prev > 0 and (str(tokens[prev]).startswith("TEMPO_") or str(tokens[prev]).startswith("KEY_")):
            prev -= 1
        is_bar_aligned = tokens[prev] == "BAR" if prev >= 0 else False
        if is_bar_aligned:
            stats["bar_aligned_phrase_total"] = int(stats.get("bar_aligned_phrase_total", 0)) + 1
        else:
            stats["mid_bar_phrase_total"] = int(stats.get("mid_bar_phrase_total", 0)) + 1
        phrase_bar_indices.append(_enclosing_bar(phrase_idx))

    eos_bar = len(bar_positions)
    for i, current_bar in enumerate(phrase_bar_indices):
        if current_bar < 0:
            continue
        next_bar = phrase_bar_indices[i + 1] if i + 1 < len(phrase_bar_indices) else eos_bar
        if next_bar < 0:
            next_bar = eos_bar
        span = next_bar - current_bar
        if span < 0:
            continue
        stats["bar_spans_sum"] = int(stats.get("bar_spans_sum", 0)) + span
        stats["bar_spans_count"] = int(stats.get("bar_spans_count", 0)) + 1


def _finalize_phrase_token_stats(stats: Dict[str, object], num_sequences: int) -> Dict[str, object]:
    """把内部统计结构转成最终输出格式。"""
    total = int(stats.get("phrase_token_total", 0))
    bar_count = int(stats.get("bar_spans_count", 0))
    bar_sum = int(stats.get("bar_spans_sum", 0))
    mid_bar = int(stats.get("mid_bar_phrase_total", 0))
    return {
        "phrase_token_total": total,
        "bar_aligned_phrase_total": int(stats.get("bar_aligned_phrase_total", 0)),
        "mid_bar_phrase_total": mid_bar,
        "mean_phrases_per_sequence": (0.0 if num_sequences == 0 else total / float(num_sequences)),
        "mid_bar_phrase_ratio": (0.0 if total == 0 else mid_bar / float(total)),
        "mean_phrase_bar_span": (0.0 if bar_count == 0 else bar_sum / float(bar_count)),
    }


def _accumulate_key_token_stats(stats: Dict[str, object], tokens: List[str]) -> None:
    """把单条 token 序列的调性统计累加到聚合结构中。"""
    counts_by_token = stats.get("counts_by_token")
    if not isinstance(counts_by_token, dict):
        return
    for token in tokens:
        token_str = str(token)
        if not is_key_token(token_str):
            continue
        counts_by_token[token_str] = int(counts_by_token.get(token_str, 0)) + 1
        stats["total_key_tokens"] = int(stats.get("total_key_tokens", 0)) + 1
        if token_str.endswith("_MAJ"):
            stats["major_total"] = int(stats.get("major_total", 0)) + 1
        elif token_str.endswith("_MIN"):
            stats["minor_total"] = int(stats.get("minor_total", 0)) + 1
        elif token_str == "KEY_UNCERTAIN":
            stats["uncertain_total"] = int(stats.get("uncertain_total", 0)) + 1


def print_velocity_table(config: TokenizerConfig) -> None:
    """打印当前 tokenizer 配置下的力度分桶代表值。"""
    velocity_config = config.velocity_config()
    reps = build_velocity_table(velocity_config)
    print("Velocity bin representatives (bin -> decoded velocity):")
    for idx, vel in enumerate(reps):
        print(f"  VEL_{idx:02d} -> {vel}")

    print("\nSample encoding (velocity -> bin):")
    for velocity in [1, 16, 32, 48, 64, 80, 96, 112, 127]:
        print(f"  {velocity:3d} -> VEL_{velocity_to_bucket(velocity, velocity_config):02d}")


def _build_empty_row_result(config: TokenizerConfig, midi_path: str, error: str) -> SplitRowResult:
    """构造一个失败或跳过时的空结果。"""
    return SplitRowResult(
        midi_path=midi_path,
        token_variants=[],
        invalid_count=1,
        oov_count=0,
        augmented_rows=0,
        skipped_transpose_rows=0,
        applied_transpose_counts={str(offset): 0 for offset in config.train_transpose_offsets},
        skipped_transpose_counts={str(offset): 0 for offset in config.train_transpose_offsets},
        error=error,
    )


def _process_split_row(
    split_name: str,
    row: Dict[str, object],
    midi_root: str,
    config: TokenizerConfig,
    vocab: Dict[str, int],
) -> SplitRowResult:
    """处理单条 split 记录，返回主进程可直接汇总的结果。"""
    rel = str(row.get("midi_path", "")).strip()
    if not rel:
        return _build_empty_row_result(config, "<empty>", "missing_midi_path")

    applied_transpose_counts = {str(offset): 0 for offset in config.train_transpose_offsets}
    skipped_transpose_counts = {str(offset): 0 for offset in config.train_transpose_offsets}
    midi_path = Path(midi_root) / Path(rel)
    try:
        midi = mido.MidiFile(midi_path, clip=True)
        notes = _collect_tokenizer_notes(midi, config)
        bar_ticks = get_bar_ticks(midi)
        tempo_events = collect_tempo_changes(midi)

        token_variants = [_tokenize_note_events(notes, tempo_events, bar_ticks, config)]
        invalid_count = 0
        oov_count = 0
        augmented_rows = 0
        skipped_transpose_rows = 0

        # 仅对 train split 做移调增强，避免验证和评测集合产生额外样本。
        if split_name == "train" and notes:
            for offset in config.train_transpose_offsets:
                shifted_notes = _transpose_notes(notes, offset, config)
                if shifted_notes is None:
                    skipped_transpose_rows += 1
                    skipped_transpose_counts[str(offset)] += 1
                    continue
                token_variants.append(
                    _tokenize_note_events(shifted_notes, tempo_events, bar_ticks, config)
                )
                augmented_rows += 1
                applied_transpose_counts[str(offset)] += 1

        for tokens in token_variants:
            valid, line_oov = validate_token_order(tokens, vocab)
            if not valid:
                invalid_count += 1
            oov_count += line_oov

        return SplitRowResult(
            midi_path=rel,
            token_variants=token_variants,
            invalid_count=invalid_count,
            oov_count=oov_count,
            augmented_rows=augmented_rows,
            skipped_transpose_rows=skipped_transpose_rows,
            applied_transpose_counts=applied_transpose_counts,
            skipped_transpose_counts=skipped_transpose_counts,
            error=None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        result = _build_empty_row_result(config, rel, str(exc))
        result.applied_transpose_counts = applied_transpose_counts
        result.skipped_transpose_counts = skipped_transpose_counts
        return result


def _process_split_batch(
    split_name: str,
    indexed_rows: List[tuple[int, Dict[str, object]]],
    midi_root: str,
    config: TokenizerConfig,
    vocab: Dict[str, int],
) -> List[tuple[int, SplitRowResult]]:
    """处理一个批次的 split 记录，并保留原始顺序索引。"""
    return [
        (row_index, _process_split_row(split_name, row, midi_root, config, vocab))
        for row_index, row in indexed_rows
    ]


def _build_indexed_batches(
    rows: List[Dict[str, object]],
    batch_size: int,
) -> List[List[tuple[int, Dict[str, object]]]]:
    """按固定批大小切分输入记录，并携带原始顺序索引。"""
    indexed_rows = list(enumerate(rows))
    if batch_size <= 0:
        batch_size = 1
    return [
        indexed_rows[start : start + batch_size]
        for start in range(0, len(indexed_rows), batch_size)
    ]


def _compute_parallel_batch_size(row_count: int, workers: int) -> int:
    """根据输入规模和进程数估算并行批大小。"""
    safe_rows = max(1, int(row_count))
    return min(128, safe_rows)


def _format_progress_line(snapshot: ProgressSnapshot) -> str:
    """把进度快照格式化为可读日志。"""
    split_remaining = max(0, int(snapshot.split_total) - int(snapshot.split_processed))
    total_remaining = max(0, int(snapshot.total_planned) - int(snapshot.total_processed))
    total_ratio = (
        0.0
        if int(snapshot.total_planned) <= 0
        else (float(snapshot.total_processed) / float(snapshot.total_planned)) * 100.0
    )
    elapsed_minutes = float(snapshot.elapsed_seconds) / 60.0
    return (
        f"[tokenize] split={snapshot.split_name} "
        f"processed={snapshot.split_processed}/{snapshot.split_total} "
        f"remaining={split_remaining} "
        f"total_progress={snapshot.total_processed}/{snapshot.total_planned} "
        f"total_remaining={total_remaining} "
        f"progress={total_ratio:.2f}% "
        f"elapsed_min={elapsed_minutes:.1f}"
    )


def _print_split_dispatch(
    *,
    split_name: str,
    workers: int,
    batch_size: int,
    batch_count: int,
) -> None:
    """打印并行分发阶段的摘要日志。"""
    print(
        f"[tokenize] split={split_name} dispatch workers={workers} "
        f"batch_size={batch_size} batches={batch_count}",
        flush=True,
    )


def _iter_split_row_results(
    split_name: str,
    rows: List[Dict[str, object]],
    midi_root: Path,
    config: TokenizerConfig,
    vocab: Dict[str, int],
    workers: int,
    total_planned_rows: int,
    total_processed_before_split: int,
    progress_callback: Callable[[ProgressSnapshot], None] | None = None,
) -> List[SplitRowResult]:
    """根据 workers 配置选择串行或多进程处理，并按批次回传实时进度。"""
    if workers <= 1 or len(rows) <= 1:
        started_at = time.monotonic()
        results: List[SplitRowResult] = []
        for row_index, row in enumerate(rows, 1):
            results.append(_process_split_row(split_name, row, str(midi_root), config, vocab))
            if progress_callback is not None:
                progress_callback(
                    ProgressSnapshot(
                        split_name=split_name,
                        split_processed=row_index,
                        split_total=len(rows),
                        total_processed=total_processed_before_split + row_index,
                        total_planned=total_planned_rows,
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )
        return results

    max_workers = max(1, min(int(workers), len(rows)))
    batch_size = _compute_parallel_batch_size(len(rows), max_workers)
    batches = _build_indexed_batches(rows, batch_size)
    _print_split_dispatch(
        split_name=split_name,
        workers=max_workers,
        batch_size=batch_size,
        batch_count=len(batches),
    )
    ready_results: Dict[int, SplitRowResult] = {}
    ordered_results: List[SplitRowResult] = []
    next_row_index = 0
    completed_rows = 0
    started_at = time.monotonic()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_split_batch,
                split_name,
                batch,
                str(midi_root),
                config,
                vocab,
            )
            for batch in batches
        ]
        for future in as_completed(futures):
            batch_results = future.result()
            completed_rows += len(batch_results)
            for row_index, row_result in batch_results:
                ready_results[int(row_index)] = row_result
            while next_row_index in ready_results:
                ordered_results.append(ready_results.pop(next_row_index))
                next_row_index += 1
            if progress_callback is not None:
                progress_callback(
                    ProgressSnapshot(
                        split_name=split_name,
                        split_processed=completed_rows,
                        split_total=len(rows),
                        total_processed=total_processed_before_split + completed_rows,
                        total_planned=total_planned_rows,
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )
    return ordered_results


def process(
    config: TokenizerConfig,
    output_dir: Path,
    vocab_path: Path,
    stats_path: Path,
    limit_per_split: Optional[int],
    workers: int = 1,
) -> None:
    """执行 tokenization 主流程。"""
    ensure_clean_directory(output_dir)
    if vocab_path.parent.resolve() != output_dir.resolve():
        remove_file_if_exists(vocab_path)
    if stats_path.parent.resolve() != output_dir.resolve():
        remove_file_if_exists(stats_path)

    midi_root = Path(config.midi_root_dir)
    vocab = build_vocab(config)
    key_vocab_tokens = build_key_vocab_tokens()
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token

    split_rows_map: Dict[str, List[Dict[str, object]]] = {}
    for split_name, split_file in config.split_files.items():
        rows = load_jsonl(Path(split_file))
        if limit_per_split is not None:
            rows = rows[:limit_per_split]
        split_rows_map[split_name] = rows

    total_planned_rows = sum(len(rows) for rows in split_rows_map.values())
    print(
        f"[tokenize] start splits={len(split_rows_map)} total_rows={total_planned_rows} "
        f"workers={max(1, int(workers))}",
        flush=True,
    )
    split_stats: Dict[str, Dict[str, object]] = {}
    total_oov = 0
    total_invalid = 0
    total_samples = 0
    total_written_rows = 0
    total_augmented_rows = 0
    total_transpose_skips = 0
    total_processed_input_rows = 0
    parse_errors: List[Dict[str, str]] = []
    total_key_token_stats = _empty_key_token_stats(key_vocab_tokens)
    total_phrase_token_stats = _empty_phrase_token_stats()

    def _progress_callback(snapshot: ProgressSnapshot) -> None:
        """按当前模式打印实时进度。"""
        if workers <= 1:
            if snapshot.split_processed % 500 != 0 and snapshot.split_processed != snapshot.split_total:
                return
        print(_format_progress_line(snapshot), flush=True)

    for split_name, _split_file in config.split_files.items():
        rows = split_rows_map[split_name]
        mode_name = "parallel" if max(1, int(workers)) > 1 and len(rows) > 1 else "serial"
        print(
            f"[tokenize] split={split_name} start rows={len(rows)} mode={mode_name} "
            f"workers={max(1, int(workers))}",
            flush=True,
        )

        tok_lines: List[str] = []
        lengths: List[int] = []
        oov_count = 0
        invalid_count = 0
        augmented_rows = 0
        skipped_transpose_rows = 0
        applied_transpose_counts = {str(offset): 0 for offset in config.train_transpose_offsets}
        skipped_transpose_counts = {str(offset): 0 for offset in config.train_transpose_offsets}
        split_key_token_stats = _empty_key_token_stats(key_vocab_tokens)
        split_phrase_token_stats = _empty_phrase_token_stats()

        row_results = _iter_split_row_results(
            split_name=split_name,
            rows=rows,
            midi_root=midi_root,
            config=config,
            vocab=vocab,
            workers=max(1, int(workers)),
            total_planned_rows=total_planned_rows,
            total_processed_before_split=total_processed_input_rows,
            progress_callback=_progress_callback,
        )

        for row_idx, row_result in enumerate(row_results, 1):
            invalid_count += row_result.invalid_count
            oov_count += row_result.oov_count
            augmented_rows += row_result.augmented_rows
            skipped_transpose_rows += row_result.skipped_transpose_rows

            for offset_key, applied_count in row_result.applied_transpose_counts.items():
                applied_transpose_counts[offset_key] += int(applied_count)
            for offset_key, skipped_count in row_result.skipped_transpose_counts.items():
                skipped_transpose_counts[offset_key] += int(skipped_count)

            if row_result.error is not None:
                parse_errors.append(
                    {
                        "split": split_name,
                        "midi_path": row_result.midi_path,
                        "error": row_result.error,
                    }
                )

            for tokens in row_result.token_variants:
                _accumulate_key_token_stats(split_key_token_stats, tokens)
                _accumulate_key_token_stats(total_key_token_stats, tokens)
                _accumulate_phrase_token_stats(split_phrase_token_stats, tokens)
                _accumulate_phrase_token_stats(total_phrase_token_stats, tokens)
                tok_lines.append(" ".join(tokens))
                lengths.append(len(tokens))

            if row_idx % 500 == 0:
                print(
                    f"[tokenize] split={split_name} processed={row_idx}/{len(rows)} "
                    f"ok={len(tok_lines)} invalid={invalid_count} aug={augmented_rows}"
                )

        out_path = output_dir / f"{split_name}.tok"
        write_tok_lines(out_path, tok_lines)

        split_stats[split_name] = {
            "input_rows": len(rows),
            "written_rows": len(tok_lines),
            "augmented_rows": augmented_rows,
            "skipped_transpose_rows": skipped_transpose_rows,
            "applied_transpose_counts": applied_transpose_counts,
            "skipped_transpose_counts": skipped_transpose_counts,
            "invalid_rows": invalid_count,
            "oov_count": oov_count,
            "length_stats": summarize_lengths(lengths),
            "key_token_stats": split_key_token_stats,
            "phrase_token_stats": _finalize_phrase_token_stats(split_phrase_token_stats, len(tok_lines)),
            "output_file": str(out_path),
        }

        total_oov += oov_count
        total_invalid += invalid_count
        total_samples += len(rows)
        total_written_rows += len(tok_lines)
        total_augmented_rows += augmented_rows
        total_transpose_skips += skipped_transpose_rows
        total_processed_input_rows += len(rows)
        print(
            f"[tokenize] split={split_name} done rows={len(rows)} written={len(tok_lines)} "
            f"aug={augmented_rows} invalid={invalid_count} oov={oov_count}",
            flush=True,
        )

    stats = {
        "tokenizer_config": asdict(config),
        "vocab_size": len(vocab),
        "oov_count": total_oov,
        "invalid_rows": total_invalid,
        "total_rows": total_samples,
        "total_written_rows": total_written_rows,
        "total_augmented_rows": total_augmented_rows,
        "total_transpose_skips": total_transpose_skips,
        "invalid_ratio": (0.0 if total_samples == 0 else total_invalid / total_samples),
        "key_token_stats": total_key_token_stats,
        "phrase_token_stats": _finalize_phrase_token_stats(total_phrase_token_stats, total_written_rows),
        "split_stats": split_stats,
        "parse_errors_head": parse_errors[:200],
    }

    dump_json_file(
        vocab_path,
        {
            "token_to_id": vocab,
            "id_to_token": id_to_token,
        },
    )
    dump_json_file(stats_path, stats)
    print(
        f"[tokenize] done vocab={len(vocab)} rows={total_samples} "
        f"written={total_written_rows} aug={total_augmented_rows} "
        f"invalid={total_invalid} oov={total_oov}"
    )
    print(f"[tokenize] vocab -> {vocab_path}")
    print(f"[tokenize] stats -> {stats_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="把 MIDI 切分清单编码为 token 序列。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/tokenizer.yaml"),
        help="tokenizer YAML 配置路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="`.tok` 输出目录。",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("data/tokenized/tokenizer_vocab.json"),
        help="词表输出路径。",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/tokenized/token_stats.json"),
        help="统计输出路径。",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="可选：每个 split 最多处理 N 条，用于烟雾测试。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="分词阶段的工作进程数；1 表示单进程。",
    )
    parser.add_argument(
        "--print-velocity-table",
        action="store_true",
        help="打印当前配置下的 velocity 分桶代表值后退出。",
    )
    return parser.parse_args()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    config = load_config(args.config)
    if args.print_velocity_table:
        print_velocity_table(config)
        return
    process(
        config=config,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        stats_path=args.stats_path,
        limit_per_split=args.limit_per_split,
        workers=max(1, int(args.workers)),
    )


if __name__ == "__main__":
    main()
