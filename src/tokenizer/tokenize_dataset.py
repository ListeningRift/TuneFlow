#!/usr/bin/env python
"""TuneFlow 数据分词脚本（当前阶段 INST 固定为 PIANO）。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：mido。请先在你当前环境中执行 `uv sync --active`。"
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


def _empty_key_token_stats(key_tokens: List[str]) -> Dict[str, object]:
    return {
        "total_key_tokens": 0,
        "counts_by_token": {token: 0 for token in key_tokens},
        "major_total": 0,
        "minor_total": 0,
        "uncertain_total": 0,
    }


def _accumulate_key_token_stats(stats: Dict[str, object], tokens: List[str]) -> None:
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
    """打印当前 tokenizer 配置下的力度桶代表值和示例编码。"""
    velocity_config = config.velocity_config()
    reps = build_velocity_table(velocity_config)
    print("Velocity bin representatives (bin -> decoded velocity):")
    for idx, vel in enumerate(reps):
        print(f"  VEL_{idx:02d} -> {vel}")

    print("\nSample encoding (velocity -> bin):")
    for velocity in [1, 16, 32, 48, 64, 80, 96, 112, 127]:
        print(f"  {velocity:3d} -> VEL_{velocity_to_bucket(velocity, velocity_config):02d}")


def process(
    config: TokenizerConfig,
    output_dir: Path,
    vocab_path: Path,
    stats_path: Path,
    limit_per_split: Optional[int],
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

    split_stats: Dict[str, Dict[str, object]] = {}
    total_oov = 0
    total_invalid = 0
    total_samples = 0
    total_written_rows = 0
    total_augmented_rows = 0
    total_transpose_skips = 0
    parse_errors: List[Dict[str, str]] = []
    total_key_token_stats = _empty_key_token_stats(key_vocab_tokens)

    for split_name, split_file in config.split_files.items():
        rows = load_jsonl(Path(split_file))
        if limit_per_split is not None:
            rows = rows[:limit_per_split]

        tok_lines: List[str] = []
        lengths: List[int] = []
        oov_count = 0
        invalid_count = 0
        augmented_rows = 0
        skipped_transpose_rows = 0
        applied_transpose_counts = {
            str(offset): 0 for offset in config.train_transpose_offsets
        }
        skipped_transpose_counts = {
            str(offset): 0 for offset in config.train_transpose_offsets
        }
        split_key_token_stats = _empty_key_token_stats(key_vocab_tokens)

        for row_idx, row in enumerate(rows, 1):
            rel = str(row.get("midi_path", "")).strip()
            if not rel:
                invalid_count += 1
                parse_errors.append(
                    {"split": split_name, "midi_path": "<empty>", "error": "missing_midi_path"}
                )
                continue

            midi_path = midi_root / Path(rel)
            try:
                midi = mido.MidiFile(midi_path, clip=True)
                notes = _collect_tokenizer_notes(midi, config)
                bar_ticks = get_bar_ticks(midi)
                tempo_events = collect_tempo_changes(midi)

                token_variants = [_tokenize_note_events(notes, tempo_events, bar_ticks, config)]
                # 仅对 train split 追加移调增强，验证与评测集保持原样。
                if split_name == "train" and notes:
                    for offset in config.train_transpose_offsets:
                        shifted_notes = _transpose_notes(notes, offset, config)
                        if shifted_notes is None:
                            # 任一音符移调后越界，则跳过该增强版本，避免静默裁剪音高。
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
                    _accumulate_key_token_stats(split_key_token_stats, tokens)
                    _accumulate_key_token_stats(total_key_token_stats, tokens)
                    tok_lines.append(" ".join(tokens))
                    lengths.append(len(tokens))
            except Exception as exc:  # pylint: disable=broad-except
                invalid_count += 1
                parse_errors.append(
                    {"split": split_name, "midi_path": rel, "error": str(exc)}
                )

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
            "output_file": str(out_path),
        }

        total_oov += oov_count
        total_invalid += invalid_count
        total_samples += len(rows)
        total_written_rows += len(tok_lines)
        total_augmented_rows += augmented_rows
        total_transpose_skips += skipped_transpose_rows

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
    parser = argparse.ArgumentParser(description="将 MIDI 切分清单编码为 token 序列。")
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
        help="可选：每个 split 最多处理 N 条（用于冒烟测试）。",
    )
    parser.add_argument(
        "--print-velocity-table",
        action="store_true",
        help="打印当前配置下的 velocity 分桶代表值并退出。",
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
    )


if __name__ == "__main__":
    main()
