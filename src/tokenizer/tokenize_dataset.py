#!/usr/bin/env python
"""TuneFlow 数据分词脚本（当前阶段 INST 固定为 PIANO）。"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：mido。请先执行 `python -m pip install mido`。"
    ) from exc

from ..utils.config_io import dump_json_file, load_yaml_mapping
from .common import (
    collect_note_events,
    collect_tempo_changes,
    get_bar_ticks,
    load_jsonl,
    nearest_value,
    summarize_lengths,
    write_tok_lines,
)


@dataclass
class TokenizerConfig:
    """分词配置。"""

    midi_root_dir: str = "data/clean"
    positions_per_bar: int = 32
    pitch_min: int = 21
    pitch_max: int = 108
    duration_bins: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    )
    velocity_bins: int = 16
    velocity_mu: float = 8.0
    velocity_center: float = 64.0
    velocity_radius: float = 63.0
    tempo_min: int = 40
    tempo_max: int = 220
    tempo_step: int = 2
    inst_classes: List[str] = field(default_factory=lambda: ["PIANO"])
    default_inst: str = "PIANO"
    special_tokens: List[str] = field(
        default_factory=lambda: ["BOS", "EOS", "FIM_HOLE", "FIM_MID"]
    )
    include_task_tokens: bool = False
    task_tokens: List[str] = field(
        default_factory=lambda: ["TASK_INFILL", "TASK_CONT", "TASK_GEN"]
    )
    recursive: bool = True
    split_files: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "data/base/train.jsonl",
            "valid": "data/base/valid.jsonl",
            "test": "data/base/test.jsonl",
            "eval": "data/eval/fixed_eval.jsonl",
        }
    )

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> "TokenizerConfig":
        """从 YAML 字典构建配置，并忽略未知字段。"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        safe = {k: v for k, v in data.items() if k in valid_keys}
        cfg = cls(**safe)
        if cfg.positions_per_bar <= 0:
            raise ValueError("positions_per_bar 必须为正数")
        if cfg.velocity_bins <= 1:
            raise ValueError("velocity_bins 必须大于 1")
        if cfg.tempo_step <= 0:
            raise ValueError("tempo_step 必须为正数")
        if cfg.pitch_min > cfg.pitch_max:
            raise ValueError("pitch_min 不能大于 pitch_max")
        if cfg.default_inst not in cfg.inst_classes:
            raise ValueError("default_inst 必须在 inst_classes 内")
        return cfg


def load_config(path: Path) -> TokenizerConfig:
    """读取 tokenizer 配置。"""
    raw = load_yaml_mapping(path, "tokenizer 配置")
    return TokenizerConfig.from_mapping(raw)


def velocity_to_bucket(velocity: int, config: TokenizerConfig) -> int:
    """将 MIDI velocity 映射到 `VEL_0..N-1`。"""
    v = min(127, max(1, int(velocity)))
    c = config.velocity_center
    r = max(1e-6, config.velocity_radius)
    mu = max(1e-6, config.velocity_mu)
    x = (v - c) / r
    sign = -1.0 if x < 0 else 1.0
    s = sign * math.log1p(mu * abs(x)) / math.log1p(mu)
    bins = config.velocity_bins - 1
    k = int(round(((s + 1.0) / 2.0) * bins))
    return min(bins, max(0, k))


def bpm_to_token(bpm: float, config: TokenizerConfig) -> str:
    """将 BPM 映射到离散 `TEMPO_x` token。"""
    clipped = min(config.tempo_max, max(config.tempo_min, int(round(bpm))))
    q = int(round((clipped - config.tempo_min) / config.tempo_step))
    value = config.tempo_min + q * config.tempo_step
    value = min(config.tempo_max, max(config.tempo_min, value))
    return f"TEMPO_{value}"


def build_vocab(config: TokenizerConfig) -> Dict[str, int]:
    """构建词表映射 token -> id。"""
    vocab: List[str] = []
    vocab.extend(config.special_tokens)
    if config.include_task_tokens:
        vocab.extend(config.task_tokens)
    vocab.append("BAR")
    vocab.extend([f"POS_{i}" for i in range(config.positions_per_bar)])
    vocab.extend([f"INST_{x}" for x in config.inst_classes])
    vocab.extend([f"PITCH_{p}" for p in range(config.pitch_min, config.pitch_max + 1)])
    vocab.extend([f"DUR_{d}" for d in config.duration_bins])
    vocab.extend([f"VEL_{i}" for i in range(config.velocity_bins)])
    for tempo in range(config.tempo_min, config.tempo_max + 1, config.tempo_step):
        vocab.append(f"TEMPO_{tempo}")

    seen = set()
    deduped: List[str] = []
    for t in vocab:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return {t: i for i, t in enumerate(deduped)}


def tokenize_midi(midi: mido.MidiFile, config: TokenizerConfig) -> List[str]:
    """将单个 MIDI 编码成 token 序列。"""
    notes = [
        n
        for n in collect_note_events(midi)
        if config.pitch_min <= n.pitch <= config.pitch_max and n.duration_tick > 0
    ]
    if not notes:
        return ["BOS", "EOS"]

    bar_ticks = get_bar_ticks(midi)
    pos_ticks = bar_ticks / config.positions_per_bar
    tempo_events = collect_tempo_changes(midi)

    per_bar: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    max_bar_idx = 0
    for n in notes:
        bar_idx = n.start_tick // bar_ticks
        max_bar_idx = max(max_bar_idx, bar_idx)
        in_bar_tick = n.start_tick % bar_ticks
        pos = int(round(in_bar_tick / max(1e-9, pos_ticks)))
        pos = min(config.positions_per_bar - 1, max(0, pos))

        dur_pos = int(round(n.duration_tick / max(1e-9, pos_ticks)))
        dur_pos = max(1, dur_pos)
        dur_bin = nearest_value(dur_pos, config.duration_bins)
        vel_bin = velocity_to_bucket(n.velocity, config)
        per_bar[bar_idx].append((pos, n.pitch, dur_bin, vel_bin))

    tokens: List[str] = ["BOS"]
    first_tempo_token = bpm_to_token(tempo_events[0][1], config)
    tokens.append(first_tempo_token)
    last_tempo_token = first_tempo_token

    tempo_idx = 0
    for bar_idx in range(max_bar_idx + 1):
        bar_start = bar_idx * bar_ticks
        while tempo_idx + 1 < len(tempo_events) and tempo_events[tempo_idx + 1][0] <= bar_start:
            tempo_idx += 1
        current_tempo_token = bpm_to_token(tempo_events[tempo_idx][1], config)

        tokens.append("BAR")
        if current_tempo_token != last_tempo_token:
            tokens.append(current_tempo_token)
            last_tempo_token = current_tempo_token

        events = per_bar.get(bar_idx, [])
        events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        for pos, pitch, dur_bin, vel_bin in events:
            tokens.append(f"POS_{pos}")
            tokens.append(f"INST_{config.default_inst}")
            tokens.append(f"PITCH_{pitch}")
            tokens.append(f"DUR_{dur_bin}")
            tokens.append(f"VEL_{vel_bin}")

    tokens.append("EOS")
    return tokens


def validate_token_order(tokens: Sequence[str], vocab: Dict[str, int]) -> Tuple[bool, int]:
    """校验 token 顺序是否合法，并返回 OOV 数量。"""
    oov = sum(1 for t in tokens if t not in vocab)
    if not tokens or tokens[0] != "BOS":
        return False, oov
    if tokens[-1] != "EOS":
        return False, oov

    i = 1
    if i < len(tokens) - 1 and tokens[i].startswith("TEMPO_"):
        i += 1

    while i < len(tokens) - 1:
        if tokens[i] != "BAR":
            return False, oov
        i += 1
        if i < len(tokens) - 1 and tokens[i].startswith("TEMPO_"):
            i += 1
        while i < len(tokens) - 1 and tokens[i].startswith("POS_"):
            if i + 4 >= len(tokens):
                return False, oov
            if not tokens[i + 1].startswith("INST_"):
                return False, oov
            if not tokens[i + 2].startswith("PITCH_"):
                return False, oov
            if not tokens[i + 3].startswith("DUR_"):
                return False, oov
            if not tokens[i + 4].startswith("VEL_"):
                return False, oov
            i += 5
        if i < len(tokens) - 1 and tokens[i] != "BAR":
            # 若不是下一个 BAR，那只能是 EOS（循环条件会处理 EOS）
            return False, oov

    return True, oov


def process(
    config: TokenizerConfig,
    output_dir: Path,
    vocab_path: Path,
    stats_path: Path,
    limit_per_split: Optional[int],
) -> None:
    """执行 tokenization 主流程。"""
    midi_root = Path(config.midi_root_dir)
    vocab = build_vocab(config)
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token

    split_stats: Dict[str, Dict[str, object]] = {}
    total_oov = 0
    total_invalid = 0
    total_samples = 0
    parse_errors: List[Dict[str, str]] = []

    for split_name, split_file in config.split_files.items():
        rows = load_jsonl(Path(split_file))
        if limit_per_split is not None:
            rows = rows[:limit_per_split]

        tok_lines: List[str] = []
        lengths: List[int] = []
        oov_count = 0
        invalid_count = 0

        for idx, row in enumerate(rows, 1):
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
                tokens = tokenize_midi(midi, config)
                valid, line_oov = validate_token_order(tokens, vocab)
                if not valid:
                    invalid_count += 1
                oov_count += line_oov
                tok_lines.append(" ".join(tokens))
                lengths.append(len(tokens))
            except Exception as exc:  # pylint: disable=broad-except
                invalid_count += 1
                parse_errors.append(
                    {"split": split_name, "midi_path": rel, "error": str(exc)}
                )

            if idx % 500 == 0:
                print(
                    f"[tokenize] split={split_name} processed={idx}/{len(rows)} "
                    f"ok={len(tok_lines)} invalid={invalid_count}"
                )

        out_path = output_dir / f"{split_name}.tok"
        write_tok_lines(out_path, tok_lines)

        split_stats[split_name] = {
            "input_rows": len(rows),
            "written_rows": len(tok_lines),
            "invalid_rows": invalid_count,
            "oov_count": oov_count,
            "length_stats": summarize_lengths(lengths),
            "output_file": str(out_path),
        }

        total_oov += oov_count
        total_invalid += invalid_count
        total_samples += len(rows)

    stats = {
        "tokenizer_config": asdict(config),
        "vocab_size": len(vocab),
        "oov_count": total_oov,
        "invalid_rows": total_invalid,
        "total_rows": total_samples,
        "invalid_ratio": (0.0 if total_samples == 0 else total_invalid / total_samples),
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
    return parser.parse_args()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    config = load_config(args.config)
    process(
        config=config,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        stats_path=args.stats_path,
        limit_per_split=args.limit_per_split,
    )


if __name__ == "__main__":
    main()
