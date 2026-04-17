#!/usr/bin/env python
"""TuneFlow 数据切分脚本（当前版本不处理风格化数据）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils.output_cleanup import ensure_clean_directory, remove_file_if_exists

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：mido。请先执行 `python -m pip install mido`。"
    ) from exc

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：pyyaml。请先执行 `python -m pip install pyyaml`。"
    ) from exc


@dataclass
class SplitConfig:
    """数据切分配置。"""

    train_ratio: float = 0.9
    valid_ratio: float = 0.05
    test_ratio: float = 0.05
    eval_from_test_ratio: float = 1.0
    seed: int = 42
    recursive: bool = True
    extensions: List[str] = field(default_factory=lambda: [".mid", ".midi"])
    dedup_tick_resolution: int = 12
    style_disabled: bool = True

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> "SplitConfig":
        """从 YAML 字典构建配置，并忽略未知字段。"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        safe_data = {k: v for k, v in data.items() if k in valid_keys}
        cfg = cls(**safe_data)
        total = cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio
        if abs(total - 1.0) > 1e-8:
            raise ValueError("train/valid/test 比例之和必须为 1.0")
        return cfg


@dataclass
class FileRecord:
    """清洗后文件元数据。"""

    path: Path
    rel_path: str
    artist: str
    title: str
    family_key: str
    content_hash: str
    note_count: int
    duration_sec: float


class DisjointSet:
    """并查集，用于把“同曲目家族或同哈希”合并成同一组。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """查找根节点（路径压缩）。"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        """合并两个节点所在集合。"""
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def load_config(path: Path) -> SplitConfig:
    """读取切分配置文件。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("切分配置必须是字典（mapping）")
    return SplitConfig.from_mapping(raw)


def discover_files(
    input_dir: Path, extensions: Sequence[str], recursive: bool
) -> List[Path]:
    """扫描输入目录下的 MIDI 文件。"""
    ext_set = {ext.lower() for ext in extensions}
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.glob("*") if p.is_file()]
    out = [p for p in files if p.suffix.lower() in ext_set]
    out.sort()
    return out


def normalize_text(text: str) -> str:
    """归一化文本，用于构建 family_key。"""
    lowered = text.lower().strip()
    lowered = re.sub(r"\.\d+$", "", lowered)
    lowered = re.sub(r"[_\-]+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered or "unknown"


def infer_artist_and_title(src: Path, input_dir: Path) -> Tuple[str, str]:
    """从路径推断 artist 与 title。"""
    rel = src.relative_to(input_dir)
    parts = rel.parts
    title = src.stem
    if len(parts) >= 2:
        artist = parts[-2]
    else:
        artist = "unknown"
    # 兼容 data/clean/clean_midi/<artist>/<song>.mid 这类目录
    if len(parts) >= 3 and normalize_text(parts[0]) == "clean midi":
        artist = parts[1]
    return artist, title


def collect_note_tuples(midi: mido.MidiFile) -> List[Tuple[int, int, int]]:
    """解析音符并返回 (start_tick, duration_tick, pitch) 列表。"""
    active: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    tuples: List[Tuple[int, int, int]] = []

    for track_idx, track in enumerate(midi.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += int(msg.time)
            if msg.is_meta or not hasattr(msg, "channel"):
                continue
            channel = int(msg.channel)
            if channel == 9:
                continue
            if msg.type == "note_on" and int(msg.velocity) > 0:
                key = (track_idx, channel, int(msg.note))
                active[key].append(abs_tick)
            elif msg.type in {"note_off", "note_on"} and (
                msg.type == "note_off" or int(msg.velocity) == 0
            ):
                key = (track_idx, channel, int(msg.note))
                if active[key]:
                    start = active[key].pop()
                    tuples.append((start, max(0, abs_tick - start), int(msg.note)))
    return tuples


def build_content_hash(
    note_tuples: Sequence[Tuple[int, int, int]],
    ticks_per_beat: int,
    dedup_tick_resolution: int,
) -> str:
    """构建标准化音符序列哈希。"""
    if not note_tuples:
        return "EMPTY"
    target_tpb = 480
    q = max(1, dedup_tick_resolution)
    normalized: List[Tuple[int, int, int]] = []
    for start, dur, pitch in note_tuples:
        start_scaled = round(start * target_tpb / max(1, ticks_per_beat))
        dur_scaled = round(dur * target_tpb / max(1, ticks_per_beat))
        start_q = int(round(start_scaled / q) * q)
        dur_q = int(round(dur_scaled / q) * q)
        normalized.append((start_q, dur_q, pitch))
    normalized.sort()
    return hashlib.sha1(
        json.dumps(normalized, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_file_record(src: Path, input_dir: Path, config: SplitConfig) -> FileRecord:
    """从单个 MIDI 文件提取切分所需元数据。"""
    midi = mido.MidiFile(src, clip=True)
    notes = collect_note_tuples(midi)
    artist, title = infer_artist_and_title(src, input_dir)
    family_key = f"{normalize_text(artist)}::{normalize_text(title)}"
    content_hash = build_content_hash(
        note_tuples=notes,
        ticks_per_beat=midi.ticks_per_beat,
        dedup_tick_resolution=config.dedup_tick_resolution,
    )
    rel_path = src.relative_to(input_dir).as_posix()
    return FileRecord(
        path=src,
        rel_path=rel_path,
        artist=artist,
        title=title,
        family_key=family_key,
        content_hash=content_hash,
        note_count=len(notes),
        duration_sec=float(midi.length),
    )


def build_leakage_safe_groups(records: Sequence[FileRecord]) -> List[List[int]]:
    """按“同 family 或同哈希”合并分组，避免跨集合泄漏。"""
    dsu = DisjointSet(len(records))
    first_by_family: Dict[str, int] = {}
    first_by_hash: Dict[str, int] = {}

    for i, rec in enumerate(records):
        if rec.family_key in first_by_family:
            dsu.union(i, first_by_family[rec.family_key])
        else:
            first_by_family[rec.family_key] = i

        if rec.content_hash in first_by_hash:
            dsu.union(i, first_by_hash[rec.content_hash])
        else:
            first_by_hash[rec.content_hash] = i

    groups_map: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(records)):
        root = dsu.find(i)
        groups_map[root].append(i)
    return list(groups_map.values())


def split_groups(
    groups: Sequence[Sequence[int]],
    total_size: int,
    config: SplitConfig,
) -> Dict[str, List[int]]:
    """按目标比例将分组分配到 train/valid/test。"""
    rng = random.Random(config.seed)
    shuffled = [list(g) for g in groups]
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)

    targets = {
        "train": int(round(total_size * config.train_ratio)),
        "valid": int(round(total_size * config.valid_ratio)),
    }
    targets["test"] = total_size - targets["train"] - targets["valid"]

    assigned = {"train": [], "valid": [], "test": []}
    counts = {"train": 0, "valid": 0, "test": 0}

    names = ["train", "valid", "test"]
    for group in shuffled:
        size = len(group)
        best_name: Optional[str] = None
        best_err: Optional[float] = None
        tie_break: float = 0.0
        for name in names:
            new_counts = counts.copy()
            new_counts[name] += size
            err = sum((new_counts[k] - targets[k]) ** 2 for k in names)
            tb = rng.random()
            if best_err is None or err < best_err or (err == best_err and tb > tie_break):
                best_err = err
                best_name = name
                tie_break = tb
        assert best_name is not None
        assigned[best_name].extend(group)
        counts[best_name] += size

    return assigned


def choose_fixed_eval_indices(
    test_indices: Sequence[int],
    config: SplitConfig,
) -> List[int]:
    """从 test 中抽取固定 eval 子集。默认比例为 1.0（即全部 test）。"""
    if not test_indices:
        return []
    ratio = max(0.0, min(1.0, config.eval_from_test_ratio))
    count = int(round(len(test_indices) * ratio))
    if ratio > 0.0 and count == 0:
        count = 1
    rng = random.Random(config.seed + 7)
    ordered = list(test_indices)
    rng.shuffle(ordered)
    chosen = ordered[:count]
    chosen.sort()
    return chosen


def record_to_json(rec: FileRecord) -> Dict[str, object]:
    """将 FileRecord 转为 JSONL 行对象。"""
    return {
        "midi_path": rec.rel_path,
        "artist": rec.artist,
        "title": rec.title,
        "family_key": rec.family_key,
        "content_hash": rec.content_hash,
        "note_count": rec.note_count,
        "duration_sec": round(rec.duration_sec, 6),
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """写出 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    """写出 JSON 报告文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process(
    input_dir: Path,
    output_base_dir: Path,
    output_eval_path: Path,
    report_path: Path,
    config: SplitConfig,
    limit: Optional[int],
) -> None:
    """执行切分流程并产出 train/valid/test + fixed_eval。"""
    files = discover_files(
        input_dir=input_dir,
        extensions=config.extensions,
        recursive=config.recursive,
    )
    if limit is not None:
        files = files[:limit]

    records: List[FileRecord] = []
    parse_errors: List[Dict[str, str]] = []

    for idx, src in enumerate(files, 1):
        try:
            rec = build_file_record(src, input_dir, config)
            records.append(rec)
        except Exception as exc:  # pylint: disable=broad-except
            parse_errors.append({"file": str(src), "error": str(exc)})
        if idx % 500 == 0:
            print(f"[split] scanned={idx}/{len(files)} valid={len(records)}")

    groups = build_leakage_safe_groups(records)
    assign = split_groups(groups=groups, total_size=len(records), config=config)

    train_records = [records[i] for i in sorted(assign["train"])]
    valid_records = [records[i] for i in sorted(assign["valid"])]
    test_records = [records[i] for i in sorted(assign["test"])]

    eval_indices = choose_fixed_eval_indices(assign["test"], config)
    eval_records = [records[i] for i in eval_indices]

    ensure_clean_directory(output_base_dir)
    if output_eval_path.parent.resolve() != output_base_dir.resolve():
        remove_file_if_exists(output_eval_path)
    if report_path.parent.resolve() != output_base_dir.resolve():
        remove_file_if_exists(report_path)

    train_path = output_base_dir / "train.jsonl"
    valid_path = output_base_dir / "valid.jsonl"
    test_path = output_base_dir / "test.jsonl"

    write_jsonl(train_path, [record_to_json(r) for r in train_records])
    write_jsonl(valid_path, [record_to_json(r) for r in valid_records])
    write_jsonl(test_path, [record_to_json(r) for r in test_records])
    write_jsonl(output_eval_path, [record_to_json(r) for r in eval_records])

    report = {
        "input_dir": str(input_dir),
        "output_base_dir": str(output_base_dir),
        "output_eval_path": str(output_eval_path),
        "total_files_scanned": len(files),
        "valid_files": len(records),
        "parse_error_files": len(parse_errors),
        "groups": len(groups),
        "split_counts": {
            "train": len(train_records),
            "valid": len(valid_records),
            "test": len(test_records),
            "fixed_eval": len(eval_records),
        },
        "ratios": {
            "train_ratio": config.train_ratio,
            "valid_ratio": config.valid_ratio,
            "test_ratio": config.test_ratio,
            "eval_from_test_ratio": config.eval_from_test_ratio,
        },
        "style_processing": "disabled",
        "config": asdict(config),
        "sample_errors": parse_errors[:50],
    }
    dump_json(report_path, report)

    print(
        "[split] done "
        f"train={len(train_records)} valid={len(valid_records)} "
        f"test={len(test_records)} eval={len(eval_records)}"
    )
    print(f"[split] report -> {report_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="切分清洗后的 MIDI 数据集。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/clean"),
        help="清洗后的 MIDI 输入目录。",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=Path("data/base"),
        help="基础数据切分输出目录（train/valid/test）。",
    )
    parser.add_argument(
        "--output-eval-path",
        type=Path,
        default=Path("data/eval/fixed_eval.jsonl"),
        help="固定评估集输出路径。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/split.yaml"),
        help="切分配置 YAML 路径。",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/data/split_report.json"),
        help="切分报告输出路径。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选：仅处理前 N 个文件（用于快速测试）。",
    )
    return parser.parse_args()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    config = load_config(args.config)
    process(
        input_dir=args.input_dir,
        output_base_dir=args.output_base_dir,
        output_eval_path=args.output_eval_path,
        report_path=args.report_path,
        config=config,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
