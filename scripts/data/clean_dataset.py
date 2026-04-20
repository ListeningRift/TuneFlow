#!/usr/bin/env python
"""TuneFlow 数据清洗脚本。

该脚本会对原始 MIDI 数据进行过滤与规范化处理，输出到 `data/clean`。
目标是为后续 tokenizer 与训练阶段提供结构更稳定、质量更可控的符号数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils.output_cleanup import ensure_clean_directory, remove_file_if_exists

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：mido。请先在你当前环境中执行 `uv sync --active`。"
    ) from exc

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：pyyaml。请先在你当前环境中执行 `uv sync --active`。"
    ) from exc


# 默认保留常见旋律/低音乐器的音高范围。
DEFAULT_MIN_PITCH = 21
DEFAULT_MAX_PITCH = 108


@dataclass
class CleaningConfig:
    """数据清洗参数配置。"""

    min_note_count: int = 50
    min_unique_pitch: int = 5
    drop_if_zero_duration_note: bool = True
    min_duration_sec: float = 10.0
    min_total_bars: int = 8
    max_total_bars: int = 512
    bpm_min: float = 40.0
    bpm_max: float = 220.0
    keep_first_tempo_only: bool = True
    keep_pitch_min: int = DEFAULT_MIN_PITCH
    keep_pitch_max: int = DEFAULT_MAX_PITCH
    track_selection_mode: str = "piano_first_then_most_notes"
    piano_program: int = 0
    max_simultaneous_notes: int = 16
    remove_drum_channel_10: bool = True
    # GM 音色表中的打击乐家族（0-based program 编号）
    drum_programs: List[int] = field(
        default_factory=lambda: [112, 113, 114, 115, 116, 117, 118, 119]
    )
    min_melody_tracks: int = 1
    min_notes_per_melody_track: int = 8
    min_unique_pitch_per_melody_track: int = 4
    deduplicate: bool = True
    dedup_tick_resolution: int = 12
    recursive: bool = True
    extensions: List[str] = field(default_factory=lambda: [".mid", ".midi"])

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> "CleaningConfig":
        """从 YAML 字典中构建配置，并忽略未知字段。"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        safe_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**safe_data)


@dataclass
class NoteEvent:
    """内部统一音符事件表示。"""

    start_tick: int
    end_tick: int
    pitch: int
    track_idx: int
    channel: int

    @property
    def duration_tick(self) -> int:
        """返回音符时值（tick）。"""
        return self.end_tick - self.start_tick


@dataclass
class FileMetrics:
    """单个 MIDI 文件的质量统计。"""

    note_count: int
    unique_pitch_count: int
    zero_duration_note_count: int
    max_simultaneous_notes: int
    melodic_track_count: int
    total_bars: int
    duration_sec: float
    bpm_median: Optional[float]
    content_hash: str


@dataclass
class TrackStats:
    """单轨统计信息，用于主轨筛选。"""

    index: int
    note_count: int
    has_explicit_piano_program: bool
    has_channel_message: bool


def load_config(path: Path) -> CleaningConfig:
    """读取并解析清洗配置文件。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("清洗配置必须是字典（mapping）")
    return CleaningConfig.from_mapping(raw)


def discover_files(
    input_dir: Path, extensions: Sequence[str], recursive: bool
) -> List[Path]:
    """扫描输入目录，返回符合扩展名的 MIDI 文件列表。"""
    ext_set = {ext.lower() for ext in extensions}
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.glob("*") if p.is_file()]
    return [p for p in files if p.suffix.lower() in ext_set]


def extract_median_bpm(midi: mido.MidiFile) -> Optional[float]:
    """提取 MIDI 中 tempo 事件对应 BPM 的中位数。"""
    bpms: List[float] = []
    for track in midi.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                bpms.append(float(mido.tempo2bpm(msg.tempo)))
    if not bpms:
        return None
    return float(median(bpms))


def get_bar_ticks(midi: mido.MidiFile) -> int:
    """获取每小节 tick 数（取第一个拍号；无拍号默认 4/4）。"""
    numerator = 4
    denominator = 4
    for track in midi.tracks:
        for msg in track:
            if msg.type == "time_signature":
                numerator = int(msg.numerator)
                denominator = int(msg.denominator)
                beats_per_bar = numerator * (4.0 / max(1, denominator))
                return max(1, int(round(midi.ticks_per_beat * beats_per_bar)))
    return max(1, int(round(midi.ticks_per_beat * 4.0)))


def collect_track_stats(
    midi: mido.MidiFile,
    config: CleaningConfig,
) -> List[TrackStats]:
    """统计每个轨道的音符数量与 piano 特征。"""
    stats: List[TrackStats] = []
    for idx, track in enumerate(midi.tracks):
        note_count = 0
        has_piano = False
        has_channel_msg = False
        for msg in track:
            if msg.is_meta or not hasattr(msg, "channel"):
                continue
            has_channel_msg = True
            channel = int(msg.channel)
            if config.remove_drum_channel_10 and channel == 9:
                continue
            if msg.type == "program_change" and int(msg.program) == config.piano_program:
                has_piano = True
            if msg.type == "note_on" and int(msg.velocity) > 0:
                note_count += 1
        stats.append(
            TrackStats(
                index=idx,
                note_count=note_count,
                has_explicit_piano_program=has_piano,
                has_channel_message=has_channel_msg,
            )
        )
    return stats


def select_primary_track_index(
    stats: Sequence[TrackStats],
    mode: str,
) -> Optional[int]:
    """根据策略选择一个主轨道索引。"""
    melodic = [s for s in stats if s.note_count > 0]
    if not melodic:
        return None

    normalized_mode = mode.strip().lower()
    if normalized_mode == "none":
        return None
    if normalized_mode == "most_notes":
        return max(melodic, key=lambda s: s.note_count).index
    if normalized_mode == "piano_only":
        piano_tracks = [s for s in melodic if s.has_explicit_piano_program]
        if not piano_tracks:
            return None
        return max(piano_tracks, key=lambda s: s.note_count).index
    if normalized_mode == "piano_first_then_most_notes":
        piano_tracks = [s for s in melodic if s.has_explicit_piano_program]
        if piano_tracks:
            return max(piano_tracks, key=lambda s: s.note_count).index
        return max(melodic, key=lambda s: s.note_count).index
    raise ValueError(
        "track_selection_mode 仅支持: none, piano_only, most_notes, piano_first_then_most_notes"
    )


def filter_midi(
    midi: mido.MidiFile, config: CleaningConfig
) -> Tuple[mido.MidiFile, bool]:
    """执行轨道筛选与事件过滤，返回清洗后的 MIDI 与是否发生改动。"""
    cleaned = mido.MidiFile(
        type=midi.type,
        ticks_per_beat=midi.ticks_per_beat,
        charset=getattr(midi, "charset", "latin1"),
    )
    changed = False
    tempo_kept = False

    drum_programs_set = set(config.drum_programs)
    track_stats = collect_track_stats(midi, config)
    primary_track_index = select_primary_track_index(
        track_stats, config.track_selection_mode
    )
    selection_enabled = config.track_selection_mode.strip().lower() != "none"

    for track_idx, track in enumerate(midi.tracks):
        track_info = track_stats[track_idx]
        keep_this_track = True
        if selection_enabled:
            if track_info.note_count > 0:
                keep_this_track = track_idx == primary_track_index
            else:
                # 仅保留纯 meta 轨（tempo/time signature 等全局信息）
                keep_this_track = not track_info.has_channel_message
        if not keep_this_track:
            changed = True
            continue

        new_track = mido.MidiTrack()
        carry_time = 0
        channel_is_drum_program: Dict[int, bool] = defaultdict(bool)

        for msg in track:
            carry_time += msg.time
            keep = True

            if msg.type == "set_tempo" and config.keep_first_tempo_only:
                if tempo_kept:
                    keep = False
                else:
                    tempo_kept = True

            if not msg.is_meta and hasattr(msg, "channel"):
                channel = int(msg.channel)

                if config.remove_drum_channel_10 and channel == 9:
                    keep = False
                elif msg.type == "program_change":
                    is_drum_program = int(msg.program) in drum_programs_set
                    channel_is_drum_program[channel] = is_drum_program
                    if is_drum_program:
                        keep = False
                elif channel_is_drum_program[channel]:
                    keep = False
                elif msg.type in {"note_on", "note_off"}:
                    note = int(msg.note)
                    if note < config.keep_pitch_min or note > config.keep_pitch_max:
                        keep = False

            if keep:
                msg_copy = msg.copy(time=carry_time)
                new_track.append(msg_copy)
                carry_time = 0
            else:
                changed = True

        if new_track:
            cleaned.tracks.append(new_track)
        else:
            changed = True

    return cleaned, changed


def collect_note_events(midi: mido.MidiFile) -> List[NoteEvent]:
    """将 MIDI 解析为音符事件列表（note_on/note_off 配对）。"""
    events: List[NoteEvent] = []
    # 活跃音符栈，键为 (track_idx, channel, pitch)
    active: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)

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
                    start_tick = active[key].pop()
                    events.append(
                        NoteEvent(
                            start_tick=start_tick,
                            end_tick=abs_tick,
                            pitch=int(msg.note),
                            track_idx=track_idx,
                            channel=channel,
                        )
                    )
    return events


def compute_max_polyphony(note_events: Sequence[NoteEvent]) -> int:
    """计算最大同时发声数（polyphony）。"""
    points: List[Tuple[int, int]] = []
    for n in note_events:
        if n.duration_tick <= 0:
            continue
        points.append((n.start_tick, 1))
        points.append((n.end_tick, -1))

    points.sort(key=lambda x: (x[0], 0 if x[1] < 0 else 1))
    current = 0
    maximum = 0
    for _, delta in points:
        current += delta
        maximum = max(maximum, current)
    return maximum


def compute_total_bars(note_events: Sequence[NoteEvent], bar_ticks: int) -> int:
    """根据音符结束位置估计总小节数。"""
    if not note_events:
        return 0
    max_tick = max(n.end_tick for n in note_events)
    return int(math.ceil(max_tick / max(1, bar_ticks)))


def truncate_midi_by_max_bars(
    midi: mido.MidiFile, max_total_bars: int
) -> Tuple[mido.MidiFile, bool]:
    """当小节数超过上限时截断到指定小节，并补齐悬挂音符 note_off。"""
    if max_total_bars <= 0:
        return midi, False

    bar_ticks = get_bar_ticks(midi)
    note_events = collect_note_events(midi)
    total_bars = compute_total_bars(note_events, bar_ticks)
    if total_bars <= max_total_bars:
        return midi, False

    cutoff_tick = bar_ticks * max_total_bars
    truncated = mido.MidiFile(
        type=midi.type,
        ticks_per_beat=midi.ticks_per_beat,
        charset=getattr(midi, "charset", "latin1"),
    )
    changed = False

    for track in midi.tracks:
        new_track = mido.MidiTrack()
        abs_tick = 0
        new_abs_tick = 0
        active_notes: Dict[Tuple[int, int], int] = defaultdict(int)
        truncated_here = False

        for msg in track:
            abs_tick += int(msg.time)
            if abs_tick > cutoff_tick:
                truncated_here = True
                break

            if not msg.is_meta and hasattr(msg, "channel"):
                channel = int(msg.channel)
                if msg.type == "note_on" and int(msg.velocity) > 0:
                    active_notes[(channel, int(msg.note))] += 1
                elif msg.type in {"note_off", "note_on"} and (
                    msg.type == "note_off" or int(msg.velocity) == 0
                ):
                    key = (channel, int(msg.note))
                    if active_notes[key] > 0:
                        active_notes[key] -= 1

            new_track.append(msg.copy(time=abs_tick - new_abs_tick))
            new_abs_tick = abs_tick

        if truncated_here:
            changed = True
            to_close: List[Tuple[int, int]] = []
            for key, cnt in active_notes.items():
                for _ in range(max(0, cnt)):
                    to_close.append(key)
            to_close.sort(key=lambda x: (x[0], x[1]))

            first = True
            for channel, note in to_close:
                delta = cutoff_tick - new_abs_tick if first else 0
                new_track.append(
                    mido.Message(
                        "note_off",
                        channel=channel,
                        note=note,
                        velocity=0,
                        time=max(0, delta),
                    )
                )
                if first:
                    new_abs_tick = cutoff_tick
                    first = False

            if not any(msg.type == "end_of_track" for msg in new_track):
                new_track.append(mido.MetaMessage("end_of_track", time=0))

        truncated.tracks.append(new_track)

    return truncated, changed


def build_content_hash(
    note_events: Sequence[NoteEvent],
    ticks_per_beat: int,
    dedup_tick_resolution: int,
) -> str:
    """构建用于去重的内容哈希（基于量化后的音符事件）。"""
    if not note_events:
        return "EMPTY"

    target_tpb = 480
    scaled: List[Tuple[int, int, int]] = []
    for n in note_events:
        start = round(n.start_tick * target_tpb / max(1, ticks_per_beat))
        dur = round(max(0, n.duration_tick) * target_tpb / max(1, ticks_per_beat))
        q = max(1, dedup_tick_resolution)
        start_q = int(round(start / q) * q)
        dur_q = int(round(dur / q) * q)
        scaled.append((start_q, dur_q, int(n.pitch)))

    scaled.sort()
    digest = hashlib.sha1(
        json.dumps(scaled, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest


def get_melodic_track_count(
    note_events: Sequence[NoteEvent],
    min_notes_per_track: int,
    min_unique_pitch_per_track: int,
) -> int:
    """统计满足“旋律轨”条件的轨道数量。"""
    track_notes: Dict[int, List[NoteEvent]] = defaultdict(list)
    for n in note_events:
        track_notes[n.track_idx].append(n)

    melodic_count = 0
    for notes in track_notes.values():
        if len(notes) < min_notes_per_track:
            continue
        unique_pitch = len({n.pitch for n in notes})
        if unique_pitch >= min_unique_pitch_per_track:
            melodic_count += 1
    return melodic_count


def evaluate_metrics(cleaned_midi: mido.MidiFile, config: CleaningConfig) -> FileMetrics:
    """计算单文件清洗后质量指标。"""
    note_events = collect_note_events(cleaned_midi)
    bar_ticks = get_bar_ticks(cleaned_midi)
    note_count = len(note_events)
    unique_pitch = len({n.pitch for n in note_events})
    zero_duration = sum(1 for n in note_events if n.duration_tick <= 0)
    max_simultaneous_notes = compute_max_polyphony(note_events)
    melodic_tracks = get_melodic_track_count(
        note_events=note_events,
        min_notes_per_track=config.min_notes_per_melody_track,
        min_unique_pitch_per_track=config.min_unique_pitch_per_melody_track,
    )
    total_bars = compute_total_bars(note_events, bar_ticks)
    bpm_median = extract_median_bpm(cleaned_midi)
    duration_sec = float(cleaned_midi.length)
    content_hash = build_content_hash(
        note_events=note_events,
        ticks_per_beat=cleaned_midi.ticks_per_beat,
        dedup_tick_resolution=config.dedup_tick_resolution,
    )
    return FileMetrics(
        note_count=note_count,
        unique_pitch_count=unique_pitch,
        zero_duration_note_count=zero_duration,
        max_simultaneous_notes=max_simultaneous_notes,
        melodic_track_count=melodic_tracks,
        total_bars=total_bars,
        duration_sec=duration_sec,
        bpm_median=bpm_median,
        content_hash=content_hash,
    )


def validate_file(metrics: FileMetrics, config: CleaningConfig) -> Optional[str]:
    """按配置阈值校验文件，返回丢弃原因；通过则返回 None。"""
    if metrics.note_count < config.min_note_count:
        return "note_count_below_min"
    if metrics.unique_pitch_count < config.min_unique_pitch:
        return "unique_pitch_below_min"
    if config.drop_if_zero_duration_note and metrics.zero_duration_note_count > 0:
        return "contains_zero_duration_note"
    if metrics.max_simultaneous_notes > config.max_simultaneous_notes:
        return "polyphony_exceeds_limit"
    if metrics.melodic_track_count < config.min_melody_tracks:
        return "no_melody_track"
    if metrics.total_bars < config.min_total_bars:
        return "bars_too_short"
    if metrics.duration_sec < config.min_duration_sec:
        return "duration_too_short"
    if metrics.bpm_median is not None and (
        metrics.bpm_median < config.bpm_min or metrics.bpm_median > config.bpm_max
    ):
        return "bpm_out_of_range"
    return None


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    """将字典以 UTF-8 JSON 格式写入磁盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    config: CleaningConfig,
    limit: Optional[int],
) -> None:
    """执行完整清洗流程并生成统计报告。"""
    files = discover_files(
        input_dir=input_dir,
        extensions=config.extensions,
        recursive=config.recursive,
    )
    files.sort()
    if limit is not None:
        files = files[:limit]

    ensure_clean_directory(output_dir)
    if report_path.parent.resolve() != output_dir.resolve():
        remove_file_if_exists(report_path)

    dedup_seen: Dict[str, str] = {}
    dropped_by_reason: Counter[str] = Counter()
    kept = 0
    changed_files = 0
    errors = 0
    samples: List[Dict[str, object]] = []

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            midi = mido.MidiFile(src, clip=True)
            cleaned_midi, changed = filter_midi(midi, config)
            cleaned_midi, truncated = truncate_midi_by_max_bars(
                cleaned_midi, config.max_total_bars
            )
            changed = changed or truncated
            if changed:
                changed_files += 1

            metrics = evaluate_metrics(cleaned_midi, config)
            reason = validate_file(metrics, config)
            if reason is None and config.deduplicate:
                if metrics.content_hash in dedup_seen:
                    reason = "duplicate_sequence"
                else:
                    dedup_seen[metrics.content_hash] = str(rel)

            if reason is not None:
                dropped_by_reason[reason] += 1
            else:
                cleaned_midi.save(dst)
                kept += 1

            if len(samples) < 200:
                samples.append(
                    {
                        "file": str(rel),
                        "kept": reason is None,
                        "drop_reason": reason,
                        "note_count": metrics.note_count,
                        "unique_pitch_count": metrics.unique_pitch_count,
                        "max_simultaneous_notes": metrics.max_simultaneous_notes,
                        "melodic_track_count": metrics.melodic_track_count,
                        "total_bars": metrics.total_bars,
                        "duration_sec": round(metrics.duration_sec, 3),
                        "bpm_median": (
                            None
                            if metrics.bpm_median is None
                            else round(metrics.bpm_median, 3)
                        ),
                    }
                )
        except Exception as exc:  # pylint: disable=broad-except
            errors += 1
            dropped_by_reason["parse_or_io_error"] += 1
            if len(samples) < 200:
                samples.append(
                    {
                        "file": str(rel),
                        "kept": False,
                        "drop_reason": "parse_or_io_error",
                        "error": str(exc),
                    }
                )

        if idx % 500 == 0:
            print(f"[clean] processed={idx}/{len(files)} kept={kept} dropped={idx-kept}")

    total = len(files)
    dropped = total - kept
    keep_rate = 0.0 if total == 0 else kept / total
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_files": total,
        "kept_files": kept,
        "dropped_files": dropped,
        "keep_rate": keep_rate,
        "changed_files": changed_files,
        "error_files": errors,
        "drop_reason_distribution": dict(dropped_by_reason.most_common()),
        "config": asdict(config),
        "sample_records": samples,
    }
    dump_json(report_path, report)
    print(f"[clean] done total={total} kept={kept} dropped={dropped}")
    print(f"[clean] report -> {report_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="清洗原始 MIDI 数据集。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="原始 MIDI 文件目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clean"),
        help="清洗后 MIDI 输出目录。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/cleaning.yaml"),
        help="清洗阈值的 YAML 配置路径。",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/data/clean_report.json"),
        help="清洗报告 JSON 输出路径。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选：限制处理文件数量（用于快速冒烟测试）。",
    )
    return parser.parse_args()


def main() -> None:
    """程序入口：加载参数与配置并启动清洗。"""
    args = parse_args()
    config = load_config(args.config)
    process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        config=config,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
