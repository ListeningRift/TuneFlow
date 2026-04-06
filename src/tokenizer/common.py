"""Tokenizer 公共方法（通用且可复用）。"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：mido。请先执行 `python -m pip install mido`。"
    ) from exc


@dataclass
class NoteEvent:
    """内部音符表示。"""

    start_tick: int
    end_tick: int
    pitch: int
    velocity: int

    @property
    def duration_tick(self) -> int:
        """返回音符 tick 时值。"""
        return max(0, self.end_tick - self.start_tick)


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    """读取 JSONL 文件。"""
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_tok_lines(path: Path, lines: Iterable[str]) -> None:
    """写出 `.tok` 文件（一行一个样本）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for line in lines:
            file.write(line + "\n")


def summarize_lengths(lengths: Sequence[int]) -> Dict[str, float]:
    """汇总长度统计。"""
    if not lengths:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = sorted(lengths)
    return {
        "count": float(len(arr)),
        "mean": float(statistics.mean(arr)),
        "p50": float(arr[int(0.50 * (len(arr) - 1))]),
        "p90": float(arr[int(0.90 * (len(arr) - 1))]),
        "p99": float(arr[int(0.99 * (len(arr) - 1))]),
    }


def nearest_value(value: int, candidates: Sequence[int]) -> int:
    """返回离给定值最近的候选值（平局取更小值）。"""
    best = candidates[0]
    best_dist = abs(value - best)
    for candidate in candidates[1:]:
        dist = abs(value - candidate)
        if dist < best_dist or (dist == best_dist and candidate < best):
            best = candidate
            best_dist = dist
    return best


def get_bar_ticks(midi: mido.MidiFile) -> int:
    """计算每小节 tick（取第一个拍号；默认 4/4）。"""
    numerator = 4
    denominator = 4
    for track in midi.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += int(msg.time)
            if msg.type == "time_signature":
                numerator = int(msg.numerator)
                denominator = int(msg.denominator)
                beats_per_bar = numerator * (4.0 / max(1, denominator))
                return max(1, int(round(midi.ticks_per_beat * beats_per_bar)))
    return max(1, int(round(midi.ticks_per_beat * 4.0)))


def collect_note_events(midi: mido.MidiFile) -> List[NoteEvent]:
    """解析 MIDI，提取 note_on/note_off 配对后的音符事件。"""
    active: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = defaultdict(list)
    notes: List[NoteEvent] = []

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
                active[key].append((abs_tick, int(msg.velocity)))
            elif msg.type in {"note_off", "note_on"} and (
                msg.type == "note_off" or int(msg.velocity) == 0
            ):
                key = (track_idx, channel, int(msg.note))
                if active[key]:
                    start_tick, velocity = active[key].pop()
                    notes.append(
                        NoteEvent(
                            start_tick=start_tick,
                            end_tick=abs_tick,
                            pitch=int(msg.note),
                            velocity=max(1, velocity),
                        )
                    )
    return notes


def collect_tempo_changes(midi: mido.MidiFile) -> List[Tuple[int, float]]:
    """提取 tempo 变化（tick, bpm）。若无 tempo 事件则返回默认 120 BPM。"""
    events: List[Tuple[int, float]] = []
    for track in midi.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += int(msg.time)
            if msg.type == "set_tempo":
                events.append((abs_tick, float(mido.tempo2bpm(msg.tempo))))
    if not events:
        return [(0, 120.0)]
    events.sort(key=lambda item: item[0])
    merged: List[Tuple[int, float]] = []
    for tick, bpm in events:
        if merged and merged[-1][0] == tick:
            merged[-1] = (tick, bpm)
        else:
            merged.append((tick, bpm))
    return merged

