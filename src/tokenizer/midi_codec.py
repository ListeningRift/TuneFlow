"""Reusable TuneFlow MIDI/token codec helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    import mido
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：mido。请先在你当前环境中执行 `uv sync --active`。"
    ) from exc

from ..utils.config_io import load_yaml_mapping
from .common import (
    NoteEvent,
    collect_note_events,
    collect_tempo_changes,
    get_bar_ticks,
    nearest_value,
)
from .velocity import VelocityConfig, bin_to_velocity, velocity_to_bin

_DEFAULT_TICKS_PER_BEAT = 480
_DEFAULT_BPM = 120.0
_MIDI_CHANNEL = 0
_FORBIDDEN_SEQUENCE_TOKENS = {"FIM_HOLE", "FIM_MID"}
_FORBIDDEN_SEQUENCE_PREFIXES = ("TASK_",)
_INST_PROGRAM_MAP = {
    "PIANO": 0,
    "GUITAR": 24,
    "BASS": 32,
    "STRINGS": 48,
    "LEAD": 80,
    "PAD": 88,
}


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
    train_transpose_offsets: List[int] = field(default_factory=list)
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
        velocity_cfg = VelocityConfig.from_mapping(data)
        safe.update(
            {
                "velocity_bins": velocity_cfg.num_bins,
                "velocity_mu": velocity_cfg.mu,
                "velocity_center": velocity_cfg.center,
                "velocity_radius": velocity_cfg.half_range,
            }
        )
        cfg = cls(**safe)
        if cfg.positions_per_bar <= 0:
            raise ValueError("positions_per_bar 必须为正数")
        if cfg.tempo_step <= 0:
            raise ValueError("tempo_step 必须为正数")
        if cfg.pitch_min > cfg.pitch_max:
            raise ValueError("pitch_min 不能大于 pitch_max")
        if cfg.default_inst not in cfg.inst_classes:
            raise ValueError("default_inst 必须在 inst_classes 内")
        cfg.train_transpose_offsets = [
            int(offset)
            for offset in dict.fromkeys(cfg.train_transpose_offsets)
            if int(offset) != 0
        ]
        return cfg

    def velocity_config(self) -> VelocityConfig:
        """返回当前 tokenizer 配置对应的力度映射配置。"""
        cfg = VelocityConfig(
            num_bins=self.velocity_bins,
            mu=self.velocity_mu,
            center=self.velocity_center,
            half_range=self.velocity_radius,
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class _DecodedNote:
    start_tick: int
    end_tick: int
    pitch: int
    velocity: int
    inst: str


def load_config(path: Path) -> TokenizerConfig:
    """读取 tokenizer 配置。"""
    raw = load_yaml_mapping(path, "tokenizer 配置")
    return TokenizerConfig.from_mapping(raw)


def velocity_to_bucket(velocity: int, velocity_config: VelocityConfig) -> int:
    """将 MIDI velocity 映射到 `VEL_0..N-1`。"""
    return velocity_to_bin(velocity, velocity_config)


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
    for token in vocab:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return {token: idx for idx, token in enumerate(deduped)}


def _collect_tokenizer_notes(midi: mido.MidiFile, config: TokenizerConfig) -> List[NoteEvent]:
    """收集落在当前音高范围内、可用于分词的音符事件。"""
    return [
        note
        for note in collect_note_events(midi)
        if config.pitch_min <= note.pitch <= config.pitch_max and note.duration_tick > 0
    ]


def _transpose_notes(
    notes: Sequence[NoteEvent], semitone_offset: int, config: TokenizerConfig
) -> List[NoteEvent] | None:
    """对音符事件做移调；若超出音域范围则返回 `None`。"""
    shifted_notes: List[NoteEvent] = []
    for note in notes:
        shifted_pitch = note.pitch + semitone_offset
        if shifted_pitch < config.pitch_min or shifted_pitch > config.pitch_max:
            return None
        shifted_notes.append(
            NoteEvent(
                start_tick=note.start_tick,
                end_tick=note.end_tick,
                pitch=shifted_pitch,
                velocity=note.velocity,
            )
        )
    return shifted_notes


def _tokenize_note_events(
    notes: Sequence[NoteEvent],
    tempo_events: Sequence[Tuple[int, float]],
    bar_ticks: int,
    config: TokenizerConfig,
) -> List[str]:
    """将音符事件与节奏信息编码成 token 序列。"""
    if not notes:
        return ["BOS", "EOS"]

    pos_ticks = bar_ticks / config.positions_per_bar
    velocity_config = config.velocity_config()

    per_bar: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    max_bar_idx = 0
    for note in notes:
        bar_idx = note.start_tick // bar_ticks
        max_bar_idx = max(max_bar_idx, bar_idx)
        in_bar_tick = note.start_tick % bar_ticks
        pos = int(round(in_bar_tick / max(1e-9, pos_ticks)))
        pos = min(config.positions_per_bar - 1, max(0, pos))

        dur_pos = int(round(note.duration_tick / max(1e-9, pos_ticks)))
        dur_pos = max(1, dur_pos)
        dur_bin = nearest_value(dur_pos, config.duration_bins)
        vel_bin = velocity_to_bucket(note.velocity, velocity_config)
        per_bar[bar_idx].append((pos, note.pitch, dur_bin, vel_bin))

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
        events.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for pos, pitch, dur_bin, vel_bin in events:
            tokens.append(f"POS_{pos}")
            tokens.append(f"INST_{config.default_inst}")
            tokens.append(f"PITCH_{pitch}")
            tokens.append(f"DUR_{dur_bin}")
            tokens.append(f"VEL_{vel_bin}")

    tokens.append("EOS")
    return tokens


def tokenize_midi(mido_file: mido.MidiFile, config: TokenizerConfig) -> List[str]:
    """将单个 MIDI 编码成 token 序列。"""
    notes = _collect_tokenizer_notes(mido_file, config)
    bar_ticks = get_bar_ticks(mido_file)
    tempo_events = collect_tempo_changes(mido_file)
    return _tokenize_note_events(notes, tempo_events, bar_ticks, config)


def validate_token_order(tokens: Sequence[str], vocab: Mapping[str, int]) -> Tuple[bool, int]:
    """校验 token 顺序是否合法，并返回 `(is_valid, oov_count)`。"""
    oov = sum(1 for token in tokens if token not in vocab)
    if not tokens or tokens[0] != "BOS":
        return False, oov
    if tokens[-1] != "EOS":
        return False, oov

    idx = 1
    if idx < len(tokens) - 1 and tokens[idx].startswith("TEMPO_"):
        idx += 1

    while idx < len(tokens) - 1:
        if tokens[idx] != "BAR":
            return False, oov
        idx += 1
        if idx < len(tokens) - 1 and tokens[idx].startswith("TEMPO_"):
            idx += 1
        while idx < len(tokens) - 1 and tokens[idx].startswith("POS_"):
            if idx + 4 >= len(tokens):
                return False, oov
            if not tokens[idx + 1].startswith("INST_"):
                return False, oov
            if not tokens[idx + 2].startswith("PITCH_"):
                return False, oov
            if not tokens[idx + 3].startswith("DUR_"):
                return False, oov
            if not tokens[idx + 4].startswith("VEL_"):
                return False, oov
            idx += 5
        if idx < len(tokens) - 1 and tokens[idx] != "BAR":
            return False, oov

    return True, oov


def _parse_token_int(token: str, prefix: str) -> int:
    if not token.startswith(prefix):
        raise ValueError(f"expected token with prefix `{prefix}`, got `{token}`")
    try:
        return int(token[len(prefix) :])
    except ValueError as exc:
        raise ValueError(f"invalid token value: `{token}`") from exc


def _tempo_from_token(token: str, config: TokenizerConfig) -> float:
    bpm = _parse_token_int(token, "TEMPO_")
    if bpm < config.tempo_min or bpm > config.tempo_max:
        raise ValueError(f"tempo token out of range: `{token}`")
    return float(bpm)


def _instrument_from_token(token: str) -> str:
    if not token.startswith("INST_"):
        raise ValueError(f"expected INST token, got `{token}`")
    inst = token[len("INST_") :].strip().upper()
    if not inst:
        raise ValueError(f"invalid instrument token: `{token}`")
    return inst


def _validate_complete_sequence(tokens: Sequence[str], config: TokenizerConfig) -> List[str]:
    normalized = [str(token).strip() for token in tokens]
    if not normalized:
        raise ValueError("token sequence is empty")

    forbidden_tokens = [token for token in normalized if token in _FORBIDDEN_SEQUENCE_TOKENS]
    if forbidden_tokens:
        raise ValueError(
            f"token sequence contains unsupported structural tokens: {', '.join(sorted(set(forbidden_tokens)))}"
        )

    forbidden_prefixes = [
        token
        for token in normalized
        if any(token.startswith(prefix) for prefix in _FORBIDDEN_SEQUENCE_PREFIXES)
    ]
    if forbidden_prefixes:
        raise ValueError(
            "token sequence contains unsupported task tokens and is not a complete musical sequence"
        )

    vocab = build_vocab(config)
    valid, oov_count = validate_token_order(normalized, vocab)
    if oov_count > 0:
        unknown_tokens = sorted({token for token in normalized if token not in vocab})
        preview = ", ".join(unknown_tokens[:8])
        raise ValueError(f"token sequence contains {oov_count} OOV token(s): {preview}")
    if not valid:
        raise ValueError("token sequence is not a valid complete TuneFlow sequence")
    return normalized


def _merge_tempo_events(tempo_events: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    merged: List[Tuple[int, float]] = []
    for tick, bpm in sorted(tempo_events, key=lambda item: item[0]):
        tick_int = max(0, int(tick))
        bpm_float = float(bpm)
        if merged and merged[-1][0] == tick_int:
            merged[-1] = (tick_int, bpm_float)
        else:
            merged.append((tick_int, bpm_float))
    return merged


def _append_absolute_events(
    track: mido.MidiTrack,
    events: Sequence[tuple[int, int, int, mido.Message | mido.MetaMessage]],
) -> None:
    last_tick = 0
    for tick, order, tie_breaker, message in sorted(events, key=lambda item: (item[0], item[1], item[2])):
        del order, tie_breaker
        event_tick = max(0, int(tick))
        delta = event_tick - last_tick
        track.append(message.copy(time=delta))
        last_tick = event_tick
    track.append(mido.MetaMessage("end_of_track", time=0))


def _program_for_inst(inst: str) -> int:
    return int(_INST_PROGRAM_MAP.get(inst.upper(), _INST_PROGRAM_MAP["PIANO"]))


def tokens_to_midi(
    tokens: Sequence[str],
    config: TokenizerConfig,
    *,
    ticks_per_beat: int = _DEFAULT_TICKS_PER_BEAT,
) -> mido.MidiFile:
    """将完整 TuneFlow token 序列反编译为 MIDI。"""
    if ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be > 0")

    normalized = _validate_complete_sequence(tokens, config)
    bar_ticks = int(round(ticks_per_beat * 4.0))
    pos_ticks = bar_ticks / config.positions_per_bar
    velocity_config = config.velocity_config()

    tempo_events: List[Tuple[int, float]] = []
    notes: List[_DecodedNote] = []

    idx = 1
    initial_bpm = _DEFAULT_BPM
    if idx < len(normalized) - 1 and normalized[idx].startswith("TEMPO_"):
        initial_bpm = _tempo_from_token(normalized[idx], config)
        idx += 1
    tempo_events.append((0, initial_bpm))

    bar_index = -1
    while idx < len(normalized) - 1:
        token = normalized[idx]
        if token != "BAR":
            raise ValueError(f"expected `BAR`, got `{token}`")
        bar_index += 1
        bar_start_tick = bar_index * bar_ticks
        idx += 1

        if idx < len(normalized) - 1 and normalized[idx].startswith("TEMPO_"):
            tempo_events.append((bar_start_tick, _tempo_from_token(normalized[idx], config)))
            idx += 1

        while idx < len(normalized) - 1 and normalized[idx].startswith("POS_"):
            if idx + 4 >= len(normalized):
                raise ValueError("incomplete note event at end of token sequence")
            pos = _parse_token_int(normalized[idx], "POS_")
            inst = _instrument_from_token(normalized[idx + 1])
            pitch = _parse_token_int(normalized[idx + 2], "PITCH_")
            dur = _parse_token_int(normalized[idx + 3], "DUR_")
            vel_bin = _parse_token_int(normalized[idx + 4], "VEL_")

            if pos < 0 or pos >= config.positions_per_bar:
                raise ValueError(f"position token out of range: `{normalized[idx]}`")
            if pitch < config.pitch_min or pitch > config.pitch_max:
                raise ValueError(f"pitch token out of range: `{normalized[idx + 2]}`")
            if dur not in config.duration_bins:
                raise ValueError(f"duration token out of range: `{normalized[idx + 3]}`")
            if vel_bin < 0 or vel_bin >= config.velocity_bins:
                raise ValueError(f"velocity token out of range: `{normalized[idx + 4]}`")

            start_tick = bar_start_tick + int(round(pos * pos_ticks))
            duration_tick = max(1, int(round(dur * pos_ticks)))
            notes.append(
                _DecodedNote(
                    start_tick=start_tick,
                    end_tick=start_tick + duration_tick,
                    pitch=pitch,
                    velocity=bin_to_velocity(vel_bin, velocity_config),
                    inst=inst,
                )
            )
            idx += 5

    midi = mido.MidiFile(type=1, ticks_per_beat=int(ticks_per_beat))
    meta_track = mido.MidiTrack()
    note_track = mido.MidiTrack()
    midi.tracks.append(meta_track)
    midi.tracks.append(note_track)

    meta_events: List[tuple[int, int, int, mido.Message | mido.MetaMessage]] = [
        (
            0,
            0,
            0,
            mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                clocks_per_click=24,
                notated_32nd_notes_per_beat=8,
                time=0,
            ),
        )
    ]
    for tempo_tick, bpm in _merge_tempo_events(tempo_events):
        meta_events.append(
            (
                tempo_tick,
                1,
                0,
                mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=0),
            )
        )
    _append_absolute_events(meta_track, meta_events)

    note_events: List[tuple[int, int, int, mido.Message | mido.MetaMessage]] = []
    sorted_notes = sorted(notes, key=lambda note: (note.start_tick, note.end_tick, note.pitch, note.velocity, note.inst))
    if sorted_notes:
        current_program = _program_for_inst(sorted_notes[0].inst)
        note_events.append(
            (
                0,
                1,
                current_program,
                mido.Message(
                    "program_change",
                    channel=_MIDI_CHANNEL,
                    program=current_program,
                    time=0,
                ),
            )
        )
        for note in sorted_notes:
            note_program = _program_for_inst(note.inst)
            if note_program != current_program:
                note_events.append(
                    (
                        note.start_tick,
                        1,
                        note_program,
                        mido.Message(
                            "program_change",
                            channel=_MIDI_CHANNEL,
                            program=note_program,
                            time=0,
                        ),
                    )
                )
                current_program = note_program
            note_events.append(
                (
                    note.end_tick,
                    0,
                    note.pitch,
                    mido.Message(
                        "note_off",
                        channel=_MIDI_CHANNEL,
                        note=note.pitch,
                        velocity=0,
                        time=0,
                    ),
                )
            )
            note_events.append(
                (
                    note.start_tick,
                    2,
                    note.pitch,
                    mido.Message(
                        "note_on",
                        channel=_MIDI_CHANNEL,
                        note=note.pitch,
                        velocity=note.velocity,
                        time=0,
                    ),
                )
            )
    _append_absolute_events(note_track, note_events)
    return midi


__all__ = [
    "TokenizerConfig",
    "build_vocab",
    "load_config",
    "tokenize_midi",
    "tokens_to_midi",
    "validate_token_order",
]
