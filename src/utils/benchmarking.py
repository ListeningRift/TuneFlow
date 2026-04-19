"""Benchmark manifest 构建与面向音乐的 token 诊断工具。"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .config_io import load_yaml_mapping
from .eval_windows import sample_bar_aligned_subsequence


def _parse_prefixed_int(token: str, prefix: str) -> int | None:
    if not token.startswith(prefix):
        return None
    try:
        return int(token[len(prefix) :])
    except ValueError:
        return None


def _quartile_thresholds(values: Sequence[float]) -> tuple[float, float, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return (0.0, 0.0, 0.0)

    def _pick(ratio: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
        return ordered[index]

    return (_pick(0.25), _pick(0.50), _pick(0.75))


def _quartile_bucket(value: float, thresholds: tuple[float, float, float]) -> int:
    if value <= thresholds[0]:
        return 0
    if value <= thresholds[1]:
        return 1
    if value <= thresholds[2]:
        return 2
    return 3


def _bucket_label(note_bucket: int, duration_bucket: int) -> str:
    return f"note_q{note_bucket}_dur_q{duration_bucket}"


def _collect_continuation_split_positions(core: list[str]) -> list[int]:
    positions: set[int] = set()
    idx = 0
    if idx < len(core) and core[idx].startswith("TEMPO_"):
        idx += 1
        positions.add(idx)

    while idx < len(core):
        if core[idx] != "BAR":
            return []
        positions.add(idx)
        idx += 1
        if idx < len(core) and core[idx].startswith("TEMPO_"):
            idx += 1

        while idx < len(core) and core[idx].startswith("POS_"):
            positions.add(idx)
            if idx + 4 >= len(core):
                return []
            if not core[idx + 1].startswith("INST_"):
                return []
            if not core[idx + 2].startswith("PITCH_"):
                return []
            if not core[idx + 3].startswith("DUR_"):
                return []
            if not core[idx + 4].startswith("VEL_"):
                return []
            idx += 5

    positions.discard(0)
    positions.discard(len(core))
    return sorted(positions)


def _collect_infill_maskable_units(core: list[str]) -> list[tuple[int, int, str, int]]:
    if not core:
        return []

    units: list[tuple[int, int, str, int]] = []
    idx = 0
    group_id = 0

    if idx < len(core) and core[idx].startswith("TEMPO_"):
        idx += 1
        group_id += 1

    while idx < len(core):
        if core[idx] != "BAR":
            return []
        units.append((idx, idx + 1, "bar", group_id))
        idx += 1

        if idx < len(core) and core[idx].startswith("TEMPO_"):
            idx += 1
            group_id += 1

        while idx < len(core) and core[idx].startswith("POS_"):
            if idx + 4 >= len(core):
                return []
            if not core[idx + 1].startswith("INST_"):
                return []
            if not core[idx + 2].startswith("PITCH_"):
                return []
            if not core[idx + 3].startswith("DUR_"):
                return []
            if not core[idx + 4].startswith("VEL_"):
                return []
            units.append((idx, idx + 5, "event", group_id))
            idx += 5
    return units


def _choose_infill_hole_bounds(
    core: list[str],
    *,
    target_hole_tokens: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    units = _collect_infill_maskable_units(core)
    if len(units) < 2:
        return None

    max_hole_tokens = max(1, min(96, len(core) - 2))
    min_hole_tokens = min(max_hole_tokens, max(1, min(target_hole_tokens, 8)))
    candidate_bounds: list[tuple[int, int, int]] = []

    for start_idx, (start_token, _, _, group_id) in enumerate(units):
        if start_token <= 0:
            continue
        end_token = start_token
        for end_idx in range(start_idx, len(units)):
            unit_start, unit_end, _, end_group_id = units[end_idx]
            if end_group_id != group_id:
                break
            if end_idx > start_idx and unit_start != end_token:
                break
            end_token = unit_end
            if end_token >= len(core):
                continue
            span = end_token - start_token
            if span < min_hole_tokens:
                continue
            if span > max_hole_tokens:
                break
            candidate_bounds.append((abs(span - target_hole_tokens), start_token, end_token))

    if not candidate_bounds:
        return None

    candidate_bounds.sort(key=lambda item: (item[0], item[1], item[2]))
    best_gap = candidate_bounds[0][0]
    near_best = [(start_cut, end_cut) for gap, start_cut, end_cut in candidate_bounds if gap <= best_gap + 4]
    return rng.choice(near_best)


def load_benchmark_config(path: Path) -> dict[str, Any]:
    """加载 benchmark YAML 配置，并补齐简单默认值。"""
    payload = load_yaml_mapping(path, "benchmark config")
    payload.setdefault("tier", path.stem.replace("benchmark_", ""))
    payload.setdefault("seed", 42)
    payload.setdefault("sample_count", None)
    payload.setdefault("per_bucket_limit", None)
    payload.setdefault("min_prefix_tokens", 32)
    payload.setdefault("continuation_prefix_ratio_min", 0.35)
    payload.setdefault("continuation_prefix_ratio_max", 0.70)
    payload.setdefault("infilling_hole_ratio_min", 0.10)
    payload.setdefault("infilling_hole_ratio_max", 0.25)
    payload.setdefault("sample_export_case_count", 12)
    payload.setdefault("sample_export_top_k", 3)
    return payload


def load_eval_rows(eval_jsonl_path: Path, eval_tok_path: Path) -> list[dict[str, Any]]:
    """加载 benchmark 元数据行与对齐后的 token 序列。"""
    with eval_jsonl_path.open("r", encoding="utf-8") as file:
        meta_rows = [json.loads(line) for line in file if line.strip()]
    token_rows = []
    with eval_tok_path.open("r", encoding="utf-8") as file:
        for line in file:
            tokens = [token for token in line.strip().split(" ") if token]
            if tokens:
                token_rows.append(tokens)

    if len(meta_rows) != len(token_rows):
        raise ValueError(
            "fixed_eval.jsonl and eval.tok row count mismatch: "
            f"{len(meta_rows)} != {len(token_rows)}"
        )

    rows: list[dict[str, Any]] = []
    for index, (meta, tokens) in enumerate(zip(meta_rows, token_rows, strict=True)):
        rows.append(
            {
                "row_id": index,
                "meta": dict(meta),
                "tokens": list(tokens),
            }
        )
    return rows


def build_continuation_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    min_prefix_tokens: int,
    prefix_ratio_min: float,
    prefix_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    """构建可复现的 continuation case。"""
    if len(source_tokens) < 30 or source_tokens[0] != "BOS" or source_tokens[-1] != "EOS":
        return None

    rng = random.Random(seed)
    min_core_len = max(int(min_prefix_tokens) + 8, 24)
    max_core_len = max(min_core_len, int(max_positions) - 8)
    sequence_window = sample_bar_aligned_subsequence(
        source_tokens,
        max_core_tokens=max_core_len,
        min_core_tokens=min_core_len,
        rng=rng,
    )
    if sequence_window is None:
        return None

    core = sequence_window[1:-1]
    core_len = len(core)
    if core_len < min_core_len:
        return None

    split_positions = _collect_continuation_split_positions(core)
    if not split_positions:
        return None

    min_prefix = max(int(min_prefix_tokens), int(round(core_len * prefix_ratio_min)))
    max_prefix = min(core_len - 8, int(round(core_len * prefix_ratio_max)))
    if min_prefix > max_prefix:
        min_prefix = max(int(min_prefix_tokens), min(split_positions))
        max_prefix = max(split_positions)

    candidate_positions = [pos for pos in split_positions if min_prefix <= pos <= max_prefix]
    if not candidate_positions:
        candidate_positions = [pos for pos in split_positions if pos >= int(min_prefix_tokens)]
    if not candidate_positions:
        return None

    target_ratio = (prefix_ratio_min + prefix_ratio_max) / 2.0
    prefix_len = min(
        candidate_positions,
        key=lambda pos: (abs((pos / float(core_len)) - target_ratio), abs(pos - int(round(core_len * target_ratio)))),
    )
    prompt_tokens = ["BOS", *core[:prefix_len]]
    target_tokens = [*core[prefix_len:], "EOS"]
    return {
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "prefix_len": len(prompt_tokens),
        "target_len": len(target_tokens),
        "window_tokens": sequence_window,
        "window_len": len(sequence_window),
    }


def build_infilling_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    hole_ratio_min: float,
    hole_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    """构建可复现的 infilling case。"""
    if len(source_tokens) < 30 or source_tokens[0] != "BOS" or source_tokens[-1] != "EOS":
        return None

    rng = random.Random(seed)
    sequence_window = sample_bar_aligned_subsequence(
        source_tokens,
        max_core_tokens=max(24, int(max_positions) - 8),
        min_core_tokens=20,
        rng=rng,
    )
    if sequence_window is None:
        return None

    core = sequence_window[1:-1]
    if len(core) < 20:
        return None

    hole_ratio = rng.uniform(float(hole_ratio_min), float(hole_ratio_max))
    target_hole_len = max(8, int(round(len(core) * hole_ratio)))
    target_hole_len = min(target_hole_len, 96, len(core) - 2)
    if target_hole_len <= 0:
        return None

    hole_bounds = _choose_infill_hole_bounds(core, target_hole_tokens=target_hole_len, rng=rng)
    if hole_bounds is None:
        return None

    hole_start_core, hole_end_core = hole_bounds
    hole_start = 1 + hole_start_core
    hole_end = 1 + hole_end_core
    prefix_tokens = sequence_window[:hole_start]
    hole_tokens = sequence_window[hole_start:hole_end]
    suffix_tokens = sequence_window[hole_end:-1]
    prompt_tokens = [*prefix_tokens, "FIM_HOLE", *suffix_tokens, "FIM_MID"]
    if len(prompt_tokens) >= max_positions:
        return None

    return {
        "prompt_tokens": prompt_tokens,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "target_hole_tokens": hole_tokens,
        "window_tokens": sequence_window,
        "window_len": len(sequence_window),
        "hole_len": len(hole_tokens),
    }


def build_benchmark_manifest(
    *,
    eval_jsonl_path: Path,
    eval_tok_path: Path,
    config: dict[str, Any],
    max_positions: int,
) -> dict[str, Any]:
    """构建可复现的 benchmark manifest。"""
    rows = load_eval_rows(eval_jsonl_path, eval_tok_path)
    note_thresholds = _quartile_thresholds([float(row["meta"]["note_count"]) for row in rows])
    duration_thresholds = _quartile_thresholds([float(row["meta"]["duration_sec"]) for row in rows])

    valid_cases: list[dict[str, Any]] = []
    for row in rows:
        meta = dict(row["meta"])
        tokens = list(row["tokens"])
        note_bucket = _quartile_bucket(float(meta["note_count"]), note_thresholds)
        duration_bucket = _quartile_bucket(float(meta["duration_sec"]), duration_thresholds)
        bucket = _bucket_label(note_bucket, duration_bucket)
        continuation_case = build_continuation_case(
            tokens,
            max_positions=max_positions,
            min_prefix_tokens=int(config["min_prefix_tokens"]),
            prefix_ratio_min=float(config["continuation_prefix_ratio_min"]),
            prefix_ratio_max=float(config["continuation_prefix_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 1,
        )
        infilling_case = build_infilling_case(
            tokens,
            max_positions=max_positions,
            hole_ratio_min=float(config["infilling_hole_ratio_min"]),
            hole_ratio_max=float(config["infilling_hole_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 7,
        )
        if continuation_case is None or infilling_case is None:
            continue

        valid_cases.append(
            {
                "row_id": int(row["row_id"]),
                "tier": str(config["tier"]),
                "bucket": bucket,
                "meta": {
                    "artist": meta.get("artist"),
                    "title": meta.get("title"),
                    "family_key": meta.get("family_key"),
                    "midi_path": meta.get("midi_path"),
                    "note_count": meta.get("note_count"),
                    "duration_sec": meta.get("duration_sec"),
                    "tok_len": len(tokens),
                },
                "continuation_case": continuation_case,
                "infilling_case": infilling_case,
            }
        )

    sample_count = config.get("sample_count")
    per_bucket_limit = config.get("per_bucket_limit")
    if sample_count is None:
        chosen_cases = valid_cases
    else:
        sample_count = int(sample_count)
        per_bucket_limit = int(per_bucket_limit) if per_bucket_limit is not None else sample_count
        rng = random.Random(int(config["seed"]))
        cases_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in valid_cases:
            cases_by_bucket[str(case["bucket"])].append(case)
        for bucket_cases in cases_by_bucket.values():
            bucket_cases.sort(key=lambda item: int(item["row_id"]))
            rng.shuffle(bucket_cases)

        chosen_cases = []
        for bucket in sorted(cases_by_bucket):
            bucket_cases = cases_by_bucket[bucket]
            chosen_cases.extend(bucket_cases[: min(len(bucket_cases), per_bucket_limit)])
        chosen_cases.sort(key=lambda item: (str(item["bucket"]), int(item["row_id"])))
        chosen_cases = chosen_cases[:sample_count]

    return {
        "tier": str(config["tier"]),
        "seed": int(config["seed"]),
        "max_positions": int(max_positions),
        "eval_jsonl_path": str(eval_jsonl_path),
        "eval_tok_path": str(eval_tok_path),
        "case_count": len(chosen_cases),
        "cases": chosen_cases,
    }


def _extract_first_unit(tokens: Sequence[str]) -> tuple[str, ...] | None:
    filtered = [token for token in tokens if token not in {"BOS", "EOS", "FIM_HOLE", "FIM_MID"}]
    idx = 0
    while idx < len(filtered) and filtered[idx].startswith("TEMPO_"):
        idx += 1
    if idx >= len(filtered):
        return None
    if filtered[idx] == "BAR":
        return ("BAR",)
    if filtered[idx].startswith("POS_") and idx + 4 < len(filtered):
        unit = filtered[idx : idx + 5]
        if (
            unit[1].startswith("INST_")
            and unit[2].startswith("PITCH_")
            and unit[3].startswith("DUR_")
            and unit[4].startswith("VEL_")
        ):
            return tuple(unit)
    return None


def duration_l1_distance(
    generated_counts: dict[str, int],
    target_counts: dict[str, int],
) -> float:
    """返回归一化时值直方图之间的 L1 距离。"""
    keys = sorted(set(generated_counts) | set(target_counts))
    if not keys:
        return 0.0
    generated_total = sum(max(0, int(generated_counts.get(key, 0))) for key in keys)
    target_total = sum(max(0, int(target_counts.get(key, 0))) for key in keys)
    if generated_total <= 0 and target_total <= 0:
        return 0.0
    distance = 0.0
    for key in keys:
        generated_prob = (float(generated_counts.get(key, 0)) / float(generated_total)) if generated_total > 0 else 0.0
        target_prob = (float(target_counts.get(key, 0)) / float(target_total)) if target_total > 0 else 0.0
        distance += abs(generated_prob - target_prob)
    return distance


_MIN_PITCH_EVENTS_FOR_COLLAPSE_METRICS = 6
_PITCH_DIVERSITY_REFERENCE_UNIQUES = 12


def _pitch_collapse_metrics(pitch_values: Sequence[int]) -> dict[str, Any]:
    """汇总生成旋律片段的 pitch 塌缩风险。"""
    pitch_list = [int(value) for value in pitch_values]
    pitch_event_count = len(pitch_list)
    unique_pitch_count = len(set(pitch_list))
    if pitch_event_count <= 0:
        return {
            "pitch_event_count": 0,
            "pitch_unique_count": 0,
            "pitch_analysis_coverage": 0.0,
            "most_common_pitch_ratio": None,
            "longest_same_pitch_run_ratio": None,
            "pitch_diversity_score": None,
        }

    if pitch_event_count < _MIN_PITCH_EVENTS_FOR_COLLAPSE_METRICS:
        return {
            "pitch_event_count": pitch_event_count,
            "pitch_unique_count": unique_pitch_count,
            "pitch_analysis_coverage": (pitch_event_count / float(_MIN_PITCH_EVENTS_FOR_COLLAPSE_METRICS)),
            "most_common_pitch_ratio": None,
            "longest_same_pitch_run_ratio": None,
            "pitch_diversity_score": None,
        }

    pitch_counter = Counter(pitch_list)
    most_common_pitch_count = max(pitch_counter.values(), default=0)

    longest_same_pitch_run = 0
    current_same_pitch_run = 0
    previous_pitch: int | None = None
    for pitch_value in pitch_list:
        if previous_pitch is not None and pitch_value == previous_pitch:
            current_same_pitch_run += 1
        else:
            current_same_pitch_run = 1
            previous_pitch = pitch_value
        longest_same_pitch_run = max(longest_same_pitch_run, current_same_pitch_run)

    normalized_unique_count = min(unique_pitch_count, _PITCH_DIVERSITY_REFERENCE_UNIQUES) / float(
        _PITCH_DIVERSITY_REFERENCE_UNIQUES
    )
    entropy = 0.0
    for count in pitch_counter.values():
        probability = float(count) / float(pitch_event_count)
        entropy -= probability * math.log(probability)
    entropy_norm = (
        entropy / math.log(float(unique_pitch_count))
        if unique_pitch_count > 1
        else 0.0
    )
    pitch_diversity_score = (0.6 * entropy_norm) + (0.4 * normalized_unique_count)

    return {
        "pitch_event_count": pitch_event_count,
        "pitch_unique_count": unique_pitch_count,
        "pitch_analysis_coverage": 1.0,
        "most_common_pitch_ratio": (most_common_pitch_count / float(pitch_event_count)),
        "longest_same_pitch_run_ratio": (longest_same_pitch_run / float(pitch_event_count)),
        "pitch_diversity_score": max(0.0, min(1.0, pitch_diversity_score)),
    }


def analyze_token_sequence(tokens: Sequence[str]) -> dict[str, Any]:
    """从可能不完整的 token 序列中分析基础音乐结构。"""
    idx = 0
    values = list(tokens)
    bar_event_counts: list[int] = []
    current_bar_events: int | None = None
    current_bar_last_pos: int | None = None
    time_order_violation_count = 0
    parsed_event_count = 0
    pitch_values: list[int] = []
    duration_counts: Counter[str] = Counter()

    def ensure_bar() -> None:
        nonlocal current_bar_events, current_bar_last_pos
        if current_bar_events is None:
            current_bar_events = 0
            current_bar_last_pos = None

    def close_bar() -> None:
        nonlocal current_bar_events, current_bar_last_pos
        if current_bar_events is not None:
            bar_event_counts.append(int(current_bar_events))
        current_bar_events = None
        current_bar_last_pos = None

    while idx < len(values):
        token = str(values[idx])
        if token in {"BOS", "FIM_HOLE", "FIM_MID"}:
            idx += 1
            continue
        if token == "EOS":
            close_bar()
            break
        if token.startswith("TEMPO_"):
            idx += 1
            continue
        if token == "BAR":
            close_bar()
            current_bar_events = 0
            current_bar_last_pos = None
            idx += 1
            continue
        if token.startswith("POS_"):
            ensure_bar()
            pos_value = _parse_prefixed_int(token, "POS_")
            if current_bar_last_pos is not None and pos_value is not None and pos_value < current_bar_last_pos:
                time_order_violation_count += 1
            if pos_value is not None:
                current_bar_last_pos = pos_value
            if idx + 4 >= len(values):
                break
            inst_token, pitch_token, dur_token, vel_token = [str(item) for item in values[idx + 1 : idx + 5]]
            if not inst_token.startswith("INST_"):
                break
            if not pitch_token.startswith("PITCH_"):
                break
            if not dur_token.startswith("DUR_"):
                break
            if not vel_token.startswith("VEL_"):
                break
            pitch_value = _parse_prefixed_int(pitch_token, "PITCH_")
            if pitch_value is not None:
                pitch_values.append(pitch_value)
            duration_counts[dur_token] += 1
            parsed_event_count += 1
            current_bar_events = 0 if current_bar_events is None else (current_bar_events + 1)
            idx += 5
            continue
        break

    close_bar()
    empty_bar_count = sum(1 for count in bar_event_counts if count == 0)
    low_density_bar_count = sum(1 for count in bar_event_counts if count <= 1)
    max_empty_run = 0
    current_empty_run = 0
    for count in bar_event_counts:
        if count == 0:
            current_empty_run += 1
            max_empty_run = max(max_empty_run, current_empty_run)
        else:
            current_empty_run = 0

    pitch_span = 0
    if pitch_values:
        pitch_span = max(pitch_values) - min(pitch_values)
    pitch_metrics = _pitch_collapse_metrics(pitch_values)

    return {
        "bar_count": len(bar_event_counts),
        "event_count": parsed_event_count,
        "empty_bar_count": empty_bar_count,
        "empty_bar_rate": (empty_bar_count / len(bar_event_counts)) if bar_event_counts else 0.0,
        "low_density_bar_count": low_density_bar_count,
        "low_density_bar_rate": (low_density_bar_count / len(bar_event_counts)) if bar_event_counts else 0.0,
        "has_multi_empty_bar_run": bool(max_empty_run >= 2),
        "max_empty_bar_run_length": max_empty_run,
        "pitch_span": pitch_span,
        "duration_counts": dict(duration_counts),
        "time_order_valid": (time_order_violation_count == 0),
        "time_order_violation_count": time_order_violation_count,
        "pitch_event_count": int(pitch_metrics["pitch_event_count"]),
        "pitch_unique_count": int(pitch_metrics["pitch_unique_count"]),
        "pitch_analysis_coverage": float(pitch_metrics["pitch_analysis_coverage"]),
        "most_common_pitch_ratio": pitch_metrics["most_common_pitch_ratio"],
        "longest_same_pitch_run_ratio": pitch_metrics["longest_same_pitch_run_ratio"],
        "pitch_diversity_score": pitch_metrics["pitch_diversity_score"],
    }


def enrich_continuation_record(record: dict[str, Any], *, target_tokens: Sequence[str]) -> dict[str, Any]:
    """为 continuation decode 轨迹补充面向音乐的诊断字段。"""
    generated_analysis = analyze_token_sequence(record.get("generated_tokens", []))
    target_analysis = analyze_token_sequence([token for token in target_tokens if token != "EOS"])
    reconstructed_analysis = analyze_token_sequence(record.get("reconstructed_tokens", []))
    first_unit_match = (
        _extract_first_unit(record.get("generated_tokens", []))
        == _extract_first_unit([token for token in target_tokens if token != "EOS"])
    )
    generated_bar_delta = generated_analysis["bar_count"] - target_analysis["bar_count"]
    generated_event_delta = generated_analysis["event_count"] - target_analysis["event_count"]
    pitch_span_delta = generated_analysis["pitch_span"] - target_analysis["pitch_span"]
    enriched = dict(record)
    enriched.update(
        {
            "stop_success": bool(record.get("reached_eos")) and bool(record.get("is_structurally_valid")),
            "structural_match_without_eos": bool(record.get("append_eos_would_validate")),
            "first_unit_match": first_unit_match,
            "time_order_valid": bool(reconstructed_analysis["time_order_valid"]),
            "time_order_violation_count": int(reconstructed_analysis["time_order_violation_count"]),
            "empty_bar_rate": float(generated_analysis["empty_bar_rate"]),
            "low_density_bar_rate": float(generated_analysis["low_density_bar_rate"]),
            "has_multi_empty_bar_run": bool(generated_analysis["has_multi_empty_bar_run"]),
            "generated_bar_count": int(generated_analysis["bar_count"]),
            "generated_event_count": int(generated_analysis["event_count"]),
            "generated_pitch_span": int(generated_analysis["pitch_span"]),
            "generated_pitch_event_count": int(generated_analysis["pitch_event_count"]),
            "generated_pitch_unique_count": int(generated_analysis["pitch_unique_count"]),
            "pitch_analysis_coverage": float(generated_analysis["pitch_analysis_coverage"]),
            "most_common_pitch_ratio": generated_analysis["most_common_pitch_ratio"],
            "longest_same_pitch_run_ratio": generated_analysis["longest_same_pitch_run_ratio"],
            "pitch_diversity_score": generated_analysis["pitch_diversity_score"],
            "target_bar_count": int(target_analysis["bar_count"]),
            "target_event_count": int(target_analysis["event_count"]),
            "target_pitch_span": int(target_analysis["pitch_span"]),
            "generated_bar_delta": int(generated_bar_delta),
            "generated_event_delta": int(generated_event_delta),
            "pitch_span_delta": int(pitch_span_delta),
            "duration_bin_l1_distance": duration_l1_distance(
                generated_analysis["duration_counts"],
                target_analysis["duration_counts"],
            ),
        }
    )
    return enriched


def enrich_infilling_record(record: dict[str, Any], *, target_hole_tokens: Sequence[str]) -> dict[str, Any]:
    """为 infilling decode 轨迹补充面向音乐的诊断字段。"""
    generated_analysis = analyze_token_sequence(record.get("generated_middle_tokens", []))
    target_analysis = analyze_token_sequence(target_hole_tokens)
    reconstructed_analysis = analyze_token_sequence(record.get("reconstructed_tokens", []))
    enriched = dict(record)
    enriched.update(
        {
            "time_order_valid": bool(reconstructed_analysis["time_order_valid"]),
            "time_order_violation_count": int(reconstructed_analysis["time_order_violation_count"]),
            "generated_bar_count": int(generated_analysis["bar_count"]),
            "generated_event_count": int(generated_analysis["event_count"]),
            "generated_pitch_span": int(generated_analysis["pitch_span"]),
            "generated_pitch_event_count": int(generated_analysis["pitch_event_count"]),
            "generated_pitch_unique_count": int(generated_analysis["pitch_unique_count"]),
            "pitch_analysis_coverage": float(generated_analysis["pitch_analysis_coverage"]),
            "most_common_pitch_ratio": generated_analysis["most_common_pitch_ratio"],
            "longest_same_pitch_run_ratio": generated_analysis["longest_same_pitch_run_ratio"],
            "pitch_diversity_score": generated_analysis["pitch_diversity_score"],
            "target_bar_count": int(target_analysis["bar_count"]),
            "target_event_count": int(target_analysis["event_count"]),
            "target_pitch_span": int(target_analysis["pitch_span"]),
            "generated_bar_delta": int(generated_analysis["bar_count"] - target_analysis["bar_count"]),
            "generated_event_delta": int(generated_analysis["event_count"] - target_analysis["event_count"]),
            "pitch_span_delta": int(generated_analysis["pitch_span"] - target_analysis["pitch_span"]),
            "duration_bin_l1_distance": duration_l1_distance(
                generated_analysis["duration_counts"],
                target_analysis["duration_counts"],
            ),
        }
    )
    return enriched


def select_export_cases(cases: Sequence[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    """为产物导出挑选尽量均匀分桶的样本 case。"""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        buckets[str(case["bucket"])].append(case)
    for bucket_cases in buckets.values():
        bucket_cases.sort(key=lambda item: int(item["row_id"]))

    chosen: list[dict[str, Any]] = []
    added = True
    while added and len(chosen) < count:
        added = False
        for bucket in sorted(buckets):
            if not buckets[bucket]:
                continue
            chosen.append(buckets[bucket].pop(0))
            added = True
            if len(chosen) >= count:
                break
    return chosen
