"""Benchmark manifest 构建与面向音乐的 token 诊断工具。"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .config_io import load_yaml_mapping
from .eval_windows import sample_phrase_aligned_subsequence

_POSITIONS_PER_BAR = 32


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
    if idx < len(core) and core[idx].startswith("KEY_"):
        idx += 1
        positions.add(idx)

    while idx < len(core):
        if core[idx] != "BAR":
            return []
        positions.add(idx)
        idx += 1
        if idx < len(core) and core[idx].startswith("TEMPO_"):
            idx += 1
        if idx < len(core) and core[idx].startswith("KEY_"):
            idx += 1
        if idx < len(core) and core[idx] == "PHRASE":
            # bar-head PHRASE: 标记为合法 split 边界（split 后 target 以 PHRASE 起首即可）
            positions.add(idx)
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
            if idx < len(core) and core[idx] == "PHRASE":
                # mid-bar PHRASE：与 bar-head 相同，记为合法 split 边界
                positions.add(idx)
                idx += 1

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
    if idx < len(core) and core[idx].startswith("KEY_"):
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
        if idx < len(core) and core[idx].startswith("KEY_"):
            idx += 1
            group_id += 1

        # bar-head PHRASE: 与紧随的 event 一起作为 `phrase_event` 6-token 单元
        if idx < len(core) and core[idx] == "PHRASE":
            if idx + 5 < len(core) and (
                core[idx + 1].startswith("POS_")
                and core[idx + 2].startswith("INST_")
                and core[idx + 3].startswith("PITCH_")
                and core[idx + 4].startswith("DUR_")
                and core[idx + 5].startswith("VEL_")
            ):
                units.append((idx, idx + 6, "phrase_event", group_id))
                idx += 6
            else:
                return []

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
            # mid-bar PHRASE + event 也作为 `phrase_event` 6-token 单元
            if idx < len(core) and core[idx] == "PHRASE":
                if idx + 5 < len(core) and (
                    core[idx + 1].startswith("POS_")
                    and core[idx + 2].startswith("INST_")
                    and core[idx + 3].startswith("PITCH_")
                    and core[idx + 4].startswith("DUR_")
                    and core[idx + 5].startswith("VEL_")
                ):
                    units.append((idx, idx + 6, "phrase_event", group_id))
                    idx += 6
                else:
                    return []
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
    sequence_window = sample_phrase_aligned_subsequence(
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
    sequence_window = sample_phrase_aligned_subsequence(
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


def build_structure_control_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    min_prefix_tokens: int,
    prefix_ratio_min: float,
    prefix_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    """基于 continuation case 构造结构控制任务样本。"""
    base_case = build_continuation_case(
        source_tokens,
        max_positions=max_positions,
        min_prefix_tokens=min_prefix_tokens,
        prefix_ratio_min=prefix_ratio_min,
        prefix_ratio_max=prefix_ratio_max,
        seed=seed,
    )
    if base_case is None:
        return None

    target_core = [token for token in base_case["target_tokens"] if token != "EOS"]
    boundary_label = "start_new_phrase" if target_core and target_core[0] == "PHRASE" else "continue_inside_phrase"
    return {
        **base_case,
        "task_name": "structure_control",
        "boundary_label": boundary_label,
    }


def build_local_development_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    min_prefix_tokens: int,
    prefix_ratio_min: float,
    prefix_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    """基于 continuation case 构造局部展开任务样本。"""
    base_case = build_continuation_case(
        source_tokens,
        max_positions=max_positions,
        min_prefix_tokens=min_prefix_tokens,
        prefix_ratio_min=prefix_ratio_min,
        prefix_ratio_max=prefix_ratio_max,
        seed=seed,
    )
    if base_case is None:
        return None

    return {
        **base_case,
        "task_name": "local_development",
        "development_label": "develop",
    }


def build_long_context_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    min_prefix_tokens: int,
    prefix_ratio_min: float,
    prefix_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    """基于 continuation case 构造长上下文续写任务样本。"""
    base_case = build_continuation_case(
        source_tokens,
        max_positions=max_positions,
        min_prefix_tokens=min_prefix_tokens,
        prefix_ratio_min=max(float(prefix_ratio_min), 0.60),
        prefix_ratio_max=max(float(prefix_ratio_max), 0.85),
        seed=seed,
    )
    if base_case is None:
        return None

    return {
        **base_case,
        "task_name": "long_context_coherence",
        "section_label": "continue_section",
    }


def build_infilling_consistency_case(
    source_tokens: list[str],
    *,
    max_positions: int,
    hole_ratio_min: float,
    hole_ratio_max: float,
    seed: int,
) -> dict[str, Any] | None:
    """基于 infilling case 构造补全一致性任务样本。"""
    base_case = build_infilling_case(
        source_tokens,
        max_positions=max_positions,
        hole_ratio_min=hole_ratio_min,
        hole_ratio_max=hole_ratio_max,
        seed=seed,
    )
    if base_case is None:
        return None

    target_hole_tokens = list(base_case["target_hole_tokens"])
    structure_label = "across_boundary" if target_hole_tokens and target_hole_tokens[0] in {"BAR", "PHRASE"} else "inside_phrase"
    return {
        **base_case,
        "task_name": "infilling_consistency",
        "structure_label": structure_label,
    }


def _required_task_case_keys(task_scope: str) -> tuple[str, ...]:
    """返回指定 benchmark scope 真正必须成功构造的任务 case 键。"""
    if task_scope == "continuation":
        return (
            "structure_control_case",
            "local_development_case",
            "long_context_case",
        )
    if task_scope == "infilling":
        return ("infilling_consistency_case",)
    if task_scope == "all":
        return (
            "structure_control_case",
            "local_development_case",
            "long_context_case",
            "infilling_consistency_case",
        )
    raise ValueError(f"Unsupported benchmark task_scope: {task_scope}")


def build_benchmark_manifest(
    *,
    eval_jsonl_path: Path,
    eval_tok_path: Path,
    config: dict[str, Any],
    max_positions: int,
    task_scope: str = "all",
) -> dict[str, Any]:
    """构建可复现的 benchmark manifest。"""
    rows = load_eval_rows(eval_jsonl_path, eval_tok_path)
    note_thresholds = _quartile_thresholds([float(row["meta"]["note_count"]) for row in rows])
    duration_thresholds = _quartile_thresholds([float(row["meta"]["duration_sec"]) for row in rows])
    required_task_case_keys = set(_required_task_case_keys(task_scope))

    valid_cases: list[dict[str, Any]] = []
    for row in rows:
        meta = dict(row["meta"])
        tokens = list(row["tokens"])
        note_bucket = _quartile_bucket(float(meta["note_count"]), note_thresholds)
        duration_bucket = _quartile_bucket(float(meta["duration_sec"]), duration_thresholds)
        bucket = _bucket_label(note_bucket, duration_bucket)
        structure_control_case = build_structure_control_case(
            tokens,
            max_positions=max_positions,
            min_prefix_tokens=int(config["min_prefix_tokens"]),
            prefix_ratio_min=float(config["continuation_prefix_ratio_min"]),
            prefix_ratio_max=float(config["continuation_prefix_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 1,
        )
        local_development_case = build_local_development_case(
            tokens,
            max_positions=max_positions,
            min_prefix_tokens=int(config["min_prefix_tokens"]),
            prefix_ratio_min=float(config["continuation_prefix_ratio_min"]),
            prefix_ratio_max=float(config["continuation_prefix_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 3,
        )
        long_context_case = build_long_context_case(
            tokens,
            max_positions=max_positions,
            min_prefix_tokens=int(config["min_prefix_tokens"]),
            prefix_ratio_min=float(config["continuation_prefix_ratio_min"]),
            prefix_ratio_max=float(config["continuation_prefix_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 5,
        )
        infilling_consistency_case = build_infilling_consistency_case(
            tokens,
            max_positions=max_positions,
            hole_ratio_min=float(config["infilling_hole_ratio_min"]),
            hole_ratio_max=float(config["infilling_hole_ratio_max"]),
            seed=int(config["seed"]) + (int(row["row_id"]) * 17) + 7,
        )
        task_cases = {
            "structure_control_case": structure_control_case,
            "local_development_case": local_development_case,
            "long_context_case": long_context_case,
            "infilling_consistency_case": infilling_consistency_case,
        }
        if any(task_cases[case_key] is None for case_key in required_task_case_keys):
            continue

        continuation_case = None
        if structure_control_case is not None:
            continuation_case = {
                "prompt_tokens": list(structure_control_case["prompt_tokens"]),
                "target_tokens": list(structure_control_case["target_tokens"]),
                "prefix_len": int(structure_control_case["prefix_len"]),
                "target_len": int(structure_control_case["target_len"]),
                "window_tokens": list(structure_control_case["window_tokens"]),
                "window_len": int(structure_control_case["window_len"]),
            }
        infilling_case = None
        if infilling_consistency_case is not None:
            infilling_case = {
                "prompt_tokens": list(infilling_consistency_case["prompt_tokens"]),
                "prefix_tokens": list(infilling_consistency_case["prefix_tokens"]),
                "suffix_tokens": list(infilling_consistency_case["suffix_tokens"]),
                "target_hole_tokens": list(infilling_consistency_case["target_hole_tokens"]),
                "window_tokens": list(infilling_consistency_case["window_tokens"]),
                "window_len": int(infilling_consistency_case["window_len"]),
                "hole_len": int(infilling_consistency_case["hole_len"]),
            }

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
                "structure_control_case": structure_control_case,
                "local_development_case": local_development_case,
                "long_context_case": long_context_case,
                "infilling_consistency_case": infilling_consistency_case,
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
    while idx < len(filtered) and (filtered[idx].startswith("TEMPO_") or filtered[idx].startswith("KEY_")):
        idx += 1
    if idx >= len(filtered):
        return None
    if filtered[idx] == "BAR":
        return ("BAR",)
    if filtered[idx] == "PHRASE" and idx + 5 < len(filtered):
        unit = filtered[idx : idx + 6]
        if (
            unit[1].startswith("POS_")
            and unit[2].startswith("INST_")
            and unit[3].startswith("PITCH_")
            and unit[4].startswith("DUR_")
            and unit[5].startswith("VEL_")
        ):
            return tuple(unit)
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


def histogram_l1_distance(
    generated_counts: dict[Any, int],
    target_counts: dict[Any, int],
) -> float:
    """返回两个离散直方图之间的归一化 L1 距离。"""
    keys = sorted(set(generated_counts) | set(target_counts), key=lambda item: str(item))
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


def duration_l1_distance(
    generated_counts: dict[str, int],
    target_counts: dict[str, int],
) -> float:
    """返回归一化时值直方图之间的 L1 距离。"""
    return histogram_l1_distance(generated_counts, target_counts)


_MIN_PITCH_EVENTS_FOR_COLLAPSE_METRICS = 6
_PITCH_DIVERSITY_REFERENCE_UNIQUES = 12
_POSITIONS_PER_BAR = 32
_STRONG_BEAT_STRIDE = 8
_MIN_RHYTHM_EVENTS_FOR_DIVERSITY_METRICS = 6
_MIN_EVENTS_FOR_REPETITION_METRICS = 8
_ONSET_DIVERSITY_REFERENCE_UNIQUES = 8
_DURATION_DIVERSITY_REFERENCE_UNIQUES = 6


def _normalized_entropy(counter: Counter[Any], total_count: int, *, normalizer: float) -> float:
    if total_count <= 0 or normalizer <= 0.0 or not counter:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = float(count) / float(total_count)
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return max(0.0, min(1.0, entropy / normalizer))


def _categorical_diversity_score(
    values: Sequence[Any],
    *,
    reference_uniques: int,
) -> tuple[int, float]:
    value_list = list(values)
    unique_count = len(set(value_list))
    if not value_list:
        return 0, 0.0
    counter = Counter(value_list)
    entropy_norm = (
        _normalized_entropy(counter, len(value_list), normalizer=math.log(float(unique_count)))
        if unique_count > 1
        else 0.0
    )
    normalized_unique_count = min(unique_count, max(1, reference_uniques)) / float(max(1, reference_uniques))
    score = (0.6 * entropy_norm) + (0.4 * normalized_unique_count)
    return unique_count, max(0.0, min(1.0, score))


def _rhythm_diversity_metrics(
    onset_positions: Sequence[int],
    duration_tokens: Sequence[str],
) -> dict[str, Any]:
    event_count = len(onset_positions)
    onset_unique_count = len(set(int(position) for position in onset_positions))
    if event_count <= 0:
        return {
            "rhythm_event_count": 0,
            "onset_unique_count": 0,
            "rhythm_analysis_coverage": 0.0,
            "onset_position_entropy": None,
            "bar_start_onset_ratio": None,
            "strong_beat_onset_ratio": None,
            "duration_diversity_score": None,
            "rhythm_diversity_score": None,
        }

    if event_count < _MIN_RHYTHM_EVENTS_FOR_DIVERSITY_METRICS:
        return {
            "rhythm_event_count": event_count,
            "onset_unique_count": onset_unique_count,
            "rhythm_analysis_coverage": (event_count / float(_MIN_RHYTHM_EVENTS_FOR_DIVERSITY_METRICS)),
            "onset_position_entropy": None,
            "bar_start_onset_ratio": None,
            "strong_beat_onset_ratio": None,
            "duration_diversity_score": None,
            "rhythm_diversity_score": None,
        }

    onset_counter = Counter(int(position) for position in onset_positions)
    onset_entropy_norm = _normalized_entropy(
        onset_counter,
        event_count,
        normalizer=math.log(float(_POSITIONS_PER_BAR)),
    )
    onset_unique_norm = min(onset_unique_count, _ONSET_DIVERSITY_REFERENCE_UNIQUES) / float(
        _ONSET_DIVERSITY_REFERENCE_UNIQUES
    )
    _duration_unique_count, duration_diversity_score = _categorical_diversity_score(
        duration_tokens,
        reference_uniques=_DURATION_DIVERSITY_REFERENCE_UNIQUES,
    )
    bar_start_onset_ratio = (
        sum(1 for position in onset_positions if int(position) == 0) / float(event_count)
    )
    strong_beat_onset_ratio = (
        sum(1 for position in onset_positions if int(position) % _STRONG_BEAT_STRIDE == 0) / float(event_count)
    )
    rhythm_diversity_score = (
        (0.35 * onset_entropy_norm)
        + (0.20 * onset_unique_norm)
        + (0.20 * float(duration_diversity_score))
        + (0.15 * (1.0 - strong_beat_onset_ratio))
        + (0.10 * (1.0 - bar_start_onset_ratio))
    )

    return {
        "rhythm_event_count": event_count,
        "onset_unique_count": onset_unique_count,
        "rhythm_analysis_coverage": 1.0,
        "onset_position_entropy": max(0.0, min(1.0, onset_entropy_norm)),
        "bar_start_onset_ratio": max(0.0, min(1.0, bar_start_onset_ratio)),
        "strong_beat_onset_ratio": max(0.0, min(1.0, strong_beat_onset_ratio)),
        "duration_diversity_score": float(duration_diversity_score),
        "rhythm_diversity_score": max(0.0, min(1.0, rhythm_diversity_score)),
    }


def _ngram_extra_repeat_ratio(items: Sequence[Any], n: int) -> float:
    if n <= 0 or len(items) < n:
        return 0.0
    windows = [tuple(items[index : index + n]) for index in range(len(items) - n + 1)]
    if not windows:
        return 0.0
    counter = Counter(windows)
    repeated_window_count = sum(max(0, count - 1) for count in counter.values())
    return repeated_window_count / float(len(windows))


def _repetition_metrics(
    event_signatures: Sequence[tuple[int, int, str]],
    rhythm_signatures: Sequence[tuple[int, str]],
) -> dict[str, Any]:
    event_count = len(event_signatures)
    if event_count <= 0:
        return {
            "repetition_event_count": 0,
            "repetition_analysis_coverage": 0.0,
            "event_ngram_repeat_ratio": None,
            "rhythm_ngram_repeat_ratio": None,
        }

    if event_count < _MIN_EVENTS_FOR_REPETITION_METRICS:
        return {
            "repetition_event_count": event_count,
            "repetition_analysis_coverage": (event_count / float(_MIN_EVENTS_FOR_REPETITION_METRICS)),
            "event_ngram_repeat_ratio": None,
            "rhythm_ngram_repeat_ratio": None,
        }

    event_ngram_repeat_ratio = 0.5 * (
        _ngram_extra_repeat_ratio(event_signatures, 2) + _ngram_extra_repeat_ratio(event_signatures, 3)
    )
    rhythm_ngram_repeat_ratio = 0.5 * (
        _ngram_extra_repeat_ratio(rhythm_signatures, 2) + _ngram_extra_repeat_ratio(rhythm_signatures, 3)
    )
    return {
        "repetition_event_count": event_count,
        "repetition_analysis_coverage": 1.0,
        "event_ngram_repeat_ratio": max(0.0, min(1.0, event_ngram_repeat_ratio)),
        "rhythm_ngram_repeat_ratio": max(0.0, min(1.0, rhythm_ngram_repeat_ratio)),
    }


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
    current_bar_index = -1
    time_order_violation_count = 0
    same_pitch_overlap_count = 0
    parsed_event_count = 0
    pitch_values: list[int] = []
    onset_positions: list[int] = []
    duration_tokens: list[str] = []
    duration_counts: Counter[str] = Counter()
    onset_position_counts: Counter[int] = Counter()
    event_signatures: list[tuple[int, int, str]] = []
    rhythm_signatures: list[tuple[int, str]] = []
    active_note_end_by_voice: dict[tuple[str, int], int] = {}

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
        if token.startswith("TEMPO_") or token.startswith("KEY_"):
            idx += 1
            continue
        if token == "BAR":
            close_bar()
            current_bar_events = 0
            current_bar_last_pos = None
            current_bar_index += 1
            idx += 1
            continue
        phrase_prefixed = token == "PHRASE"
        event_start_idx = idx + 1 if phrase_prefixed else idx
        if phrase_prefixed or token.startswith("POS_"):
            if phrase_prefixed and idx + 5 >= len(values):
                break
            if event_start_idx >= len(values):
                break
            pos_token = str(values[event_start_idx])
            if not pos_token.startswith("POS_"):
                break
            ensure_bar()
            pos_value = _parse_prefixed_int(pos_token, "POS_")
            if current_bar_last_pos is not None and pos_value is not None and pos_value < current_bar_last_pos:
                time_order_violation_count += 1
            if pos_value is not None:
                current_bar_last_pos = pos_value
            if event_start_idx + 4 >= len(values):
                break
            inst_token, pitch_token, dur_token, vel_token = [
                str(item) for item in values[event_start_idx + 1 : event_start_idx + 5]
            ]
            if not inst_token.startswith("INST_"):
                break
            if not pitch_token.startswith("PITCH_"):
                break
            if not dur_token.startswith("DUR_"):
                break
            if not vel_token.startswith("VEL_"):
                break
            pitch_value = _parse_prefixed_int(pitch_token, "PITCH_")
            dur_value = _parse_prefixed_int(dur_token, "DUR_")
            if pos_value is not None:
                onset_positions.append(pos_value)
                onset_position_counts[pos_value] += 1
            duration_tokens.append(dur_token)
            if pitch_value is not None:
                pitch_values.append(pitch_value)
                if pos_value is not None:
                    event_signatures.append((pos_value, pitch_value, dur_token))
            if pos_value is not None:
                rhythm_signatures.append((pos_value, dur_token))
            if pos_value is not None and pitch_value is not None and dur_value is not None:
                absolute_start = (max(0, current_bar_index) * _POSITIONS_PER_BAR) + pos_value
                voice_key = (inst_token, pitch_value)
                active_end = active_note_end_by_voice.get(voice_key)
                if active_end is not None and absolute_start < active_end:
                    same_pitch_overlap_count += 1
                active_note_end_by_voice[voice_key] = max(
                    active_note_end_by_voice.get(voice_key, absolute_start + max(1, dur_value)),
                    absolute_start + max(1, dur_value),
                )
            duration_counts[dur_token] += 1
            parsed_event_count += 1
            current_bar_events = 0 if current_bar_events is None else (current_bar_events + 1)
            idx += 6 if phrase_prefixed else 5
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
    rhythm_metrics = _rhythm_diversity_metrics(onset_positions, duration_tokens)
    repetition_metrics = _repetition_metrics(event_signatures, rhythm_signatures)

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
        "onset_position_counts": dict(onset_position_counts),
        "time_order_valid": (time_order_violation_count == 0),
        "time_order_violation_count": time_order_violation_count,
        "same_pitch_overlap_count": same_pitch_overlap_count,
        "same_pitch_overlap_rate": (
            same_pitch_overlap_count / float(parsed_event_count) if parsed_event_count > 0 else 0.0
        ),
        "pitch_event_count": int(pitch_metrics["pitch_event_count"]),
        "pitch_unique_count": int(pitch_metrics["pitch_unique_count"]),
        "pitch_analysis_coverage": float(pitch_metrics["pitch_analysis_coverage"]),
        "most_common_pitch_ratio": pitch_metrics["most_common_pitch_ratio"],
        "longest_same_pitch_run_ratio": pitch_metrics["longest_same_pitch_run_ratio"],
        "pitch_diversity_score": pitch_metrics["pitch_diversity_score"],
        "rhythm_event_count": int(rhythm_metrics["rhythm_event_count"]),
        "onset_unique_count": int(rhythm_metrics["onset_unique_count"]),
        "rhythm_analysis_coverage": float(rhythm_metrics["rhythm_analysis_coverage"]),
        "onset_position_entropy": rhythm_metrics["onset_position_entropy"],
        "bar_start_onset_ratio": rhythm_metrics["bar_start_onset_ratio"],
        "strong_beat_onset_ratio": rhythm_metrics["strong_beat_onset_ratio"],
        "duration_diversity_score": rhythm_metrics["duration_diversity_score"],
        "rhythm_diversity_score": rhythm_metrics["rhythm_diversity_score"],
        "repetition_event_count": int(repetition_metrics["repetition_event_count"]),
        "repetition_analysis_coverage": float(repetition_metrics["repetition_analysis_coverage"]),
        "event_ngram_repeat_ratio": repetition_metrics["event_ngram_repeat_ratio"],
        "rhythm_ngram_repeat_ratio": repetition_metrics["rhythm_ngram_repeat_ratio"],
    }


def _last_pos_in_active_bar(tokens: Sequence[str]) -> int | None:
    values = list(tokens)
    idx = 0
    current_bar_last_pos: int | None = None

    while idx < len(values):
        token = str(values[idx])
        if token in {"BOS", "EOS", "FIM_HOLE", "FIM_MID"}:
            idx += 1
            continue
        if token.startswith("TEMPO_") or token.startswith("KEY_"):
            idx += 1
            continue
        if token == "BAR":
            current_bar_last_pos = None
            idx += 1
            continue
        phrase_prefixed = token == "PHRASE"
        event_start_idx = idx + 1 if phrase_prefixed else idx
        if phrase_prefixed or token.startswith("POS_"):
            if phrase_prefixed and idx + 5 >= len(values):
                break
            if event_start_idx >= len(values):
                break
            pos_token = str(values[event_start_idx])
            if not pos_token.startswith("POS_"):
                break
            pos_value = _parse_prefixed_int(pos_token, "POS_")
            if pos_value is not None:
                current_bar_last_pos = pos_value
            if event_start_idx + 4 >= len(values):
                break
            inst_token, pitch_token, dur_token, vel_token = [
                str(item) for item in values[event_start_idx + 1 : event_start_idx + 5]
            ]
            if (
                not inst_token.startswith("INST_")
                or not pitch_token.startswith("PITCH_")
                or not dur_token.startswith("DUR_")
                or not vel_token.startswith("VEL_")
            ):
                break
            idx += 6 if phrase_prefixed else 5
            continue
        break

    return current_bar_last_pos


def _first_pos_before_bar(tokens: Sequence[str]) -> int | None:
    values = list(tokens)
    idx = 0

    while idx < len(values):
        token = str(values[idx])
        if token in {"BOS", "EOS", "FIM_HOLE", "FIM_MID"}:
            idx += 1
            continue
        if token.startswith("TEMPO_") or token.startswith("KEY_"):
            idx += 1
            continue
        if token == "BAR":
            return None
        if token == "PHRASE":
            if idx + 5 >= len(values):
                break
            pos_token = str(values[idx + 1])
            inst_token, pitch_token, dur_token, vel_token = [str(item) for item in values[idx + 2 : idx + 6]]
            if (
                pos_token.startswith("POS_")
                and inst_token.startswith("INST_")
                and pitch_token.startswith("PITCH_")
                and dur_token.startswith("DUR_")
                and vel_token.startswith("VEL_")
            ):
                return _parse_prefixed_int(pos_token, "POS_")
            break
        if token.startswith("POS_"):
            return _parse_prefixed_int(token, "POS_")
        break

    return None


def _infilling_boundary_time_order_stats(
    prefix_tokens: Sequence[str],
    generated_middle_tokens: Sequence[str],
    suffix_tokens: Sequence[str],
) -> dict[str, int]:
    prefix_last_pos = _last_pos_in_active_bar(prefix_tokens)
    middle_first_pos = _first_pos_before_bar(generated_middle_tokens)
    middle_last_pos = _last_pos_in_active_bar(generated_middle_tokens)
    suffix_first_pos = _first_pos_before_bar(suffix_tokens)

    prefix_to_middle_violation_count = int(
        prefix_last_pos is not None
        and middle_first_pos is not None
        and middle_first_pos < prefix_last_pos
    )
    middle_to_suffix_violation_count = int(
        middle_last_pos is not None
        and suffix_first_pos is not None
        and suffix_first_pos < middle_last_pos
    )

    return {
        "prefix_to_middle_violation_count": prefix_to_middle_violation_count,
        "middle_to_suffix_violation_count": middle_to_suffix_violation_count,
        "boundary_violation_count": (
            prefix_to_middle_violation_count + middle_to_suffix_violation_count
        ),
    }


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _finite_score(value: Any) -> float | None:
    """把可用分值统一转换成有限浮点数。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _mean_score(
    values: Sequence[Any],
    *,
    expected_count: int | None = None,
    missing_value: float = 0.0,
) -> float:
    """对有限分数求均值，并允许把缺失证据按指定分值补齐。"""
    numeric_values: list[float] = []
    for value in values:
        numeric = _finite_score(value)
        if numeric is not None:
            numeric_values.append(numeric)
    if expected_count is None:
        if not numeric_values:
            return 0.0
        return _clamp_score(sum(numeric_values) / float(len(numeric_values)))

    normalized_expected_count = max(0, int(expected_count))
    if normalized_expected_count <= 0:
        return 0.0
    missing_count = max(0, normalized_expected_count - len(numeric_values))
    total_score = sum(numeric_values) + (float(missing_value) * float(missing_count))
    denominator = max(normalized_expected_count, len(numeric_values), 1)
    return _clamp_score(total_score / float(denominator))


def _bool_hit(value: Any) -> float:
    """把布尔型命中结果转换成 0/1 分值。"""
    return 1.0 if bool(value) else 0.0


def _normalized_l1_similarity(distance: Any) -> float:
    """把 [0, 2] 区间附近的 L1 距离映射成 [0, 1] 相似度。"""
    numeric = _finite_score(distance)
    if numeric is None:
        return 0.0
    return _clamp_score(1.0 - max(0.0, numeric))


def _pitch_span_similarity(generated_pitch_span: Any, target_pitch_span: Any) -> float:
    """根据生成段与目标段的音高跨度差异估计相似度。"""
    generated_value = _finite_score(generated_pitch_span)
    target_value = _finite_score(target_pitch_span)
    if generated_value is None or target_value is None:
        return 0.0
    scale = max(abs(generated_value), abs(target_value), 1.0)
    return _clamp_score(1.0 - (abs(generated_value - target_value) / scale))


def _boundary_kind(tokens: Sequence[str]) -> str | None:
    """提取序列起始边界的结构类型。"""
    first_unit = _extract_first_unit(tokens)
    if first_unit is None:
        return None
    if first_unit[0] == "BAR":
        return "bar"
    if first_unit[0] == "PHRASE":
        return "phrase"
    return "event"


def _first_event_position(tokens: Sequence[str]) -> int | None:
    """提取序列中的首个事件位置，用于比较边界后的落点时序。"""
    values = list(tokens)
    idx = 0
    while idx < len(values):
        token = str(values[idx])
        if token in {"BOS", "EOS", "FIM_HOLE", "FIM_MID", "BAR"}:
            idx += 1
            continue
        if token.startswith("TEMPO_") or token.startswith("KEY_"):
            idx += 1
            continue
        if token == "PHRASE":
            if idx + 5 >= len(values):
                return None
            return _parse_prefixed_int(str(values[idx + 1]), "POS_")
        if token.startswith("POS_"):
            return _parse_prefixed_int(token, "POS_")
        idx += 1
    return None


def _has_generated_content(
    enriched: dict[str, Any],
    *,
    generated_len_key: str,
    generated_event_count_key: str = "generated_event_count",
) -> bool:
    """判断当前任务输出是否包含足够的实际生成内容。"""
    generated_len = _finite_score(enriched.get(generated_len_key))
    if generated_len is not None and generated_len > 0.0:
        return True
    generated_event_count = _finite_score(enriched.get(generated_event_count_key))
    return bool(generated_event_count is not None and generated_event_count > 0.0)


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
            "same_pitch_overlap_count": int(reconstructed_analysis["same_pitch_overlap_count"]),
            "same_pitch_overlap_rate": float(reconstructed_analysis["same_pitch_overlap_rate"]),
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
            "rhythm_analysis_coverage": float(generated_analysis["rhythm_analysis_coverage"]),
            "repetition_analysis_coverage": float(generated_analysis["repetition_analysis_coverage"]),
            "onset_position_entropy": generated_analysis["onset_position_entropy"],
            "bar_start_onset_ratio": generated_analysis["bar_start_onset_ratio"],
            "strong_beat_onset_ratio": generated_analysis["strong_beat_onset_ratio"],
            "duration_diversity_score": generated_analysis["duration_diversity_score"],
            "rhythm_diversity_score": generated_analysis["rhythm_diversity_score"],
            "event_ngram_repeat_ratio": generated_analysis["event_ngram_repeat_ratio"],
            "rhythm_ngram_repeat_ratio": generated_analysis["rhythm_ngram_repeat_ratio"],
            "target_bar_count": int(target_analysis["bar_count"]),
            "target_event_count": int(target_analysis["event_count"]),
            "target_pitch_span": int(target_analysis["pitch_span"]),
            "target_same_pitch_overlap_count": int(target_analysis["same_pitch_overlap_count"]),
            "target_same_pitch_overlap_rate": float(target_analysis["same_pitch_overlap_rate"]),
            "generated_bar_delta": int(generated_bar_delta),
            "generated_event_delta": int(generated_event_delta),
            "pitch_span_delta": int(pitch_span_delta),
            "onset_position_l1_distance": histogram_l1_distance(
                generated_analysis["onset_position_counts"],
                target_analysis["onset_position_counts"],
            ),
            "duration_bin_l1_distance": duration_l1_distance(
                generated_analysis["duration_counts"],
                target_analysis["duration_counts"],
            ),
        }
    )
    return enriched


def enrich_structure_control_record(record: dict[str, Any], *, target_tokens: Sequence[str]) -> dict[str, Any]:
    """补充结构控制任务的任务级原始指标。"""
    enriched = dict(record)
    if "first_unit_match" not in enriched or "duration_bin_l1_distance" not in enriched:
        enriched = enrich_continuation_record(enriched, target_tokens=target_tokens)
    target_core = [token for token in target_tokens if token != "EOS"]
    target_boundary_kind = _boundary_kind(target_core)
    boundary_type_hit = _bool_hit(
        _boundary_kind(enriched.get("generated_tokens", [])) == target_boundary_kind
        and target_boundary_kind is not None
    )
    target_boundary_pos = _first_event_position(target_core)
    boundary_timing_hit = _bool_hit(
        _first_event_position(enriched.get("generated_tokens", [])) == target_boundary_pos
        and target_boundary_pos is not None
    )
    post_boundary_realization_score = _mean_score(
        [
            boundary_type_hit,
            boundary_timing_hit,
            _bool_hit(enriched.get("first_unit_match")),
        ]
    )
    enriched.update(
        {
            "boundary_type_hit": boundary_type_hit,
            "boundary_timing_hit": boundary_timing_hit,
            "post_boundary_realization_score": post_boundary_realization_score,
        }
    )
    return enriched


def enrich_local_development_record(record: dict[str, Any], *, target_tokens: Sequence[str]) -> dict[str, Any]:
    """补充局部展开任务的任务级原始指标。"""
    enriched = dict(record)
    if "first_unit_match" not in enriched or "duration_bin_l1_distance" not in enriched:
        enriched = enrich_continuation_record(enriched, target_tokens=target_tokens)
    motif_relation_hit = _bool_hit(enriched.get("first_unit_match"))
    copy_overuse_penalty = _mean_score(
        [
            enriched.get("event_ngram_repeat_ratio"),
            enriched.get("rhythm_ngram_repeat_ratio"),
            enriched.get("most_common_pitch_ratio"),
            enriched.get("longest_same_pitch_run_ratio"),
        ],
        expected_count=4,
        missing_value=1.0,
    )
    unrelated_drift_penalty = _mean_score(
        [
            1.0 - _normalized_l1_similarity(enriched.get("onset_position_l1_distance")),
            1.0 - _normalized_l1_similarity(enriched.get("duration_bin_l1_distance")),
            1.0 - _pitch_span_similarity(
                enriched.get("generated_pitch_span"),
                enriched.get("target_pitch_span"),
            ),
        ],
        expected_count=3,
        missing_value=1.0,
    )
    quality_score = _mean_score(
        [
            motif_relation_hit,
            1.0 - copy_overuse_penalty,
            1.0 - unrelated_drift_penalty,
            enriched.get("pitch_diversity_score"),
            enriched.get("rhythm_diversity_score"),
        ],
        expected_count=5,
        missing_value=0.0,
    )
    enriched.update(
        {
            "motif_relation_hit": motif_relation_hit,
            "copy_overuse_penalty": copy_overuse_penalty,
            "unrelated_drift_penalty": unrelated_drift_penalty,
            "quality_score": quality_score,
        }
    )
    return enriched


def enrich_long_context_record(record: dict[str, Any], *, target_tokens: Sequence[str]) -> dict[str, Any]:
    """补充长上下文任务的任务级原始指标。"""
    enriched = dict(record)
    if "first_unit_match" not in enriched or "duration_bin_l1_distance" not in enriched:
        enriched = enrich_continuation_record(enriched, target_tokens=target_tokens)
    has_generated_content = _has_generated_content(enriched, generated_len_key="generated_len")
    completion_rate = _bool_hit(enriched.get("stop_success") and has_generated_content)
    theme_retention_score = (
        _mean_score(
            [
                _normalized_l1_similarity(enriched.get("onset_position_l1_distance")),
                _normalized_l1_similarity(enriched.get("duration_bin_l1_distance")),
                _pitch_span_similarity(
                    enriched.get("generated_pitch_span"),
                    enriched.get("target_pitch_span"),
                ),
            ],
            expected_count=3,
            missing_value=0.0,
        )
        if has_generated_content
        else 0.0
    )
    section_continuity_score = (
        _mean_score(
            [
                _bool_hit(enriched.get("time_order_valid")),
                _bool_hit(enriched.get("structural_match_without_eos")),
                _bool_hit(enriched.get("first_unit_match")),
            ],
            expected_count=3,
            missing_value=0.0,
        )
        if has_generated_content
        else 0.0
    )
    degeneration_penalty = _mean_score(
        [
            enriched.get("same_pitch_overlap_rate"),
            enriched.get("most_common_pitch_ratio"),
            enriched.get("longest_same_pitch_run_ratio"),
            enriched.get("event_ngram_repeat_ratio"),
            enriched.get("rhythm_ngram_repeat_ratio"),
            (
                1.0 - float(enriched["pitch_diversity_score"])
                if _finite_score(enriched.get("pitch_diversity_score")) is not None
                else None
            ),
            (
                1.0 - float(enriched["rhythm_diversity_score"])
                if _finite_score(enriched.get("rhythm_diversity_score")) is not None
                else None
            ),
        ],
        expected_count=7,
        missing_value=1.0,
    )
    enriched.update(
        {
            "completion_rate": completion_rate,
            "theme_retention_score": theme_retention_score,
            "section_continuity_score": section_continuity_score,
            "degeneration_penalty": degeneration_penalty,
        }
    )
    return enriched


def enrich_infilling_record(record: dict[str, Any], *, target_hole_tokens: Sequence[str]) -> dict[str, Any]:
    """为 infilling decode 轨迹补充面向音乐的诊断字段。"""
    prefix_tokens = list(record.get("prefix_tokens", []))
    generated_middle_tokens = list(record.get("generated_middle_tokens", []))
    suffix_tokens = list(record.get("suffix_tokens", []))
    generated_analysis = analyze_token_sequence(record.get("generated_middle_tokens", []))
    target_analysis = analyze_token_sequence(target_hole_tokens)
    reconstructed_analysis = analyze_token_sequence(record.get("reconstructed_tokens", []))
    boundary_order_stats = _infilling_boundary_time_order_stats(
        prefix_tokens,
        generated_middle_tokens,
        suffix_tokens,
    )
    internal_time_order_violation_count = int(generated_analysis["time_order_violation_count"])
    boundary_time_order_violation_count = int(boundary_order_stats["boundary_violation_count"])
    enriched = dict(record)
    enriched.update(
        {
            "time_order_valid": bool(reconstructed_analysis["time_order_valid"]),
            "time_order_violation_count": int(reconstructed_analysis["time_order_violation_count"]),
            "same_pitch_overlap_count": int(reconstructed_analysis["same_pitch_overlap_count"]),
            "same_pitch_overlap_rate": float(reconstructed_analysis["same_pitch_overlap_rate"]),
            "internal_time_order_valid": (internal_time_order_violation_count == 0),
            "internal_time_order_violation_count": internal_time_order_violation_count,
            "boundary_time_order_valid": (boundary_time_order_violation_count == 0),
            "boundary_time_order_violation_count": boundary_time_order_violation_count,
            "prefix_to_middle_time_order_violation_count": int(
                boundary_order_stats["prefix_to_middle_violation_count"]
            ),
            "middle_to_suffix_time_order_violation_count": int(
                boundary_order_stats["middle_to_suffix_violation_count"]
            ),
            "generated_bar_count": int(generated_analysis["bar_count"]),
            "generated_event_count": int(generated_analysis["event_count"]),
            "generated_pitch_span": int(generated_analysis["pitch_span"]),
            "generated_pitch_event_count": int(generated_analysis["pitch_event_count"]),
            "generated_pitch_unique_count": int(generated_analysis["pitch_unique_count"]),
            "pitch_analysis_coverage": float(generated_analysis["pitch_analysis_coverage"]),
            "most_common_pitch_ratio": generated_analysis["most_common_pitch_ratio"],
            "longest_same_pitch_run_ratio": generated_analysis["longest_same_pitch_run_ratio"],
            "pitch_diversity_score": generated_analysis["pitch_diversity_score"],
            "rhythm_analysis_coverage": float(generated_analysis["rhythm_analysis_coverage"]),
            "repetition_analysis_coverage": float(generated_analysis["repetition_analysis_coverage"]),
            "onset_position_entropy": generated_analysis["onset_position_entropy"],
            "bar_start_onset_ratio": generated_analysis["bar_start_onset_ratio"],
            "strong_beat_onset_ratio": generated_analysis["strong_beat_onset_ratio"],
            "duration_diversity_score": generated_analysis["duration_diversity_score"],
            "rhythm_diversity_score": generated_analysis["rhythm_diversity_score"],
            "event_ngram_repeat_ratio": generated_analysis["event_ngram_repeat_ratio"],
            "rhythm_ngram_repeat_ratio": generated_analysis["rhythm_ngram_repeat_ratio"],
            "target_bar_count": int(target_analysis["bar_count"]),
            "target_event_count": int(target_analysis["event_count"]),
            "target_pitch_span": int(target_analysis["pitch_span"]),
            "target_same_pitch_overlap_count": int(target_analysis["same_pitch_overlap_count"]),
            "target_same_pitch_overlap_rate": float(target_analysis["same_pitch_overlap_rate"]),
            "generated_bar_delta": int(generated_analysis["bar_count"] - target_analysis["bar_count"]),
            "generated_event_delta": int(generated_analysis["event_count"] - target_analysis["event_count"]),
            "pitch_span_delta": int(generated_analysis["pitch_span"] - target_analysis["pitch_span"]),
            "onset_position_l1_distance": histogram_l1_distance(
                generated_analysis["onset_position_counts"],
                target_analysis["onset_position_counts"],
            ),
            "duration_bin_l1_distance": duration_l1_distance(
                generated_analysis["duration_counts"],
                target_analysis["duration_counts"],
            ),
        }
    )
    return enriched


def enrich_infilling_consistency_record(
    record: dict[str, Any],
    *,
    target_hole_tokens: Sequence[str],
) -> dict[str, Any]:
    """补充补全一致性任务的任务级原始指标。"""
    enriched = dict(record)
    if "boundary_time_order_valid" not in enriched or "duration_bin_l1_distance" not in enriched:
        enriched = enrich_infilling_record(enriched, target_hole_tokens=target_hole_tokens)
    has_generated_content = _has_generated_content(
        enriched,
        generated_len_key="generated_middle_len",
    )
    bridge_validity = _bool_hit(
        bool(enriched.get("is_structurally_valid"))
        and bool(enriched.get("time_order_valid"))
        and bool(enriched.get("internal_time_order_valid"))
        and has_generated_content
    )
    boundary_compatibility_hit = _bool_hit(enriched.get("boundary_time_order_valid") and has_generated_content)
    rhythmic_connection_score = (
        _mean_score(
            [
                _normalized_l1_similarity(enriched.get("onset_position_l1_distance")),
                _normalized_l1_similarity(enriched.get("duration_bin_l1_distance")),
            ],
            expected_count=2,
            missing_value=0.0,
        )
        if has_generated_content
        else 0.0
    )
    pitch_connection_score = (
        _mean_score(
            [
                _pitch_span_similarity(
                    enriched.get("generated_pitch_span"),
                    enriched.get("target_pitch_span"),
                ),
                (
                    1.0 - float(enriched["same_pitch_overlap_rate"])
                    if _finite_score(enriched.get("same_pitch_overlap_rate")) is not None
                    else None
                ),
            ],
            expected_count=2,
            missing_value=0.0,
        )
        if has_generated_content
        else 0.0
    )
    structural_fit_score = _mean_score(
        [
            bridge_validity,
            boundary_compatibility_hit,
            rhythmic_connection_score,
            pitch_connection_score,
        ],
        expected_count=4,
        missing_value=0.0,
    )
    enriched.update(
        {
            "bridge_validity": bridge_validity,
            "boundary_compatibility_hit": boundary_compatibility_hit,
            "rhythmic_connection_score": rhythmic_connection_score,
            "pitch_connection_score": pitch_connection_score,
            "structural_fit_score": structural_fit_score,
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
