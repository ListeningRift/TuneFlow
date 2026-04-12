#!/usr/bin/env python
"""评估 base 训练产物的最小闭环指标（valid_loss / ppl / 结构合法率）。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


def _ensure_project_root_on_path() -> Path:
    """确保仓库根目录可导入，并返回该路径。"""
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args() -> argparse.Namespace:
    """解析评估命令行参数。"""
    parser = argparse.ArgumentParser(description="对一个 run 下的全部 checkpoint 执行 infilling 评估。")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/checkpoints/base/train_base_run_small"),
        help="某次训练 run 的 checkpoint 目录（包含 *.pt）。",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="报告文件名中的 run_id；默认使用 checkpoint 目录名。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="可选报告输出路径；默认 outputs/reports/eval_infilling/<run_id>.json。",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/train/model_base.yaml"),
        help="当 checkpoint 不含 model_config 时使用的兜底模型配置。",
    )
    parser.add_argument(
        "--valid-idx",
        type=Path,
        default=Path("data/tokenized/valid.idx.json"),
        help="用于计算 valid_loss/ppl 的验证集 idx 路径。",
    )
    parser.add_argument(
        "--valid-bin",
        type=Path,
        default=None,
        help="可选：覆盖验证集 bin 路径。",
    )
    parser.add_argument(
        "--eval-tok",
        type=Path,
        default=Path("data/tokenized/eval.tok"),
        help="用于构造 infilling prompt 的评估 token 文件。",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("data/tokenized/tokenizer_vocab.json"),
        help="tokenizer 词表 JSON 路径。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="评估设备。",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="数值精度模式。",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="计算 valid_loss 采样时使用的序列长度。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="验证 micro-batch 大小。",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=10,
        help="每个 checkpoint 评估的验证 batch 数。",
    )
    parser.add_argument(
        "--num-infilling-samples",
        type=int,
        default=32,
        help="每个 checkpoint 用于结构合法率评估的 infilling 样本数。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="每条 infilling prompt 最大生成 token 数。",
    )
    parser.add_argument(
        "--limit-checkpoints",
        type=int,
        default=None,
        help="可选：限制评估的 checkpoint 数量（用于冒烟测试）。",
    )
    parser.add_argument(
        "--checkpoint-policy",
        type=str,
        default="all",
        choices=["all", "sampled"],
        help="checkpoint 选择策略：all=全量评估，sampled=抽样评估。",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=6,
        help="当 --checkpoint-policy=sampled 时，抽样的 step_* checkpoint 数量。",
    )
    parser.add_argument(
        "--valid-loss-source",
        type=str,
        default="recompute",
        choices=["recompute", "metrics", "auto"],
        help="valid_loss/ppl 来源：recompute=重新评估，metrics=复用训练期 metrics.jsonl，auto=优先复用失败时再重算。",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="可选：训练期 metrics.jsonl 路径；默认尝试使用 checkpoint 目录下的 metrics.jsonl。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--debug-invalid-samples",
        type=int,
        default=0,
        help="每个 checkpoint 额外保留多少条非法样本；大于 0 时会打印摘要并写出 debug JSON。",
    )
    parser.add_argument(
        "--debug-preview-tokens",
        type=int,
        default=16,
        help="控制台打印非法样本时，每段 token 预览的最大数量。",
    )
    return parser.parse_args()


_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    """定义 checkpoint 排序规则：step_* 优先，其次 best/last/latest。"""
    match = _STEP_RE.match(path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    if path.name == "best.pt":
        return (1, 0, path.name)
    if path.name == "last.pt":
        return (2, 0, path.name)
    if path.name == "latest.pt":
        return (3, 0, path.name)
    return (4, 0, path.name)


def _sample_step_checkpoints(step_paths: list[Path], sample_count: int) -> list[Path]:
    """对 `step_*.pt` 做均匀抽样，保留头尾和中间代表点。"""
    if sample_count <= 0 or len(step_paths) <= sample_count:
        return step_paths
    if sample_count == 1:
        return [step_paths[-1]]

    indices = {
        round(index * (len(step_paths) - 1) / (sample_count - 1))
        for index in range(sample_count)
    }
    return [step_paths[index] for index in sorted(indices)]


def _discover_checkpoints(
    checkpoint_dir: Path,
    limit: int | None,
    policy: str,
    sample_count: int,
) -> list[Path]:
    """扫描目录中的 checkpoint 文件，并按规则排序或抽样。"""
    paths = sorted([p for p in checkpoint_dir.glob("*.pt") if p.is_file()], key=_checkpoint_sort_key)
    if policy == "sampled":
        step_paths = [p for p in paths if _STEP_RE.match(p.name)]
        extra_paths = [p for p in paths if not _STEP_RE.match(p.name)]
        sampled = _sample_step_checkpoints(step_paths, sample_count)
        deduped: dict[str, Path] = {p.name: p for p in [*sampled, *extra_paths]}
        paths = sorted(deduped.values(), key=_checkpoint_sort_key)
    if limit is not None:
        paths = paths[: max(0, limit)]
    return paths


def _resolve_metrics_path(checkpoint_dir: Path, metrics_path: Path | None) -> Path | None:
    """解析 metrics.jsonl 路径；若不存在则返回 `None`。"""
    candidate = metrics_path if metrics_path is not None else (checkpoint_dir / "metrics.jsonl")
    if not candidate.exists():
        return None
    return candidate.resolve()


def _load_valid_loss_by_step(metrics_path: Path | None) -> dict[int, float]:
    """从训练期 metrics.jsonl 中提取 `step -> valid_loss` 映射。"""
    if metrics_path is None:
        return {}

    mapping: dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("event") != "eval":
                continue
            try:
                step = int(payload["step"])
                valid_loss = float(payload["valid_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(valid_loss):
                mapping[step] = valid_loss
    return mapping


def _load_vocab(vocab_path: Path) -> tuple[dict[str, int], list[str]]:
    """
    读取词表并返回双向映射所需结构。

    返回：
    - token_to_id: dict[str, int]
    - id_to_token: list[str]
    """
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    token_to_id = payload.get("token_to_id")
    if not isinstance(token_to_id, dict):
        raise ValueError(f"Invalid vocab file: missing token_to_id in {vocab_path}")
    token_to_id = {str(k): int(v) for k, v in token_to_id.items()}

    id_to_token_raw = payload.get("id_to_token")
    if isinstance(id_to_token_raw, list) and id_to_token_raw:
        id_to_token = [str(x) for x in id_to_token_raw]
    else:
        max_id = max(token_to_id.values()) if token_to_id else -1
        id_to_token = ["<UNK>"] * (max_id + 1)
        for token, idx in token_to_id.items():
            if 0 <= idx < len(id_to_token):
                id_to_token[idx] = token
    return token_to_id, id_to_token


def _load_eval_tok_lines(path: Path) -> list[list[str]]:
    """读取 `eval.tok`，每行拆分成 token 列表。"""
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            tokens = [tok for tok in line.strip().split(" ") if tok]
            if tokens:
                rows.append(tokens)
    return rows


def _validate_token_order(tokens: list[str], vocab: dict[str, int]) -> bool:
    """
    校验 token 序列结构是否合法。

    语法约束与 tokenizer 训练数据保持一致：
    - BOS ... EOS
    - BAR 分段
    - 音符事件按 `POS -> INST -> PITCH -> DUR -> VEL` 5 元组出现
    """
    if any(token not in vocab for token in tokens):
        return False
    if not tokens or tokens[0] != "BOS" or tokens[-1] != "EOS":
        return False

    idx = 1
    if idx < len(tokens) - 1 and tokens[idx].startswith("TEMPO_"):
        idx += 1

    while idx < len(tokens) - 1:
        if tokens[idx] != "BAR":
            return False
        idx += 1
        if idx < len(tokens) - 1 and tokens[idx].startswith("TEMPO_"):
            idx += 1
        while idx < len(tokens) - 1 and tokens[idx].startswith("POS_"):
            if idx + 4 >= len(tokens):
                return False
            if not tokens[idx + 1].startswith("INST_"):
                return False
            if not tokens[idx + 2].startswith("PITCH_"):
                return False
            if not tokens[idx + 3].startswith("DUR_"):
                return False
            if not tokens[idx + 4].startswith("VEL_"):
                return False
            idx += 5
        if idx < len(tokens) - 1 and tokens[idx] != "BAR":
            return False
    return True


def _inspect_token_order(tokens: list[str], vocab: dict[str, int]) -> tuple[bool, str]:
    """
    校验 token 序列结构是否合法，并返回首个失败原因。
    语法约束与 tokenizer 训练数据保持一致：
    - BOS ... EOS
    - BAR 分段
    - 音符事件按 `POS -> INST -> PITCH -> DUR -> VEL` 五元组出现
    """
    for idx, token in enumerate(tokens):
        if token not in vocab:
            return False, f"unknown_token@{idx}:{token}"
    if not tokens:
        return False, "empty_sequence"
    if tokens[0] != "BOS":
        return False, f"missing_bos:{tokens[0]}"
    if tokens[-1] != "EOS":
        return False, f"missing_eos:{tokens[-1]}"

    idx = 1
    if idx < len(tokens) - 1 and tokens[idx].startswith("TEMPO_"):
        idx += 1

    while idx < len(tokens) - 1:
        if tokens[idx] != "BAR":
            return False, f"expected_bar@{idx}:{tokens[idx]}"
        idx += 1
        if idx < len(tokens) - 1 and tokens[idx].startswith("TEMPO_"):
            idx += 1
        while idx < len(tokens) - 1 and tokens[idx].startswith("POS_"):
            if idx + 4 >= len(tokens):
                return False, f"incomplete_note_tuple@{idx}"
            if not tokens[idx + 1].startswith("INST_"):
                return False, f"expected_inst@{idx + 1}:{tokens[idx + 1]}"
            if not tokens[idx + 2].startswith("PITCH_"):
                return False, f"expected_pitch@{idx + 2}:{tokens[idx + 2]}"
            if not tokens[idx + 3].startswith("DUR_"):
                return False, f"expected_dur@{idx + 3}:{tokens[idx + 3]}"
            if not tokens[idx + 4].startswith("VEL_"):
                return False, f"expected_vel@{idx + 4}:{tokens[idx + 4]}"
            idx += 5
        if idx < len(tokens) - 1 and tokens[idx] != "BAR":
            return False, f"expected_bar_or_eos@{idx}:{tokens[idx]}"
    return True, "ok"


def _preview_tokens(tokens: list[str], limit: int, *, from_tail: bool = False) -> str:
    """生成用于控制台日志的 token 预览字符串。"""
    if limit <= 0:
        return ""
    if len(tokens) <= limit:
        return " ".join(tokens)
    if from_tail:
        return "... " + " ".join(tokens[-limit:])
    return " ".join(tokens[:limit]) + " ..."


def _print_invalid_sample_preview(
    *,
    step: int,
    checkpoint_name: str,
    invalid_samples: list[dict[str, Any]],
    preview_limit: int,
) -> None:
    """在控制台打印非法样本的简要摘要。"""
    if not invalid_samples:
        return

    reason_counts: dict[str, int] = {}
    for sample in invalid_samples:
        raw_reason = str(sample.get("raw", {}).get("failure_reason", "unknown"))
        fsm_reason = str(sample.get("fsm", {}).get("failure_reason", "unknown"))
        reason_counts[f"raw:{raw_reason}"] = reason_counts.get(f"raw:{raw_reason}", 0) + 1
        reason_counts[f"fsm:{fsm_reason}"] = reason_counts.get(f"fsm:{fsm_reason}", 0) + 1
    reasons_text = ", ".join(f"{key}={value}" for key, value in sorted(reason_counts.items()))
    print(
        f"[eval_infilling][debug] step={step} checkpoint={checkpoint_name} "
        f"captured_invalid={len(invalid_samples)} reasons={reasons_text}"
    )
    for index, sample in enumerate(invalid_samples, start=1):
        prompt_tail = _preview_tokens(list(sample.get("prompt_tokens", [])), preview_limit, from_tail=True)
        raw_record = sample.get("raw", {})
        fsm_record = sample.get("fsm", {})
        raw_generated_head = _preview_tokens(list(raw_record.get("generated_middle_tokens", [])), preview_limit)
        fsm_generated_head = _preview_tokens(list(fsm_record.get("generated_middle_tokens", [])), preview_limit)
        print(
            f"[eval_infilling][debug] sample={index} attempt={sample.get('attempt_index')} "
            f"raw_failure={raw_record.get('failure_reason')} fsm_failure={fsm_record.get('failure_reason')} "
            f"raw_reason={raw_record.get('syntax_reason')} fsm_reason={fsm_record.get('syntax_reason')} "
            f"prompt_len={sample.get('prompt_len')} "
            f"raw_generated_len={raw_record.get('generated_middle_len')} "
            f"fsm_generated_len={fsm_record.get('generated_middle_len')} "
            f"prompt_tail={prompt_tail}"
        )
        print(f"[eval_infilling][debug] sample={index} raw_generated={raw_generated_head}")
        print(f"[eval_infilling][debug] sample={index} fsm_generated={fsm_generated_head}")


def _collect_infill_maskable_units(core: list[str]) -> list[tuple[int, int, str, int]]:
    """
    收集允许被 mask 的完整单元。

    约束如下：
    - 只允许 mask `BAR`
    - 只允许 mask 完整音符事件 `POS -> INST -> PITCH -> DUR -> VEL`
    - `TEMPO_*` 属于不可 mask 的特殊 token，同时也是不可跨越的边界
    """
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
    """
    在允许 mask 的完整单元之间选择一个连续洞区间。

    洞只会覆盖 `BAR` 或完整音符事件，且不会跨过 `TEMPO_*`。
    """
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


def _safe_perplexity(loss: float) -> float:
    """将 loss 转为 ppl，并在极端值时做安全处理。"""
    if not math.isfinite(loss):
        return float("nan")
    if loss >= 80.0:
        return float("inf")
    return math.exp(loss)


def _resolve_valid_loss(
    *,
    step: int,
    args: argparse.Namespace,
    valid_loss_by_step: dict[int, float],
    recompute_fn,
) -> tuple[float, str]:
    """
    解析当前 checkpoint 的 valid_loss 来源。

    返回：
    - valid_loss
    - source，取值 `metrics` 或 `recompute`
    """
    metrics_loss = valid_loss_by_step.get(step)
    if args.valid_loss_source == "metrics":
        if metrics_loss is None:
            raise ValueError(
                f"--valid-loss-source=metrics 但在 metrics.jsonl 中未找到 step={step} 的 valid_loss。"
            )
        return metrics_loss, "metrics"

    if args.valid_loss_source == "auto" and metrics_loss is not None:
        return metrics_loss, "metrics"

    return recompute_fn(), "recompute"


def _evaluate_valid_loss(
    model,
    dataset,
    torch_mod,
    rng: random.Random,
    device,
    id_to_token: list[str],
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    use_amp: bool,
    amp_dtype,
    autocast_context_fn,
) -> float:
    """在验证集上采样若干 batch，计算平均 valid_loss。"""
    if eval_batches <= 0:
        return float("nan")

    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch_mod.no_grad():
        for _ in range(eval_batches):
            input_ids, labels = dataset.sample_batch(
                torch_mod=torch_mod,
                rng=rng,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                id_to_token=id_to_token,
            )
            with autocast_context_fn(
                torch_mod=torch_mod,
                use_amp=use_amp,
                device_type=device.type,
                amp_dtype=amp_dtype,
            ):
                outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
            loss_val = float(outputs.loss.item())
            if math.isfinite(loss_val):
                losses.append(loss_val)
    if was_training:
        model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def _build_infilling_prompt(
    source_tokens: list[str],
    max_positions: int,
    rng: random.Random,
) -> tuple[list[str], list[str], int] | None:
    """
    从完整 token 序列构造 infilling prompt。

    形式：
    `prefix + FIM_HOLE + suffix + FIM_MID`
    """
    if len(source_tokens) < 30:
        return None
    if source_tokens[0] != "BOS" or source_tokens[-1] != "EOS":
        return None

    from src.utils.eval_windows import sample_bar_aligned_subsequence

    seq = sample_bar_aligned_subsequence(
        source_tokens,
        max_core_tokens=max(24, max_positions - 8),
        min_core_tokens=20,
        rng=rng,
    )
    if seq is None:
        return None
    core = seq[1:-1]
    core_len = len(core)
    target_hole_len = max(8, int(round(core_len * 0.2)))
    target_hole_len = min(target_hole_len, 96, core_len - 2)
    if target_hole_len <= 0:
        return None

    hole_bounds = _choose_infill_hole_bounds(core, target_hole_tokens=target_hole_len, rng=rng)
    if hole_bounds is None:
        return None
    hole_start_core, hole_end_core = hole_bounds
    hole_start = 1 + hole_start_core
    hole_end = 1 + hole_end_core

    prefix = seq[:hole_start]
    # suffix 不含 EOS，要求模型先生成 middle，再生成 EOS 结束。
    suffix = seq[hole_end:-1]
    prompt_tokens = [*prefix, "FIM_HOLE", *suffix, "FIM_MID"]
    if len(prompt_tokens) >= max_positions:
        return None
    return prompt_tokens, prefix, len(suffix)


def _generate_middle_tokens(
    model,
    torch_mod,
    prompt_tokens: list[str],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    grammar_fsm,
    prefix_tokens: list[str],
    suffix_tokens: list[str],
    device,
    use_amp: bool,
    amp_dtype,
    autocast_context_fn,
    max_positions: int,
    max_new_tokens: int,
) -> tuple[list[str], bool, dict[str, float | int]]:
    """基于 prompt 执行贪心解码，返回 middle token 与是否遇到 EOS。"""
    from src.decoding import select_masked_argmax

    prompt_ids: list[int] = []
    for token in prompt_tokens:
        token_id = token_to_id.get(token)
        if token_id is None:
            return [], False, {
                "step_count": 0,
                "illegal_top1_count": 0,
                "mask_intervention_count": 0,
                "legal_mass_sum": 0.0,
                "dead_end_count": 0,
            }
        prompt_ids.append(token_id)

    input_ids = torch_mod.tensor([prompt_ids], dtype=torch_mod.long, device=device)
    middle_tokens: list[str] = []
    reached_eos = False
    stats: dict[str, float | int] = {
        "step_count": 0,
        "illegal_top1_count": 0,
        "mask_intervention_count": 0,
        "legal_mass_sum": 0.0,
        "dead_end_count": 0,
    }

    fsm_state = None
    compatible_states: set[str] | None = None
    if grammar_fsm is not None:
        fsm_state = grammar_fsm.state_after_prefix_tokens(prefix_tokens)
        compatible_states = grammar_fsm.compatible_states_for_suffix_tokens(suffix_tokens)
        if fsm_state is None or not compatible_states:
            stats["dead_end_count"] = 1
            return middle_tokens, False, stats

    max_can_generate = max(0, min(max_new_tokens, max_positions - int(input_ids.shape[1])))
    if max_can_generate <= 0:
        return middle_tokens, reached_eos, stats

    with torch_mod.no_grad():
        for _ in range(max_can_generate):
            with autocast_context_fn(
                torch_mod=torch_mod,
                use_amp=use_amp,
                device_type=device.type,
                amp_dtype=amp_dtype,
            ):
                outputs = model(input_ids=input_ids, return_dict=True)
            step_logits = outputs.logits[0, -1, :]
            if grammar_fsm is None:
                next_id = int(torch_mod.argmax(step_logits, dim=-1).item())
            else:
                allowed_ids: list[int] = []
                for token_id in grammar_fsm.allowed_token_ids(fsm_state):
                    if token_id == grammar_fsm.eos_id:
                        continue
                    next_state = grammar_fsm.transition(fsm_state, token_id)
                    if next_state is not None and next_state in compatible_states:
                        allowed_ids.append(token_id)
                if fsm_state in compatible_states:
                    allowed_ids.append(grammar_fsm.eos_id)

                decision = select_masked_argmax(step_logits, allowed_ids)
                stats["step_count"] = int(stats["step_count"]) + 1
                stats["legal_mass_sum"] = float(stats["legal_mass_sum"]) + float(decision.legal_mass)
                if not decision.raw_top1_is_legal:
                    stats["illegal_top1_count"] = int(stats["illegal_top1_count"]) + 1
                if (
                    decision.raw_top1_id is not None
                    and decision.next_id is not None
                    and decision.raw_top1_id != decision.next_id
                ):
                    stats["mask_intervention_count"] = int(stats["mask_intervention_count"]) + 1
                if decision.next_id is None:
                    stats["dead_end_count"] = int(stats["dead_end_count"]) + 1
                    return middle_tokens, False, stats
                next_id = int(decision.next_id)

            if next_id < 0 or next_id >= len(id_to_token):
                return middle_tokens, False, stats

            next_token = id_to_token[next_id]
            if next_token == "EOS":
                reached_eos = True
                break
            if grammar_fsm is not None:
                next_state = grammar_fsm.transition(fsm_state, next_id)
                if next_state is None or next_state not in compatible_states:
                    stats["dead_end_count"] = int(stats["dead_end_count"]) + 1
                    return middle_tokens, False, stats
                fsm_state = next_state
            middle_tokens.append(next_token)

            next_ids = torch_mod.tensor([[next_id]], dtype=torch_mod.long, device=device)
            input_ids = torch_mod.cat([input_ids, next_ids], dim=1)
            if int(input_ids.shape[1]) >= max_positions:
                break

    return middle_tokens, reached_eos, stats


def _build_infilling_trace(
    *,
    prefix_tokens: list[str],
    suffix_tokens: list[str],
    generated_middle_tokens: list[str],
    reached_eos: bool,
    prompt_tokens: list[str],
    source_tokens: list[str],
    grammar_fsm,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured record for one infilling decode."""
    reconstructed = [*prefix_tokens, *generated_middle_tokens, *suffix_tokens, "EOS"]
    is_valid, syntax_reason = grammar_fsm.inspect_complete_tokens(reconstructed)
    failure_reason = ("ok" if is_valid else "syntax_invalid")

    record = {
        "failure_reason": failure_reason,
        "syntax_reason": syntax_reason,
        "generated_middle_len": len(generated_middle_tokens),
        "reconstructed_len": len(reconstructed),
        "reached_eos": reached_eos,
        "is_structurally_valid": is_valid,
        "prompt_len": len(prompt_tokens),
        "prefix_len": len(prefix_tokens),
        "suffix_len": len(suffix_tokens),
        "prompt_tokens": list(prompt_tokens),
        "prefix_tokens": list(prefix_tokens),
        "generated_middle_tokens": list(generated_middle_tokens),
        "suffix_tokens": list(suffix_tokens),
        "reconstructed_tokens": list(reconstructed),
        "source_tokens": list(source_tokens),
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def _evaluate_structural_validity(
    model,
    torch_mod,
    eval_rows: list[list[str]],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    grammar_fsm,
    device,
    use_amp: bool,
    amp_dtype,
    autocast_context_fn,
    max_positions: int,
    num_samples: int,
    max_new_tokens: int,
    rng: random.Random,
    debug_invalid_samples: int = 0,
) -> dict[str, Any]:
    """
    评估结构合法率。

    流程：
    1) 从 eval 样本随机构造 infilling prompt；
    2) 让模型生成 middle；
    3) 重建为完整序列并做结构语法校验。
    """
    if num_samples <= 0 or not eval_rows:
        return {
            "raw_structural_validity_rate": float("nan"),
            "raw_structural_valid_count": 0,
            "raw_structural_total_count": 0,
            "raw_eos_reached_count": 0,
            "raw_syntax_pass_count": 0,
            "raw_invalid_reason_counts": {},
            "raw_invalid_syntax_reason_counts": {},
            "fsm_structural_validity_rate": float("nan"),
            "fsm_structural_valid_count": 0,
            "fsm_structural_total_count": 0,
            "fsm_eos_reached_count": 0,
            "fsm_syntax_pass_count": 0,
            "fsm_invalid_reason_counts": {},
            "fsm_invalid_syntax_reason_counts": {},
            "fsm_decoding_step_count": 0,
            "fsm_illegal_top1_count": 0,
            "fsm_mask_intervention_count": 0,
            "fsm_legal_mass_mean": float("nan"),
            "fsm_dead_end_count": 0,
            "invalid_samples": [],
        }

    raw_valid_count = 0
    attempted = 0
    raw_eos_hits = 0
    raw_syntax_hits = 0
    fsm_valid_count = 0
    fsm_eos_hits = 0
    fsm_syntax_hits = 0
    invalid_samples: list[dict[str, Any]] = []
    raw_invalid_reason_counts: Counter[str] = Counter()
    raw_invalid_syntax_reason_counts: Counter[str] = Counter()
    fsm_invalid_reason_counts: Counter[str] = Counter()
    fsm_invalid_syntax_reason_counts: Counter[str] = Counter()
    fsm_step_count = 0
    fsm_illegal_top1_count = 0
    fsm_mask_intervention_count = 0
    fsm_legal_mass_sum = 0.0
    fsm_dead_end_count = 0

    retries = 0
    max_retries = max(num_samples * 10, 100)
    while attempted < num_samples and retries < max_retries:
        retries += 1
        source_tokens = eval_rows[rng.randrange(len(eval_rows))]
        built = _build_infilling_prompt(source_tokens=source_tokens, max_positions=max_positions, rng=rng)
        if built is None:
            continue

        prompt_tokens, prefix, suffix_len = built
        suffix = prompt_tokens[len(prefix) + 1 : len(prompt_tokens) - 1]
        if len(suffix) != suffix_len:
            suffix = suffix[:suffix_len]
        dyn_max_new = min(max_new_tokens, max(0, max_positions - len(prompt_tokens)))
        # 生成预算受“剩余上下文长度”约束，避免越界。
        dyn_max_new = min(max_new_tokens, max(0, max_positions - len(prompt_tokens)))
        if dyn_max_new <= 0:
            continue
        raw_middle_tokens, raw_reached_eos, _ = _generate_middle_tokens(
            model=model,
            torch_mod=torch_mod,
            prompt_tokens=prompt_tokens,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            grammar_fsm=None,
            prefix_tokens=prefix,
            suffix_tokens=suffix,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            autocast_context_fn=autocast_context_fn,
            max_positions=max_positions,
            max_new_tokens=dyn_max_new,
        )
        fsm_middle_tokens, fsm_reached_eos, fsm_stats = _generate_middle_tokens(
            model=model,
            torch_mod=torch_mod,
            prompt_tokens=prompt_tokens,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            grammar_fsm=grammar_fsm,
            prefix_tokens=prefix,
            suffix_tokens=suffix,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            autocast_context_fn=autocast_context_fn,
            max_positions=max_positions,
            max_new_tokens=dyn_max_new,
        )

        # 从 prompt 中提取 suffix（去掉 FIM_HOLE 与 FIM_MID）。
        suffix = prompt_tokens[len(prefix) + 1 : len(prompt_tokens) - 1]
        if len(suffix) != suffix_len:
            suffix = suffix[:suffix_len]

        raw_record = _build_infilling_trace(
            prefix_tokens=prefix,
            suffix_tokens=suffix,
            generated_middle_tokens=raw_middle_tokens,
            reached_eos=raw_reached_eos,
            prompt_tokens=prompt_tokens,
            source_tokens=source_tokens,
            grammar_fsm=grammar_fsm,
        )
        fsm_record = _build_infilling_trace(
            prefix_tokens=prefix,
            suffix_tokens=suffix,
            generated_middle_tokens=fsm_middle_tokens,
            reached_eos=fsm_reached_eos,
            prompt_tokens=prompt_tokens,
            source_tokens=source_tokens,
            grammar_fsm=grammar_fsm,
            extra_fields={
                "decoding_step_count": int(fsm_stats["step_count"]),
                "illegal_top1_count": int(fsm_stats["illegal_top1_count"]),
                "mask_intervention_count": int(fsm_stats["mask_intervention_count"]),
                "legal_mass_mean": (
                    float(fsm_stats["legal_mass_sum"]) / float(fsm_stats["step_count"])
                    if int(fsm_stats["step_count"]) > 0
                    else float("nan")
                ),
                "dead_end_count": int(fsm_stats["dead_end_count"]),
            },
        )

        if raw_record["reached_eos"]:
            raw_eos_hits += 1
        if raw_record["is_structurally_valid"]:
            raw_syntax_hits += 1
            raw_valid_count += 1
        else:
            raw_invalid_reason_counts[str(raw_record["failure_reason"])] += 1
            raw_invalid_syntax_reason_counts[str(raw_record["syntax_reason"])] += 1

        if fsm_record["reached_eos"]:
            fsm_eos_hits += 1
        if fsm_record["is_structurally_valid"]:
            fsm_syntax_hits += 1
            fsm_valid_count += 1
        else:
            fsm_invalid_reason_counts[str(fsm_record["failure_reason"])] += 1
            fsm_invalid_syntax_reason_counts[str(fsm_record["syntax_reason"])] += 1

        fsm_step_count += int(fsm_stats["step_count"])
        fsm_illegal_top1_count += int(fsm_stats["illegal_top1_count"])
        fsm_mask_intervention_count += int(fsm_stats["mask_intervention_count"])
        fsm_legal_mass_sum += float(fsm_stats["legal_mass_sum"])
        fsm_dead_end_count += int(fsm_stats["dead_end_count"])

        if (
            (not raw_record["is_structurally_valid"]) or (not fsm_record["is_structurally_valid"])
        ) and len(invalid_samples) < debug_invalid_samples:
            invalid_samples.append(
                {
                    "attempt_index": attempted + 1,
                    "prompt_len": len(prompt_tokens),
                    "prefix_len": len(prefix),
                    "suffix_len": len(suffix),
                    "prompt_tokens": list(prompt_tokens),
                    "prefix_tokens": list(prefix),
                    "suffix_tokens": list(suffix),
                    "source_tokens": list(source_tokens),
                    "raw": raw_record,
                    "fsm": fsm_record,
                }
            )
        attempted += 1

    return {
        "raw_structural_validity_rate": ((raw_valid_count / attempted) if attempted else float("nan")),
        "raw_structural_valid_count": raw_valid_count,
        "raw_structural_total_count": attempted,
        "raw_eos_reached_count": raw_eos_hits,
        "raw_syntax_pass_count": raw_syntax_hits,
        "raw_invalid_reason_counts": dict(raw_invalid_reason_counts),
        "raw_invalid_syntax_reason_counts": dict(raw_invalid_syntax_reason_counts),
        "fsm_structural_validity_rate": ((fsm_valid_count / attempted) if attempted else float("nan")),
        "fsm_structural_valid_count": fsm_valid_count,
        "fsm_structural_total_count": attempted,
        "fsm_eos_reached_count": fsm_eos_hits,
        "fsm_syntax_pass_count": fsm_syntax_hits,
        "fsm_invalid_reason_counts": dict(fsm_invalid_reason_counts),
        "fsm_invalid_syntax_reason_counts": dict(fsm_invalid_syntax_reason_counts),
        "fsm_decoding_step_count": fsm_step_count,
        "fsm_illegal_top1_count": fsm_illegal_top1_count,
        "fsm_mask_intervention_count": fsm_mask_intervention_count,
        "fsm_legal_mass_mean": ((fsm_legal_mass_sum / fsm_step_count) if fsm_step_count else float("nan")),
        "fsm_dead_end_count": fsm_dead_end_count,
        "invalid_samples": invalid_samples,
    }


def _resolve_report_path(run_id: str, output_path: Path | None, project_root: Path) -> Path:
    """解析最终报告输出路径。"""
    if output_path is not None:
        return output_path if output_path.is_absolute() else (project_root / output_path)
    return project_root / "outputs" / "reports" / "eval_infilling" / f"{run_id}.json"


def main() -> None:
    """
    程序入口：按 checkpoint 逐个评估，并输出结构化 JSON 报告。
    """
    project_root = _ensure_project_root_on_path()
    os.chdir(project_root)
    args = _parse_args()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else (project_root / args.checkpoint_dir)
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    run_id = args.run_id if args.run_id else checkpoint_dir.name
    report_path = _resolve_report_path(run_id=run_id, output_path=args.output_path, project_root=project_root).resolve()
    invalid_samples_path = report_path.with_name(f"{report_path.stem}.invalid_samples.json")

    # 懒加载 torch，确保错误信息更可读，并减少无关场景启动开销。
    from src.utils.config_io import dump_json_file
    from src.utils.config_io import dump_json_file
    from src.utils.report_plots import write_eval_report_plot
    from src.utils.torch_utils import lazy_import_torch, resolve_torch_device

    from src.decoding import TuneFlowGrammarFSM

    torch = lazy_import_torch()
    from src.model.configuration import DecoderConfig
    from src.model.modeling import DecoderForCausalLM
    from src.training.train_base import (
        TokenBinDataset,
        _autocast_context,
        _load_checkpoint,
        _resolve_precision,
    )

    device = resolve_torch_device(torch, args.device)
    precision_name, use_amp, amp_dtype, _ = _resolve_precision(torch_mod=torch, requested=args.precision, device=device)

    checkpoints = _discover_checkpoints(
        checkpoint_dir=checkpoint_dir,
        limit=args.limit_checkpoints,
        policy=args.checkpoint_policy,
        sample_count=args.sample_count,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No *.pt checkpoints found under: {checkpoint_dir}")

    # 1) 准备验证集采样器（用于 valid_loss / ppl）。
    valid_idx = args.valid_idx if args.valid_idx.is_absolute() else (project_root / args.valid_idx)
    valid_idx = args.valid_idx if args.valid_idx.is_absolute() else (project_root / args.valid_idx)
    valid_bin = None
    if args.valid_bin is not None:
        valid_bin = args.valid_bin if args.valid_bin.is_absolute() else (project_root / args.valid_bin)
    valid_dataset = TokenBinDataset(valid_idx.resolve(), None if valid_bin is None else valid_bin.resolve())

    # 2) 准备 infilling 评估样本与词表。
    eval_tok_path = args.eval_tok if args.eval_tok.is_absolute() else (project_root / args.eval_tok)
    eval_tok_path = args.eval_tok if args.eval_tok.is_absolute() else (project_root / args.eval_tok)
    eval_rows = _load_eval_tok_lines(eval_tok_path.resolve())
    if not eval_rows:
        raise ValueError(f"No eval samples found in {eval_tok_path}")

    vocab_path = args.vocab_path if args.vocab_path.is_absolute() else (project_root / args.vocab_path)
    vocab, id_to_token = _load_vocab(vocab_path.resolve())
    grammar_fsm = TuneFlowGrammarFSM.from_vocab(vocab)
    metrics_path = None
    if args.metrics_path is not None:
        metrics_path = args.metrics_path if args.metrics_path.is_absolute() else (project_root / args.metrics_path)
    metrics_path = _resolve_metrics_path(checkpoint_dir, metrics_path)
    valid_loss_by_step = _load_valid_loss_by_step(metrics_path)

    results: list[dict[str, Any]] = []
    invalid_debug_results: list[dict[str, Any]] = []
    started_at = time.time()
    print(
        f"[eval_infilling] run_id={run_id} checkpoints={len(checkpoints)} "
        f"device={device} precision={precision_name} checkpoint_policy={args.checkpoint_policy} "
        f"valid_loss_source={args.valid_loss_source}"
    )

    try:
        for index, ckpt_path in enumerate(checkpoints, start=1):
            print(f"[eval_infilling] ({index}/{len(checkpoints)}) checkpoint={ckpt_path.name}")
            ckpt_payload = _load_checkpoint(torch, ckpt_path)

            # 优先使用 checkpoint 内部保存的模型配置，保证评估与训练一致。
            ckpt_model_cfg = ckpt_payload.get("model_config")
            ckpt_model_cfg = ckpt_payload.get("model_config")
            if isinstance(ckpt_model_cfg, dict):
                config = DecoderConfig.from_dict(ckpt_model_cfg)
            else:
                fallback_model_cfg = args.model_config if args.model_config.is_absolute() else (project_root / args.model_config)
                config = DecoderConfig.from_yaml(fallback_model_cfg.resolve())

            model = DecoderForCausalLM(config).to(device)
            model.load_state_dict(ckpt_payload["model_state_dict"])
            model.eval()

            # 每个 checkpoint 的有效 seq_len 受对应模型最大位置长度约束。
            eval_seq_len = min(max(1, args.seq_len), int(config.max_position_embeddings))
            eval_seq_len = min(max(1, args.seq_len), int(config.max_position_embeddings))
            valid_rng = random.Random(args.seed)
            infill_rng = random.Random(args.seed + 1)

            step = int(ckpt_payload.get("step", -1))
            valid_loss, valid_loss_source = _resolve_valid_loss(
                step=step,
                args=args,
                valid_loss_by_step=valid_loss_by_step,
                recompute_fn=lambda: _evaluate_valid_loss(
                    model=model,
                    dataset=valid_dataset,
                    torch_mod=torch,
                    rng=valid_rng,
                    device=device,
                    id_to_token=id_to_token,
                    batch_size=args.batch_size,
                    seq_len=eval_seq_len,
                    eval_batches=args.eval_batches,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    autocast_context_fn=_autocast_context,
                ),
            )
            ppl = _safe_perplexity(valid_loss)
            infilling_metrics = _evaluate_structural_validity(
                model=model,
                torch_mod=torch,
                eval_rows=eval_rows,
                token_to_id=vocab,
                id_to_token=id_to_token,
                grammar_fsm=grammar_fsm,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                autocast_context_fn=_autocast_context,
                max_positions=int(config.max_position_embeddings),
                num_samples=args.num_infilling_samples,
                max_new_tokens=args.max_new_tokens,
                rng=infill_rng,
                debug_invalid_samples=max(0, int(args.debug_invalid_samples)),
            )
            struct_rate = infilling_metrics["raw_structural_validity_rate"]
            struct_valid = infilling_metrics["raw_structural_valid_count"]
            struct_total = infilling_metrics["raw_structural_total_count"]
            eos_hits = infilling_metrics["raw_eos_reached_count"]
            syntax_hits = infilling_metrics["raw_syntax_pass_count"]
            invalid_samples = infilling_metrics["invalid_samples"]
            invalid_reason_counts = infilling_metrics["raw_invalid_reason_counts"]
            invalid_syntax_reason_counts = infilling_metrics["raw_invalid_syntax_reason_counts"]
            fsm_struct_rate = infilling_metrics["fsm_structural_validity_rate"]
            fsm_struct_valid = infilling_metrics["fsm_structural_valid_count"]
            fsm_struct_total = infilling_metrics["fsm_structural_total_count"]
            fsm_eos_hits = infilling_metrics["fsm_eos_reached_count"]
            fsm_syntax_hits = infilling_metrics["fsm_syntax_pass_count"]
            fsm_invalid_reason_counts = infilling_metrics["fsm_invalid_reason_counts"]
            fsm_invalid_syntax_reason_counts = infilling_metrics["fsm_invalid_syntax_reason_counts"]

            result = {
                "checkpoint_name": ckpt_path.name,
                "checkpoint_path": str(ckpt_path),
                "step": step,
                "valid_loss": valid_loss,
                "valid_loss_source": valid_loss_source,
                "ppl": ppl,
                "structural_validity_rate": struct_rate,
                "structural_valid_count": struct_valid,
                "structural_total_count": struct_total,
                "eos_reached_count": eos_hits,
                "eos_reached_rate": ((eos_hits / struct_total) if struct_total else float("nan")),
                "syntax_pass_count": syntax_hits,
                "syntax_pass_rate": ((syntax_hits / struct_total) if struct_total else float("nan")),
                "fsm_structural_validity_rate": fsm_struct_rate,
                "fsm_structural_valid_count": fsm_struct_valid,
                "fsm_structural_total_count": fsm_struct_total,
                "fsm_eos_reached_count": fsm_eos_hits,
                "fsm_eos_reached_rate": ((fsm_eos_hits / fsm_struct_total) if fsm_struct_total else float("nan")),
                "fsm_syntax_pass_count": fsm_syntax_hits,
                "fsm_syntax_pass_rate": ((fsm_syntax_hits / fsm_struct_total) if fsm_struct_total else float("nan")),
                "fsm_decoding_step_count": infilling_metrics["fsm_decoding_step_count"],
                "fsm_illegal_top1_count": infilling_metrics["fsm_illegal_top1_count"],
                "fsm_illegal_top1_rate": (
                    (infilling_metrics["fsm_illegal_top1_count"] / infilling_metrics["fsm_decoding_step_count"])
                    if infilling_metrics["fsm_decoding_step_count"]
                    else float("nan")
                ),
                "fsm_mask_intervention_count": infilling_metrics["fsm_mask_intervention_count"],
                "fsm_mask_intervention_rate": (
                    (
                        infilling_metrics["fsm_mask_intervention_count"]
                        / infilling_metrics["fsm_decoding_step_count"]
                    )
                    if infilling_metrics["fsm_decoding_step_count"]
                    else float("nan")
                ),
                "fsm_legal_mass_mean": infilling_metrics["fsm_legal_mass_mean"],
                "fsm_dead_end_count": infilling_metrics["fsm_dead_end_count"],
                "seq_len": eval_seq_len,
                "eval_batches": args.eval_batches,
                "num_infilling_samples": args.num_infilling_samples,
                "invalid_debug_sample_count": len(invalid_samples),
                "invalid_reason_counts": invalid_reason_counts,
                "invalid_syntax_reason_counts": invalid_syntax_reason_counts,
                "fsm_invalid_reason_counts": fsm_invalid_reason_counts,
                "fsm_invalid_syntax_reason_counts": fsm_invalid_syntax_reason_counts,
            }
            results.append(result)
            invalid_reason_text = ", ".join(
                f"{key}={value}" for key, value in sorted(invalid_reason_counts.items())
            ) or "none"
            invalid_syntax_text = ", ".join(
                f"{key}={value}" for key, value in sorted(invalid_syntax_reason_counts.items())
            ) or "none"
            fsm_invalid_reason_text = ", ".join(
                f"{key}={value}" for key, value in sorted(fsm_invalid_reason_counts.items())
            ) or "none"
            fsm_invalid_syntax_text = ", ".join(
                f"{key}={value}" for key, value in sorted(fsm_invalid_syntax_reason_counts.items())
            ) or "none"
            print(
                f"[eval_infilling] step={result['step']} "
                f"valid_loss={valid_loss:.6f}({valid_loss_source}) ppl={ppl:.6f} "
                f"raw_struct={struct_rate:.4f} ({struct_valid}/{struct_total}) "
                f"fsm_struct={fsm_struct_rate:.4f} ({fsm_struct_valid}/{fsm_struct_total}) "
                f"raw_invalid={invalid_reason_text} "
                f"fsm_invalid={fsm_invalid_reason_text} "
                f"fsm_illegal_top1={result['fsm_illegal_top1_rate']:.4f}"
            )
            if invalid_syntax_reason_counts:
                print(f"[eval_infilling] step={result['step']} raw_invalid_syntax={invalid_syntax_text}")
            if fsm_invalid_syntax_reason_counts:
                print(f"[eval_infilling] step={result['step']} fsm_invalid_syntax={fsm_invalid_syntax_text}")
            if invalid_samples:
                invalid_debug_results.append(
                    {
                        "checkpoint_name": ckpt_path.name,
                        "checkpoint_path": str(ckpt_path),
                        "step": step,
                        "invalid_reason_counts": invalid_reason_counts,
                        "invalid_syntax_reason_counts": invalid_syntax_reason_counts,
                        "fsm_invalid_reason_counts": fsm_invalid_reason_counts,
                        "fsm_invalid_syntax_reason_counts": fsm_invalid_syntax_reason_counts,
                        "samples": invalid_samples,
                    }
                )
                _print_invalid_sample_preview(
                    step=step,
                    checkpoint_name=ckpt_path.name,
                    invalid_samples=invalid_samples,
                    preview_limit=max(1, int(args.debug_preview_tokens)),
                )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
                # 防止多 checkpoint 连续评估时显存积压。
                torch.cuda.empty_cache()
    finally:
        valid_dataset.close()

    # 汇总 run 级别统计，便于快速看趋势与最优点。
    finite_losses = [r["valid_loss"] for r in results if math.isfinite(float(r["valid_loss"]))]
    finite_losses = [r["valid_loss"] for r in results if math.isfinite(float(r["valid_loss"]))]
    finite_struct = [r["structural_validity_rate"] for r in results if math.isfinite(float(r["structural_validity_rate"]))]
    finite_fsm_struct = [
        r["fsm_structural_validity_rate"]
        for r in results
        if math.isfinite(float(r["fsm_structural_validity_rate"]))
    ]
    summary_invalid_reason_counts: Counter[str] = Counter()
    summary_invalid_syntax_reason_counts: Counter[str] = Counter()
    summary_fsm_invalid_reason_counts: Counter[str] = Counter()
    summary_fsm_invalid_syntax_reason_counts: Counter[str] = Counter()
    for result in results:
        summary_invalid_reason_counts.update(result.get("invalid_reason_counts", {}))
        summary_invalid_syntax_reason_counts.update(result.get("invalid_syntax_reason_counts", {}))
        summary_fsm_invalid_reason_counts.update(result.get("fsm_invalid_reason_counts", {}))
        summary_fsm_invalid_syntax_reason_counts.update(result.get("fsm_invalid_syntax_reason_counts", {}))
    summary = {
        "checkpoint_count": len(results),
        "best_valid_loss": (min(finite_losses) if finite_losses else float("nan")),
        "best_structural_validity_rate": (max(finite_struct) if finite_struct else float("nan")),
        "best_fsm_structural_validity_rate": (max(finite_fsm_struct) if finite_fsm_struct else float("nan")),
        "elapsed_sec": max(0.0, time.time() - started_at),
        "invalid_reason_counts": dict(summary_invalid_reason_counts),
        "invalid_syntax_reason_counts": dict(summary_invalid_syntax_reason_counts),
        "fsm_invalid_reason_counts": dict(summary_fsm_invalid_reason_counts),
        "fsm_invalid_syntax_reason_counts": dict(summary_fsm_invalid_syntax_reason_counts),
    }

    report = {
        "run_id": run_id,
        "created_at": time.time(),
        "checkpoint_dir": str(checkpoint_dir),
        "output_path": str(report_path),
        "eval_config": {
            "device": args.device,
            "precision": args.precision,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "num_infilling_samples": args.num_infilling_samples,
            "max_new_tokens": args.max_new_tokens,
            "debug_invalid_samples": args.debug_invalid_samples,
            "debug_preview_tokens": args.debug_preview_tokens,
            "seed": args.seed,
            "checkpoint_policy": args.checkpoint_policy,
            "sample_count": args.sample_count,
            "valid_loss_source": args.valid_loss_source,
            "valid_idx": str(valid_idx.resolve()),
            "valid_bin": None if valid_bin is None else str(valid_bin.resolve()),
            "eval_tok": str(eval_tok_path.resolve()),
            "vocab_path": str(vocab_path.resolve()),
            "metrics_path": None if metrics_path is None else str(metrics_path),
        },
        "summary": summary,
        "results": results,
        "artifacts": {
            "plot_path": str(report_path.with_suffix(".png")),
            "invalid_samples_path": (str(invalid_samples_path) if args.debug_invalid_samples > 0 else None),
        },
    }

    if args.debug_invalid_samples > 0:
        dump_json_file(
            invalid_samples_path,
            {
                "run_id": run_id,
                "created_at": time.time(),
                "checkpoint_dir": str(checkpoint_dir),
                "report_path": str(report_path),
                "debug_invalid_samples": args.debug_invalid_samples,
                "results": invalid_debug_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    dump_json_file(report_path, report, ensure_ascii=False, indent=2)
    plot_path = write_eval_report_plot(
        report_path=report_path,
        report=report,
        title="Infilling Eval Report",
        metric_specs=[
            {"key": "valid_loss", "label": "Valid Loss", "color": "#2563eb"},
            {"key": "ppl", "label": "Perplexity", "color": "#dc2626"},
            {"key": "structural_validity_rate", "label": "Raw Structural Validity Rate", "color": "#059669", "percent": True},
            {"key": "fsm_structural_validity_rate", "label": "FSM Structural Validity Rate", "color": "#0f766e", "percent": True},
        ],
    )
    print(f"[eval_infilling] report -> {report_path}")
    if args.debug_invalid_samples > 0:
        print(f"[eval_infilling] invalid samples -> {invalid_samples_path}")
    print(f"[eval_infilling] plot -> {plot_path}")


if __name__ == "__main__":
    main()
