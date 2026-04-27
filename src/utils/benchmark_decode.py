"""TuneFlow benchmark 评估专用的解码适配与 trace 构建。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..inference import generate_continuation_tokens, generate_middle_tokens, load_vocab


_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    """按 step 优先、再按特殊别名顺序对 checkpoint 排序。"""
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


def sample_step_checkpoints(step_paths: list[Path], sample_count: int) -> list[Path]:
    """在保留首尾 checkpoint 的前提下，对 step checkpoint 做均匀抽样。"""
    if sample_count <= 0 or len(step_paths) <= sample_count:
        return step_paths
    if sample_count == 1:
        return [step_paths[-1]]

    indices = {
        round(index * (len(step_paths) - 1) / (sample_count - 1))
        for index in range(sample_count)
    }
    return [step_paths[index] for index in sorted(indices)]


def discover_checkpoints(
    checkpoint_dir: Path,
    limit: int | None,
    policy: str,
    sample_count: int,
    *,
    include_aliases: bool = False,
) -> list[Path]:
    """在 run 目录下发现并筛选 checkpoint。"""
    paths = sorted([path for path in checkpoint_dir.glob("*.pt") if path.is_file()], key=checkpoint_sort_key)
    step_paths = [path for path in paths if _STEP_RE.match(path.name)]
    extra_paths = [path for path in paths if not _STEP_RE.match(path.name)]
    if step_paths:
        paths = list(step_paths)
        if include_aliases:
            deduped: dict[str, Path] = {path.name: path for path in [*paths, *extra_paths]}
            paths = sorted(deduped.values(), key=checkpoint_sort_key)
    if policy == "sampled":
        sampled = sample_step_checkpoints(step_paths if step_paths else paths, sample_count)
        deduped: dict[str, Path] = {path.name: path for path in [*sampled, *(extra_paths if include_aliases else [])]}
        paths = sorted(deduped.values(), key=checkpoint_sort_key)
    if limit is not None:
        paths = paths[: max(0, limit)]
    return paths


def build_continuation_trace(
    *,
    prompt_tokens: list[str],
    target_tokens: list[str],
    generated_tokens: list[str],
    reached_eos: bool,
    source_tokens: list[str],
    grammar_fsm,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """为单条续写解码构建结构化记录。"""
    reconstructed = [*prompt_tokens, *generated_tokens, "EOS"] if reached_eos else [*prompt_tokens, *generated_tokens]
    appended_eos_tokens = [*prompt_tokens, *generated_tokens, "EOS"]
    append_eos_would_validate, append_eos_syntax_reason = grammar_fsm.inspect_complete_tokens(appended_eos_tokens)
    syntax_reason = "missing_eos"
    is_valid = False
    failure_reason = "missing_eos"
    if reached_eos:
        is_valid = append_eos_would_validate
        syntax_reason = append_eos_syntax_reason
        failure_reason = ("ok" if is_valid else "syntax_invalid")
    elif append_eos_would_validate:
        failure_reason = "missing_eos_only"
    else:
        syntax_reason = append_eos_syntax_reason
        failure_reason = "missing_eos_plus_syntax"

    target_prefixless = [token for token in target_tokens if token != "EOS"]
    first_token_match = (
        bool(generated_tokens and target_prefixless and generated_tokens[0] == target_prefixless[0])
        if target_prefixless
        else None
    )

    record = {
        "prompt_len": len(prompt_tokens),
        "target_len": len(target_tokens),
        "generated_len": len(generated_tokens),
        "reconstructed_len": len(reconstructed),
        "reached_eos": reached_eos,
        "is_structurally_valid": is_valid,
        "failure_reason": failure_reason,
        "syntax_reason": syntax_reason,
        "append_eos_would_validate": append_eos_would_validate,
        "append_eos_syntax_reason": append_eos_syntax_reason,
        "first_token_match": first_token_match,
        "generated_tokens": list(generated_tokens),
        "reconstructed_tokens": list(reconstructed),
        "target_tokens": list(target_tokens),
        "source_tokens": list(source_tokens),
    }
    if extra_fields:
        record.update(extra_fields)
    return record


def build_infilling_trace(
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
    """为单条补全解码构建结构化记录。"""
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
