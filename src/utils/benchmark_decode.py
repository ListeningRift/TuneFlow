"""TuneFlow benchmark 评估共用的解码辅助函数。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_STEP_RE = re.compile(r"^step_(\d+)\.pt$")
_SAFE_CLOSE_STATES = {
    "after_head_tempo",
    "after_bar",
    "after_bar_tempo",
    "after_vel",
}


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


def load_vocab(vocab_path: Path) -> tuple[dict[str, int], list[str]]:
    """加载 tokenizer 词表，并构建 token/id 双向映射。"""
    import json

    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    token_to_id = payload.get("token_to_id")
    if not isinstance(token_to_id, dict):
        raise ValueError(f"Invalid vocab file: missing token_to_id in {vocab_path}")
    token_to_id = {str(key): int(value) for key, value in token_to_id.items()}

    id_to_token_raw = payload.get("id_to_token")
    if isinstance(id_to_token_raw, list) and id_to_token_raw:
        id_to_token = [str(item) for item in id_to_token_raw]
    else:
        max_id = max(token_to_id.values()) if token_to_id else -1
        id_to_token = ["<UNK>"] * (max_id + 1)
        for token, idx in token_to_id.items():
            if 0 <= idx < len(id_to_token):
                id_to_token[idx] = token
    return token_to_id, id_to_token


def continuation_eos_bias(*, generated_len: int, max_can_generate: int, fsm_state: str | None) -> float:
    """为 FSM 续写解码提供与长度相关的 EOS 偏置。"""
    if max_can_generate <= 0 or generated_len < 0:
        return 0.0

    progress = float(generated_len) / float(max_can_generate)
    remaining_budget = max_can_generate - generated_len
    bias = 0.0
    if progress >= 0.50:
        bias += 0.35
    if progress >= 0.70:
        bias += 0.75
    if progress >= 0.85:
        bias += 1.50
    if remaining_budget <= 8:
        bias += 0.75
    if remaining_budget <= 4:
        bias += 1.00
    if fsm_state in _SAFE_CLOSE_STATES and generated_len >= 8:
        bias += 0.50
    return bias


def should_force_safe_boundary_stop(*, generated_len: int, max_can_generate: int, fsm_state: str | None) -> bool:
    """在预算将耗尽时，是否应在安全边界处确定性停止。"""
    if fsm_state not in _SAFE_CLOSE_STATES:
        return False
    if generated_len <= 0 or max_can_generate <= 0:
        return False
    remaining_budget = max_can_generate - generated_len
    if remaining_budget > max(4, min(8, max_can_generate // 8)):
        return False
    return generated_len >= max(8, min(24, max_can_generate // 4))


def _forward_decode_step(
    *,
    model,
    torch_mod,
    input_ids,
    past_key_values,
    device,
    use_amp: bool,
    amp_dtype,
    autocast_context_fn,
):
    with autocast_context_fn(
        torch_mod=torch_mod,
        use_amp=use_amp,
        device_type=device.type,
        amp_dtype=amp_dtype,
    ):
        return model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )


def _fallback_bridgeable_states_for_suffix_tokens(
    grammar_fsm,
    *,
    suffix_tokens: list[str],
    start_state: str | None,
) -> set[str]:
    """为缺少桥接辅助接口的旧版/简化 FSM 近似推断可桥接状态集合。"""
    if start_state is None:
        return set()

    compatible_states = grammar_fsm.compatible_states_for_suffix_tokens(suffix_tokens)
    if not compatible_states:
        return set()

    # 先从当前起始状态出发，只沿着非 EOS 转移做一次可达搜索，
    # 避免把理论上兼容、但实际上从当前前缀根本走不到的状态误算进去。
    reachable_states: set[str] = {start_state}
    frontier = [start_state]
    while frontier:
        state = frontier.pop()
        for token_id in grammar_fsm.allowed_token_ids(state):
            if token_id == grammar_fsm.eos_id:
                continue
            next_state = grammar_fsm.transition(state, token_id)
            if next_state is None or next_state in reachable_states:
                continue
            reachable_states.add(next_state)
            frontier.append(next_state)

    # 先取“既兼容 suffix，又从当前前缀能到达”的状态作为种子，
    # 再反向扩一层，把能够通过若干个非 EOS token 走到这些种子状态的前置状态也纳入 bridgeable。
    bridgeable_states = {state for state in compatible_states if state in reachable_states}
    changed = True
    while changed:
        changed = False
        for state in list(reachable_states):
            if state in bridgeable_states:
                continue
            for token_id in grammar_fsm.allowed_token_ids(state):
                if token_id == grammar_fsm.eos_id:
                    continue
                next_state = grammar_fsm.transition(state, token_id)
                if next_state in bridgeable_states:
                    bridgeable_states.add(state)
                    changed = True
                    break
    return bridgeable_states


def generate_continuation_tokens(
    *,
    model,
    torch_mod,
    prompt_tokens: list[str],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    grammar_fsm,
    device,
    use_amp: bool,
    amp_dtype,
    autocast_context_fn,
    max_positions: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[list[str], bool, dict[str, float | int]]:
    """使用温度和 top-p 采样执行续写解码，可选启用语法约束。"""
    from src.decoding import select_masked_token, select_token

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
                "eos_bias_step_count": 0,
                "safe_boundary_stop_count": 0,
                "auto_close_count": 0,
            }
        prompt_ids.append(token_id)

    input_ids = torch_mod.tensor([prompt_ids], dtype=torch_mod.long, device=device)
    generated_tokens: list[str] = []
    reached_eos = False
    past_key_values = None
    stats: dict[str, float | int] = {
        "step_count": 0,
        "illegal_top1_count": 0,
        "mask_intervention_count": 0,
        "legal_mass_sum": 0.0,
        "dead_end_count": 0,
        "auto_close_count": 0,
        "eos_bias_step_count": 0,
        "safe_boundary_stop_count": 0,
    }

    fsm_state = None
    if grammar_fsm is not None:
        fsm_state = grammar_fsm.state_after_prefix_ids(prompt_ids)
        if fsm_state is None:
            stats["dead_end_count"] = 1
            return generated_tokens, False, stats

    max_can_generate = max(0, min(max_new_tokens, max_positions - int(input_ids.shape[1])))
    if max_can_generate <= 0:
        if grammar_fsm is not None and grammar_fsm.eos_id in grammar_fsm.allowed_token_ids(fsm_state):
            reached_eos = True
            stats["auto_close_count"] = 1
        return generated_tokens, reached_eos, stats

    stopped_without_eos = False
    with torch_mod.no_grad():
        for _ in range(max_can_generate):
            if (
                grammar_fsm is not None
                and grammar_fsm.eos_id in grammar_fsm.allowed_token_ids(fsm_state)
                and should_force_safe_boundary_stop(
                    generated_len=len(generated_tokens),
                    max_can_generate=max_can_generate,
                    fsm_state=fsm_state,
                )
            ):
                reached_eos = True
                stats["safe_boundary_stop_count"] = int(stats["safe_boundary_stop_count"]) + 1
                break

            outputs = _forward_decode_step(
                model=model,
                torch_mod=torch_mod,
                input_ids=input_ids,
                past_key_values=past_key_values,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                autocast_context_fn=autocast_context_fn,
            )
            past_key_values = outputs.past_key_values
            step_logits = outputs.logits[0, -1, :]
            if grammar_fsm is None:
                decision = select_token(step_logits, temperature=temperature, top_p=top_p)
                if decision.next_id is None:
                    return generated_tokens, False, stats
                next_id = int(decision.next_id)
            else:
                if grammar_fsm.eos_id in grammar_fsm.allowed_token_ids(fsm_state):
                    eos_bias = continuation_eos_bias(
                        generated_len=len(generated_tokens),
                        max_can_generate=max_can_generate,
                        fsm_state=fsm_state,
                    )
                    if eos_bias > 0.0:
                        step_logits = step_logits.clone()
                        step_logits[grammar_fsm.eos_id] = step_logits[grammar_fsm.eos_id] + float(eos_bias)
                        stats["eos_bias_step_count"] = int(stats["eos_bias_step_count"]) + 1
                allowed_ids = grammar_fsm.allowed_token_ids(fsm_state)
                decision = select_masked_token(
                    step_logits,
                    allowed_ids,
                    temperature=temperature,
                    top_p=top_p,
                )
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
                    return generated_tokens, False, stats
                next_id = int(decision.next_id)

            if next_id < 0 or next_id >= len(id_to_token):
                return generated_tokens, False, stats

            next_token = id_to_token[next_id]
            if grammar_fsm is not None:
                next_state = grammar_fsm.transition(fsm_state, next_id)
                if next_state is None:
                    stats["dead_end_count"] = int(stats["dead_end_count"]) + 1
                    return generated_tokens, False, stats
                fsm_state = next_state
            if next_token == "EOS":
                reached_eos = True
                break
            generated_tokens.append(next_token)

            next_ids = torch_mod.tensor([[next_id]], dtype=torch_mod.long, device=device)
            input_ids = next_ids if past_key_values is not None else torch_mod.cat([input_ids, next_ids], dim=1)
            if (len(prompt_ids) + len(generated_tokens)) >= max_positions:
                stopped_without_eos = True
                break
        else:
            if not reached_eos:
                stopped_without_eos = True

    if (
        grammar_fsm is not None
        and not reached_eos
        and stopped_without_eos
        and grammar_fsm.eos_id in grammar_fsm.allowed_token_ids(fsm_state)
    ):
        reached_eos = True
        stats["auto_close_count"] = 1

    return generated_tokens, reached_eos, stats


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


def generate_middle_tokens(
    *,
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
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[list[str], bool, dict[str, float | int]]:
    """使用温度和 top-p 采样执行中间补全解码，可选启用语法约束。"""
    from src.decoding import select_masked_token, select_token

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
    past_key_values = None
    stats: dict[str, float | int] = {
        "step_count": 0,
        "illegal_top1_count": 0,
        "mask_intervention_count": 0,
        "legal_mass_sum": 0.0,
        "dead_end_count": 0,
    }

    fsm_state = None
    compatible_states: set[str] | None = None
    bridgeable_states: set[str] | None = None
    if grammar_fsm is not None:
        fsm_state = grammar_fsm.state_after_prefix_tokens(prefix_tokens)
        compatible_states = grammar_fsm.compatible_states_for_suffix_tokens(suffix_tokens)
        if hasattr(grammar_fsm, "bridgeable_states_for_suffix_tokens"):
            bridgeable_states = grammar_fsm.bridgeable_states_for_suffix_tokens(suffix_tokens)
        else:
            bridgeable_states = _fallback_bridgeable_states_for_suffix_tokens(
                grammar_fsm,
                suffix_tokens=suffix_tokens,
                start_state=fsm_state,
            )
        if fsm_state is None or not compatible_states or not bridgeable_states or fsm_state not in bridgeable_states:
            stats["dead_end_count"] = 1
            return middle_tokens, False, stats

    max_can_generate = max(0, min(max_new_tokens, max_positions - int(input_ids.shape[1])))
    if max_can_generate <= 0:
        return middle_tokens, reached_eos, stats

    with torch_mod.no_grad():
        for _ in range(max_can_generate):
            outputs = _forward_decode_step(
                model=model,
                torch_mod=torch_mod,
                input_ids=input_ids,
                past_key_values=past_key_values,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                autocast_context_fn=autocast_context_fn,
            )
            past_key_values = outputs.past_key_values
            step_logits = outputs.logits[0, -1, :]
            if grammar_fsm is None:
                decision = select_token(step_logits, temperature=temperature, top_p=top_p)
                if decision.next_id is None:
                    return middle_tokens, False, stats
                next_id = int(decision.next_id)
            else:
                allowed_ids: list[int] = []
                for token_id in grammar_fsm.allowed_token_ids(fsm_state):
                    if token_id == grammar_fsm.eos_id:
                        continue
                    next_state = grammar_fsm.transition(fsm_state, token_id)
                    if next_state is not None and next_state in bridgeable_states:
                        allowed_ids.append(token_id)
                if fsm_state in compatible_states:
                    allowed_ids.append(grammar_fsm.eos_id)

                decision = select_masked_token(
                    step_logits,
                    allowed_ids,
                    temperature=temperature,
                    top_p=top_p,
                )
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
                if next_state is None or next_state not in bridgeable_states:
                    stats["dead_end_count"] = int(stats["dead_end_count"]) + 1
                    return middle_tokens, False, stats
                fsm_state = next_state
            middle_tokens.append(next_token)

            next_ids = torch_mod.tensor([[next_id]], dtype=torch_mod.long, device=device)
            input_ids = next_ids if past_key_values is not None else torch_mod.cat([input_ids, next_ids], dim=1)
            if (len(prompt_ids) + len(middle_tokens)) >= max_positions:
                break

    return middle_tokens, reached_eos, stats


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
