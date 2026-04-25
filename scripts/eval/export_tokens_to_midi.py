#!/usr/bin/env python
"""Export benchmark sample token sequences to MIDI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_ALLOWED_COMPLETE_FIELDS = {
    "fsm_reconstructed_tokens",
    "raw_reconstructed_tokens",
}
_COMPLETE_TO_FRAGMENT_FIELD = {
    "fsm_reconstructed_tokens": "fsm_output_tokens",
    "raw_reconstructed_tokens": "raw_output_tokens",
}
_TASK_TO_SUFFIX = {
    "continuation": "continuation",
    "infilling": "infilling",
}
_TASK_TO_TARGET_FIELD = {
    "continuation": "target_tokens",
    "infilling": "target_hole_tokens",
}


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 benchmark sample JSON 导出完整 token 序列为 MIDI。"
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="输入 benchmark sample JSON（continuation.json 或 infilling.json）。",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=None,
        help="可选：要导出的 case 索引；不传时导出全部 case。",
    )
    parser.add_argument(
        "--token-field",
        type=str,
        default="fsm_reconstructed_tokens",
        help="要导出的完整序列字段，默认 fsm_reconstructed_tokens。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出 MIDI 路径。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/tokenizer.yaml"),
        help="tokenizer YAML 配置路径。",
    )
    parser.add_argument(
        "--ticks-per-beat",
        type=int,
        default=480,
        help="导出 MIDI 的 ticks_per_beat，默认 480。",
    )
    return parser.parse_args(argv)


def _resolve_case_tokens(case: dict[str, Any], token_field: str) -> list[str]:
    if token_field not in _ALLOWED_COMPLETE_FIELDS:
        supported = ", ".join(sorted(_ALLOWED_COMPLETE_FIELDS))
        raise ValueError(
            f"`{token_field}` is not supported; only complete sequence fields are supported: {supported}"
        )
    tokens = case.get(token_field)
    if not isinstance(tokens, list):
        raise ValueError(f"`{token_field}` is missing or is not a token array")
    if not all(isinstance(item, str) and item.strip() for item in tokens):
        raise ValueError(f"`{token_field}` must be a non-empty token string array")
    return [str(item).strip() for item in tokens]


def _resolve_output_targets(output_path: Path, case_index: int | None, case_count: int) -> list[tuple[int, Path]]:
    if case_index is not None:
        return [(case_index, output_path)]

    if output_path.suffix:
        raise ValueError("when `--case-index` is omitted, `--output` must be a directory path")
    return [(index, output_path / f"{index}_full.mid") for index in range(case_count)]


def _normalize_sequence_tokens(tokens: list[str]) -> list[str]:
    normalized = [str(token).strip() for token in tokens if str(token).strip()]
    if normalized and normalized[-1] == "EOS":
        normalized = normalized[:-1]
    return normalized


def _build_structure_only_prefix(prompt_tokens: list[str], *, stop_at_hole: bool) -> list[str]:
    normalized = [str(token).strip() for token in prompt_tokens if str(token).strip()]
    if not normalized or normalized[0] != "BOS":
        raise ValueError("prompt_tokens must start with BOS")

    prefix: list[str] = ["BOS"]
    idx = 1
    if idx < len(normalized) and normalized[idx].startswith("TEMPO_"):
        prefix.append(normalized[idx])
        idx += 1
    if idx < len(normalized) and normalized[idx].startswith("KEY_"):
        prefix.append(normalized[idx])
        idx += 1

    while idx < len(normalized):
        token = normalized[idx]
        if stop_at_hole and token == "FIM_HOLE":
            break
        if token in {"FIM_HOLE", "FIM_MID"}:
            raise ValueError("prompt_tokens contains unsupported FIM markers for this export mode")
        if token == "EOS":
            break
        if token == "BAR":
            prefix.append(token)
            idx += 1
            if idx < len(normalized) and normalized[idx].startswith("TEMPO_"):
                prefix.append(normalized[idx])
                idx += 1
            if idx < len(normalized) and normalized[idx].startswith("KEY_"):
                prefix.append(normalized[idx])
                idx += 1
            continue
        if token.startswith("POS_"):
            if idx + 4 >= len(normalized):
                raise ValueError("prompt_tokens contains an incomplete note event")
            idx += 5
            continue
        raise ValueError(f"unsupported token in prompt_tokens: `{token}`")

    return prefix


def _resolve_fragment_tokens(case: dict[str, Any], token_field: str) -> list[str]:
    fragment_field = _COMPLETE_TO_FRAGMENT_FIELD.get(token_field)
    if fragment_field is None:
        raise ValueError(f"cannot resolve fragment field for `{token_field}`")
    tokens = case.get(fragment_field)
    if not isinstance(tokens, list):
        raise ValueError(f"`{fragment_field}` is missing or is not a token array")
    if not all(isinstance(item, str) and item.strip() for item in tokens):
        raise ValueError(f"`{fragment_field}` must be a non-empty token string array")
    return [str(item).strip() for item in tokens]


def _resolve_target_tokens(case: dict[str, Any], *, task_name: str) -> list[str]:
    target_field = _TASK_TO_TARGET_FIELD.get(task_name)
    if target_field is None:
        raise ValueError(f"unsupported task for target export: `{task_name}`")
    tokens = case.get(target_field)
    if not isinstance(tokens, list):
        raise ValueError(f"`{target_field}` is missing or is not a token array")
    if not all(isinstance(item, str) and item.strip() for item in tokens):
        raise ValueError(f"`{target_field}` must be a non-empty token string array")
    return [str(item).strip() for item in tokens]


def _build_partial_sequence(case: dict[str, Any], *, task_name: str, token_field: str) -> list[str]:
    prompt_tokens = case.get("prompt_tokens")
    if not isinstance(prompt_tokens, list):
        raise ValueError("`prompt_tokens` is missing or is not a token array")
    if not all(isinstance(item, str) and item.strip() for item in prompt_tokens):
        raise ValueError("`prompt_tokens` must be a non-empty token string array")

    fragment_tokens = _normalize_sequence_tokens(_resolve_fragment_tokens(case, token_field))
    if task_name == "continuation":
        prefix = _build_structure_only_prefix([str(item).strip() for item in prompt_tokens], stop_at_hole=False)
    elif task_name == "infilling":
        prefix = _build_structure_only_prefix([str(item).strip() for item in prompt_tokens], stop_at_hole=True)
    else:
        raise ValueError(f"unsupported task for partial export: `{task_name}`")

    return [*prefix, *fragment_tokens, "EOS"]


def _split_infilling_prompt(prompt_tokens: list[str]) -> tuple[list[str], list[str]]:
    normalized = [str(token).strip() for token in prompt_tokens if str(token).strip()]
    if not normalized or normalized[0] != "BOS":
        raise ValueError("prompt_tokens must start with BOS")
    if "FIM_HOLE" not in normalized or "FIM_MID" not in normalized:
        raise ValueError("infilling prompt_tokens must contain FIM_HOLE and FIM_MID")

    hole_index = normalized.index("FIM_HOLE")
    mid_index = normalized.index("FIM_MID")
    if mid_index <= hole_index:
        raise ValueError("FIM_MID must appear after FIM_HOLE in prompt_tokens")
    prefix_tokens = _normalize_sequence_tokens(normalized[:hole_index])
    suffix_tokens = _normalize_sequence_tokens(normalized[hole_index + 1 : mid_index])
    return prefix_tokens, suffix_tokens


def _build_reference_full_sequence(case: dict[str, Any], *, task_name: str) -> list[str]:
    prompt_tokens = case.get("prompt_tokens")
    if not isinstance(prompt_tokens, list):
        raise ValueError("`prompt_tokens` is missing or is not a token array")
    if not all(isinstance(item, str) and item.strip() for item in prompt_tokens):
        raise ValueError("`prompt_tokens` must be a non-empty token string array")

    target_tokens = _normalize_sequence_tokens(_resolve_target_tokens(case, task_name=task_name))
    prompt_normalized = [str(item).strip() for item in prompt_tokens if str(item).strip()]

    if task_name == "continuation":
        prefix_tokens = _normalize_sequence_tokens(prompt_normalized)
        return [*prefix_tokens, *target_tokens, "EOS"]
    if task_name == "infilling":
        prefix_tokens, suffix_tokens = _split_infilling_prompt(prompt_normalized)
        return [*prefix_tokens, *target_tokens, *suffix_tokens, "EOS"]
    raise ValueError(f"unsupported task for reference export: `{task_name}`")


def _build_target_sequence(case: dict[str, Any], *, task_name: str) -> list[str]:
    prompt_tokens = case.get("prompt_tokens")
    if not isinstance(prompt_tokens, list):
        raise ValueError("`prompt_tokens` is missing or is not a token array")
    if not all(isinstance(item, str) and item.strip() for item in prompt_tokens):
        raise ValueError("`prompt_tokens` must be a non-empty token string array")

    target_tokens = _normalize_sequence_tokens(_resolve_target_tokens(case, task_name=task_name))
    prompt_normalized = [str(item).strip() for item in prompt_tokens if str(item).strip()]

    if task_name == "continuation":
        prefix = _build_structure_only_prefix(prompt_normalized, stop_at_hole=False)
    elif task_name == "infilling":
        prefix = _build_structure_only_prefix(prompt_normalized, stop_at_hole=True)
    else:
        raise ValueError(f"unsupported task for target export: `{task_name}`")
    return [*prefix, *target_tokens, "EOS"]


def _resolve_task_name(payload: dict[str, Any], input_json: Path) -> str:
    task_name = str(payload.get("task", "")).strip().lower()
    if task_name in _TASK_TO_SUFFIX:
        return task_name
    input_name = input_json.name.lower()
    if "continuation" in input_name:
        return "continuation"
    if "infilling" in input_name:
        return "infilling"
    raise ValueError("unable to determine task type from sample JSON; expected continuation or infilling")


def _partial_output_path(full_output_path: Path, *, task_name: str, single_case: bool) -> Path:
    task_suffix = _TASK_TO_SUFFIX[task_name]
    if single_case:
        return full_output_path.with_name(f"{full_output_path.stem}_{task_suffix}.mid")
    stem = full_output_path.stem
    if stem.endswith("_full"):
        stem = stem[:-5]
    return full_output_path.with_name(f"{stem}_{task_suffix}.mid")


def _sibling_output_path(full_output_path: Path, *, suffix: str, single_case: bool) -> Path:
    if single_case:
        return full_output_path.with_name(f"{full_output_path.stem}_{suffix}.mid")
    stem = full_output_path.stem
    if stem.endswith("_full"):
        stem = stem[:-5]
    return full_output_path.with_name(f"{stem}_{suffix}.mid")


def main(argv: list[str] | None = None) -> int:
    try:
        project_root = _ensure_project_root_on_path()
        args = _parse_args(argv)

        from src.tokenizer import load_config, tokens_to_midi
        from src.utils.config_io import load_json_file

        input_json = args.input_json
        if not input_json.is_absolute():
            input_json = (project_root / input_json).resolve()
        output_path = args.output
        if not output_path.is_absolute():
            output_path = (project_root / output_path).resolve()
        config_path = args.config
        if not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()

        payload = load_json_file(input_json, "benchmark sample json")
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise ValueError("benchmark sample json must contain a `cases` array")
        if args.case_index is not None and (args.case_index < 0 or args.case_index >= len(cases)):
            raise IndexError(
                f"case_index {args.case_index} is out of range for {input_json} (cases={len(cases)})"
            )

        config = load_config(config_path)
        output_targets = _resolve_output_targets(output_path, args.case_index, len(cases))
        token_field = str(args.token_field).strip()
        task_name = _resolve_task_name(payload, input_json)
        exported_full_paths: list[Path] = []
        exported_partial_paths: list[Path] = []
        exported_target_paths: list[Path] = []
        exported_reference_paths: list[Path] = []
        single_case = args.case_index is not None
        for index, target_path in output_targets:
            case = cases[index]
            if not isinstance(case, dict):
                raise ValueError(f"case at index {index} is not an object")
            tokens = _resolve_case_tokens(case, token_field)
            midi = tokens_to_midi(tokens, config, ticks_per_beat=int(args.ticks_per_beat))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            midi.save(target_path)
            exported_full_paths.append(target_path)

            partial_tokens = _build_partial_sequence(case, task_name=task_name, token_field=token_field)
            partial_path = _partial_output_path(target_path, task_name=task_name, single_case=single_case)
            partial_midi = tokens_to_midi(partial_tokens, config, ticks_per_beat=int(args.ticks_per_beat))
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_midi.save(partial_path)
            exported_partial_paths.append(partial_path)

            target_sequence = _build_target_sequence(case, task_name=task_name)
            target_path_midi = _sibling_output_path(target_path, suffix="target", single_case=single_case)
            target_midi = tokens_to_midi(target_sequence, config, ticks_per_beat=int(args.ticks_per_beat))
            target_path_midi.parent.mkdir(parents=True, exist_ok=True)
            target_midi.save(target_path_midi)
            exported_target_paths.append(target_path_midi)

            reference_sequence = _build_reference_full_sequence(case, task_name=task_name)
            reference_path = _sibling_output_path(target_path, suffix="reference_full", single_case=single_case)
            reference_midi = tokens_to_midi(reference_sequence, config, ticks_per_beat=int(args.ticks_per_beat))
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_midi.save(reference_path)
            exported_reference_paths.append(reference_path)

        if args.case_index is None:
            print(
                "[export_tokens_to_midi] "
                f"input={input_json} case_count={len(exported_full_paths)} "
                f"task={task_name} token_field={token_field} output_dir={output_path}"
            )
        else:
            print(
                "[export_tokens_to_midi] "
                f"input={input_json} case_index={args.case_index} "
                f"task={task_name} token_field={token_field} "
                f"full={exported_full_paths[0]} partial={exported_partial_paths[0]} "
                f"target={exported_target_paths[0]} reference={exported_reference_paths[0]}"
            )
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[export_tokens_to_midi] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
