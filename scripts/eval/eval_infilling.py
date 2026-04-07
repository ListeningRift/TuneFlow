#!/usr/bin/env python
"""评估 base 训练产物的最小闭环指标（valid_loss / ppl / 结构合法率）。"""

from __future__ import annotations

import argparse
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
        default=Path("outputs/checkpoints/base/train_base_run"),
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
        help="可选报告输出路径；默认 outputs/reports/eval/<run_id>.json。",
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
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
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


def _discover_checkpoints(checkpoint_dir: Path, limit: int | None) -> list[Path]:
    """扫描目录中的 checkpoint 文件，并按规则排序。"""
    paths = sorted([p for p in checkpoint_dir.glob("*.pt") if p.is_file()], key=_checkpoint_sort_key)
    if limit is not None:
        paths = paths[: max(0, limit)]
    return paths


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


def _safe_perplexity(loss: float) -> float:
    """将 loss 转为 ppl，并在极端值时做安全处理。"""
    if not math.isfinite(loss):
        return float("nan")
    if loss >= 80.0:
        return float("inf")
    return math.exp(loss)


def _evaluate_valid_loss(
    model,
    dataset,
    torch_mod,
    rng: random.Random,
    device,
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

    core = source_tokens[1:-1]
    if len(core) < 20:
        return None

    # 控制序列长度，避免超过模型上下文窗口。
    max_core_len = max(24, max_positions - 8)
    if len(core) > max_core_len:
        start = rng.randrange(len(core) - max_core_len + 1)
        core = core[start : start + max_core_len]

    seq = ["BOS", *core, "EOS"]
    core_len = len(core)
    hole_len = max(8, int(round(core_len * 0.2)))
    hole_len = min(hole_len, 96, core_len - 4)
    if hole_len <= 0:
        return None

    hole_start_core = rng.randrange(0, core_len - hole_len + 1)
    hole_start = 1 + hole_start_core
    hole_end = hole_start + hole_len

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
    device,
    use_amp: bool,
    amp_dtype,
    autocast_context_fn,
    max_positions: int,
    max_new_tokens: int,
) -> tuple[list[str], bool]:
    """基于 prompt 执行贪心解码，返回 middle token 与是否遇到 EOS。"""
    prompt_ids: list[int] = []
    for token in prompt_tokens:
        token_id = token_to_id.get(token)
        if token_id is None:
            return [], False
        prompt_ids.append(token_id)

    input_ids = torch_mod.tensor([prompt_ids], dtype=torch_mod.long, device=device)
    middle_tokens: list[str] = []
    reached_eos = False

    max_can_generate = max(0, min(max_new_tokens, max_positions - int(input_ids.shape[1])))
    if max_can_generate <= 0:
        return middle_tokens, reached_eos

    with torch_mod.no_grad():
        for _ in range(max_can_generate):
            with autocast_context_fn(
                torch_mod=torch_mod,
                use_amp=use_amp,
                device_type=device.type,
                amp_dtype=amp_dtype,
            ):
                outputs = model(input_ids=input_ids, return_dict=True)
            next_id = int(torch_mod.argmax(outputs.logits[:, -1, :], dim=-1).item())
            if next_id < 0 or next_id >= len(id_to_token):
                return middle_tokens, False

            next_token = id_to_token[next_id]
            if next_token == "EOS":
                reached_eos = True
                break
            middle_tokens.append(next_token)

            next_ids = torch_mod.tensor([[next_id]], dtype=torch_mod.long, device=device)
            input_ids = torch_mod.cat([input_ids, next_ids], dim=1)
            if int(input_ids.shape[1]) >= max_positions:
                break

    return middle_tokens, reached_eos


def _evaluate_structural_validity(
    model,
    torch_mod,
    eval_rows: list[list[str]],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    vocab: dict[str, int],
    device,
    use_amp: bool,
    amp_dtype,
    autocast_context_fn,
    max_positions: int,
    num_samples: int,
    max_new_tokens: int,
    rng: random.Random,
) -> tuple[float, int, int]:
    """
    评估结构合法率。

    流程：
    1) 从 eval 样本随机构造 infilling prompt；
    2) 让模型生成 middle；
    3) 重建为完整序列并做结构语法校验。
    """
    if num_samples <= 0 or not eval_rows:
        return float("nan"), 0, 0

    valid_count = 0
    attempted = 0

    retries = 0
    max_retries = max(num_samples * 10, 100)
    while attempted < num_samples and retries < max_retries:
        retries += 1
        source_tokens = eval_rows[rng.randrange(len(eval_rows))]
        built = _build_infilling_prompt(source_tokens=source_tokens, max_positions=max_positions, rng=rng)
        if built is None:
            continue

        prompt_tokens, prefix, suffix_len = built
        # 生成预算受“剩余上下文长度”约束，避免越界。
        dyn_max_new = min(max_new_tokens, max(8, int(round((max_positions - len(prompt_tokens)) * 0.9))))
        middle_tokens, reached_eos = _generate_middle_tokens(
            model=model,
            torch_mod=torch_mod,
            prompt_tokens=prompt_tokens,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
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

        reconstructed = [*prefix, *middle_tokens, *suffix, "EOS"]
        is_valid = reached_eos and _validate_token_order(reconstructed, vocab)
        if is_valid:
            valid_count += 1
        attempted += 1

    if attempted == 0:
        return float("nan"), 0, 0
    return valid_count / attempted, valid_count, attempted


def _resolve_report_path(run_id: str, output_path: Path | None, project_root: Path) -> Path:
    """解析最终报告输出路径。"""
    if output_path is not None:
        return output_path if output_path.is_absolute() else (project_root / output_path)
    return project_root / "outputs" / "reports" / "eval" / f"{run_id}.json"


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

    # 懒加载 torch，确保错误信息更可读，并减少无关场景启动开销。
    from src.utils.config_io import dump_json_file
    from src.utils.torch_utils import lazy_import_torch, resolve_torch_device

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

    checkpoints = _discover_checkpoints(checkpoint_dir, args.limit_checkpoints)
    if not checkpoints:
        raise FileNotFoundError(f"No *.pt checkpoints found under: {checkpoint_dir}")

    # 1) 准备验证集采样器（用于 valid_loss / ppl）。
    valid_idx = args.valid_idx if args.valid_idx.is_absolute() else (project_root / args.valid_idx)
    valid_bin = None
    if args.valid_bin is not None:
        valid_bin = args.valid_bin if args.valid_bin.is_absolute() else (project_root / args.valid_bin)
    valid_dataset = TokenBinDataset(valid_idx.resolve(), None if valid_bin is None else valid_bin.resolve())

    # 2) 准备 infilling 评估样本与词表。
    eval_tok_path = args.eval_tok if args.eval_tok.is_absolute() else (project_root / args.eval_tok)
    eval_rows = _load_eval_tok_lines(eval_tok_path.resolve())
    if not eval_rows:
        raise ValueError(f"No eval samples found in {eval_tok_path}")

    vocab_path = args.vocab_path if args.vocab_path.is_absolute() else (project_root / args.vocab_path)
    vocab, id_to_token = _load_vocab(vocab_path.resolve())

    results: list[dict[str, Any]] = []
    started_at = time.time()
    print(f"[eval_infilling] run_id={run_id} checkpoints={len(checkpoints)} device={device} precision={precision_name}")

    try:
        for index, ckpt_path in enumerate(checkpoints, start=1):
            print(f"[eval_infilling] ({index}/{len(checkpoints)}) checkpoint={ckpt_path.name}")
            ckpt_payload = _load_checkpoint(torch, ckpt_path)

            # 优先使用 checkpoint 内部保存的模型配置，保证评估与训练一致。
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
            rng = random.Random(args.seed + index)

            valid_loss = _evaluate_valid_loss(
                model=model,
                dataset=valid_dataset,
                torch_mod=torch,
                rng=rng,
                device=device,
                batch_size=args.batch_size,
                seq_len=eval_seq_len,
                eval_batches=args.eval_batches,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                autocast_context_fn=_autocast_context,
            )
            ppl = _safe_perplexity(valid_loss)
            struct_rate, struct_valid, struct_total = _evaluate_structural_validity(
                model=model,
                torch_mod=torch,
                eval_rows=eval_rows,
                token_to_id=vocab,
                id_to_token=id_to_token,
                vocab=vocab,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                autocast_context_fn=_autocast_context,
                max_positions=int(config.max_position_embeddings),
                num_samples=args.num_infilling_samples,
                max_new_tokens=args.max_new_tokens,
                rng=rng,
            )

            result = {
                "checkpoint_name": ckpt_path.name,
                "checkpoint_path": str(ckpt_path),
                "step": int(ckpt_payload.get("step", -1)),
                "valid_loss": valid_loss,
                "ppl": ppl,
                "structural_validity_rate": struct_rate,
                "structural_valid_count": struct_valid,
                "structural_total_count": struct_total,
                "seq_len": eval_seq_len,
                "eval_batches": args.eval_batches,
                "num_infilling_samples": args.num_infilling_samples,
            }
            results.append(result)
            print(
                f"[eval_infilling] step={result['step']} "
                f"valid_loss={valid_loss:.6f} ppl={ppl:.6f} "
                f"struct_valid={struct_rate:.4f} ({struct_valid}/{struct_total})"
            )

            del model
            if device.type == "cuda":
                # 防止多 checkpoint 连续评估时显存积压。
                torch.cuda.empty_cache()
    finally:
        valid_dataset.close()

    # 汇总 run 级别统计，便于快速看趋势与最优点。
    finite_losses = [r["valid_loss"] for r in results if math.isfinite(float(r["valid_loss"]))]
    finite_struct = [r["structural_validity_rate"] for r in results if math.isfinite(float(r["structural_validity_rate"]))]
    summary = {
        "checkpoint_count": len(results),
        "best_valid_loss": (min(finite_losses) if finite_losses else float("nan")),
        "best_structural_validity_rate": (max(finite_struct) if finite_struct else float("nan")),
        "elapsed_sec": max(0.0, time.time() - started_at),
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
            "seed": args.seed,
            "valid_idx": str(valid_idx.resolve()),
            "valid_bin": None if valid_bin is None else str(valid_bin.resolve()),
            "eval_tok": str(eval_tok_path.resolve()),
            "vocab_path": str(vocab_path.resolve()),
        },
        "summary": summary,
        "results": results,
    }

    dump_json_file(report_path, report, ensure_ascii=False, indent=2)
    print(f"[eval_infilling] report -> {report_path}")


if __name__ == "__main__":
    main()
