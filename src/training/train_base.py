"""Base 训练入口文件"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

from ..utils.torch_utils import count_parameters, lazy_import_torch, resolve_torch_device


class TokenBinDataset:
    """基于 `.bin + .idx.json` 的随机窗口采样器。"""

    _DTYPE_TO_TYPECODE = {
        "uint16": "H",
        "uint32": "I",
    }

    def __init__(self, idx_path: Path, bin_path_override: Path | None = None):
        self.idx_path = idx_path
        # idx 文件里记录了 dtype / offsets / lengths / num_tokens 等元信息
        idx_payload = json.loads(idx_path.read_text(encoding="utf-8"))
        self.dtype = str(idx_payload.get("dtype", ""))
        if self.dtype not in self._DTYPE_TO_TYPECODE:
            raise ValueError(
                f"Unsupported dtype in {idx_path}: {self.dtype!r}. "
                "Expected one of: uint16, uint32."
            )
        self.typecode = self._DTYPE_TO_TYPECODE[self.dtype]

        offsets = idx_payload.get("offsets")
        lengths = idx_payload.get("lengths")
        if not isinstance(offsets, list) or not isinstance(lengths, list):
            raise ValueError(f"{idx_path} must contain list fields `offsets` and `lengths`.")
        self.offsets = [int(x) for x in offsets]
        self.lengths = [int(x) for x in lengths]
        if len(self.offsets) != len(self.lengths):
            raise ValueError(f"offsets/lengths size mismatch in {idx_path}")
        self.num_sequences = len(self.lengths)
        self.num_tokens = int(idx_payload.get("num_tokens", 0))

        # 解析 bin 路径，允许命令行显式覆盖
        self.bin_path = self._resolve_bin_path(idx_path, idx_payload, bin_path_override)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary token file not found: {self.bin_path}")

        # 使用 mmap 做零拷贝读取，避免把整份语料一次性加载到内存
        self._bin_file = self.bin_path.open("rb")
        self._mmap = mmap.mmap(self._bin_file.fileno(), length=0, access=mmap.ACCESS_READ)
        self._token_view = memoryview(self._mmap).cast(self.typecode)
        # 缓存“长度足够采样”的序列索引，减少重复扫描开销
        self._eligible_cache: dict[int, list[int]] = {}

    @staticmethod
    def _resolve_bin_path(idx_path: Path, idx_payload: dict, bin_path_override: Path | None) -> Path:
        """解析 `.bin` 文件路径。"""
        if bin_path_override is not None:
            return bin_path_override

        idx_bin = Path(str(idx_payload.get("bin_file", "")))
        if idx_bin.is_absolute():
            return idx_bin

        # 优先使用 idx.json 中给出的路径（通常是仓库相对路径），
        # 若不存在，再退化到“相对 idx 文件目录”的解析策略。
        if idx_bin.exists():
            return idx_bin.resolve()
        return (idx_path.parent / idx_bin).resolve()

    def close(self) -> None:
        """显式释放 mmap 与文件句柄。"""
        self._token_view.release()
        self._mmap.close()
        self._bin_file.close()

    def _eligible_indices(self, min_len: int) -> list[int]:
        """返回长度 >= min_len 的序列索引列表。"""
        cached = self._eligible_cache.get(min_len)
        if cached is not None:
            return cached
        indices = [i for i, length in enumerate(self.lengths) if length >= min_len]
        self._eligible_cache[min_len] = indices
        return indices

    def sample_batch(self, torch_mod, rng: random.Random, batch_size: int, seq_len: int, device):
        """随机采样 batch，构造自回归训练所需的 input_ids / labels。"""
        min_len = seq_len + 1
        candidates = self._eligible_indices(min_len)
        if not candidates:
            raise ValueError(
                f"No sequence in {self.idx_path} has length >= {min_len}. "
                "Please lower --seq-len or regenerate data."
            )

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for _ in range(batch_size):
            # 先随机抽一条“足够长”的序列，再随机抽一个连续窗口
            seq_idx = candidates[rng.randrange(len(candidates))]
            seq_offset = self.offsets[seq_idx]
            seq_total_len = self.lengths[seq_idx]
            start_in_seq = rng.randrange(seq_total_len - min_len + 1)
            abs_start = seq_offset + start_in_seq
            window = self._token_view[abs_start : abs_start + min_len]
            # 经典 next-token 训练：输入是前 N，标签是后 N
            input_rows.append(list(window[:-1]))
            label_rows.append(list(window[1:]))

        input_ids = torch_mod.tensor(input_rows, dtype=torch_mod.long, device=device)
        labels = torch_mod.tensor(label_rows, dtype=torch_mod.long, device=device)
        return input_ids, labels


def _parse_args() -> argparse.Namespace:
    """命令行参数：覆盖训练、评估、保存、恢复等核心开关。"""
    parser = argparse.ArgumentParser(description="TuneFlow base training (real-data loop).")
    # 配置/数据路径
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/train/model_base.yaml"),
        help="Model config YAML path.",
    )
    parser.add_argument(
        "--train-idx",
        type=Path,
        default=Path("data/tokenized/train.idx.json"),
        help="Path to train `.idx.json`.",
    )
    parser.add_argument(
        "--train-bin",
        type=Path,
        default=None,
        help="Optional override for train `.bin` path.",
    )
    parser.add_argument(
        "--valid-idx",
        type=Path,
        default=Path("data/tokenized/valid.idx.json"),
        help="Path to valid `.idx.json`.",
    )
    parser.add_argument(
        "--valid-bin",
        type=Path,
        default=None,
        help="Optional override for valid `.bin` path.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    # 设备与精度
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="Numerical precision mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # 训练超参
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total optimizer-update steps (not micro-steps).",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Training sequence length (must be <= max_position_embeddings).",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    # 学习率调度
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "linear"],
        help="Learning-rate schedule type.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps for scheduler.")
    parser.add_argument(
        "--min-lr-scale",
        type=float,
        default=0.1,
        help="Final LR scale for linear/cosine scheduler.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Train-log interval in update steps.")
    parser.add_argument("--eval-every", type=int, default=50, help="Validation interval; <=0 disables.")
    parser.add_argument("--eval-batches", type=int, default=5, help="Validation micro-batches per eval.")
    parser.add_argument("--save-every", type=int, default=100, help="Checkpoint interval; <=0 disables.")
    # 保存/恢复行为
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save best checkpoint by validation loss.",
    )
    parser.add_argument(
        "--no-restore-rng",
        action="store_true",
        help="Do not restore RNG state when resuming from checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoints/base/minimal_real_train"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Optional JSONL metrics path; default is `<output-dir>/metrics.jsonl`.",
    )
    return parser.parse_args()


def _autocast_context(torch_mod, use_amp: bool, device_type: str, amp_dtype):
    """根据配置返回 autocast 上下文；未启用时返回空上下文。"""
    if not use_amp:
        return nullcontext()
    return torch_mod.autocast(device_type=device_type, dtype=amp_dtype)


def _resolve_precision(torch_mod, requested: str, device):
    """
    解析精度策略，返回:
    (effective_name, use_amp, amp_dtype, use_grad_scaler)
    """
    if requested == "fp32":
        return "fp32", False, None, False

    if device.type != "cuda":
        # CPU 不启用半精度 autocast（保持行为明确且稳定）
        if requested in {"bf16", "fp16"}:
            print(f"[train_base] precision={requested} requested on {device.type}; fallback to fp32.")
        return "fp32", False, None, False

    bf16_supported = bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)())
    if requested == "auto":
        if bf16_supported:
            return "bf16", True, torch_mod.bfloat16, False
        return "fp16", True, torch_mod.float16, True
    if requested == "bf16":
        if bf16_supported:
            return "bf16", True, torch_mod.bfloat16, False
        print("[train_base] bf16 is not supported on this GPU; fallback to fp16.")
        return "fp16", True, torch_mod.float16, True
    if requested == "fp16":
        return "fp16", True, torch_mod.float16, True

    return "fp32", False, None, False


def _build_scheduler(torch_mod, optimizer, name: str, total_steps: int, warmup_steps: int, min_lr_scale: float):
    """构建 LR scheduler（none / linear / cosine）。"""
    if name == "none":
        return None

    if total_steps <= 0:
        raise ValueError("total_steps must be > 0 when scheduler is enabled.")

    warmup = max(0, min(int(warmup_steps), int(total_steps)))
    floor = min(1.0, max(0.0, float(min_lr_scale)))

    def lr_lambda(current_step_zero_based: int) -> float:
        # LambdaLR 传入的是从 0 开始的 step，这里统一换算到从 1 开始
        step = current_step_zero_based + 1
        # 1) warmup 阶段线性升温
        if warmup > 0 and step <= warmup:
            return float(step) / float(warmup)

        if total_steps <= warmup:
            return floor

        # 2) warmup 后按 schedule 衰减到 floor
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        progress = min(1.0, max(0.0, progress))

        if name == "linear":
            return floor + (1.0 - floor) * (1.0 - progress)
        if name == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cosine
        return 1.0

    return torch_mod.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _append_metrics(path: Path, payload: dict) -> None:
    """以 JSONL 追加一条结构化指标记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _collect_rng_states(torch_mod) -> dict:
    """采集 Python / Torch / CUDA RNG 状态，支持可复现实验恢复。"""
    payload = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch_mod.get_rng_state(),
    }
    if torch_mod.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch_mod.cuda.get_rng_state_all()
    return payload


def _restore_rng_states(torch_mod, payload: dict) -> None:
    """恢复 RNG 状态。"""
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    if "torch_rng_state" in payload:
        torch_mod.set_rng_state(payload["torch_rng_state"])
    if torch_mod.cuda.is_available() and "torch_cuda_rng_state_all" in payload:
        torch_mod.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])


def _load_checkpoint(torch_mod, path: Path):
    """加载 checkpoint，并兼容是否支持 `weights_only` 参数的 torch 版本。"""
    try:
        return torch_mod.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_mod.load(path, map_location="cpu")


def _save_checkpoint(
    torch_mod,
    path: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    scaler,
    best_valid_loss: float,
    model_config,
    args: argparse.Namespace,
) -> None:
    """保存训练状态（模型/优化器/scheduler/scaler/RNG/配置）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "best_valid_loss": best_valid_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "model_config": model_config.to_dict(),
        "train_args": vars(args),
    }
    payload.update(_collect_rng_states(torch_mod))
    torch_mod.save(payload, path)
    print(f"[train_base] checkpoint -> {path}")


def _evaluate(
    model,
    dataset: TokenBinDataset,
    torch_mod,
    rng: random.Random,
    device,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    use_amp: bool,
    amp_dtype,
) -> float:
    """在验证集上采样若干 batch，返回平均 loss。"""
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
            # 评估路径与训练保持一致的精度策略，避免统计口径不一致
            with _autocast_context(torch_mod, use_amp=use_amp, device_type=device.type, amp_dtype=amp_dtype):
                outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
            losses.append(float(outputs.loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main() -> None:
    """训练主流程：初始化 ->（可选恢复）-> 训练 -> 评估 -> 保存。"""
    args = _parse_args()
    torch = lazy_import_torch()

    from src.model import DecoderConfig, DecoderForCausalLM

    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad-accum-steps must be > 0.")
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0.")

    # 先设全局种子，再创建独立 run_rng（用于数据采样）
    random.seed(args.seed)
    run_rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = DecoderConfig.from_yaml(args.model_config)
    if args.seq_len > config.max_position_embeddings:
        raise SystemExit(
            f"--seq-len ({args.seq_len}) exceeds model max_position_embeddings "
            f"({config.max_position_embeddings})."
        )

    train_dataset = TokenBinDataset(args.train_idx, args.train_bin)
    valid_dataset = TokenBinDataset(args.valid_idx, args.valid_bin) if args.valid_idx.exists() else None

    device = resolve_torch_device(torch, args.device)
    # 精度解析：决定是否使用 AMP、使用哪种 dtype、是否启用 GradScaler
    precision_name, use_amp, amp_dtype, use_scaler = _resolve_precision(
        torch_mod=torch, requested=args.precision, device=device
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if use_scaler else None

    model = DecoderForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(
        torch_mod=torch,
        optimizer=optimizer,
        name=args.scheduler,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        min_lr_scale=args.min_lr_scale,
    )

    start_step = 0
    best_valid_loss = float("inf")
    if args.resume_from is not None:
        # 恢复训练状态（含 scheduler/scaler/RNG）
        if not args.resume_from.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume_from}")
        ckpt = _load_checkpoint(torch, args.resume_from)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        best_valid_loss = float(ckpt.get("best_valid_loss", float("inf")))
        if not args.no_restore_rng:
            _restore_rng_states(torch, ckpt)
        print(f"[train_base] resumed from {args.resume_from} at step={start_step}")

    if start_step >= args.steps:
        print(
            f"[train_base] start_step ({start_step}) >= --steps ({args.steps}); "
            "nothing to train."
        )
        return

    total_params = count_parameters(model)
    effective_batch = args.batch_size * args.grad_accum_steps
    print(f"[train_base] model_type={config.model_type} vocab={config.vocab_size}")
    print(
        f"[train_base] params={total_params:,} device={device} precision={precision_name} "
        f"steps={args.steps} batch={args.batch_size} grad_accum={args.grad_accum_steps} "
        f"effective_batch={effective_batch} seq_len={args.seq_len}"
    )
    print(
        f"[train_base] train={train_dataset.idx_path} ({train_dataset.num_sequences} seqs, "
        f"{train_dataset.num_tokens} tokens)"
    )
    if valid_dataset is not None:
        print(
            f"[train_base] valid={valid_dataset.idx_path} ({valid_dataset.num_sequences} seqs, "
            f"{valid_dataset.num_tokens} tokens)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 默认把指标写到 output_dir/metrics.jsonl
    metrics_path = args.metrics_path if args.metrics_path is not None else (args.output_dir / "metrics.jsonl")
    _append_metrics(
        metrics_path,
        {
            "event": "run_start",
            "time": time.time(),
            "start_step": start_step,
            "target_steps": args.steps,
            "precision": precision_name,
            "scheduler": args.scheduler,
            "lr": args.lr,
            "effective_batch": effective_batch,
            "seq_len": args.seq_len,
            "resume_from": None if args.resume_from is None else str(args.resume_from),
        },
    )

    model.train()
    run_start = time.perf_counter()
    interval_start = run_start
    interval_tokens = 0

    try:
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            for _ in range(args.grad_accum_steps):
                input_ids, labels = train_dataset.sample_batch(
                    torch_mod=torch,
                    rng=run_rng,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=device,
                )
                with _autocast_context(
                    torch_mod=torch, use_amp=use_amp, device_type=device.type, amp_dtype=amp_dtype
                ):
                    outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
                    raw_loss = outputs.loss
                    # 梯度累积：每个 micro-step loss 按累积步数做缩放
                    scaled_loss = raw_loss / args.grad_accum_steps

                if not torch.isfinite(raw_loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {float(raw_loss.item())}")

                if scaler is not None:
                    # fp16 路径：先 scale 再 backward
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                step_loss += float(raw_loss.item())

            step_loss /= args.grad_accum_steps

            if scaler is not None:
                # clip 前先 unscale，保证梯度范数语义正确
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler is not None:
                # fp16 路径：step + update
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            current_lr = float(optimizer.param_groups[0]["lr"])
            interval_tokens += args.batch_size * args.seq_len * args.grad_accum_steps

            should_log = step == start_step + 1 or (args.log_every > 0 and step % args.log_every == 0)
            if should_log:
                now = time.perf_counter()
                elapsed = max(1e-9, now - interval_start)
                toks_per_sec = interval_tokens / elapsed
                print(
                    f"[train_base] step={step}/{args.steps} "
                    f"loss={step_loss:.6f} "
                    f"lr={current_lr:.6e} "
                    f"tok/s={toks_per_sec:.1f}"
                )
                _append_metrics(
                    metrics_path,
                    {
                        "event": "train",
                        "time": time.time(),
                        "step": step,
                        "loss": step_loss,
                        "lr": current_lr,
                        "tok_per_sec": toks_per_sec,
                    },
                )
                interval_start = now
                interval_tokens = 0

            should_eval = (
                valid_dataset is not None
                and args.eval_every > 0
                and (step % args.eval_every == 0 or step == args.steps)
            )
            if should_eval:
                val_loss = _evaluate(
                    model=model,
                    dataset=valid_dataset,
                    torch_mod=torch,
                    rng=run_rng,
                    device=device,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    eval_batches=args.eval_batches,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                print(f"[train_base] eval step={step} valid_loss={val_loss:.6f}")
                _append_metrics(
                    metrics_path,
                    {
                        "event": "eval",
                        "time": time.time(),
                        "step": step,
                        "valid_loss": val_loss,
                    },
                )
                if args.save_best and val_loss < best_valid_loss:
                    # 仅在指标变优时覆盖 best.pt
                    best_valid_loss = val_loss
                    best_path = args.output_dir / "best.pt"
                    _save_checkpoint(
                        torch_mod=torch,
                        path=best_path,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        best_valid_loss=best_valid_loss,
                        model_config=config,
                        args=args,
                    )

            should_save = args.save_every > 0 and step % args.save_every == 0
            if should_save:
                step_ckpt = args.output_dir / f"step_{step}.pt"
                _save_checkpoint(
                    torch_mod=torch,
                    path=step_ckpt,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_valid_loss=best_valid_loss,
                    model_config=config,
                    args=args,
                )
                # 同步一个 latest.pt 便于自动恢复
                shutil.copy2(step_ckpt, args.output_dir / "latest.pt")

        final_ckpt = args.output_dir / "last.pt"
        _save_checkpoint(
            torch_mod=torch,
            path=final_ckpt,
            step=args.steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_valid_loss=best_valid_loss,
            model_config=config,
            args=args,
        )
        shutil.copy2(final_ckpt, args.output_dir / "latest.pt")

        total_elapsed = time.perf_counter() - run_start
        print(f"[train_base] done in {total_elapsed:.2f}s")
    finally:
        # 无论训练是否异常退出，都确保释放数据句柄
        train_dataset.close()
        if valid_dataset is not None:
            valid_dataset.close()


if __name__ == "__main__":
    main()
