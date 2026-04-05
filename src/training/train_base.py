"""Base 训练核心入口（当前为最小可运行烟测版）。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from ..utils.torch_utils import count_parameters, lazy_import_torch, resolve_torch_device


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TuneFlow Base 训练入口（最小烟测）。")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/train/model_base.yaml"),
        help="模型配置 YAML 路径（默认读取 model 段）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=3,
        help="烟测训练步数（仅用于验证前后向链路）。设为 0 仅构建模型。",
    )
    parser.add_argument(
        "--smoke-batch-size",
        type=int,
        default=2,
        help="烟测 batch 大小。",
    )
    parser.add_argument(
        "--smoke-seq-len",
        type=int,
        default=64,
        help="烟测序列长度。",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="烟测学习率。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch = lazy_import_torch()

    from src.model import DecoderConfig, DecoderForCausalLM

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = DecoderConfig.from_yaml(args.model_config)
    model = DecoderForCausalLM(config)
    device = resolve_torch_device(torch, args.device)
    model.to(device)

    total_params = count_parameters(model)
    print(f"[train_base] model_type={config.model_type} vocab={config.vocab_size}")
    print(f"[train_base] params={total_params:,} device={device}")

    if args.smoke_steps <= 0:
        print("[train_base] smoke-steps<=0，已完成模型构建检查。")
        return

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.smoke_steps + 1):
        input_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(args.smoke_batch_size, args.smoke_seq_len),
            device=device,
            dtype=torch.long,
        )
        labels = input_ids.clone()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"[train_base] step={step}/{args.smoke_steps} loss={loss.item():.6f}")

    print("[train_base] smoke training finished.")


if __name__ == "__main__":
    main()
