"""LoRA 训练核心入口（脚手架）。"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TuneFlow LoRA 训练入口（脚手架）。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/train_lora_rnb.yaml",
        help="LoRA 训练配置路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[train_lora] config={args.config}")
    print("[train_lora] 当前仅完成入口重构，LoRA 训练逻辑待实现。")


if __name__ == "__main__":
    main()
