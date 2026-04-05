"""训练核心模块。"""

from .train_base import main as train_base_main
from .train_lora import main as train_lora_main

__all__ = ["train_base_main", "train_lora_main"]
