# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Copyright 2026 TuneFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from Hugging Face Transformers Qwen2 configuration:
# https://github.com/huggingface/transformers/tree/374d44d54adb1c5f52e68aff97d1675f56d657a8/src/transformers/models/qwen2

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from ..utils.config_io import load_json_file, load_yaml_mapping

DEFAULT_TOKENIZER_VOCAB_PATH = "data/tokenized/tokenizer_vocab.json"


@dataclass
class DecoderConfig:
    """
    TuneFlow 本地 Decoder-only 模型配置。

    默认值与当前仓库阶段对齐：
    - 词表来自 `data/tokenized/tokenizer_vocab.json`
    - 上下文长度为 1024（design.md 的 v0 基线）
    - 采用约 50M 参数规模与 GQA 结构
    """

    model_type: str = "tuneflow_decoder"
    architecture_hint: str = "qwen2-style"

    # 分词器 / 词表相关
    vocab_path: str | None = DEFAULT_TOKENIZER_VOCAB_PATH
    sync_vocab_with_file: bool = True
    sync_special_tokens_with_file: bool = True
    strict_vocab_file: bool = False
    vocab_size: int = 243
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | None = 1
    special_token_ids: dict[str, int] = field(default_factory=lambda: {"BOS": 0, "EOS": 1, "FIM_HOLE": 2, "FIM_MID": 3})

    # v0 基线规模（约 50M）
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 12
    num_key_value_heads: int | None = 4
    hidden_act: str = "silu"

    # 序列 / 位置编码
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0

    # 初始化 / 归一化 / 注意力行为
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    use_cache: bool = True
    tie_word_embeddings: bool = False

    # 可选滑窗注意力模式
    use_sliding_window: bool = False
    sliding_window: int | None = 512
    max_window_layers: int = 0
    layer_types: list[str] | None = None

    _name_or_path: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """初始化后统一做配置修正与约束校验。"""
        # 先尝试按词表文件同步动态字段（词表大小、特殊 token）。
        self._sync_from_vocab_file_if_needed()

        # 未显式指定 KV 头时，退化为普通多头注意力（KV 头数=Q 头数）。
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # 头维度必须整除，否则无法正确 reshape 到多头格式。
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )

        # GQA 要求 Q 头数能被 KV 头数整除，便于 KV 复制分组。
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )

        # 滑窗模式下保证窗口参数有效；关闭时统一清空窗口值。
        if self.use_sliding_window:
            if self.sliding_window is None or self.sliding_window <= 0:
                raise ValueError("sliding_window must be a positive integer when use_sliding_window=True.")
        else:
            self.sliding_window = None

        # 若未显式给每层类型，则按规则自动生成 layer_types。
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    @property
    def head_dim(self) -> int:
        """单个注意力头的维度。"""
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        """导出为可序列化字典，便于落盘和实验记录。"""
        return {
            "model_type": self.model_type,
            "architecture_hint": self.architecture_hint,
            "vocab_path": self.vocab_path,
            "sync_vocab_with_file": self.sync_vocab_with_file,
            "sync_special_tokens_with_file": self.sync_special_tokens_with_file,
            "strict_vocab_file": self.strict_vocab_file,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "special_token_ids": dict(self.special_token_ids),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "hidden_act": self.hidden_act,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "initializer_range": self.initializer_range,
            "rms_norm_eps": self.rms_norm_eps,
            "attention_dropout": self.attention_dropout,
            "use_cache": self.use_cache,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window": self.sliding_window,
            "max_window_layers": self.max_window_layers,
            "layer_types": list(self.layer_types) if self.layer_types is not None else None,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "DecoderConfig":
        """从字典构建配置对象。"""
        return cls(**config_dict)

    @classmethod
    def from_tokenizer_vocab(
        cls,
        vocab_path: str | Path = DEFAULT_TOKENIZER_VOCAB_PATH,
        **overrides: Any,
    ) -> "DecoderConfig":
        """直接从词表文件构建配置，并自动写入词表相关字段。"""
        vocab_path = Path(vocab_path)
        payload = load_json_file(vocab_path, "词表")
        token_to_id = payload.get("token_to_id")
        if not isinstance(token_to_id, dict):
            raise ValueError(f"Invalid tokenizer vocab format: {vocab_path}")

        # 仅收集当前阶段关键特殊 token，避免把无关 token 混入配置核心字段。
        special_token_ids: dict[str, int] = {}
        for token in ("BOS", "EOS", "FIM_HOLE", "FIM_MID"):
            if token in token_to_id:
                special_token_ids[token] = int(token_to_id[token])

        config_values = {
            "vocab_path": str(vocab_path),
            "vocab_size": len(token_to_id),
            "bos_token_id": token_to_id.get("BOS"),
            "eos_token_id": token_to_id.get("EOS"),
            "special_token_ids": special_token_ids,
            "sync_vocab_with_file": True,
            "sync_special_tokens_with_file": True,
        }
        config_values.update(overrides)
        return cls(**config_values)

    @classmethod
    def from_yaml(cls, config_path: str | Path, key: str = "model") -> "DecoderConfig":
        """从 YAML 读取配置，默认读取 `model` 段。"""
        config_path = Path(config_path)
        payload = load_yaml_mapping(config_path, "模型配置")
        if key in payload and isinstance(payload[key], dict):
            payload = payload[key]
        if not isinstance(payload, dict):
            raise ValueError(f"Expected mapping in {config_path}, got {type(payload)!r}.")
        return cls.from_dict(payload)

    def to_yaml(self, config_path: str | Path, key: str = "model") -> None:
        """把当前配置写回 YAML，便于实验可复现。"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: self.to_dict()}
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")

    def reload_vocab(self) -> None:
        """手动强制重新加载词表同步字段。"""
        self._sync_from_vocab_file_if_needed(force=True)

    def _sync_from_vocab_file_if_needed(self, force: bool = False) -> None:
        """
        按需从词表文件同步配置。

        同步内容：
        - `vocab_size`
        - `bos_token_id` / `eos_token_id`
        - `special_token_ids`（含 FIM/TASK token）
        """
        if not self.vocab_path:
            return
        if not force and not (self.sync_vocab_with_file or self.sync_special_tokens_with_file):
            return

        vocab_file = Path(self.vocab_path)
        # strict 模式下缺词表即报错；非 strict 模式容忍缺失（便于早期搭脚手架）。
        if not vocab_file.exists():
            if self.strict_vocab_file:
                raise FileNotFoundError(f"Tokenizer vocab file not found: {vocab_file}")
            return

        payload = load_json_file(vocab_file, "词表")
        token_to_id = payload.get("token_to_id")
        if not isinstance(token_to_id, dict):
            if self.strict_vocab_file:
                raise ValueError(f"Invalid tokenizer vocab format: {vocab_file}")
            return

        if self.sync_vocab_with_file:
            self.vocab_size = len(token_to_id)

        if self.sync_special_tokens_with_file:
            self.bos_token_id = token_to_id.get("BOS", self.bos_token_id)
            self.eos_token_id = token_to_id.get("EOS", self.eos_token_id)

            # 保留已有映射并覆盖可识别 token，避免外部自定义字段被意外丢失。
            merged = dict(self.special_token_ids)
            for token in ("BOS", "EOS", "FIM_HOLE", "FIM_MID", "TASK_INFILL", "TASK_CONT", "TASK_GEN"):
                if token in token_to_id:
                    merged[token] = int(token_to_id[token])
            self.special_token_ids = merged
