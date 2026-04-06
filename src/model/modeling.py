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
# Adapted from Hugging Face Transformers Qwen2 modeling:
# https://github.com/huggingface/transformers/tree/374d44d54adb1c5f52e68aff97d1675f56d657a8/src/transformers/models/qwen2

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import DecoderConfig


def _get_activation_fn(name: str):
    """根据字符串返回激活函数实现。"""
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation: {name}")


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV 头复制到与 Q 头匹配的数量（GQA 关键步骤）。

    输入形状: [B, num_kv_heads, T, D]
    输出形状: [B, num_kv_heads * n_rep, T, D]
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE 辅助函数：把最后一维前后两半做旋转拼接。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q/K 应用旋转位置编码。

    q/k 形状: [B, H, T, D]
    cos/sin 形状: [B, T, D]
    """
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    query_len: int,
    key_len: int,
    past_key_values_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    构建 4D 因果注意力掩码，并按需叠加 padding 掩码。

    返回形状: [B, 1, query_len, key_len]
    """
    min_value = torch.finfo(dtype).min

    # 因果掩码：未来位置不可见。
    q_positions = torch.arange(query_len, device=device) + past_key_values_length
    k_positions = torch.arange(key_len, device=device)
    causal = (k_positions[None, :] > q_positions[:, None]).to(dtype) * min_value
    causal = causal[None, None, :, :].expand(batch_size, 1, query_len, key_len)

    if attention_mask is None:
        return causal

    if attention_mask.dim() == 2:
        # 2D 掩码默认是 [B, key_len] 的有效位标记（1=可见，0=padding）。
        if attention_mask.shape[-1] != key_len:
            raise ValueError(
                f"attention_mask length ({attention_mask.shape[-1]}) does not match key_len ({key_len})."
            )
        padding_mask = (1.0 - attention_mask.to(dtype=dtype))[:, None, None, :] * min_value
        return causal + padding_mask

    if attention_mask.dim() == 4:
        # 若外部已传 4D 掩码，则只做形状校验并叠加因果掩码。
        if attention_mask.shape != (batch_size, 1, query_len, key_len):
            raise ValueError(
                f"Expected 4D attention_mask shape {(batch_size, 1, query_len, key_len)}, "
                f"but got {tuple(attention_mask.shape)}."
            )
        return causal + attention_mask.to(dtype=dtype)

    raise ValueError("attention_mask must be None, 2D, or 4D.")


@dataclass
class DecoderModelOutput:
    """Backbone 输出结构。"""
    last_hidden_state: torch.Tensor
    past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None


@dataclass
class CausalLMOutput:
    """Causal LM 输出结构。"""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None


class RMSNorm(nn.Module):
    """RMSNorm：仅按均方根缩放，不做均值中心化。"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """保持输入 dtype 输出，内部计算临时转 float 提升稳定性。"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(dtype=input_dtype)


class RotaryEmbedding(nn.Module):
    """生成 RoPE 所需 cos/sin 表达。"""
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据位置索引计算 cos/sin。"""
        freqs = torch.einsum("bs,d->bsd", position_ids.float(), self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype, device=x.device)
        sin = emb.sin().to(dtype=x.dtype, device=x.device)
        return cos, sin


class GatedMLP(nn.Module):
    """门控 MLP（SwiGLU 风格）。"""
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _get_activation_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """down( act(gate(x)) * up(x) )"""
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecoderAttention(nn.Module):
    """支持 GQA + RoPE + KV Cache 的自注意力模块。"""
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """返回注意力输出与可选缓存 `(key, value)`。"""
        bsz, q_len, _ = hidden_states.size()

        # 1) 线性映射到 Q/K/V 并转成多头布局。
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # 2) 对 Q/K 注入旋转位置编码。
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3) 推理场景拼接历史缓存，实现增量解码。
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # 4) GQA：把较少的 KV 头复制到与 Q 头数一致。
        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        # 5) 标准缩放点积注意力。
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights / (self.head_dim**0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # softmax 用 float32 计算更稳定，再转回原 dtype。
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value


class DecoderLayer(nn.Module):
    """单层 Decoder Block：PreNorm Attention + PreNorm MLP。"""
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.self_attn = DecoderAttention(config)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """返回该层输出与该层 KV cache。"""
        # 子层 1：注意力残差块。
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 子层 2：MLP 残差块。
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class LocalPreTrainedModel(nn.Module):
    """本地预训练模型基类：负责统一参数初始化。"""
    config_class = DecoderConfig

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

    def _init_weights(self, module: nn.Module) -> None:
        """按配置的 `initializer_range` 初始化线性层和词嵌入。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if self.config.pad_token_id is not None:
                with torch.no_grad():
                    module.weight[self.config.pad_token_id].zero_()

    def post_init(self) -> None:
        """对子模块递归应用初始化逻辑。"""
        self.apply(self._init_weights)


class DecoderBackbone(LocalPreTrainedModel):
    """Decoder 主干：Embedding + N 层 DecoderLayer + 最终 RMSNorm。"""
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入词嵌入层（供外部替换/共享）。"""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """替换输入词嵌入层。"""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> DecoderModelOutput | tuple:
        """执行主干前向，支持训练和增量推理两种路径。"""
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds.")

        # use_cache 未显式传入时，沿用配置默认值。
        use_cache = self.config.use_cache if use_cache is None else use_cache

        # 输入支持 token id 或外部嵌入二选一。
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            batch_size, seq_length = input_ids.shape
        else:
            hidden_states = inputs_embeds
            batch_size, seq_length, _ = inputs_embeds.shape

        # 统一 past_key_values 格式，便于层内直接索引。
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        first_past = past_key_values[0]
        past_key_values_length = 0 if first_past is None else first_past[0].shape[2]

        # 未传 position_ids 时按连续位置自动生成。
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                dtype=torch.long,
                device=hidden_states.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 计算当前步 RoPE 参数和注意力掩码。
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        key_length = past_key_values_length + seq_length
        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            query_len=seq_length,
            key_len=key_length,
            past_key_values_length=past_key_values_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        # 逐层前向传播。
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, present_kv = decoder_layer(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=causal_mask,
                past_key_value=past_key_values[layer_idx],
                use_cache=use_cache,
            )
            # 仅在推理缓存开启时收集本层 cache。
            if use_cache:
                next_decoder_cache = next_decoder_cache + (present_kv,)

        # 最后一层归一化。
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 兼容 tuple 返回风格，便于与常见训练脚本接轨。
        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs = outputs + (next_decoder_cache,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            return outputs

        return DecoderModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
        )


class DecoderForCausalLM(LocalPreTrainedModel):
    """在 DecoderBackbone 之上加语言模型头并提供训练损失。"""
    def __init__(self, config: DecoderConfig):
        super().__init__(config)
        self.model = DecoderBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """透传 backbone 的输入嵌入接口。"""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """透传 backbone 的输入嵌入替换接口。"""
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        """获取 lm_head。"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """替换 lm_head。"""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> CausalLMOutput | tuple:
        """执行 Causal LM 前向，按需计算 teacher-forcing loss。"""
        # 先跑主干拿到最后隐藏状态。
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 右移一位计算自回归交叉熵。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # 兼容 tuple 返回风格。
        if not return_dict:
            result = (logits,)
            if loss is not None:
                result = (loss,) + result
            if outputs.past_key_values is not None:
                result = result + (outputs.past_key_values,)
            if outputs.hidden_states is not None:
                result = result + (outputs.hidden_states,)
            return result

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
