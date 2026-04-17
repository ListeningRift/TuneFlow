"""Base 训练入口文件。"""

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

from ..utils.output_cleanup import ensure_clean_directory, remove_file_if_exists
from ..utils.torch_utils import count_parameters, lazy_import_torch, resolve_torch_device

_TRAIN_LOSS_EMA_ALPHA = 0.1


def _load_id_to_token(vocab_path: Path) -> list[str]:
    """从词表 JSON 读取 `id -> token` 映射。"""
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    id_to_token_raw = payload.get("id_to_token")
    if isinstance(id_to_token_raw, list) and id_to_token_raw:
        return [str(token) for token in id_to_token_raw]

    token_to_id = payload.get("token_to_id")
    if not isinstance(token_to_id, dict) or not token_to_id:
        raise ValueError(f"Invalid tokenizer vocab file: {vocab_path}")

    max_id = max(int(idx) for idx in token_to_id.values())
    id_to_token = ["<UNK>"] * (max_id + 1)
    for token, idx in token_to_id.items():
        int_idx = int(idx)
        if 0 <= int_idx < len(id_to_token):
            id_to_token[int_idx] = str(token)
    return id_to_token


class TokenBinDataset:
    """基于 `.bin + .idx.json` 的随机窗口采样器。"""

    _DTYPE_TO_TYPECODE = {
        "uint16": "H",
        "uint32": "I",
    }

    def __init__(self, idx_path: Path, bin_path_override: Path | None = None):
        self.idx_path = idx_path
        # idx 文件中记录了 dtype / offsets / lengths / num_tokens 等元信息。
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

        # 解析 bin 路径，允许命令行显式覆盖。
        self.bin_path = self._resolve_bin_path(idx_path, idx_payload, bin_path_override)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary token file not found: {self.bin_path}")

        # 使用 mmap 做零拷贝读取，避免把整份语料一次性加载到内存。
        self._bin_file = self.bin_path.open("rb")
        self._mmap = mmap.mmap(self._bin_file.fileno(), length=0, access=mmap.ACCESS_READ)
        self._token_view = memoryview(self._mmap).cast(self.typecode)
        # 缓存“长度足够可采样”的序列索引，减少重复扫描开销。
        self._eligible_cache: dict[int, list[int]] = {}

    @staticmethod
    def _resolve_bin_path(idx_path: Path, idx_payload: dict, bin_path_override: Path | None) -> Path:
        """解析 `.bin` 文件路径。"""
        if bin_path_override is not None:
            return bin_path_override

        idx_bin = Path(str(idx_payload.get("bin_file", "")))
        if idx_bin.is_absolute():
            return idx_bin

        # 优先使用 idx.json 中记录的路径；不存在时退化到“相对 idx 文件目录”的解析策略。
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

    def _sequence_tokens(self, seq_idx: int) -> list[int]:
        """按序列索引读取完整 token 序列。"""
        seq_offset = self.offsets[seq_idx]
        seq_total_len = self.lengths[seq_idx]
        return list(self._token_view[seq_offset : seq_offset + seq_total_len])

    @staticmethod
    def _collect_window_cut_positions(sequence_tokens: list[int], id_to_token: list[str]) -> list[int]:
        """
        收集 NEXT/validation 允许切分的边界位置。

        目标是保证窗口首尾都落在完整结构单元边界上，不在完整音符事件内部截断。
        """
        positions: list[int] = [0]
        idx = 0

        while idx < len(sequence_tokens):
            token_id = int(sequence_tokens[idx])
            if token_id < 0 or token_id >= len(id_to_token):
                return []

            token = id_to_token[token_id]
            if token in {"BOS", "EOS", "FIM_HOLE", "FIM_MID"} or token.startswith("TEMPO_") or token == "BAR":
                idx += 1
                positions.append(idx)
                continue

            if token.startswith("POS_") and idx + 4 < len(sequence_tokens):
                inst_id = int(sequence_tokens[idx + 1])
                pitch_id = int(sequence_tokens[idx + 2])
                dur_id = int(sequence_tokens[idx + 3])
                vel_id = int(sequence_tokens[idx + 4])
                if (
                    0 <= inst_id < len(id_to_token)
                    and 0 <= pitch_id < len(id_to_token)
                    and 0 <= dur_id < len(id_to_token)
                    and 0 <= vel_id < len(id_to_token)
                    and id_to_token[inst_id].startswith("INST_")
                    and id_to_token[pitch_id].startswith("PITCH_")
                    and id_to_token[dur_id].startswith("DUR_")
                    and id_to_token[vel_id].startswith("VEL_")
                ):
                    idx += 5
                    positions.append(idx)
                    continue
            return []

        return positions

    def _sample_aligned_window(
        self,
        rng: random.Random,
        window_len: int,
        *,
        id_to_token: list[str],
        anchor: str = "random",
        exclude_terminal_eos: bool = False,
        max_attempts: int = 128,
    ) -> list[int]:
        """
        采样一个精确长度的结构对齐窗口。

        窗口首尾都落在合法边界上，不会把完整音符事件截成半段。
        """
        if window_len <= 0:
            raise ValueError("window_len must be > 0.")

        min_seq_len = window_len + (1 if exclude_terminal_eos else 0)
        candidates = self._eligible_indices(min_seq_len)
        if not candidates:
            raise ValueError(
                f"No sequence in {self.idx_path} has length >= {min_seq_len}. "
                "Please lower --seq-len or regenerate data."
            )

        for _ in range(max_attempts):
            seq_idx = candidates[rng.randrange(len(candidates))]
            seq_tokens = self._sequence_tokens(seq_idx)
            cut_positions = self._collect_window_cut_positions(seq_tokens, id_to_token)
            if len(cut_positions) < 2:
                continue
            cut_set = set(cut_positions)
            exact_starts = [start for start in cut_positions[:-1] if start + window_len in cut_set]
            if exclude_terminal_eos:
                exact_starts = [start for start in exact_starts if start + window_len == len(seq_tokens) - 1]
            elif anchor == "start":
                exact_starts = [start for start in exact_starts if start == 0]
            elif anchor == "end":
                exact_starts = [start for start in exact_starts if start + window_len == len(seq_tokens)]

            if not exact_starts:
                continue

            start = exact_starts[rng.randrange(len(exact_starts))]
            end = start + window_len
            return seq_tokens[start:end]

        raise ValueError("Unable to sample an aligned window after multiple retries.")

    def _sample_window(self, rng: random.Random, window_len: int, anchor: str = "random") -> list[int]:
        """随机抽取一个连续窗口（长度为 `window_len`）。"""
        if window_len <= 0:
            raise ValueError("window_len must be > 0.")

        candidates = self._eligible_indices(window_len)
        if not candidates:
            raise ValueError(
                f"No sequence in {self.idx_path} has length >= {window_len}. "
                "Please lower --seq-len or regenerate data."
            )

        seq_idx = candidates[rng.randrange(len(candidates))]
        seq_offset = self.offsets[seq_idx]
        seq_total_len = self.lengths[seq_idx]
        if anchor == "start":
            start_in_seq = 0
        elif anchor == "end":
            start_in_seq = seq_total_len - window_len
        else:
            start_in_seq = rng.randrange(seq_total_len - window_len + 1)
        abs_start = seq_offset + start_in_seq
        return list(self._token_view[abs_start : abs_start + window_len])

    def _sample_window_before_eos(self, rng: random.Random, window_len: int) -> list[int]:
        """抽取一个以真实序列末尾为锚点、但不包含最终 EOS 的窗口。"""
        if window_len <= 0:
            raise ValueError("window_len must be > 0.")

        candidates = self._eligible_indices(window_len + 1)
        if not candidates:
            raise ValueError(
                f"No sequence in {self.idx_path} has length >= {window_len + 1}. "
                "Please lower --seq-len or regenerate data."
            )

        seq_idx = candidates[rng.randrange(len(candidates))]
        seq_offset = self.offsets[seq_idx]
        seq_total_len = self.lengths[seq_idx]
        abs_start = seq_offset + seq_total_len - window_len - 1
        return list(self._token_view[abs_start : abs_start + window_len])

    @staticmethod
    def _collect_fim_maskable_units(base_tokens: list[int], id_to_token: list[str]) -> list[tuple[int, int, str, int]]:
        """
        收集训练时允许被 FIM mask 的完整结构单元。

        只允许 mask：
        - `BAR`
        - 完整音符事件 `POS -> INST -> PITCH -> DUR -> VEL`

        `BOS/EOS/TEMPO_*` 以及无法识别的残缺片段都视为不可 mask 边界。
        """
        units: list[tuple[int, int, str, int]] = []
        idx = 0
        group_id = 0

        while idx < len(base_tokens):
            token_id = int(base_tokens[idx])
            if token_id < 0 or token_id >= len(id_to_token):
                idx += 1
                group_id += 1
                continue

            token = id_to_token[token_id]
            if token in {"BOS", "EOS", "FIM_HOLE", "FIM_MID"} or token.startswith("TEMPO_"):
                idx += 1
                group_id += 1
                continue

            if token == "BAR":
                units.append((idx, idx + 1, "bar", group_id))
                idx += 1
                continue

            if token.startswith("POS_") and idx + 4 < len(base_tokens):
                inst_id = int(base_tokens[idx + 1])
                pitch_id = int(base_tokens[idx + 2])
                dur_id = int(base_tokens[idx + 3])
                vel_id = int(base_tokens[idx + 4])
                if (
                    0 <= inst_id < len(id_to_token)
                    and 0 <= pitch_id < len(id_to_token)
                    and 0 <= dur_id < len(id_to_token)
                    and 0 <= vel_id < len(id_to_token)
                    and id_to_token[inst_id].startswith("INST_")
                    and id_to_token[pitch_id].startswith("PITCH_")
                    and id_to_token[dur_id].startswith("DUR_")
                    and id_to_token[vel_id].startswith("VEL_")
                ):
                    units.append((idx, idx + 5, "event", group_id))
                    idx += 5
                    continue

            idx += 1
            group_id += 1

        return units

    @classmethod
    def _choose_fim_hole_bounds(
        cls,
        base_tokens: list[int],
        *,
        id_to_token: list[str],
        rng: random.Random,
        fim_min_span: int,
        fim_max_span: int,
    ) -> tuple[int, int] | None:
        """
        在完整结构单元之间选择连续洞区间。

        洞不会覆盖 `BOS/EOS/TEMPO_*`，也不会跨过这些边界。
        """
        units = cls._collect_fim_maskable_units(base_tokens=base_tokens, id_to_token=id_to_token)
        if len(units) < 2:
            return None

        max_span = min(max(1, fim_max_span), len(base_tokens) - 2)
        min_span = min(max(1, fim_min_span), max_span)
        target_span = min_span if min_span == max_span else rng.randint(min_span, max_span)
        candidate_bounds: list[tuple[int, int, int]] = []

        for start_idx, (start_token, _, _, group_id) in enumerate(units):
            if start_token <= 0:
                continue
            end_token = start_token
            for unit_idx in range(start_idx, len(units)):
                unit_start, unit_end, _, end_group_id = units[unit_idx]
                if end_group_id != group_id:
                    break
                if unit_idx > start_idx and unit_start != end_token:
                    break
                end_token = unit_end
                if end_token >= len(base_tokens):
                    continue
                span = end_token - start_token
                if span < min_span:
                    continue
                if span > max_span:
                    break
                candidate_bounds.append((abs(span - target_span), start_token, end_token))

        if not candidate_bounds:
            return None

        candidate_bounds.sort(key=lambda item: (item[0], item[1], item[2]))
        best_gap = candidate_bounds[0][0]
        near_best = [(start, end) for gap, start, end in candidate_bounds if gap <= best_gap + 4]
        return near_best[rng.randrange(len(near_best))]

    @classmethod
    def _build_fim_example(
        cls,
        base_tokens: list[int],
        rng: random.Random,
        id_to_token: list[str],
        fim_hole_token_id: int,
        fim_mid_token_id: int,
        fim_min_span: int,
        fim_max_span: int,
        append_eos: bool = False,
        eos_token_id: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """
        基于基础序列构造 FIM 样本：
        `prefix + FIM_HOLE + suffix + FIM_MID + middle`

        labels 在 `FIM_MID` 之前全部置为 -100，仅对 middle 区域计入损失。
        """
        length = len(base_tokens)
        if length < 4:
            raise ValueError("FIM base sequence must contain at least 4 tokens.")

        hole_bounds = cls._choose_fim_hole_bounds(
            base_tokens=base_tokens,
            id_to_token=id_to_token,
            rng=rng,
            fim_min_span=fim_min_span,
            fim_max_span=fim_max_span,
        )
        if hole_bounds is None:
            raise ValueError("Unable to find a valid FIM hole made of complete structural units.")

        # 约束 hole 左右都至少保留 1 个 token，避免退化为纯前缀/纯后缀。
        start, end = hole_bounds

        prefix = base_tokens[:start]
        middle = base_tokens[start:end]
        suffix = base_tokens[end:]
        fim_tokens = [*prefix, fim_hole_token_id, *suffix, fim_mid_token_id, *middle]
        if append_eos:
            if eos_token_id is None:
                raise ValueError("append_eos=True requires eos_token_id.")
            fim_tokens.append(eos_token_id)

        labels = fim_tokens.copy()
        fim_mid_pos = len(prefix) + 1 + len(suffix)
        for idx in range(fim_mid_pos + 1):
            labels[idx] = -100
        return fim_tokens, labels

    def sample_batch(
        self,
        torch_mod,
        rng: random.Random,
        batch_size: int,
        seq_len: int,
        device,
        id_to_token: list[str],
        bos_sample_ratio: float = 0.0,
        eos_sample_ratio: float = 0.0,
    ):
        """采样 NEXT batch（labels 与 input_ids 对齐，由模型内部完成 shift）。"""
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for _ in range(batch_size):
            pick = rng.random()
            if pick < bos_sample_ratio:
                anchor = "start"
            elif pick < bos_sample_ratio + eos_sample_ratio:
                anchor = "end"
            else:
                anchor = "random"
            window = self._sample_aligned_window(
                rng=rng,
                window_len=seq_len,
                id_to_token=id_to_token,
                anchor=anchor,
            )
            input_rows.append(window)
            label_rows.append(window.copy())

        input_ids = torch_mod.tensor(input_rows, dtype=torch_mod.long, device=device)
        labels = torch_mod.tensor(label_rows, dtype=torch_mod.long, device=device)
        return input_ids, labels

    def sample_mixed_batch(
        self,
        torch_mod,
        rng: random.Random,
        batch_size: int,
        seq_len: int,
        device,
        id_to_token: list[str],
        fim_ratio: float,
        fim_hole_token_id: int | None,
        fim_mid_token_id: int | None,
        fim_min_span: int,
        fim_max_span: int,
        bos_sample_ratio: float,
        eos_sample_ratio: float,
        fim_eos_ratio: float,
        eos_token_id: int | None,
    ):
        """
        采样 NEXT + FIM 混合 batch。

        返回 `(input_ids, labels, fim_examples)`，其中 `fim_examples` 为当前 batch 的 FIM 条数。
        """
        if not (0.0 <= fim_ratio <= 1.0):
            raise ValueError(f"fim_ratio must be within [0, 1], got {fim_ratio}.")

        use_fim = fim_ratio > 0.0 and fim_hole_token_id is not None and fim_mid_token_id is not None
        if use_fim and seq_len <= 2:
            raise ValueError("seq_len must be > 2 when FIM is enabled.")

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        fim_examples = 0

        for _ in range(batch_size):
            pick_fim = use_fim and (rng.random() < fim_ratio)
            if pick_fim:
                use_fim_eos = (
                    fim_eos_ratio > 0.0
                    and eos_token_id is not None
                    and seq_len > 3
                    and (rng.random() < fim_eos_ratio)
                )
                fim_input = None
                fim_labels = None
                for _ in range(16):
                    if use_fim_eos:
                        base_tokens = self._sample_aligned_window(
                            rng=rng,
                            window_len=seq_len - 3,
                            id_to_token=id_to_token,
                            exclude_terminal_eos=True,
                        )
                    else:
                        base_tokens = self._sample_aligned_window(
                            rng=rng,
                            window_len=seq_len - 2,
                            id_to_token=id_to_token,
                        )
                    try:
                        fim_input, fim_labels = self._build_fim_example(
                            base_tokens=base_tokens,
                            rng=rng,
                            id_to_token=id_to_token,
                            fim_hole_token_id=fim_hole_token_id,
                            fim_mid_token_id=fim_mid_token_id,
                            fim_min_span=fim_min_span,
                            fim_max_span=fim_max_span,
                            append_eos=use_fim_eos,
                            eos_token_id=eos_token_id,
                        )
                        break
                    except ValueError:
                        continue
                if fim_input is None or fim_labels is None:
                    raise ValueError("Unable to sample a valid FIM example after multiple retries.")
                input_rows.append(fim_input)
                label_rows.append(fim_labels)
                fim_examples += 1
            else:
                pick = rng.random()
                if pick < bos_sample_ratio:
                    anchor = "start"
                elif pick < bos_sample_ratio + eos_sample_ratio:
                    anchor = "end"
                else:
                    anchor = "random"
                window = self._sample_aligned_window(
                    rng=rng,
                    window_len=seq_len,
                    id_to_token=id_to_token,
                    anchor=anchor,
                )
                input_rows.append(window)
                label_rows.append(window.copy())

        input_ids = torch_mod.tensor(input_rows, dtype=torch_mod.long, device=device)
        labels = torch_mod.tensor(label_rows, dtype=torch_mod.long, device=device)
        return input_ids, labels, fim_examples


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数：覆盖训练、评估、保存、恢复等核心开关。"""
    parser = argparse.ArgumentParser(description="TuneFlow base training (real-data loop).")
    # 配置与数据路径
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
    # 训练超参数
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
    parser.add_argument(
        "--fim-ratio",
        type=float,
        default=0.15,
        help="Fraction of FIM samples in each training batch, within [0, 1].",
    )
    parser.add_argument(
        "--fim-min-span",
        type=int,
        default=8,
        help="Minimum FIM hole length in tokens.",
    )
    parser.add_argument(
        "--fim-max-span",
        type=int,
        default=64,
        help="Maximum FIM hole length in tokens.",
    )
    parser.add_argument(
        "--bos-sample-ratio",
        type=float,
        default=0.1,
        help="Fraction of NEXT samples anchored at sequence start to strengthen BOS-prefix continuation.",
    )
    parser.add_argument(
        "--eos-sample-ratio",
        type=float,
        default=0.1,
        help="Fraction of NEXT samples anchored at sequence end to strengthen EOS prediction.",
    )
    parser.add_argument(
        "--fim-eos-ratio",
        type=float,
        default=1.0,
        help="Fraction of FIM samples that explicitly supervise EOS after middle generation.",
    )
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
    # 保存与恢复行为
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
        default=Path("outputs/checkpoints/minimal_real_train"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Optional JSONL metrics path; default is `<output-dir>/metrics.jsonl`.",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args(argv)


def _autocast_context(torch_mod, use_amp: bool, device_type: str, amp_dtype):
    """根据配置返回 autocast 上下文；未启用时返回空上下文。"""
    if not use_amp:
        return nullcontext()
    return torch_mod.autocast(device_type=device_type, dtype=amp_dtype)


def _resolve_precision(torch_mod, requested: str, device):
    """
    解析精度策略，返回：
    (effective_name, use_amp, amp_dtype, use_grad_scaler)
    """
    if requested == "fp32":
        return "fp32", False, None, False

    if device.type != "cuda":
        # CPU 不启用半精度 autocast，半精度请求会自动回退到 fp32。
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
        # LambdaLR 传入 0-based step，这里统一换算到 1-based。
        step = current_step_zero_based + 1
        # 1) warmup 阶段线性升温。
        if warmup > 0 and step <= warmup:
            return float(step) / float(warmup)

        if total_steps <= warmup:
            return floor

        # 2) warmup 后按 schedule 衰减到 floor。
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
    """向 JSONL 追加一条结构化指标记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
def _tokens_seen_for_step(step: int, *, effective_batch: int, seq_len: int) -> int:
    """Derive the approximate number of training tokens consumed by a step."""
    return max(0, int(step)) * max(1, int(effective_batch)) * max(1, int(seq_len))


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
    id_to_token: list[str],
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
                id_to_token=id_to_token,
            )
            # 评估阶段与训练保持一致的精度策略，避免统计口径偏差。
            with _autocast_context(torch_mod, use_amp=use_amp, device_type=device.type, amp_dtype=amp_dtype):
                outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
            losses.append(float(outputs.loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main(argv: list[str] | None = None) -> None:
    """训练主流程：初始化 ->（可选恢复）-> 训练 -> 评估 -> 保存。"""
    args = _parse_args(argv)
    torch = lazy_import_torch()

    from src.model import DecoderConfig, DecoderForCausalLM

    if args.grad_accum_steps <= 0:
        raise SystemExit("--grad-accum-steps must be > 0.")
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0.")
    if not (0.0 <= args.fim_ratio <= 1.0):
        raise SystemExit("--fim-ratio must be within [0, 1].")
    if not (0.0 <= args.bos_sample_ratio <= 1.0):
        raise SystemExit("--bos-sample-ratio must be within [0, 1].")
    if not (0.0 <= args.eos_sample_ratio <= 1.0):
        raise SystemExit("--eos-sample-ratio must be within [0, 1].")
    if args.bos_sample_ratio + args.eos_sample_ratio > 1.0:
        raise SystemExit("--bos-sample-ratio + --eos-sample-ratio must be <= 1.")
    if not (0.0 <= args.fim_eos_ratio <= 1.0):
        raise SystemExit("--fim-eos-ratio must be within [0, 1].")
    if args.fim_min_span <= 0:
        raise SystemExit("--fim-min-span must be > 0.")
    if args.fim_max_span <= 0:
        raise SystemExit("--fim-max-span must be > 0.")

    # 先设全局随机种子，再创建独立 run_rng（用于数据采样）。
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
    fim_hole_token_id = config.special_token_ids.get("FIM_HOLE")
    fim_mid_token_id = config.special_token_ids.get("FIM_MID")
    eos_token_id = config.eos_token_id
    if args.fim_ratio > 0 and (fim_hole_token_id is None or fim_mid_token_id is None):
        raise SystemExit(
            "--fim-ratio > 0 but FIM_HOLE/FIM_MID token id is missing in model config."
        )
    if not config.vocab_path:
        raise SystemExit("Training window sampling requires model config to provide vocab_path.")
    vocab_path = Path(config.vocab_path)
    if not vocab_path.is_absolute():
        vocab_path = vocab_path.resolve()
    if not vocab_path.exists():
        raise SystemExit(f"Tokenizer vocab file not found for window sampling: {vocab_path}")
    id_to_token = _load_id_to_token(vocab_path)

    train_dataset = TokenBinDataset(args.train_idx, args.train_bin)
    valid_dataset = TokenBinDataset(args.valid_idx, args.valid_bin) if args.valid_idx.exists() else None

    device = resolve_torch_device(torch, args.device)
    # 精度解析：决定是否使用 AMP、使用哪种 dtype、是否启用 GradScaler。
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
        # 恢复训练状态（模型/优化器/scheduler/scaler/RNG）。
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
    total_planned_tokens = args.steps * effective_batch * args.seq_len
    remaining_steps = max(0, args.steps - start_step)
    remaining_planned_tokens = remaining_steps * effective_batch * args.seq_len
    approx_total_data_passes = total_planned_tokens / max(1, train_dataset.num_tokens)
    approx_remaining_data_passes = remaining_planned_tokens / max(1, train_dataset.num_tokens)
    print(f"[train_base] model_type={config.model_type} vocab={config.vocab_size}")
    print(
        f"[train_base] params={total_params:,} device={device} precision={precision_name} "
        f"steps={args.steps} batch={args.batch_size} grad_accum={args.grad_accum_steps} "
        f"effective_batch={effective_batch} seq_len={args.seq_len} fim_ratio={args.fim_ratio:.2f} "
        f"bos_ratio={args.bos_sample_ratio:.2f} eos_ratio={args.eos_sample_ratio:.2f} fim_eos_ratio={args.fim_eos_ratio:.2f}"
    )
    print(
        f"[train_base] train={train_dataset.idx_path} ({train_dataset.num_sequences} seqs, "
        f"{train_dataset.num_tokens} tokens)"
    )
    print(
        f"[train_base] planned_train_tokens={total_planned_tokens:,} "
        f"(remaining={remaining_planned_tokens:,}) "
        f"approx_data_passes={approx_total_data_passes:.4f} "
        f"(remaining={approx_remaining_data_passes:.4f})"
    )
    if valid_dataset is not None:
        print(
            f"[train_base] valid={valid_dataset.idx_path} ({valid_dataset.num_sequences} seqs, "
            f"{valid_dataset.num_tokens} tokens)"
        )

    if args.resume_from is None:
        ensure_clean_directory(args.output_dir)
        if args.metrics_path is not None:
            remove_file_if_exists(args.metrics_path)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    # 默认把指标写到 output_dir/metrics.jsonl。
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
            "train_num_sequences": train_dataset.num_sequences,
            "train_num_tokens": train_dataset.num_tokens,
            "planned_train_tokens": total_planned_tokens,
            "remaining_planned_tokens": remaining_planned_tokens,
            "approx_total_data_passes": approx_total_data_passes,
            "approx_remaining_data_passes": approx_remaining_data_passes,
            "fim_ratio": args.fim_ratio,
            "bos_sample_ratio": args.bos_sample_ratio,
            "eos_sample_ratio": args.eos_sample_ratio,
            "fim_eos_ratio": args.fim_eos_ratio,
            "fim_min_span": args.fim_min_span,
            "fim_max_span": args.fim_max_span,
            "resume_from": None if args.resume_from is None else str(args.resume_from),
        },
    )

    model.train()
    run_start = time.perf_counter()
    interval_start = run_start
    interval_tokens = 0
    train_loss_ema: float | None = None

    try:
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0
            step_fim_examples = 0
            for _ in range(args.grad_accum_steps):
                input_ids, labels, fim_examples = train_dataset.sample_mixed_batch(
                    torch_mod=torch,
                    rng=run_rng,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    device=device,
                    id_to_token=id_to_token,
                    fim_ratio=args.fim_ratio,
                    fim_hole_token_id=fim_hole_token_id,
                    fim_mid_token_id=fim_mid_token_id,
                    fim_min_span=args.fim_min_span,
                    fim_max_span=args.fim_max_span,
                    bos_sample_ratio=args.bos_sample_ratio,
                    eos_sample_ratio=args.eos_sample_ratio,
                    fim_eos_ratio=args.fim_eos_ratio,
                    eos_token_id=eos_token_id,
                )
                with _autocast_context(
                    torch_mod=torch, use_amp=use_amp, device_type=device.type, amp_dtype=amp_dtype
                ):
                    outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
                    raw_loss = outputs.loss
                    # 梯度累积：每个 micro-step 的 loss 按累积步数缩放。
                    scaled_loss = raw_loss / args.grad_accum_steps

                if not torch.isfinite(raw_loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {float(raw_loss.item())}")

                if scaler is not None:
                    # fp16 路径：先 scale 再 backward。
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                step_loss += float(raw_loss.item())
                step_fim_examples += int(fim_examples)

            step_loss /= args.grad_accum_steps
            if train_loss_ema is None:
                train_loss_ema = step_loss
            else:
                train_loss_ema = (
                    (_TRAIN_LOSS_EMA_ALPHA * step_loss)
                    + ((1.0 - _TRAIN_LOSS_EMA_ALPHA) * float(train_loss_ema))
                )

            if scaler is not None:
                # clip 前先 unscale，确保梯度范数语义正确。
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler is not None:
                # fp16 路径：step + update。
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            current_lr = float(optimizer.param_groups[0]["lr"])
            interval_tokens += args.batch_size * args.seq_len * args.grad_accum_steps
            step_total_examples = args.batch_size * args.grad_accum_steps
            step_fim_ratio = step_fim_examples / max(1, step_total_examples)
            tokens_seen = _tokens_seen_for_step(
                step,
                effective_batch=effective_batch,
                seq_len=args.seq_len,
            )

            should_log = step == start_step + 1 or (args.log_every > 0 and step % args.log_every == 0)
            if should_log:
                now = time.perf_counter()
                elapsed = max(1e-9, now - interval_start)
                toks_per_sec = interval_tokens / elapsed
                print(
                    f"[train_base] step={step}/{args.steps} "
                    f"loss={step_loss:.6f} "
                    f"lr={current_lr:.6e} "
                    f"tok/s={toks_per_sec:.1f} "
                    f"fim_in_batch={step_fim_examples}/{step_total_examples}({step_fim_ratio:.2f})"
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
                        "fim_examples": step_fim_examples,
                        "fim_ratio_in_batch": step_fim_ratio,
                        "tokens_seen": tokens_seen,
                        "train_loss_ema": train_loss_ema,
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
                    id_to_token=id_to_token,
                )
                print(f"[train_base] eval step={step} valid_loss={val_loss:.6f}")
                best_valid_loss_so_far = min(best_valid_loss, val_loss)
                overfit_gap = (
                    (val_loss - float(train_loss_ema))
                    if train_loss_ema is not None
                    else float("nan")
                )
                _append_metrics(
                    metrics_path,
                    {
                        "event": "eval",
                        "time": time.time(),
                        "step": step,
                        "valid_loss": val_loss,
                        "tokens_seen": tokens_seen,
                        "train_loss_ema": train_loss_ema,
                        "best_valid_loss_so_far": best_valid_loss_so_far,
                        "overfit_gap": overfit_gap,
                    },
                )
                if args.save_best and val_loss < best_valid_loss:
                    # 仅在指标变优时覆盖 best.pt。
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
                # 同步一份 latest.pt，便于自动恢复。
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
        # 无论训练是否异常退出，都确保释放数据句柄。
        train_dataset.close()
        if valid_dataset is not None:
            valid_dataset.close()


if __name__ == "__main__":
    main()



