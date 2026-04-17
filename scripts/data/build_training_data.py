#!/usr/bin/env python
"""TuneFlow 训练数据打包脚本。"""

from __future__ import annotations

import argparse
import array
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils.output_cleanup import remove_file_if_exists, remove_matching_children

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "缺少依赖：pyyaml。请先执行 `python -m pip install pyyaml`。"
    ) from exc


@dataclass
class BuildConfig:
    """打包配置。"""

    tokenized_dir: str = "data/tokenized"
    vocab_path: str = "data/tokenized/tokenizer_vocab.json"
    splits: List[str] = None  # type: ignore[assignment]
    dtype: str = "auto"
    strict_oov: bool = True

    def __post_init__(self) -> None:
        if self.splits is None:
            self.splits = ["train", "valid", "test", "eval"]

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> "BuildConfig":
        """从 YAML 字典构建配置，并忽略未知字段。"""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        safe = {k: v for k, v in data.items() if k in valid_keys}
        cfg = cls(**safe)
        if cfg.dtype not in {"auto", "uint16", "uint32"}:
            raise ValueError("dtype 仅支持 auto/uint16/uint32")
        return cfg


def load_config(path: Path) -> BuildConfig:
    """读取 build 配置。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("build 配置必须是字典（mapping）")
    return BuildConfig.from_mapping(raw)


def load_vocab(path: Path) -> Dict[str, int]:
    """读取词表 token->id。"""
    if not path.exists():
        raise FileNotFoundError(f"词表不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    token_to_id = data.get("token_to_id")
    if not isinstance(token_to_id, dict):
        raise ValueError("词表格式错误：缺少 token_to_id")
    return {str(k): int(v) for k, v in token_to_id.items()}


def choose_array_type(dtype: str, vocab_size: int) -> Tuple[str, str]:
    """根据配置与词表大小选择二进制存储类型。"""
    if dtype == "uint16":
        return "H", "uint16"
    if dtype == "uint32":
        return "I", "uint32"
    if vocab_size <= 65535:
        return "H", "uint16"
    return "I", "uint32"


def encode_tok_line(line: str, vocab: Dict[str, int]) -> Tuple[List[int], int]:
    """将一行 token 文本编码成 id 序列，并返回 OOV 数量。"""
    tokens = [t for t in line.strip().split(" ") if t]
    ids: List[int] = []
    oov = 0
    for t in tokens:
        if t in vocab:
            ids.append(vocab[t])
        else:
            oov += 1
    return ids, oov


def write_bin_idx(
    tok_path: Path,
    bin_path: Path,
    idx_path: Path,
    vocab: Dict[str, int],
    type_code: str,
    dtype_name: str,
    strict_oov: bool,
) -> Dict[str, object]:
    """读取 `.tok` 并写出 `.bin` 与 `.idx.json`。"""
    offsets: List[int] = []
    lengths: List[int] = []
    total_tokens = 0
    total_oov = 0
    total_lines = 0
    empty_lines = 0

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    with tok_path.open("r", encoding="utf-8") as fin, bin_path.open("wb") as fout:
        for line in fin:
            total_lines += 1
            ids, oov = encode_tok_line(line, vocab)
            total_oov += oov
            if not ids:
                empty_lines += 1
                continue
            if strict_oov and oov > 0:
                raise ValueError(
                    f"检测到 OOV（strict_oov=true）：file={tok_path} line={total_lines} oov={oov}"
                )
            offsets.append(total_tokens)
            lengths.append(len(ids))
            total_tokens += len(ids)
            arr = array.array(type_code, ids)
            arr.tofile(fout)

    idx_payload = {
        "tok_file": str(tok_path),
        "bin_file": str(bin_path),
        "dtype": dtype_name,
        "num_sequences": len(lengths),
        "num_tokens": total_tokens,
        "offsets": offsets,
        "lengths": lengths,
        "total_lines": total_lines,
        "empty_lines": empty_lines,
        "oov_count": total_oov,
    }
    idx_path.write_text(json.dumps(idx_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return idx_payload


def dump_json(path: Path, payload: Dict[str, object]) -> None:
    """写出 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process(config: BuildConfig, report_path: Path) -> None:
    """执行训练数据打包流程。"""
    tokenized_dir = Path(config.tokenized_dir)
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    remove_matching_children(tokenized_dir, ["*.bin", "*.idx.json"])
    if report_path.parent.resolve() != tokenized_dir.resolve():
        remove_file_if_exists(report_path)
    vocab = load_vocab(Path(config.vocab_path))
    type_code, dtype_name = choose_array_type(config.dtype, len(vocab))

    split_reports: Dict[str, Dict[str, object]] = {}
    for split in config.splits:
        tok_path = tokenized_dir / f"{split}.tok"
        if not tok_path.exists():
            split_reports[split] = {"status": "skipped", "reason": "tok_file_not_found"}
            continue
        bin_path = tokenized_dir / f"{split}.bin"
        idx_path = tokenized_dir / f"{split}.idx.json"
        idx_payload = write_bin_idx(
            tok_path=tok_path,
            bin_path=bin_path,
            idx_path=idx_path,
            vocab=vocab,
            type_code=type_code,
            dtype_name=dtype_name,
            strict_oov=config.strict_oov,
        )
        split_reports[split] = {"status": "ok", **idx_payload}
        print(
            f"[build] split={split} seqs={idx_payload['num_sequences']} "
            f"tokens={idx_payload['num_tokens']} oov={idx_payload['oov_count']}"
        )

    report = {
        "config": asdict(config),
        "vocab_size": len(vocab),
        "dtype": dtype_name,
        "split_reports": split_reports,
    }
    dump_json(report_path, report)
    print(f"[build] report -> {report_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="将 `.tok` 转为训练可加载的 `.bin/.idx`。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/build_training.yaml"),
        help="build 配置路径。",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/data/build_training_report.json"),
        help="打包报告输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    config = load_config(args.config)
    process(config=config, report_path=args.report_path)


if __name__ == "__main__":
    main()
