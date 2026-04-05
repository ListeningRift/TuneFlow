#!/usr/bin/env python
"""TuneFlow 数据构建验收检查脚本。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


DTYPE_BYTES = {"uint16": 2, "uint32": 4}


@dataclass
class ValidateConfig:
    """验收检查配置。"""

    base_dir: Path = Path("data/base")
    eval_file: Path = Path("data/eval/fixed_eval.jsonl")
    tokenized_dir: Path = Path("data/tokenized")
    token_stats_path: Path = Path("data/tokenized/token_stats.json")
    vocab_path: Path = Path("data/tokenized/tokenizer_vocab.json")
    build_report_path: Path = Path("outputs/reports/data/build_training_report.json")
    report_path: Path = Path("outputs/reports/data/validate_data_report.json")
    required_splits: List[str] = field(default_factory=lambda: ["train", "valid", "test", "eval"])


def count_non_empty_lines(path: Path) -> int:
    """统计文件中非空行数。"""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_json(path: Path) -> Dict[str, object]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def check_exists(path: Path, failures: List[str], label: str) -> None:
    """检查路径存在。"""
    if not path.exists():
        failures.append(f"{label} 不存在: {path}")


def check_token_stats(stats: Dict[str, object], failures: List[str]) -> None:
    """检查 token_stats 关键指标。"""
    oov_count = int(stats.get("oov_count", -1))
    invalid_rows = int(stats.get("invalid_rows", -1))
    if oov_count != 0:
        failures.append(f"token_stats.oov_count 应为 0，实际为 {oov_count}")
    if invalid_rows != 0:
        failures.append(f"token_stats.invalid_rows 应为 0，实际为 {invalid_rows}")


def validate_idx_payload(
    split: str,
    idx_payload: Dict[str, object],
    tok_path: Path,
    bin_path: Path,
    failures: List[str],
) -> None:
    """检查单个 split 的 idx/bin 一致性。"""
    dtype = str(idx_payload.get("dtype", ""))
    if dtype not in DTYPE_BYTES:
        failures.append(f"{split}.idx.json dtype 非法: {dtype}")
        return
    bytes_per_token = DTYPE_BYTES[dtype]

    num_sequences = int(idx_payload.get("num_sequences", -1))
    num_tokens = int(idx_payload.get("num_tokens", -1))
    offsets = idx_payload.get("offsets", [])
    lengths = idx_payload.get("lengths", [])

    if not isinstance(offsets, list) or not isinstance(lengths, list):
        failures.append(f"{split}.idx.json offsets/lengths 不是列表")
        return
    if len(offsets) != len(lengths):
        failures.append(f"{split}.idx.json offsets 与 lengths 长度不一致")
        return
    if num_sequences != len(lengths):
        failures.append(
            f"{split}.idx.json num_sequences={num_sequences} 与 lengths={len(lengths)} 不一致"
        )

    if lengths:
        if int(offsets[0]) != 0:
            failures.append(f"{split}.idx.json offsets[0] 应为 0")
        for i in range(1, len(lengths)):
            prev_end = int(offsets[i - 1]) + int(lengths[i - 1])
            if int(offsets[i]) != prev_end:
                failures.append(f"{split}.idx.json offsets 在位置 {i} 不连续")
                break
        token_sum = sum(int(x) for x in lengths)
        if token_sum != num_tokens:
            failures.append(
                f"{split}.idx.json num_tokens={num_tokens} 与 lengths 求和={token_sum} 不一致"
            )
    else:
        if num_tokens != 0:
            failures.append(f"{split}.idx.json lengths 为空但 num_tokens={num_tokens} 非 0")

    if tok_path.exists():
        line_count = count_non_empty_lines(tok_path)
        if line_count != num_sequences:
            failures.append(
                f"{split}.tok 非空行数={line_count} 与 idx.num_sequences={num_sequences} 不一致"
            )

    if bin_path.exists():
        expected_size = num_tokens * bytes_per_token
        actual_size = bin_path.stat().st_size
        if actual_size != expected_size:
            failures.append(
                f"{split}.bin 大小={actual_size} 与期望={expected_size}（num_tokens*dtype_bytes）不一致"
            )


def run_checks(cfg: ValidateConfig) -> Tuple[List[str], Dict[str, object]]:
    """执行全部验收检查，返回失败列表与汇总详情。"""
    failures: List[str] = []
    details: Dict[str, object] = {"required_splits": cfg.required_splits}

    required_paths = {
        "base/train.jsonl": cfg.base_dir / "train.jsonl",
        "base/valid.jsonl": cfg.base_dir / "valid.jsonl",
        "base/test.jsonl": cfg.base_dir / "test.jsonl",
        "eval/fixed_eval.jsonl": cfg.eval_file,
        "tokenizer_vocab.json": cfg.vocab_path,
        "token_stats.json": cfg.token_stats_path,
        "build_training_report.json": cfg.build_report_path,
    }
    for label, path in required_paths.items():
        check_exists(path, failures, label)

    if cfg.token_stats_path.exists():
        token_stats = load_json(cfg.token_stats_path)
        check_token_stats(token_stats, failures)
        details["token_stats_head"] = {
            "oov_count": token_stats.get("oov_count"),
            "invalid_rows": token_stats.get("invalid_rows"),
            "total_rows": token_stats.get("total_rows"),
            "vocab_size": token_stats.get("vocab_size"),
        }

    build_report: Dict[str, object] = {}
    split_reports: Dict[str, object] = {}
    if cfg.build_report_path.exists():
        build_report = load_json(cfg.build_report_path)
        split_reports = build_report.get("split_reports", {})
        if not isinstance(split_reports, dict):
            failures.append("build_training_report.json 的 split_reports 字段格式错误")
            split_reports = {}

    details["split_checks"] = {}
    for split in cfg.required_splits:
        tok_path = cfg.tokenized_dir / f"{split}.tok"
        bin_path = cfg.tokenized_dir / f"{split}.bin"
        idx_path = cfg.tokenized_dir / f"{split}.idx.json"

        check_exists(tok_path, failures, f"{split}.tok")
        check_exists(bin_path, failures, f"{split}.bin")
        check_exists(idx_path, failures, f"{split}.idx.json")

        split_detail: Dict[str, object] = {
            "tok_path": str(tok_path),
            "bin_path": str(bin_path),
            "idx_path": str(idx_path),
        }

        if split in split_reports and isinstance(split_reports[split], dict):
            status = str(split_reports[split].get("status"))
            split_detail["build_report_status"] = status
            if status != "ok":
                failures.append(f"build_report 中 split={split} 状态不是 ok: {status}")
        else:
            failures.append(f"build_report 中缺少 split={split} 记录")

        if idx_path.exists():
            idx_payload = load_json(idx_path)
            validate_idx_payload(
                split=split,
                idx_payload=idx_payload,
                tok_path=tok_path,
                bin_path=bin_path,
                failures=failures,
            )
            split_detail["num_sequences"] = idx_payload.get("num_sequences")
            split_detail["num_tokens"] = idx_payload.get("num_tokens")
            split_detail["dtype"] = idx_payload.get("dtype")
        details["split_checks"][split] = split_detail

    return failures, details


def dump_report(path: Path, payload: Dict[str, object]) -> None:
    """写出验收报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> ValidateConfig:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="验收检查数据构建产物完整性与一致性。")
    parser.add_argument("--base-dir", type=Path, default=Path("data/base"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval/fixed_eval.jsonl"))
    parser.add_argument("--tokenized-dir", type=Path, default=Path("data/tokenized"))
    parser.add_argument("--token-stats-path", type=Path, default=Path("data/tokenized/token_stats.json"))
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("data/tokenized/tokenizer_vocab.json"),
    )
    parser.add_argument(
        "--build-report-path",
        type=Path,
        default=Path("outputs/reports/data/build_training_report.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/data/validate_data_report.json"),
    )
    parser.add_argument(
        "--required-splits",
        nargs="+",
        default=["train", "valid", "test", "eval"],
    )
    ns = parser.parse_args()
    return ValidateConfig(
        base_dir=ns.base_dir,
        eval_file=ns.eval_file,
        tokenized_dir=ns.tokenized_dir,
        token_stats_path=ns.token_stats_path,
        vocab_path=ns.vocab_path,
        build_report_path=ns.build_report_path,
        report_path=ns.report_path,
        required_splits=list(ns.required_splits),
    )


def main() -> None:
    """程序入口。"""
    cfg = parse_args()
    failures, details = run_checks(cfg)
    payload = {
        "passed": len(failures) == 0,
        "failure_count": len(failures),
        "failures": failures,
        "details": details,
        "config": {
            "base_dir": str(cfg.base_dir),
            "eval_file": str(cfg.eval_file),
            "tokenized_dir": str(cfg.tokenized_dir),
            "token_stats_path": str(cfg.token_stats_path),
            "vocab_path": str(cfg.vocab_path),
            "build_report_path": str(cfg.build_report_path),
            "required_splits": cfg.required_splits,
        },
    }
    dump_report(cfg.report_path, payload)

    if failures:
        print(f"[validate] failed count={len(failures)}")
        for msg in failures[:20]:
            print(f"[validate] - {msg}")
        print(f"[validate] report -> {cfg.report_path}")
        raise SystemExit(1)

    print("[validate] passed")
    print(f"[validate] report -> {cfg.report_path}")


if __name__ == "__main__":
    main()
