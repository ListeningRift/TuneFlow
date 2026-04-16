"""Helpers for reading and aligning training metrics.jsonl files."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

_STEP_CHECKPOINT_RE = re.compile(r"^step_(\d+)\.pt$")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tokens_seen_for_step(step: int, *, effective_batch: int | None, seq_len: int | None) -> int | None:
    if effective_batch is None or seq_len is None:
        return None
    return max(0, int(step)) * max(1, int(effective_batch)) * max(1, int(seq_len))


def resolve_metrics_path(checkpoint_dir: Path, metrics_path: Path | None = None) -> Path | None:
    """Resolve a metrics.jsonl path relative to a checkpoint directory."""
    candidate = metrics_path if metrics_path is not None else (checkpoint_dir / "metrics.jsonl")
    if not candidate.exists():
        return None
    return candidate.resolve()


def load_training_metrics(metrics_path: Path | None) -> dict[str, Any]:
    """Load and align training/eval events from metrics.jsonl."""
    if metrics_path is None:
        return {
            "metrics_path": None,
            "run": {},
            "train_by_step": {},
            "eval_by_step": {},
        }

    run_payload: dict[str, Any] = {}
    train_by_step: dict[int, dict[str, Any]] = {}
    eval_by_step: dict[int, dict[str, Any]] = {}
    effective_batch: int | None = None
    seq_len: int | None = None
    train_loss_ema: float | None = None
    best_valid_loss = float("inf")

    with metrics_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            event = str(payload.get("event", ""))
            if event == "run_start":
                run_payload = dict(payload)
                effective_batch = _safe_int(payload.get("effective_batch"))
                seq_len = _safe_int(payload.get("seq_len"))
                continue

            step = _safe_int(payload.get("step"))
            if step is None:
                continue

            if event == "train":
                loss_value = _safe_float(payload.get("loss"))
                logged_ema = _safe_float(payload.get("train_loss_ema"))
                if logged_ema is not None:
                    train_loss_ema = logged_ema
                elif loss_value is not None:
                    if train_loss_ema is None:
                        train_loss_ema = loss_value
                    else:
                        train_loss_ema = (0.1 * loss_value) + (0.9 * float(train_loss_ema))

                entry = dict(payload)
                entry["tokens_seen"] = (
                    _safe_int(payload.get("tokens_seen"))
                    if _safe_int(payload.get("tokens_seen")) is not None
                    else _tokens_seen_for_step(step, effective_batch=effective_batch, seq_len=seq_len)
                )
                entry["train_loss_ema"] = train_loss_ema
                train_by_step[step] = entry
                continue

            if event == "eval":
                valid_loss = _safe_float(payload.get("valid_loss"))
                if valid_loss is not None:
                    best_valid_loss = min(best_valid_loss, valid_loss)

                logged_ema = _safe_float(payload.get("train_loss_ema"))
                aligned_train_ema = logged_ema if logged_ema is not None else train_loss_ema
                overfit_gap = _safe_float(payload.get("overfit_gap"))
                if overfit_gap is None and valid_loss is not None and aligned_train_ema is not None:
                    overfit_gap = valid_loss - aligned_train_ema

                entry = dict(payload)
                entry["tokens_seen"] = (
                    _safe_int(payload.get("tokens_seen"))
                    if _safe_int(payload.get("tokens_seen")) is not None
                    else _tokens_seen_for_step(step, effective_batch=effective_batch, seq_len=seq_len)
                )
                entry["train_loss_ema"] = aligned_train_ema
                entry["best_valid_loss_so_far"] = (
                    _safe_float(payload.get("best_valid_loss_so_far"))
                    if _safe_float(payload.get("best_valid_loss_so_far")) is not None
                    else (best_valid_loss if math.isfinite(best_valid_loss) else None)
                )
                entry["overfit_gap"] = overfit_gap
                eval_by_step[step] = entry

    return {
        "metrics_path": str(metrics_path),
        "run": run_payload,
        "train_by_step": train_by_step,
        "eval_by_step": eval_by_step,
    }


def training_metrics_history(metrics_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return step-sorted train/eval metric histories for reporting and plotting."""
    train_history = [
        {"step": int(step), **dict(payload)}
        for step, payload in sorted(metrics_payload.get("train_by_step", {}).items(), key=lambda item: int(item[0]))
    ]
    eval_history = [
        {"step": int(step), **dict(payload)}
        for step, payload in sorted(metrics_payload.get("eval_by_step", {}).items(), key=lambda item: int(item[0]))
    ]
    return {
        "train": train_history,
        "eval": eval_history,
    }


def summarize_training_metrics(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a compact training-health summary from metrics.jsonl history."""
    history = training_metrics_history(metrics_payload)
    train_history = history["train"]
    eval_history = history["eval"]

    last_train = train_history[-1] if train_history else {}
    last_eval = eval_history[-1] if eval_history else {}

    best_eval: dict[str, Any] | None = None
    best_valid_loss: float | None = None
    for row in eval_history:
        valid_loss = _safe_float(row.get("valid_loss"))
        if valid_loss is None:
            continue
        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_eval = row

    plateau_eval_streak = 0
    if best_eval is not None and eval_history:
        seen_best = False
        for row in reversed(eval_history):
            if int(row.get("step", -1)) == int(best_eval.get("step", -2)):
                seen_best = True
                break
            plateau_eval_streak += 1
        if not seen_best:
            plateau_eval_streak = 0

    train_losses = [_safe_float(row.get("loss")) for row in train_history]
    train_emas = [_safe_float(row.get("train_loss_ema")) for row in train_history]
    tok_per_sec_values = [_safe_float(row.get("tok_per_sec")) for row in train_history]
    fim_ratio_values = [_safe_float(row.get("fim_ratio_in_batch")) for row in train_history]

    finite_train_losses = [value for value in train_losses if value is not None]
    finite_train_emas = [value for value in train_emas if value is not None]
    finite_tok_per_sec = [value for value in tok_per_sec_values if value is not None]
    finite_fim_ratios = [value for value in fim_ratio_values if value is not None]

    previous_eval = eval_history[-2] if len(eval_history) >= 2 else {}
    previous_train = train_history[-2] if len(train_history) >= 2 else {}

    latest_valid_loss = _safe_float(last_eval.get("valid_loss"))
    previous_valid_loss = _safe_float(previous_eval.get("valid_loss"))
    latest_train_loss_ema = _safe_float(last_eval.get("train_loss_ema", last_train.get("train_loss_ema")))
    previous_train_loss_ema = _safe_float(previous_train.get("train_loss_ema"))

    return {
        "run": dict(metrics_payload.get("run", {})),
        "metrics_path": metrics_payload.get("metrics_path"),
        "train_event_count": len(train_history),
        "eval_event_count": len(eval_history),
        "last_train_step": last_train.get("step"),
        "last_eval_step": last_eval.get("step"),
        "tokens_seen_last": last_eval.get("tokens_seen", last_train.get("tokens_seen")),
        "latest_train_loss": _safe_float(last_train.get("loss")),
        "latest_train_loss_ema": latest_train_loss_ema,
        "latest_valid_loss": latest_valid_loss,
        "best_valid_loss": best_valid_loss,
        "best_valid_step": None if best_eval is None else best_eval.get("step"),
        "best_valid_tokens_seen": None if best_eval is None else best_eval.get("tokens_seen"),
        "latest_overfit_gap": _safe_float(last_eval.get("overfit_gap")),
        "latest_valid_loss_delta": (
            None
            if latest_valid_loss is None or previous_valid_loss is None
            else (latest_valid_loss - previous_valid_loss)
        ),
        "latest_train_loss_ema_delta": (
            None
            if latest_train_loss_ema is None or previous_train_loss_ema is None
            else (latest_train_loss_ema - previous_train_loss_ema)
        ),
        "plateau_eval_streak": plateau_eval_streak,
        "tok_per_sec_mean": (
            None if not finite_tok_per_sec else (sum(finite_tok_per_sec) / float(len(finite_tok_per_sec)))
        ),
        "tok_per_sec_median": (None if not finite_tok_per_sec else float(statistics.median(finite_tok_per_sec))),
        "fim_ratio_mean": (
            None if not finite_fim_ratios else (sum(finite_fim_ratios) / float(len(finite_fim_ratios)))
        ),
        "fim_ratio_std": (
            None
            if len(finite_fim_ratios) <= 1
            else float(statistics.pstdev(finite_fim_ratios))
        ),
        "train_loss_min": (None if not finite_train_losses else min(finite_train_losses)),
        "train_loss_ema_min": (None if not finite_train_emas else min(finite_train_emas)),
    }


def prefilter_checkpoints_by_valid_loss(
    checkpoint_paths: list[Path],
    metrics_payload: dict[str, Any],
    *,
    top_k: int,
    preserve_earliest: int = 0,
) -> tuple[list[Path], dict[str, Any]]:
    """Keep the lowest-valid-loss checkpoints first, while preserving a few early checkpoints."""
    if top_k <= 0 or len(checkpoint_paths) <= top_k:
        return list(checkpoint_paths), {
            "enabled": bool(top_k > 0),
            "requested_top_k": int(top_k),
            "preserve_earliest": int(max(0, preserve_earliest)),
            "original_count": len(checkpoint_paths),
            "selected_count": len(checkpoint_paths),
            "used_valid_loss_count": 0,
            "preserved_earliest_count": 0,
            "fallback_count": len(checkpoint_paths),
            "selected": [],
        }

    eval_by_step = dict(metrics_payload.get("eval_by_step", {}))
    ranked: list[tuple[float, int, Path]] = []
    chronological: list[tuple[int, Path]] = []
    fallback: list[Path] = []
    selected_meta: list[dict[str, Any]] = []

    for checkpoint_path in checkpoint_paths:
        match = _STEP_CHECKPOINT_RE.match(checkpoint_path.name)
        if match is None:
            fallback.append(checkpoint_path)
            continue
        step = int(match.group(1))
        eval_payload = dict(eval_by_step.get(step, {}))
        valid_loss = _safe_float(eval_payload.get("valid_loss"))
        if valid_loss is None:
            fallback.append(checkpoint_path)
            continue
        ranked.append((valid_loss, step, checkpoint_path))
        chronological.append((step, checkpoint_path))

    ranked.sort(key=lambda item: (float(item[0]), int(item[1])))
    chronological.sort(key=lambda item: int(item[0]))

    requested_top_k = int(top_k)
    selected_paths: list[Path] = []
    preserved_earliest = max(0, min(int(preserve_earliest), requested_top_k))
    for _step, checkpoint_path in chronological[:preserved_earliest]:
        if checkpoint_path not in selected_paths:
            selected_paths.append(checkpoint_path)
    for _loss, _step, checkpoint_path in ranked:
        if checkpoint_path in selected_paths:
            continue
        selected_paths.append(checkpoint_path)
        if len(selected_paths) >= requested_top_k:
            break
    if len(selected_paths) < top_k:
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path in selected_paths:
                continue
            selected_paths.append(checkpoint_path)
            if len(selected_paths) >= top_k:
                break

    selected_paths = selected_paths[:requested_top_k]
    for checkpoint_path in selected_paths:
        match = _STEP_CHECKPOINT_RE.match(checkpoint_path.name)
        step = None if match is None else int(match.group(1))
        valid_loss = None
        if step is not None:
            valid_loss = _safe_float(dict(eval_by_step.get(step, {})).get("valid_loss"))
        selected_meta.append(
            {
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path),
                "step": step,
                "valid_loss_from_training": valid_loss,
            }
        )

    return selected_paths, {
        "enabled": True,
        "requested_top_k": requested_top_k,
        "preserve_earliest": preserved_earliest,
        "original_count": len(checkpoint_paths),
        "selected_count": len(selected_paths),
        "used_valid_loss_count": sum(
            1 for item in selected_meta if _safe_float(item.get("valid_loss_from_training")) is not None
        ),
        "preserved_earliest_count": min(preserved_earliest, len(selected_paths)),
        "fallback_count": sum(
            1 for item in selected_meta if _safe_float(item.get("valid_loss_from_training")) is None
        ),
        "selected": selected_meta,
    }


def training_metrics_for_step(metrics_payload: dict[str, Any], step: int) -> dict[str, Any]:
    """Return aligned training/eval metrics for a checkpoint step."""
    eval_payload = dict(metrics_payload.get("eval_by_step", {}).get(int(step), {}))
    train_payload = dict(metrics_payload.get("train_by_step", {}).get(int(step), {}))
    return {
        "step": int(step),
        "train": train_payload,
        "eval": eval_payload,
        "valid_loss_from_training": eval_payload.get("valid_loss"),
        "train_loss_ema": eval_payload.get("train_loss_ema", train_payload.get("train_loss_ema")),
        "best_valid_loss_so_far": eval_payload.get("best_valid_loss_so_far"),
        "overfit_gap": eval_payload.get("overfit_gap"),
        "tokens_seen": eval_payload.get("tokens_seen", train_payload.get("tokens_seen")),
    }
