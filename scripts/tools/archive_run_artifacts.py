#!/usr/bin/env python
"""Archive a valuable TuneFlow checkpoint plus benchmark artifacts into a versioned snapshot."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_REPORT_CANDIDATES = (
    "benchmark_report.json",
    "benchmark_continuation_report.json",
    "benchmark_infilling_report.json",
)


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive the current valuable run artifacts before future training/eval clears them. "
            "By default this copies the benchmark-recommended checkpoint and the whole benchmark run folder."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="Train config YAML path.")
    parser.add_argument(
        "--preset",
        type=str,
        default="small",
        choices=["small", "full"],
        help="Built-in train config preset used when --config is omitted.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. When omitted, read the benchmark recommended checkpoint first.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("outputs/archive"),
        help="Root directory that stores archived snapshots.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional extra label appended to the archive directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved archive plan without copying files.",
    )
    return parser.parse_args()


def _load_train_mapping(config_path: Path) -> dict[str, Any]:
    from src.utils.config_io import load_yaml_mapping

    payload = load_yaml_mapping(config_path, "train run config")
    if "train" in payload:
        train_payload = payload["train"]
        if not isinstance(train_payload, dict):
            raise ValueError(f"`train` section in {config_path} must be a mapping.")
        return train_payload
    return payload


def _resolve_preset_config(project_root: Path, preset: str) -> Path:
    mapping = {
        "small": project_root / "configs" / "train" / "train_base_run_small.yaml",
        "full": project_root / "configs" / "train" / "train_base_run_full.yaml",
    }
    config_path = mapping[preset].resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Preset config not found for --preset {preset}: {config_path}")
    return config_path


def _legacy_checkpoint_aliases(checkpoint_name: str) -> list[str]:
    aliases = [checkpoint_name]
    legacy_map = {
        "base_small": "train_base_run_small",
        "base_full": "train_base_run_full",
    }
    legacy_name = legacy_map.get(checkpoint_name)
    if legacy_name is not None:
        aliases.append(legacy_name)
    return aliases


def _resolve_run(project_root: Path, args: argparse.Namespace) -> tuple[Path, Path, Path, str, dict[str, Any]]:
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else (project_root / args.config)
        config_path = config_path.resolve()
    else:
        config_path = _resolve_preset_config(project_root, args.preset)

    train_mapping = _load_train_mapping(config_path)
    output_dir_value = train_mapping.get("output_dir")
    if output_dir_value is None:
        raise ValueError(f"Missing output_dir in train config: {config_path}")

    configured_checkpoint_dir = Path(str(output_dir_value))
    if not configured_checkpoint_dir.is_absolute():
        configured_checkpoint_dir = (project_root / configured_checkpoint_dir).resolve()

    checkpoint_candidates = [configured_checkpoint_dir]
    if configured_checkpoint_dir.parent.name == "checkpoints":
        for alias_name in _legacy_checkpoint_aliases(configured_checkpoint_dir.name):
            legacy_candidate = configured_checkpoint_dir.parent / "base" / alias_name
            alias_candidate = configured_checkpoint_dir.parent / alias_name
            if legacy_candidate not in checkpoint_candidates:
                checkpoint_candidates.append(legacy_candidate)
            if alias_candidate not in checkpoint_candidates:
                checkpoint_candidates.append(alias_candidate)

    checkpoint_dir = configured_checkpoint_dir
    for candidate in checkpoint_candidates:
        if candidate.exists():
            checkpoint_dir = candidate
            break

    run_id = configured_checkpoint_dir.name
    benchmark_root = (project_root / "outputs" / "benchmark" / run_id).resolve()
    return config_path, checkpoint_dir, benchmark_root, run_id, train_mapping


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _find_benchmark_report(benchmark_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    for report_name in _REPORT_CANDIDATES:
        report_path = benchmark_root / report_name
        payload = _load_json_if_exists(report_path)
        if payload is not None:
            return report_path, payload
    return None, None


def _recommended_checkpoint_from_report(report_payload: dict[str, Any] | None) -> tuple[Path | None, dict[str, Any] | None]:
    if report_payload is None:
        return None, None

    candidate = None
    final_selection = report_payload.get("final_selection")
    if isinstance(final_selection, dict):
        candidate = final_selection.get("recommended_checkpoint")
    if not isinstance(candidate, dict):
        summary = report_payload.get("summary")
        if isinstance(summary, dict):
            candidate = summary.get("recommended_checkpoint")
    if not isinstance(candidate, dict):
        return None, None

    checkpoint_path = candidate.get("checkpoint_path")
    if not checkpoint_path:
        return None, None
    return Path(str(checkpoint_path)).resolve(), candidate


def _resolve_checkpoint_path(
    *,
    explicit_checkpoint_path: Path | None,
    checkpoint_dir: Path,
    report_payload: dict[str, Any] | None,
) -> tuple[Path, str, dict[str, Any] | None]:
    if explicit_checkpoint_path is not None:
        checkpoint_path = explicit_checkpoint_path.resolve()
        return checkpoint_path, "explicit", None

    recommended_path, recommended_meta = _recommended_checkpoint_from_report(report_payload)
    if recommended_path is not None:
        return recommended_path, "benchmark_recommended", recommended_meta

    for fallback_name in ("best.pt", "latest.pt", "last.pt"):
        fallback_path = checkpoint_dir / fallback_name
        if fallback_path.exists():
            return fallback_path.resolve(), f"fallback_{fallback_name}", None

    raise FileNotFoundError(
        "Unable to resolve a checkpoint to archive. "
        "Please run benchmark first or pass --checkpoint-path explicitly."
    )


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag.strip())
    return cleaned.strip("-._")


def _build_archive_dir(archive_root: Path, *, run_id: str, checkpoint_path: Path, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [run_id, checkpoint_path.stem, timestamp]
    safe_tag = _sanitize_tag(tag)
    if safe_tag:
        parts.append(safe_tag)
    base_name = "__".join(parts)
    archive_dir = archive_root / run_id / base_name

    counter = 1
    while archive_dir.exists():
        archive_dir = archive_root / run_id / f"{base_name}__{counter:02d}"
        counter += 1
    return archive_dir


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_directory(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"archive target already exists: {dst}")
    shutil.copytree(src, dst)


def _best_aliases(checkpoint_dir: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name in ("best.pt", "latest.pt", "last.pt"):
        path = checkpoint_dir / name
        if path.exists():
            aliases[name] = str(path.resolve())
    return aliases


def main() -> None:
    project_root = _ensure_project_root_on_path()
    os.chdir(project_root)
    args = _parse_args()

    config_path, checkpoint_dir, benchmark_root, run_id, train_mapping = _resolve_run(project_root, args)
    report_path, report_payload = _find_benchmark_report(benchmark_root)
    checkpoint_path, checkpoint_resolution, recommended_meta = _resolve_checkpoint_path(
        explicit_checkpoint_path=args.checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        report_payload=report_payload,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint to archive not found: {checkpoint_path}")

    archive_root = args.archive_root if args.archive_root.is_absolute() else (project_root / args.archive_root)
    archive_root = archive_root.resolve()
    archive_dir = _build_archive_dir(
        archive_root,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        tag=args.tag,
    )

    metrics_path = checkpoint_dir / "metrics.jsonl"
    archive_checkpoint_dir = archive_dir / "checkpoint"
    archive_benchmark_dir = archive_dir / "benchmark"
    archive_config_dir = archive_dir / "config"

    manifest = {
        "archive_version": 1,
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "archive_dir": str(archive_dir),
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "benchmark_root": str(benchmark_root),
        "selected_checkpoint": {
            "path": str(checkpoint_path),
            "name": checkpoint_path.name,
            "resolution": checkpoint_resolution,
            "recommended_checkpoint_meta": recommended_meta,
        },
        "benchmark_report_path": (None if report_path is None else str(report_path)),
        "train_mapping": train_mapping,
        "checkpoint_aliases_present": _best_aliases(checkpoint_dir),
        "copied_artifacts": {
            "checkpoint_file": str(archive_checkpoint_dir / checkpoint_path.name),
            "metrics_file": (str(archive_checkpoint_dir / "metrics.jsonl") if metrics_path.exists() else None),
            "benchmark_dir": (str(archive_benchmark_dir / benchmark_root.name) if benchmark_root.exists() else None),
            "config_file": str(archive_config_dir / config_path.name),
        },
    }

    print(f"[archive] run_id={run_id}")
    print(f"[archive] checkpoint={checkpoint_path}")
    print(f"[archive] benchmark_root={benchmark_root}")
    print(f"[archive] archive_dir={archive_dir}")

    if args.dry_run:
        print("[archive] dry-run only; no files copied.")
        return

    archive_dir.mkdir(parents=True, exist_ok=False)
    _copy_file(config_path, archive_config_dir / config_path.name)
    _copy_file(checkpoint_path, archive_checkpoint_dir / checkpoint_path.name)
    if metrics_path.exists():
        _copy_file(metrics_path, archive_checkpoint_dir / "metrics.jsonl")
    if benchmark_root.exists():
        _copy_directory(benchmark_root, archive_benchmark_dir / benchmark_root.name)
    else:
        print(f"[archive] warning: benchmark root not found, skipped benchmark copy: {benchmark_root}")

    manifest_path = archive_dir / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[archive] manifest -> {manifest_path}")
    print(f"[archive] archived checkpoint -> {archive_checkpoint_dir / checkpoint_path.name}")
    if benchmark_root.exists():
        print(f"[archive] archived benchmark -> {archive_benchmark_dir / benchmark_root.name}")


if __name__ == "__main__":
    main()
