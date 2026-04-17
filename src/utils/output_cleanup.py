"""Helpers for cleaning managed output locations before regenerating artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


def clear_directory_contents(path: Path) -> None:
    """Remove all children under an existing managed directory."""
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"expected directory output target, got file: {path}")
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def ensure_clean_directory(path: Path) -> None:
    """Create a directory if needed and make sure it starts empty."""
    path.mkdir(parents=True, exist_ok=True)
    clear_directory_contents(path)


def remove_file_if_exists(path: Path) -> None:
    """Delete a file artifact when it already exists."""
    if path.exists():
        if path.is_dir():
            raise IsADirectoryError(f"expected file artifact, got directory: {path}")
        path.unlink()


def remove_matching_children(path: Path, patterns: Iterable[str]) -> None:
    """Delete matching direct children under a directory without touching other files."""
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(f"expected directory output target, got file: {path}")
    seen: set[Path] = set()
    for pattern in patterns:
        for child in path.glob(pattern):
            resolved = child.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if child.is_dir():
                shutil.rmtree(child)
            elif child.exists():
                child.unlink()
