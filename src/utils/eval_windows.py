"""Helpers for building valid evaluation windows from token sequences."""

from __future__ import annotations

import random
from typing import Sequence

from src.music_analysis import PhraseAnalysis, analyze_phrase_candidates


def _scan_structural_positions(tokens: Sequence[str]) -> tuple[list[int], list[int]]:
    """One-pass scan returning (bar_positions, phrase_positions)."""
    bar_positions: list[int] = []
    phrase_positions: list[int] = []
    for idx, token in enumerate(tokens):
        if token == "BAR":
            bar_positions.append(idx)
        elif token == "PHRASE":
            phrase_positions.append(idx)
    return bar_positions, phrase_positions


def _build_window_at_positions(
    source_tokens: Sequence[str],
    analysis: PhraseAnalysis,
    *,
    start_index: int,
    end_index: int,
) -> list[str] | None:
    """Materialize a window starting at `start_index` (a BAR cut) and ending
    just before `end_index` (a BAR cut or EOS index).

    Header is `BOS [TEMPO] [KEY]` derived from the effective context at start.
    Body BAR headers are normalized: redundant TEMPO/KEY stripped, PHRASE preserved.
    Window body always starts with `BAR` to satisfy `validate_token_order`.
    """
    if not analysis.bars:
        return None

    bar_index = -1
    for idx, bar in enumerate(analysis.bars):
        if bar.start_token == start_index:
            bar_index = idx
            break
    if bar_index < 0:
        return None

    tempo_token = analysis.bars[bar_index].effective_tempo_token
    key_token = analysis.bars[bar_index].effective_key_token

    body = [str(token) for token in source_tokens[start_index:end_index]]
    normalized: list[str] = []
    idx = 0
    while idx < len(body):
        token = body[idx]
        if token == "BAR":
            normalized.append("BAR")
            idx += 1
            while idx < len(body) and (body[idx].startswith("TEMPO_") or body[idx].startswith("KEY_")):
                idx += 1
            if idx < len(body) and body[idx] == "PHRASE":
                normalized.append("PHRASE")
                idx += 1
            continue
        normalized.append(token)
        idx += 1

    window: list[str] = ["BOS"]
    if tempo_token is not None:
        window.append(tempo_token)
    if key_token is not None:
        window.append(key_token)
    window.extend(normalized)
    window.append("EOS")
    return window


def sample_phrase_aligned_subsequence(
    source_tokens: Sequence[str],
    *,
    max_core_tokens: int,
    min_core_tokens: int,
    rng: random.Random,
    max_attempts: int = 64,
) -> list[str] | None:
    """Sample a valid normalized subsequence in `BOS [TEMPO] [KEY] body EOS` form.

    `body` always starts with `BAR` (required by `validate_token_order`).
    PHRASE priority is expressed as "prefer bars whose interior carries a PHRASE
    token"; bars without inline PHRASE serve as fallback.
    """
    if max_core_tokens <= 0:
        return None
    if min_core_tokens <= 0:
        min_core_tokens = 1
    if not source_tokens or source_tokens[0] != "BOS" or source_tokens[-1] != "EOS":
        return None

    bar_positions, phrase_positions = _scan_structural_positions(source_tokens)
    if not bar_positions:
        return None
    eos_index = len(source_tokens) - 1
    analysis = analyze_phrase_candidates(source_tokens)
    if not analysis.bars:
        return None

    # Map PHRASE token index → enclosing BAR start index.
    phrase_bar_starts: set[int] = set()
    for phrase_idx in phrase_positions:
        enclosing = -1
        for bar_pos in bar_positions:
            if bar_pos < phrase_idx:
                enclosing = bar_pos
            else:
                break
        if enclosing >= 0:
            phrase_bar_starts.add(enclosing)

    # Cut points: all BAR starts; "preferred" subset is those that carry a PHRASE.
    all_cuts = [*bar_positions, eos_index]
    preferred_starts = [pos for pos in bar_positions if pos in phrase_bar_starts]
    fallback_starts = list(bar_positions)

    def _try_pick(starts: list[int]) -> list[str] | None:
        if not starts:
            return None
        for _ in range(max_attempts):
            start = starts[rng.randrange(len(starts))]
            valid_ends = [
                end for end in all_cuts
                if end > start and min_core_tokens <= (end - start) <= max_core_tokens
            ]
            if not valid_ends:
                continue
            end = valid_ends[rng.randrange(len(valid_ends))]
            window = _build_window_at_positions(
                source_tokens, analysis, start_index=start, end_index=end,
            )
            if window is None:
                continue
            body_len = len(window) - 2
            if min_core_tokens <= body_len <= max_core_tokens:
                return window
        return None

    return _try_pick(preferred_starts) or _try_pick(fallback_starts)
