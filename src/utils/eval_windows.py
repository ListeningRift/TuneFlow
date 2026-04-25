"""Helpers for building valid evaluation windows from token sequences."""

from __future__ import annotations

import random
from typing import Sequence

from src.music_analysis import analyze_phrase_candidates


def _build_bar_span_window(
    source_tokens: Sequence[str],
    *,
    start_bar: int,
    end_bar: int,
) -> list[str] | None:
    analysis = analyze_phrase_candidates(source_tokens)
    if not analysis.bars or not (0 <= start_bar < end_bar <= len(analysis.bars)):
        return None

    window_tokens: list[str] = ["BOS"]
    leading_tempo = analysis.bars[start_bar].effective_tempo_token
    leading_key = analysis.bars[start_bar].effective_key_token
    if leading_tempo is not None:
        window_tokens.append(leading_tempo)
    if leading_key is not None:
        window_tokens.append(leading_key)

    for bar_index in range(start_bar, end_bar):
        bar = analysis.bars[bar_index]
        raw_bar_tokens = [str(token) for token in source_tokens[bar.start_token : bar.end_token]]
        if raw_bar_tokens and raw_bar_tokens[0] == "BAR":
            idx = 1
            if idx < len(raw_bar_tokens) and raw_bar_tokens[idx].startswith("TEMPO_"):
                idx += 1
            if idx < len(raw_bar_tokens) and raw_bar_tokens[idx].startswith("KEY_"):
                idx += 1
            raw_bar_tokens = ["BAR", *raw_bar_tokens[idx:]]
        window_tokens.extend(raw_bar_tokens)

    window_tokens.append("EOS")
    return window_tokens


def sample_bar_aligned_subsequence(
    source_tokens: Sequence[str],
    *,
    max_core_tokens: int,
    min_core_tokens: int,
    rng: random.Random,
    max_attempts: int = 64,
) -> list[str] | None:
    """
    Sample a valid normalized subsequence in `BOS [TEMPO] BAR ... EOS` form.

    The window keeps only the tempo active at the window start, matching the
    phrase-oriented training view.
    """
    if max_core_tokens <= 0:
        return None
    if min_core_tokens <= 0:
        min_core_tokens = 1
    if not source_tokens or source_tokens[0] != "BOS" or source_tokens[-1] != "EOS":
        return None

    analysis = analyze_phrase_candidates(source_tokens)
    bars = list(analysis.bars)

    if not bars:
        body = [str(token) for token in source_tokens[1:-1]]
        if not (min_core_tokens <= len(body) <= max_core_tokens):
            return None
        return ["BOS", *body, "EOS"]

    def _build_window(start_bar: int, choose_random_end: bool) -> list[str] | None:
        candidate_ends: list[int] = []
        for end_bar in range(start_bar + 1, len(bars) + 1):
            window = _build_bar_span_window(source_tokens, start_bar=start_bar, end_bar=end_bar)
            if window is None:
                return None
            body_len = len(window) - 2
            if body_len > max_core_tokens:
                break
            if body_len >= min_core_tokens:
                candidate_ends.append(end_bar)

        if not candidate_ends:
            return None

        chosen_end = rng.choice(candidate_ends) if choose_random_end else candidate_ends[-1]
        return _build_bar_span_window(source_tokens, start_bar=start_bar, end_bar=chosen_end)

    for _ in range(max_attempts):
        start_bar = rng.randrange(len(bars))
        window = _build_window(start_bar, choose_random_end=True)
        if window is not None:
            return window

    for start_bar in range(len(bars)):
        window = _build_window(start_bar, choose_random_end=False)
        if window is not None:
            return window
    return None
