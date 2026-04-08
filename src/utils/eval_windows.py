"""用于从 token 序列中构造语法保持型评估窗口的辅助函数。"""

from __future__ import annotations

import random
from typing import Sequence


def sample_bar_aligned_subsequence(
    source_tokens: Sequence[str],
    *,
    max_core_tokens: int,
    min_core_tokens: int,
    rng: random.Random,
    max_attempts: int = 64,
) -> list[str] | None:
    """
    截取一段自洽且结构合法的子序列。

    返回结果始终满足如下形式：
    `BOS [TEMPO_*] BAR ... BAR ... EOS`

    这样可以避免旧逻辑把长样本从任意中间位置截断后，再强行包上
    `BOS/EOS`，从而把“真值重建”本身裁成语法非法序列的问题。
    """
    if max_core_tokens <= 0:
        return None
    if min_core_tokens <= 0:
        min_core_tokens = 1
    if not source_tokens or source_tokens[0] != "BOS" or source_tokens[-1] != "EOS":
        return None

    initial_tempo = None
    if len(source_tokens) > 2 and str(source_tokens[1]).startswith("TEMPO_"):
        initial_tempo = str(source_tokens[1])

    idx = 2 if initial_tempo is not None else 1
    seq_end = len(source_tokens) - 1
    bars: list[tuple[int, int, str | None]] = []
    current_tempo = initial_tempo

    while idx < seq_end:
        if source_tokens[idx] != "BAR":
            return None
        bar_start = idx
        leading_tempo = current_tempo
        idx += 1
        if idx < seq_end and str(source_tokens[idx]).startswith("TEMPO_"):
            current_tempo = str(source_tokens[idx])
            idx += 1
        while idx < seq_end and source_tokens[idx] != "BAR":
            idx += 1
        bars.append((bar_start, idx, leading_tempo))

    if not bars:
        body = [str(token) for token in source_tokens[1:-1]]
        if not (min_core_tokens <= len(body) <= max_core_tokens):
            return None
        return ["BOS", *body, "EOS"]

    def _build_window(start_bar: int, choose_random_end: bool) -> list[str] | None:
        leading_tempo = bars[start_bar][2]
        body: list[str] = [] if leading_tempo is None else [leading_tempo]
        candidate_lengths: list[int] = []

        for end_bar in range(start_bar, len(bars)):
            bar_start, bar_end, _ = bars[end_bar]
            bar_tokens = [str(token) for token in source_tokens[bar_start:bar_end]]
            if len(body) + len(bar_tokens) > max_core_tokens:
                break
            body.extend(bar_tokens)
            if len(body) >= min_core_tokens:
                candidate_lengths.append(len(body))

        if not candidate_lengths:
            return None

        body_len = rng.choice(candidate_lengths) if choose_random_end else candidate_lengths[-1]
        return ["BOS", *body[:body_len], "EOS"]

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
