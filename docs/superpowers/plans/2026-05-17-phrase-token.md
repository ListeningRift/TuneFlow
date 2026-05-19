# PHRASE Token 引入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把"乐句开始"作为一类显式 token `PHRASE` 写入数据；拿掉训练侧基于启发式的乐句感知采样与 phrase-aware FIM；评测窗口切分升级为 PHRASE 优先、BAR 兜底。

**Architecture:**
- 词表与 FSM：新增 `PHRASE` token 与 `AFTER_PHRASE` 状态；语法规则约束 `PHRASE` 后必须紧跟完整 event。
- 编码端：`phrase_analysis` 把"边界"提升为一等数据（`PhraseBoundary`），新增 `inject_phrase_tokens(tokens)` 在 `inject_key_tokens` 之后注入；`tokens_to_midi` 把 PHRASE 当作可忽略结构 token。
- 训练管线：删除 `PhraseSamplingConfig` 与三类窗口采样路径；保留 NEXT 随机锚点 + 通用 FIM；FIM unit 新增 `phrase_event`（长度 6）。
- 评测窗口：`sample_bar_aligned_subsequence` → `sample_phrase_aligned_subsequence`，PHRASE 优先、BAR 兜底。

**Tech Stack:** Python 3.11, mido, PyTorch (惰性导入), pytest/unittest.

**Breaking change:** 词表偏移，落地后必须重新 `tokenize_dataset` → `build_training_data` → 从 scratch 重训。

**Commit policy:** 全部 task 完成后一次大 commit。

**Spec 源文档:** `docs/superpowers/specs/2026-05-17-phrase-token-design.md`

---

## 文件影响清单

**新增：**
- `tests/test_decoding_grammar_fsm.py`：FSM 中 PHRASE 状态的单测

**修改（源代码）：**
- `src/music_analysis/phrase_analysis.py`：boundary-first 重构 + 删除窗口采样函数
- `src/music_analysis/__init__.py`：调整 re-export
- `src/tokenizer/midi_codec.py`：vocab 加 PHRASE / `validate_token_order` / `inject_phrase_tokens` / `_tokenize_note_events` / `tokens_to_midi`
- `src/tokenizer/tokenize_dataset.py`：新增 phrase 统计
- `src/decoding/grammar_fsm.py`：新增 `AFTER_PHRASE` 状态
- `src/utils/eval_windows.py`：函数改名 + 重新实现
- `src/utils/benchmarking.py`：调用点改名
- `src/training/train_base.py`：删除 phrase sampling，新增 `phrase_event` unit
- `configs/train/train_base_run_full.yaml`、`configs/train/train_base_run_small.yaml`：删除 phrase 采样配置项
- `docs/todo.md`：第 1 项标注完成

**修改（测试）：**
- `tests/test_music_analysis.py`：删除窗口采样测试、修正 imports、新增 boundary / mid-bar anchor 断言
- `tests/test_tokenizer_midi_codec.py`：新增 PHRASE roundtrip 与 validate 测试

---

## Task 1: phrase_analysis 引入 PhraseBoundary 与 onset_positions

**Files:**
- Modify: `src/music_analysis/phrase_analysis.py`
- Test: `tests/test_music_analysis.py`

- [ ] **Step 1：在 `phrase_analysis.py` 顶部增加 `PhraseBoundary` 数据类、扩展 `PhraseAnalysisConfig.mid_bar_min_rest_pos`、为 `BarInfo` 增加 `onset_positions` 字段**

在 `PhraseAnalysisConfig` 后追加 `mid_bar_min_rest_pos: int = 8`（默认 `positions_per_bar // 4`，硬编码到 32 ppb 对应 8）。

`BarInfo` 增加字段：
```python
@dataclass(frozen=True)
class BarInfo:
    start_token: int
    end_token: int
    note_count: int
    onset_count: int
    rest_ratio: float
    pitch_span: int
    mean_duration: float
    effective_tempo_token: str | None
    effective_key_token: str | None
    onset_positions: tuple[int, ...]
```

在 `PhraseAnalysisConfig` 之后新增：
```python
@dataclass(frozen=True)
class PhraseBoundary:
    """PHRASE 落点。anchor_pos==0 表示 bar-aligned；>0 表示 mid-bar POS 槽前。"""

    bar_index: int
    anchor_pos: int
```

`PhraseAnalysis` 字段调整为：
```python
@dataclass(frozen=True)
class PhraseAnalysis:
    bars: tuple[BarInfo, ...]
    boundary_scores: tuple[BoundaryScore, ...]
    boundaries: tuple[PhraseBoundary, ...]
    phrase_spans: tuple[PhraseSpan, ...]
```

- [ ] **Step 2：修改 `_build_bar_info` 让它把 `onset_positions` 返回**

在原 `_build_bar_info` 中，把局部变量 `onset_positions: set[int]` 改为有序记录：每次匹配完整 event 5-tuple 后 `onset_positions.add(pos_value)`。构造 `BarInfo` 时把 `tuple(sorted(onset_positions))` 传入。

```python
bars.append(
    BarInfo(
        start_token=start_token,
        end_token=end_token,
        note_count=note_count,
        onset_count=len(onset_positions),
        rest_ratio=min(1.0, max(0.0, rest_ratio)),
        pitch_span=pitch_span,
        mean_duration=mean_duration,
        effective_tempo_token=effective_tempo,
        effective_key_token=effective_key,
        onset_positions=tuple(sorted(onset_positions)),
    )
)
```

- [ ] **Step 3：写失败测试 `test_bar_info_exposes_onset_positions`**

在 `tests/test_music_analysis.py` 的 `MusicAnalysisTests` 中新增：
```python
def test_bar_info_exposes_onset_positions(self) -> None:
    analysis = analyze_phrase_candidates(_phrase_source_tokens())
    self.assertEqual(analysis.bars[0].onset_positions, (0, 8))
    self.assertEqual(analysis.bars[3].onset_positions, (0, 16))
```

- [ ] **Step 4：跑测试，期望 PASS（之前的 step 1-2 已落实）**

Run: `python -m pytest tests/test_music_analysis.py::MusicAnalysisTests::test_bar_info_exposes_onset_positions -v`
Expected: PASS

---

## Task 2: phrase_analysis 重写为 boundary-first 流水线

**Files:**
- Modify: `src/music_analysis/phrase_analysis.py`

- [ ] **Step 1：实现 `_pick_in_bar_anchor`**

放在 `_pick_candidate_boundaries` 之后：
```python
def _pick_in_bar_anchor(left_bar: BarInfo, right_bar: BarInfo, config: PhraseAnalysisConfig) -> int:
    """v1：只在右 bar 起首有 >=1 拍留白时把 PHRASE 推迟到首个 onset。"""
    if not right_bar.onset_positions:
        return 0
    first_onset = right_bar.onset_positions[0]
    if first_onset >= config.mid_bar_min_rest_pos:
        return first_onset
    return 0
```

- [ ] **Step 2：实现 `_assemble_final_boundaries`**

替换原先的 `_build_phrase_spans` 内 boundaries 装配逻辑：
```python
def _assemble_final_boundaries(
    bars: Sequence[BarInfo],
    candidate_boundary_bars: Sequence[int],
    config: PhraseAnalysisConfig,
) -> tuple[PhraseBoundary, ...]:
    if not bars:
        return tuple()

    first_content_bar = next((i for i, bar in enumerate(bars) if bar.note_count > 0), None)
    if first_content_bar is None:
        return tuple()

    boundary_set: dict[tuple[int, int], PhraseBoundary] = {}
    forced = PhraseBoundary(bar_index=first_content_bar, anchor_pos=0)
    boundary_set[(forced.bar_index, forced.anchor_pos)] = forced

    for bar_index in candidate_boundary_bars:
        if bar_index <= first_content_bar or bar_index >= len(bars):
            continue
        anchor_pos = _pick_in_bar_anchor(bars[bar_index - 1], bars[bar_index], config)
        key = (bar_index, anchor_pos)
        if key not in boundary_set:
            boundary_set[key] = PhraseBoundary(bar_index=bar_index, anchor_pos=anchor_pos)

    ordered = sorted(boundary_set.values(), key=lambda b: (b.bar_index, b.anchor_pos))

    # 长 span 切分：相邻 boundary 之间 bar 跨度 > max_phrase_bars 时，按 preferred_phrase_bars 插合成 boundary
    expanded: list[PhraseBoundary] = []
    for idx, current in enumerate(ordered):
        expanded.append(current)
        next_bar = ordered[idx + 1].bar_index if idx + 1 < len(ordered) else len(bars)
        gap = next_bar - current.bar_index
        cursor = current.bar_index
        while gap > config.max_phrase_bars:
            synth_bar = cursor + config.preferred_phrase_bars
            if synth_bar >= next_bar:
                break
            if bars[synth_bar].note_count == 0:
                synth_bar += 1
                if synth_bar >= next_bar:
                    break
            expanded.append(PhraseBoundary(bar_index=synth_bar, anchor_pos=0))
            cursor = synth_bar
            gap = next_bar - cursor

    # 短 span 合并：相邻 boundary < min_phrase_bars 时删除后一个；首句 boundary 永不删除
    merged: list[PhraseBoundary] = []
    for boundary in expanded:
        if not merged:
            merged.append(boundary)
            continue
        prev = merged[-1]
        if boundary.bar_index - prev.bar_index < config.min_phrase_bars and boundary != forced:
            continue
        merged.append(boundary)
    return tuple(merged)
```

- [ ] **Step 3：实现 `_derive_phrase_spans`**

替换旧 `_build_phrase_spans` 体：
```python
def _derive_phrase_spans(
    bars: Sequence[BarInfo],
    boundaries: Sequence[PhraseBoundary],
) -> tuple[PhraseSpan, ...]:
    if not bars or not boundaries:
        return tuple()
    sorted_boundaries = sorted(boundaries, key=lambda b: (b.bar_index, b.anchor_pos))
    spans: list[PhraseSpan] = []
    for idx, current in enumerate(sorted_boundaries):
        start_bar = current.bar_index
        end_bar = sorted_boundaries[idx + 1].bar_index if idx + 1 < len(sorted_boundaries) else len(bars)
        if end_bar <= start_bar:
            continue
        start_token = bars[start_bar].start_token
        end_token = bars[end_bar - 1].end_token
        spans.append(
            PhraseSpan(
                start_bar=start_bar,
                end_bar=end_bar,
                start_token=start_token,
                end_token=end_token,
                tempo_token=bars[start_bar].effective_tempo_token,
                key_token=bars[start_bar].effective_key_token,
                tokens=tuple(),
                source_kind="single_phrase",
            )
        )
    return tuple(spans)
```

- [ ] **Step 4：重写 `analyze_phrase_candidates`**

```python
def analyze_phrase_candidates(
    tokens: Sequence[str],
    config: PhraseAnalysisConfig | None = None,
) -> PhraseAnalysis:
    """从单条 token 序列中分析乐句候选区间。"""
    config = PhraseAnalysisConfig() if config is None else config
    bars = _build_bar_info(tokens, config)
    boundary_scores = _build_boundary_scores(bars, config)
    candidate_boundary_bars = _pick_candidate_boundaries(boundary_scores, config)
    boundaries = _assemble_final_boundaries(bars, candidate_boundary_bars, config)
    phrase_spans = _derive_phrase_spans(bars, boundaries)
    return PhraseAnalysis(
        bars=bars,
        boundary_scores=boundary_scores,
        boundaries=boundaries,
        phrase_spans=phrase_spans,
    )
```

- [ ] **Step 5：删除以下旧符号（已被取代或不再使用）**

从 `src/music_analysis/phrase_analysis.py` 删除：
- `PhraseWindowPolicy` dataclass
- `SampledWindow` dataclass
- `_find_best_split`
- `_merge_short_spans`
- `_build_phrase_spans`（已被 `_assemble_final_boundaries` + `_derive_phrase_spans` 取代）
- `_normalized_bar_tokens`
- `_build_phrase_view_tokens`
- `_build_phrase_span`
- `extract_phrase`
- `_phrase_boundaries_from_spans`
- `_count_phrase_boundaries`
- `_choose_single_phrase_window`
- `_choose_cross_boundary_window`
- `_choose_long_context_window`
- `sample_phrase_window`

模块顶部 `import random` 不再需要，移除该 import。

- [ ] **Step 6：跑现有的 boundary 相关测试以确认重构不破坏 high-level 行为**

Run: `python -m pytest tests/test_music_analysis.py::MusicAnalysisTests::test_analyze_phrase_candidates_detects_boundaries_and_lengths tests/test_music_analysis.py::MusicAnalysisTests::test_analyze_phrase_candidates_accepts_missing_terminal_eos -v`
Expected: PASS（断言只用到 `analysis.bars` / `analysis.boundary_scores` / `analysis.phrase_spans`，新签名兼容）

---

## Task 3: music_analysis `__init__` 调整 re-export

**Files:**
- Modify: `src/music_analysis/__init__.py`

- [ ] **Step 1：把 `__init__.py` 改为**

```python
"""Music analysis helpers for TuneFlow token sequences."""

from .key_analysis import (
    KeyAnalysisConfig,
    KeyFrame,
    KeySegment,
    KeyTimelineAnalysis,
    ModulationPoint,
    analyze_key_timeline,
)
from .phrase_analysis import (
    BarInfo,
    BoundaryScore,
    PhraseAnalysis,
    PhraseAnalysisConfig,
    PhraseBoundary,
    PhraseSpan,
    analyze_phrase_candidates,
)

__all__ = [
    "BarInfo",
    "BoundaryScore",
    "KeyAnalysisConfig",
    "KeyFrame",
    "KeySegment",
    "KeyTimelineAnalysis",
    "ModulationPoint",
    "PhraseAnalysis",
    "PhraseAnalysisConfig",
    "PhraseBoundary",
    "PhraseSpan",
    "analyze_key_timeline",
    "analyze_phrase_candidates",
]
```

- [ ] **Step 2：执行 import 冒烟以验证 module 加载**

Run: `python -c "from src.music_analysis import PhraseBoundary, analyze_phrase_candidates; print('ok')"`
Expected: 输出 `ok`

---

## Task 4: midi_codec 新增 PHRASE 到 vocab

**Files:**
- Modify: `src/tokenizer/midi_codec.py` (build_vocab)
- Test: `tests/test_tokenizer_midi_codec.py`

- [ ] **Step 1：写失败测试 `test_phrase_token_in_vocab`**

在 `tests/test_tokenizer_midi_codec.py` 中找到现有 `TokenizerVocabTests` 一类（如不存在，就在文件末尾追加一个 `class PhraseTokenVocabTests(unittest.TestCase):`），添加：
```python
def test_phrase_token_in_vocab(self) -> None:
    config = TokenizerConfig()
    vocab = build_vocab(config)
    self.assertIn("PHRASE", vocab)
    self.assertIn("BAR", vocab)
    self.assertLess(vocab["BAR"], vocab["PHRASE"], "PHRASE must follow BAR in vocab order")
    self.assertLess(vocab["PHRASE"], vocab["POS_0"], "PHRASE must precede POS_* tokens")
```

- [ ] **Step 2：跑测试，期望 FAIL**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k test_phrase_token_in_vocab -v`
Expected: FAIL (assertIn "PHRASE")

- [ ] **Step 3：在 `build_vocab` 中加入 PHRASE**

在 `src/tokenizer/midi_codec.py` `build_vocab` 内 `vocab.append("BAR")` 后追加一行：
```python
    vocab.append("BAR")
    vocab.append("PHRASE")
```

- [ ] **Step 4：跑测试，期望 PASS**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k test_phrase_token_in_vocab -v`
Expected: PASS

---

## Task 5: midi_codec `validate_token_order` 支持 PHRASE

**Files:**
- Modify: `src/tokenizer/midi_codec.py` (validate_token_order)
- Test: `tests/test_tokenizer_midi_codec.py`

- [ ] **Step 1：写失败测试 `test_validate_token_order_accepts_phrase` / `test_validate_token_order_rejects_invalid_phrase`**

```python
def test_validate_token_order_accepts_bar_head_phrase(self) -> None:
    config = TokenizerConfig()
    vocab = build_vocab(config)
    tokens = [
        "BOS", "TEMPO_120", "KEY_UNCERTAIN",
        "BAR", "PHRASE", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "EOS",
    ]
    valid, oov = validate_token_order(tokens, vocab)
    self.assertTrue(valid)
    self.assertEqual(oov, 0)

def test_validate_token_order_accepts_mid_bar_phrase(self) -> None:
    config = TokenizerConfig()
    vocab = build_vocab(config)
    tokens = [
        "BOS", "TEMPO_120", "KEY_UNCERTAIN",
        "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "PHRASE", "POS_8", "INST_PIANO", "PITCH_64", "DUR_4", "VEL_8",
        "EOS",
    ]
    valid, oov = validate_token_order(tokens, vocab)
    self.assertTrue(valid)

def test_validate_token_order_rejects_consecutive_phrase(self) -> None:
    config = TokenizerConfig()
    vocab = build_vocab(config)
    tokens = [
        "BOS", "TEMPO_120", "KEY_UNCERTAIN",
        "BAR", "PHRASE", "PHRASE", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "EOS",
    ]
    valid, _ = validate_token_order(tokens, vocab)
    self.assertFalse(valid)

def test_validate_token_order_rejects_phrase_before_bar_or_eos(self) -> None:
    config = TokenizerConfig()
    vocab = build_vocab(config)
    tokens_phrase_before_bar = [
        "BOS", "TEMPO_120", "KEY_UNCERTAIN",
        "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "PHRASE", "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "EOS",
    ]
    valid, _ = validate_token_order(tokens_phrase_before_bar, vocab)
    self.assertFalse(valid)

def test_validate_token_order_rejects_phrase_at_bos(self) -> None:
    config = TokenizerConfig()
    vocab = build_vocab(config)
    tokens = [
        "BOS", "PHRASE", "TEMPO_120", "KEY_UNCERTAIN",
        "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "EOS",
    ]
    valid, _ = validate_token_order(tokens, vocab)
    self.assertFalse(valid)
```

- [ ] **Step 2：跑测试，期望 FAIL（旧逻辑不识别 PHRASE）**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k validate_token_order -v`
Expected: 上述新增的 5 个测试至少有 2 个 FAIL（`accepts_*` 会失败因为 PHRASE 走到 BAR-while 循环里被当作非 POS）。

- [ ] **Step 3：重写 `validate_token_order` 主体支持 PHRASE**

替换函数体：
```python
def validate_token_order(tokens: Sequence[str], vocab: Mapping[str, int]) -> Tuple[bool, int]:
    """校验 token 顺序是否合法，并返回 `(is_valid, oov_count)`。"""
    oov = sum(1 for token in tokens if token not in vocab)
    if not tokens or tokens[0] != "BOS":
        return False, oov
    if tokens[-1] != "EOS":
        return False, oov

    last_index = len(tokens) - 1
    idx = 1
    if idx < last_index and tokens[idx].startswith("TEMPO_"):
        idx += 1
    if idx < last_index and is_key_token(tokens[idx]):
        idx += 1

    while idx < last_index:
        if tokens[idx] != "BAR":
            return False, oov
        idx += 1
        if idx < last_index and tokens[idx].startswith("TEMPO_"):
            idx += 1
        if idx < last_index and is_key_token(tokens[idx]):
            idx += 1
        # Optional bar-head PHRASE: must be followed by a complete event tuple
        if idx < last_index and tokens[idx] == "PHRASE":
            if idx + 5 >= last_index:
                return False, oov
            if not tokens[idx + 1].startswith("POS_"):
                return False, oov
            idx += 1
        # event loop with optional mid-bar PHRASE
        while idx < last_index and tokens[idx].startswith("POS_"):
            if idx + 4 >= last_index:
                return False, oov
            if not tokens[idx + 1].startswith("INST_"):
                return False, oov
            if not tokens[idx + 2].startswith("PITCH_"):
                return False, oov
            if not tokens[idx + 3].startswith("DUR_"):
                return False, oov
            if not tokens[idx + 4].startswith("VEL_"):
                return False, oov
            idx += 5
            if idx < last_index and tokens[idx] == "PHRASE":
                # mid-bar PHRASE must be followed by another complete event
                if idx + 5 >= last_index:
                    return False, oov
                if not tokens[idx + 1].startswith("POS_"):
                    return False, oov
                idx += 1
        if idx < last_index and tokens[idx] != "BAR":
            return False, oov

    return True, oov
```

- [ ] **Step 4：跑测试，期望全部 PASS**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k validate_token_order -v`
Expected: PASS

---

## Task 6: midi_codec 实现 `inject_phrase_tokens`

**Files:**
- Modify: `src/tokenizer/midi_codec.py`
- Test: `tests/test_tokenizer_midi_codec.py`

- [ ] **Step 1：写失败测试 `test_inject_phrase_tokens_forces_first_phrase`**

```python
def test_inject_phrase_tokens_forces_first_phrase(self) -> None:
    from src.tokenizer.midi_codec import inject_phrase_tokens
    tokens = [
        "BOS", "TEMPO_120", "KEY_UNCERTAIN",
        "BAR",
        "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "BAR", "POS_0", "INST_PIANO", "PITCH_62", "DUR_4", "VEL_8",
        "EOS",
    ]
    out = inject_phrase_tokens(tokens)
    # The first content bar (index 1) should get a bar-head PHRASE
    bar_positions = [i for i, t in enumerate(out) if t == "BAR"]
    self.assertEqual(out[bar_positions[1] + 1], "PHRASE")
    self.assertEqual(out[bar_positions[1] + 2], "POS_0")

def test_inject_phrase_tokens_no_phrase_on_empty_bar(self) -> None:
    from src.tokenizer.midi_codec import inject_phrase_tokens
    tokens = ["BOS", "TEMPO_120", "KEY_UNCERTAIN", "BAR", "EOS"]
    out = inject_phrase_tokens(tokens)
    self.assertNotIn("PHRASE", out)

def test_inject_phrase_tokens_dedups_adjacent(self) -> None:
    from src.tokenizer.midi_codec import inject_phrase_tokens
    # Force a scenario where a candidate boundary lands on first_content_bar — adjacency dedup
    # We trust _assemble_final_boundaries to also dedup; here we just check tokens are valid afterwards.
    tokens = [
        "BOS", "TEMPO_120", "KEY_UNCERTAIN",
        "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "EOS",
    ]
    out = inject_phrase_tokens(tokens)
    for i in range(len(out) - 1):
        self.assertFalse(out[i] == "PHRASE" and out[i + 1] == "PHRASE")
```

- [ ] **Step 2：跑测试，期望 FAIL**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k inject_phrase_tokens -v`
Expected: FAIL（函数不存在）

- [ ] **Step 3：在 `src/tokenizer/midi_codec.py` 顶部添加 import 并实现 `inject_phrase_tokens`**

在 `from ..music_analysis.key_analysis import analyze_key_timeline` 之后追加：
```python
from ..music_analysis import analyze_phrase_candidates
```

在 `inject_key_tokens` 之后添加：
```python
def inject_phrase_tokens(tokens: Sequence[str]) -> List[str]:
    """Inject `PHRASE` structural tokens based on phrase analysis boundaries.

    Input must already have KEY tokens injected. PHRASE 在 bar-aligned 位置插在
    `BAR [TEMPO] [KEY]` 头部之后、首个 POS 之前；mid-bar anchor 插在指定 POS 之前。
    """
    base_tokens = [str(token) for token in tokens]
    if not base_tokens or base_tokens[0] != "BOS":
        return base_tokens

    analysis = analyze_phrase_candidates(base_tokens)
    if not analysis.boundaries:
        return base_tokens

    out = list(base_tokens)
    # Insert from tail to head so earlier insertion indices stay valid.
    for boundary in sorted(analysis.boundaries, key=lambda b: (b.bar_index, b.anchor_pos), reverse=True):
        if boundary.bar_index >= len(analysis.bars):
            continue
        bar = analysis.bars[boundary.bar_index]
        if bar.note_count == 0:
            continue
        # Find bar header end in `out` (start_token is index in original tokens; after prior
        # inserts, indices for *earlier* bars are unchanged because we walk tail->head).
        bar_start = bar.start_token
        header_end = bar_start + 1
        while header_end < len(out) and (
            out[header_end].startswith("TEMPO_") or out[header_end].startswith("KEY_")
        ):
            header_end += 1

        if boundary.anchor_pos == 0:
            insert_at = header_end
        else:
            target = f"POS_{boundary.anchor_pos}"
            scan = header_end
            found = -1
            first_pos = -1
            while scan < len(out) and out[scan] != "BAR":
                if out[scan].startswith("POS_"):
                    if first_pos < 0:
                        first_pos = scan
                    if out[scan] == target:
                        found = scan
                        break
                scan += 1
            if found >= 0:
                insert_at = found
            elif first_pos >= 0:
                insert_at = first_pos
            else:
                continue
        out.insert(insert_at, "PHRASE")

    # Idempotent adjacency dedup
    deduped: List[str] = []
    for token in out:
        if deduped and deduped[-1] == "PHRASE" and token == "PHRASE":
            continue
        deduped.append(token)
    return deduped
```

- [ ] **Step 4：跑测试，期望 PASS**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k inject_phrase_tokens -v`
Expected: PASS

---

## Task 7: `_tokenize_note_events` 串联 inject_phrase_tokens

**Files:**
- Modify: `src/tokenizer/midi_codec.py`
- Test: `tests/test_tokenizer_midi_codec.py`

- [ ] **Step 1：写失败测试 `test_tokenize_midi_injects_phrase`**

```python
def test_tokenize_midi_emits_phrase_tokens(self) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120.0), time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=240))
    track.append(mido.Message("note_on", note=62, velocity=80, time=1920))  # next bar
    track.append(mido.Message("note_off", note=62, velocity=0, time=240))
    config = TokenizerConfig()
    tokens = tokenize_midi(midi, config)
    self.assertIn("PHRASE", tokens)
```

- [ ] **Step 2：跑测试，期望 FAIL**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k test_tokenize_midi_emits_phrase_tokens -v`
Expected: FAIL

- [ ] **Step 3：在 `_tokenize_note_events` 末尾 wrap inject_phrase_tokens**

替换原 `return inject_key_tokens(tokens)` 为：
```python
    return inject_phrase_tokens(inject_key_tokens(tokens))
```

并把空序列分支也保护起来：
```python
    if not notes:
        return inject_phrase_tokens(inject_key_tokens(["BOS", "EOS"]))
```

- [ ] **Step 4：跑测试，期望 PASS**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k test_tokenize_midi_emits_phrase_tokens -v`
Expected: PASS

---

## Task 8: `tokens_to_midi` 跳过 PHRASE

**Files:**
- Modify: `src/tokenizer/midi_codec.py` (tokens_to_midi)
- Test: `tests/test_tokenizer_midi_codec.py`

- [ ] **Step 1：写失败测试 `test_tokens_to_midi_ignores_phrase`**

```python
def test_tokens_to_midi_ignores_phrase(self) -> None:
    config = TokenizerConfig()
    base = inject_key_tokens([
        "BOS", "TEMPO_120",
        "BAR", "POS_0", "INST_PIANO", "PITCH_60", "DUR_4", "VEL_8",
        "BAR", "POS_0", "INST_PIANO", "PITCH_62", "DUR_4", "VEL_8",
        "EOS",
    ])
    with_phrase = list(base)
    # inject manually: bar-head PHRASE on first content bar
    bar_idx = with_phrase.index("BAR")
    header_end = bar_idx + 1
    while header_end < len(with_phrase) and (
        with_phrase[header_end].startswith("TEMPO_") or with_phrase[header_end].startswith("KEY_")
    ):
        header_end += 1
    with_phrase.insert(header_end, "PHRASE")
    a = tokens_to_midi(base, config)
    b = tokens_to_midi(with_phrase, config)
    self.assertEqual([m.bytes() for m in a.tracks[1] if hasattr(m, "bytes") and not m.is_meta],
                     [m.bytes() for m in b.tracks[1] if hasattr(m, "bytes") and not m.is_meta])
```

- [ ] **Step 2：跑测试，期望 FAIL**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k test_tokens_to_midi_ignores_phrase -v`
Expected: FAIL（PHRASE 进 `_validate_complete_sequence` 失败或在主循环里被当成非 POS 报错）

- [ ] **Step 3：修改 `tokens_to_midi` 主循环**

在 BAR 头部消费 TEMPO/KEY 之后再加一段：
```python
        if idx < len(normalized) - 1 and normalized[idx] == "PHRASE":
            idx += 1
```

把 event loop 的 `while` 条件改为 PHRASE 也可以推进：
```python
        while idx < len(normalized) - 1 and (
            normalized[idx].startswith("POS_") or normalized[idx] == "PHRASE"
        ):
            if normalized[idx] == "PHRASE":
                idx += 1
                continue
            if idx + 4 >= len(normalized):
                raise ValueError("incomplete note event at end of token sequence")
            # ... 现有 POS/INST/PITCH/DUR/VEL 解析保持不变 ...
```

完整替换示意（只展示改动后的循环体）：
```python
        while idx < len(normalized) - 1 and (
            normalized[idx].startswith("POS_") or normalized[idx] == "PHRASE"
        ):
            if normalized[idx] == "PHRASE":
                idx += 1
                continue
            if idx + 4 >= len(normalized):
                raise ValueError("incomplete note event at end of token sequence")
            pos = _parse_token_int(normalized[idx], "POS_")
            inst = _instrument_from_token(normalized[idx + 1])
            pitch = _parse_token_int(normalized[idx + 2], "PITCH_")
            dur = _parse_token_int(normalized[idx + 3], "DUR_")
            vel_bin = _parse_token_int(normalized[idx + 4], "VEL_")
            if pos < 0 or pos >= config.positions_per_bar:
                raise ValueError(f"position token out of range: `{normalized[idx]}`")
            if pitch < config.pitch_min or pitch > config.pitch_max:
                raise ValueError(f"pitch token out of range: `{normalized[idx + 2]}`")
            if dur not in config.duration_bins:
                raise ValueError(f"duration token out of range: `{normalized[idx + 3]}`")
            if vel_bin < 0 or vel_bin >= config.velocity_bins:
                raise ValueError(f"velocity token out of range: `{normalized[idx + 4]}`")
            start_tick = bar_start_tick + int(round(pos * pos_ticks))
            duration_tick = max(1, int(round(dur * pos_ticks)))
            notes.append(
                _DecodedNote(
                    start_tick=start_tick,
                    end_tick=start_tick + duration_tick,
                    pitch=pitch,
                    velocity=bin_to_velocity(vel_bin, velocity_config),
                    inst=inst,
                )
            )
            idx += 5
```

`_validate_complete_sequence` 调用走更新后的 `validate_token_order`，会接受 PHRASE。

- [ ] **Step 4：跑测试，期望 PASS**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -k test_tokens_to_midi_ignores_phrase -v`
Expected: PASS

- [ ] **Step 5：跑现有 tokenize roundtrip 测试，确保未引入回归**

Run: `python -m pytest tests/test_tokenizer_midi_codec.py -v`
Expected: 所有现有用例 PASS（PHRASE 已合法）

---

## Task 9: tokenize_dataset 加 phrase 统计

**Files:**
- Modify: `src/tokenizer/tokenize_dataset.py`

- [ ] **Step 1：在 `_accumulate_key_token_stats` 之后新增 `_empty_phrase_token_stats` / `_accumulate_phrase_token_stats`**

```python
def _empty_phrase_token_stats() -> Dict[str, object]:
    return {
        "phrase_token_total": 0,
        "bar_aligned_phrase_total": 0,
        "mid_bar_phrase_total": 0,
        "bar_spans_sum": 0,
        "bar_spans_count": 0,
    }


def _accumulate_phrase_token_stats(stats: Dict[str, object], tokens: List[str]) -> None:
    bar_positions: List[int] = []
    phrase_indices: List[int] = []
    for idx, token in enumerate(tokens):
        if token == "BAR":
            bar_positions.append(idx)
        elif token == "PHRASE":
            phrase_indices.append(idx)
    if not phrase_indices:
        return
    stats["phrase_token_total"] = int(stats.get("phrase_token_total", 0)) + len(phrase_indices)

    bar_index_by_token: Dict[int, int] = {pos: i for i, pos in enumerate(bar_positions)}

    def _enclosing_bar(token_index: int) -> int:
        bar_index = -1
        for pos, idx in bar_index_by_token.items():
            if pos < token_index:
                bar_index = max(bar_index, idx)
            else:
                break
        return bar_index

    phrase_bar_indices: List[int] = []
    for phrase_idx in phrase_indices:
        # bar-aligned PHRASE: previous non-TEMPO/KEY token is BAR or its trailing TEMPO/KEY
        prev = phrase_idx - 1
        while prev > 0 and (tokens[prev].startswith("TEMPO_") or tokens[prev].startswith("KEY_")):
            prev -= 1
        is_bar_aligned = tokens[prev] == "BAR" if prev >= 0 else False
        if is_bar_aligned:
            stats["bar_aligned_phrase_total"] = int(stats.get("bar_aligned_phrase_total", 0)) + 1
        else:
            stats["mid_bar_phrase_total"] = int(stats.get("mid_bar_phrase_total", 0)) + 1
        phrase_bar_indices.append(_enclosing_bar(phrase_idx))

    eos_bar = len(bar_positions)
    sorted_phrase_bars = sorted(set(b for b in phrase_bar_indices if b >= 0))
    for i, bar_index in enumerate(sorted_phrase_bars):
        next_bar = sorted_phrase_bars[i + 1] if i + 1 < len(sorted_phrase_bars) else eos_bar
        stats["bar_spans_sum"] = int(stats.get("bar_spans_sum", 0)) + (next_bar - bar_index)
        stats["bar_spans_count"] = int(stats.get("bar_spans_count", 0)) + 1


def _finalize_phrase_token_stats(stats: Dict[str, object], num_sequences: int) -> Dict[str, object]:
    total = int(stats.get("phrase_token_total", 0))
    bar_count = int(stats.get("bar_spans_count", 0))
    bar_sum = int(stats.get("bar_spans_sum", 0))
    mid_bar = int(stats.get("mid_bar_phrase_total", 0))
    return {
        "phrase_token_total": total,
        "bar_aligned_phrase_total": int(stats.get("bar_aligned_phrase_total", 0)),
        "mid_bar_phrase_total": mid_bar,
        "mean_phrases_per_sequence": (0.0 if num_sequences == 0 else total / float(num_sequences)),
        "mid_bar_phrase_ratio": (0.0 if total == 0 else mid_bar / float(total)),
        "mean_phrase_bar_span": (0.0 if bar_count == 0 else bar_sum / float(bar_count)),
    }
```

- [ ] **Step 2：把 phrase stats 接入 split 与 total 累加**

在 `process(...)` 中：
- 在 `total_key_token_stats = _empty_key_token_stats(...)` 之后追加 `total_phrase_token_stats = _empty_phrase_token_stats()`
- 每个 split 内 `split_key_token_stats` 之后追加 `split_phrase_token_stats = _empty_phrase_token_stats()`
- 每条成功的 `tokens` 上同时调用 `_accumulate_phrase_token_stats(split_phrase_token_stats, tokens)` 与 `_accumulate_phrase_token_stats(total_phrase_token_stats, tokens)`
- `split_stats[split_name]` 字典里增加 `"phrase_token_stats": _finalize_phrase_token_stats(split_phrase_token_stats, len(tok_lines))`
- 最终 `stats` 字典里增加 `"phrase_token_stats": _finalize_phrase_token_stats(total_phrase_token_stats, total_written_rows)`

- [ ] **Step 3：冒烟运行（小 limit）**

Run: `python -m src.tokenizer.tokenize_dataset --config configs/tokenizer/tokenizer.yaml --limit-per-split 1 --output-dir /tmp/tok_smoke --vocab-path /tmp/tok_smoke/tokenizer_vocab.json --stats-path /tmp/tok_smoke/token_stats.json` （若 Windows 下 `/tmp` 不可写，改为 `outputs/_smoke`）

Expected: 退出码 0；`token_stats.json` 包含 `phrase_token_stats.mean_phrases_per_sequence` 字段。

---

## Task 10: FSM 新增 AFTER_PHRASE 状态

**Files:**
- Modify: `src/decoding/grammar_fsm.py`
- Test: `tests/test_decoding_grammar_fsm.py` (new file)

- [ ] **Step 1：新建测试文件 `tests/test_decoding_grammar_fsm.py`**

```python
from __future__ import annotations

import unittest

from src.tokenizer import TokenizerConfig, build_vocab
from src.decoding.grammar_fsm import (
    AFTER_BAR,
    AFTER_BAR_TEMPO,
    AFTER_BAR_KEY,
    AFTER_BAR_TEMPO_KEY,
    AFTER_POS,
    AFTER_VEL,
    TuneFlowGrammarFSM,
)


class FsmPhraseTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = build_vocab(TokenizerConfig())
        self.fsm = TuneFlowGrammarFSM(self.vocab)
        self.phrase_id = self.vocab["PHRASE"]
        self.pos0_id = self.vocab["POS_0"]
        self.bar_id = self.vocab["BAR"]
        self.eos_id = self.vocab["EOS"]

    def test_phrase_allowed_after_bar_head(self) -> None:
        for state in (AFTER_BAR, AFTER_BAR_TEMPO, AFTER_BAR_KEY, AFTER_BAR_TEMPO_KEY):
            self.assertIn(self.phrase_id, self.fsm.allowed_token_ids(state),
                          msg=f"PHRASE must be allowed from {state}")
            self.assertEqual(self.fsm.transition(state, self.phrase_id), "after_phrase")

    def test_phrase_allowed_after_vel(self) -> None:
        self.assertIn(self.phrase_id, self.fsm.allowed_token_ids(AFTER_VEL))
        self.assertEqual(self.fsm.transition(AFTER_VEL, self.phrase_id), "after_phrase")

    def test_after_phrase_accepts_only_pos(self) -> None:
        allowed = self.fsm.allowed_token_ids("after_phrase")
        self.assertEqual(set(allowed), set(self.fsm._pos_ids))
        self.assertEqual(self.fsm.transition("after_phrase", self.pos0_id), AFTER_POS)
        self.assertIsNone(self.fsm.transition("after_phrase", self.bar_id))
        self.assertIsNone(self.fsm.transition("after_phrase", self.eos_id))

    def test_compatible_states_for_phrase_suffix(self) -> None:
        suffix_ids = [self.phrase_id, self.pos0_id]
        compatible = self.fsm.compatible_states_for_suffix_ids(suffix_ids)
        # AFTER_VEL should be able to consume PHRASE+POS+(rest)+EOS
        self.assertIn(AFTER_VEL, compatible)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2：跑测试，期望 FAIL**

Run: `python -m pytest tests/test_decoding_grammar_fsm.py -v`
Expected: FAIL（`AFTER_PHRASE` 不存在 / PHRASE 不在 allowed list）

- [ ] **Step 3：在 `grammar_fsm.py` 加 AFTER_PHRASE 状态、转移、allowed 表**

顶部状态常量新增：
```python
AFTER_PHRASE = "after_phrase"
```

`_NON_TERMINAL_STATES` 追加 `AFTER_PHRASE`。

在 `__init__` 里收集 PHRASE id：
```python
        self._phrase_ids: tuple[int, ...] = self._collect_prefix_ids("PHRASE")
        self._register_ids(self._phrase_ids, "PHRASE")
```

`_allowed_ids_by_state` 改动：
```python
            AFTER_BAR: (
                *self._tempo_ids, *self._key_ids, *self._phrase_ids,
                *self._pos_ids, self.bar_id, self.eos_id,
            ),
            AFTER_BAR_TEMPO: (
                *self._key_ids, *self._phrase_ids,
                *self._pos_ids, self.bar_id, self.eos_id,
            ),
            AFTER_BAR_KEY: (*self._phrase_ids, *self._pos_ids, self.bar_id, self.eos_id),
            AFTER_BAR_TEMPO_KEY: (*self._phrase_ids, *self._pos_ids, self.bar_id, self.eos_id),
            AFTER_VEL: (*self._phrase_ids, *self._pos_ids, self.bar_id, self.eos_id),
            AFTER_PHRASE: self._pos_ids,
```

注意 `_collect_prefix_ids("PHRASE")` 会匹配前缀，但词表里只有恰好一个 token 命中（`PHRASE` 本身），这是预期的。

在 `transition` 中：
- `AFTER_BAR / AFTER_BAR_TEMPO / AFTER_BAR_KEY / AFTER_BAR_TEMPO_KEY` 分支增加：
```python
            if category == "PHRASE":
                return AFTER_PHRASE
```
- `AFTER_VEL` 分支增加同样的 PHRASE 转移
- 新增 AFTER_PHRASE 分支：
```python
        if state == AFTER_PHRASE:
            return AFTER_POS if category == "POS" else None
```

`_unexpected_reason` 与 `_unfinished_reason` 增加 AFTER_PHRASE 分支：
```python
        if state == AFTER_PHRASE:
            return f"expected_pos@{index}:{token}"
```
```python
        if state in {... AFTER_VEL, AFTER_PHRASE}:
            return "missing_eos"
```
（实际上 AFTER_PHRASE 只能转 POS，所以 unfinished 走 missing_eos 分支没有意义；改为）：
```python
        if state == AFTER_PHRASE:
            return "incomplete_phrase_expected_pos"
```

- [ ] **Step 4：跑测试，期望 PASS**

Run: `python -m pytest tests/test_decoding_grammar_fsm.py -v`
Expected: PASS

- [ ] **Step 5：跑既有 FSM 用例（若仓库有 grammar_fsm 相关历史用例则一并验证）**

Run: `python -m pytest tests/ -k grammar_fsm -v`
Expected: PASS

---

## Task 11: 评测窗口改名 + PHRASE 优先实现

**Files:**
- Modify: `src/utils/eval_windows.py`
- Modify: `src/utils/benchmarking.py`
- Test: `tests/test_music_analysis.py`

- [ ] **Step 1：写失败测试 `test_eval_window_prefers_phrase_anchor`**

在 `tests/test_music_analysis.py` 把 line 17 的 import 改为：
```python
from src.utils.eval_windows import sample_phrase_aligned_subsequence
```

把 line 206 / 216 的两处调用同步改名。然后追加测试：
```python
def test_eval_window_starts_on_phrase_anchor_when_possible(self) -> None:
    from src.tokenizer.midi_codec import inject_phrase_tokens
    raw = _phrase_source_tokens()
    tokens = inject_phrase_tokens(inject_key_tokens(raw))
    rng = random.Random(0)
    window = sample_phrase_aligned_subsequence(tokens, max_core_tokens=80, min_core_tokens=12, rng=rng)
    self.assertIsNotNone(window)
    assert window is not None
    # The body's first non-header token should be either PHRASE or BAR (fallback)
    self.assertIn(window[0], {"BOS"})
    # header: skip BOS, optional TEMPO_, optional KEY_
    idx = 1
    while idx < len(window) and (window[idx].startswith("TEMPO_") or window[idx].startswith("KEY_")):
        idx += 1
    self.assertIn(window[idx], {"PHRASE", "BAR"})

def test_eval_window_keeps_inline_phrases(self) -> None:
    from src.tokenizer.midi_codec import inject_phrase_tokens
    raw = _long_phrase_source_tokens()
    tokens = inject_phrase_tokens(inject_key_tokens(raw))
    rng = random.Random(1)
    window = sample_phrase_aligned_subsequence(tokens, max_core_tokens=160, min_core_tokens=24, rng=rng)
    self.assertIsNotNone(window)
    assert window is not None
    # If source had any PHRASE in selected span, it should still be present
    if "PHRASE" in tokens[1:-1]:
        # at least one PHRASE should survive in some window
        # we only assert non-zero count over a small batch of seeds
        survived = 0
        for seed in range(8):
            w = sample_phrase_aligned_subsequence(
                tokens, max_core_tokens=160, min_core_tokens=24, rng=random.Random(seed),
            )
            if w is not None and "PHRASE" in w:
                survived += 1
        self.assertGreater(survived, 0)
```

- [ ] **Step 2：跑测试，期望 FAIL（函数未改名 / 行为未实现）**

Run: `python -m pytest tests/test_music_analysis.py -k "phrase_anchor or inline_phrases" -v`
Expected: FAIL（NameError 或 PHRASE 没保留）

- [ ] **Step 3：重写 `src/utils/eval_windows.py`**

完整替换文件内容：
```python
"""Helpers for building valid evaluation windows from token sequences."""

from __future__ import annotations

import random
from typing import Sequence

from src.music_analysis import analyze_phrase_candidates


def _iter_bar_token_positions(tokens: Sequence[str]) -> list[int]:
    return [idx for idx, token in enumerate(tokens) if token == "BAR"]


def _iter_phrase_token_positions(tokens: Sequence[str]) -> list[int]:
    return [idx for idx, token in enumerate(tokens) if token == "PHRASE"]


def _build_window_at_positions(
    source_tokens: Sequence[str],
    *,
    start_index: int,
    end_index: int,
) -> list[str] | None:
    """Materialize a window starting at `start_index` (a PHRASE/BAR cut) and ending
    just before `end_index` (a PHRASE/BAR cut or EOS index).

    Header is `BOS [TEMPO] [KEY]` derived from the effective context at start.
    """
    analysis = analyze_phrase_candidates(source_tokens)
    if not analysis.bars:
        return None

    # Find bar that contains start_index
    bar_index = -1
    for idx, bar in enumerate(analysis.bars):
        if bar.start_token <= start_index < bar.end_token:
            bar_index = idx
            break
    if bar_index < 0:
        return None

    tempo_token = analysis.bars[bar_index].effective_tempo_token
    key_token = analysis.bars[bar_index].effective_key_token

    body = [str(token) for token in source_tokens[start_index:end_index]]
    # Normalize BAR headers in body: strip trailing TEMPO/KEY duplicates but keep PHRASE
    normalized: list[str] = []
    idx = 0
    while idx < len(body):
        token = body[idx]
        if token == "BAR":
            normalized.append("BAR")
            idx += 1
            # Drop redundant in-body TEMPO/KEY (already represented in the leading window header)
            while idx < len(body) and (body[idx].startswith("TEMPO_") or body[idx].startswith("KEY_")):
                idx += 1
            # Keep optional PHRASE
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

    Cut points are PHRASE positions first, then BAR positions as fallback.
    """
    if max_core_tokens <= 0:
        return None
    if min_core_tokens <= 0:
        min_core_tokens = 1
    if not source_tokens or source_tokens[0] != "BOS" or source_tokens[-1] != "EOS":
        return None

    phrase_positions = _iter_phrase_token_positions(source_tokens)
    bar_positions = _iter_bar_token_positions(source_tokens)
    eos_index = len(source_tokens) - 1
    # cut points = phrase positions ∪ bar positions; type tag for diagnostics
    cut_points: list[tuple[int, str]] = [(pos, "phrase") for pos in phrase_positions]
    cut_points.extend((pos, "bar") for pos in bar_positions)
    cut_points.append((eos_index, "eos"))
    cut_points.sort(key=lambda item: (item[0], 0 if item[1] == "phrase" else 1))
    if not cut_points:
        return None

    def _try_pick(prefer: str) -> list[str] | None:
        candidates_start = [pos for pos, kind in cut_points if kind == prefer]
        candidates_end = [pos for pos, kind in cut_points if kind in {prefer, "eos"}]
        if not candidates_start or not candidates_end:
            return None
        for _ in range(max_attempts):
            start = candidates_start[rng.randrange(len(candidates_start))]
            ends = [end for end in candidates_end if end > start and (end - start) <= max_core_tokens]
            ends = [end for end in ends if (end - start) >= min_core_tokens]
            if not ends:
                continue
            end = ends[rng.randrange(len(ends))]
            window = _build_window_at_positions(source_tokens, start_index=start, end_index=end)
            if window is None:
                continue
            body_len = len(window) - 2
            if min_core_tokens <= body_len <= max_core_tokens:
                return window
        return None

    return _try_pick("phrase") or _try_pick("bar")
```

- [ ] **Step 4：把 `benchmarking.py` 调用点改名**

在 `src/utils/benchmarking.py`：
- line 13：`from .eval_windows import sample_bar_aligned_subsequence` → `from .eval_windows import sample_phrase_aligned_subsequence`
- line 241、line 301：函数名同步改

Run: `python -m pytest tests/test_benchmarking.py -v`
Expected: PASS（仅函数改名，行为兼容）

- [ ] **Step 5：跑测试，期望 PASS**

Run: `python -m pytest tests/test_music_analysis.py -k "phrase_anchor or inline_phrases or window_start_tempo or window_start_key" -v`
Expected: PASS

---

## Task 12: 训练管线删除 phrase sampling + 新增 phrase_event 单元

**Files:**
- Modify: `src/training/train_base.py`
- Modify: `configs/train/train_base_run_full.yaml`
- Modify: `configs/train/train_base_run_small.yaml`

- [ ] **Step 1：从 `train_base.py` 顶部 import 删除 `PhraseAnalysisConfig / PhraseWindowPolicy / analyze_phrase_candidates / sample_phrase_window`**

把：
```python
from ..music_analysis import (
    PhraseAnalysisConfig,
    PhraseWindowPolicy,
    analyze_phrase_candidates,
    sample_phrase_window,
)
```
整段删除。

- [ ] **Step 2：删除 `PhraseSamplingConfig` dataclass 与所有引用**

删除：
- `PhraseSamplingConfig` 定义
- `TokenBinDataset._pick_phrase_policy_kind`
- `TokenBinDataset._phrase_policy_fallback_order`
- `TokenBinDataset._phrase_policy_for_kind`
- `TokenBinDataset._sample_phrase_window`
- `TokenBinDataset._choose_phrase_hole_bar_span`
- `TokenBinDataset._build_phrase_aware_fim_example`
- `TokenBinDataset._build_phrase_or_fallback_fim_example`

- [ ] **Step 3：简化 `sample_batch`**

替换为：
```python
    def sample_batch(
        self,
        torch_mod,
        rng: random.Random,
        batch_size: int,
        seq_len: int,
        device,
        id_to_token: list[str],
        token_to_id: dict[str, int] | None = None,
        eos_token_id: int | None = None,
    ):
        """采样 NEXT batch（labels 与 input_ids 对齐，由模型内部完成 shift）。"""
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for _ in range(batch_size):
            window = self._sample_aligned_window(
                rng=rng,
                window_len=seq_len,
                id_to_token=id_to_token,
                anchor="random",
            )
            input_rows.append(window)
            label_rows.append(window.copy())

        if eos_token_id is not None:
            input_rows, label_rows = self._pad_rows(
                input_rows, label_rows, seq_len=seq_len, eos_token_id=eos_token_id,
            )
        input_ids = torch_mod.tensor(input_rows, dtype=torch_mod.long, device=device)
        labels = torch_mod.tensor(label_rows, dtype=torch_mod.long, device=device)
        return input_ids, labels
```

- [ ] **Step 4：简化 `sample_mixed_batch`**

替换为：
```python
    def sample_mixed_batch(
        self,
        torch_mod,
        rng: random.Random,
        batch_size: int,
        seq_len: int,
        device,
        id_to_token: list[str],
        token_to_id: dict[str, int] | None,
        fim_ratio: float,
        fim_hole_token_id: int | None,
        fim_mid_token_id: int | None,
        fim_min_span: int,
        fim_max_span: int,
        fim_eos_ratio: float,
        eos_token_id: int | None,
    ):
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
                            rng=rng, window_len=seq_len - 3,
                            id_to_token=id_to_token, exclude_terminal_eos=True,
                        )
                    else:
                        base_tokens = self._sample_aligned_window(
                            rng=rng, window_len=seq_len - 2, id_to_token=id_to_token,
                        )
                    try:
                        fim_input, fim_labels = self._build_fim_example(
                            base_tokens=base_tokens, rng=rng, id_to_token=id_to_token,
                            fim_hole_token_id=fim_hole_token_id,
                            fim_mid_token_id=fim_mid_token_id,
                            fim_min_span=fim_min_span, fim_max_span=fim_max_span,
                            append_eos=use_fim_eos, eos_token_id=eos_token_id,
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
                window = self._sample_aligned_window(
                    rng=rng, window_len=seq_len, id_to_token=id_to_token, anchor="random",
                )
                input_rows.append(window)
                label_rows.append(window.copy())

        if eos_token_id is not None:
            input_rows, label_rows = self._pad_rows(
                input_rows, label_rows, seq_len=seq_len, eos_token_id=eos_token_id,
            )
        input_ids = torch_mod.tensor(input_rows, dtype=torch_mod.long, device=device)
        labels = torch_mod.tensor(label_rows, dtype=torch_mod.long, device=device)
        return input_ids, labels, fim_examples
```

- [ ] **Step 5：`_evaluate` 签名删除 `phrase_sampling`**

把 `_evaluate(... phrase_sampling: PhraseSamplingConfig)` 改为不再接收该参数；并在内部对 `sample_batch` 调用同步删除该 kwarg。

- [ ] **Step 6：`_collect_window_cut_positions` 增加 PHRASE+event 整体推进**

在 `token == "BAR"` 分支前加入 PHRASE 处理：
```python
            if token == "PHRASE":
                # PHRASE must be followed by a full event 5-tuple. Treat as 1+5=6-token unit.
                if idx + 5 >= len(sequence_tokens):
                    return []
                inst_id = int(sequence_tokens[idx + 2])
                pitch_id = int(sequence_tokens[idx + 3])
                dur_id = int(sequence_tokens[idx + 4])
                vel_id = int(sequence_tokens[idx + 5])
                pos_id = int(sequence_tokens[idx + 1])
                if not (
                    0 <= pos_id < len(id_to_token)
                    and 0 <= inst_id < len(id_to_token)
                    and 0 <= pitch_id < len(id_to_token)
                    and 0 <= dur_id < len(id_to_token)
                    and 0 <= vel_id < len(id_to_token)
                    and id_to_token[pos_id].startswith("POS_")
                    and id_to_token[inst_id].startswith("INST_")
                    and id_to_token[pitch_id].startswith("PITCH_")
                    and id_to_token[dur_id].startswith("DUR_")
                    and id_to_token[vel_id].startswith("VEL_")
                ):
                    return []
                idx += 6
                positions.append(idx)
                continue
```

并在 `token in {"BOS", "EOS", ...}` 那个早期 branch 中**不要**单独列 PHRASE。

- [ ] **Step 7：`_collect_fim_maskable_units` 新增 `phrase_event` 单元**

在 `token == "BAR"` 分支前加：
```python
            if token == "PHRASE":
                if idx + 5 >= len(base_tokens):
                    idx += 1
                    group_id += 1
                    continue
                pos_id = int(base_tokens[idx + 1])
                inst_id = int(base_tokens[idx + 2])
                pitch_id = int(base_tokens[idx + 3])
                dur_id = int(base_tokens[idx + 4])
                vel_id = int(base_tokens[idx + 5])
                if (
                    0 <= pos_id < len(id_to_token)
                    and 0 <= inst_id < len(id_to_token)
                    and 0 <= pitch_id < len(id_to_token)
                    and 0 <= dur_id < len(id_to_token)
                    and 0 <= vel_id < len(id_to_token)
                    and id_to_token[pos_id].startswith("POS_")
                    and id_to_token[inst_id].startswith("INST_")
                    and id_to_token[pitch_id].startswith("PITCH_")
                    and id_to_token[dur_id].startswith("DUR_")
                    and id_to_token[vel_id].startswith("VEL_")
                ):
                    units.append((idx, idx + 6, "phrase_event", group_id))
                    idx += 6
                    continue
                idx += 1
                group_id += 1
                continue
```

- [ ] **Step 8：清理 CLI 与 `main()` 中 phrase 参数**

从 `build_arg_parser` 删除 8 个 phrase-related 参数：`--use-phrase-window-sampling`、`--single-phrase-sample-ratio`、`--cross-phrase-sample-ratio`、`--long-context-sample-ratio`、`--phrase-min-bars`、`--phrase-max-bars`、`--single-phrase-bar-min/max`、`--cross-phrase-bar-min/max`、`--long-context-bar-min/max`。

`main()` 中：
- 删除 `total_phrase_ratio` 求和校验、`phrase_min_bars` 校验、`single/cross/long_phrase_bar_*` 校验
- 删除 `phrase_sampling = PhraseSamplingConfig(...)` 实例化
- `sample_mixed_batch(...)` 调用去掉 `phrase_sampling=phrase_sampling`
- `_evaluate(...)` 调用去掉 `phrase_sampling=phrase_sampling`
- `run_start` metrics payload 中去掉 `use_phrase_window_sampling / single_phrase_sample_ratio / cross_phrase_sample_ratio / long_context_sample_ratio`
- 训练日志 print/metrics 中去掉 `phrase(single/cross/long)`、`phrase_fim`、`fallback`、`single_phrase_examples`、`cross_boundary_examples`、`long_context_examples`、`phrase_fim_examples`、`phrase_fim_fallback_examples` 字段
- `step_sample_stats` 字典与 `sample_stats` 字段删除
- `sample_mixed_batch` 返回值由 `(input_ids, labels, fim_examples, sample_stats)` 改为 `(input_ids, labels, fim_examples)`；调用方同步只解三元组

- [ ] **Step 9：更新 yaml 配置**

`configs/train/train_base_run_full.yaml` 与 `configs/train/train_base_run_small.yaml`：
- 删除字段 `use_phrase_window_sampling / single_phrase_sample_ratio / cross_phrase_sample_ratio / long_context_sample_ratio / phrase_min_bars / phrase_max_bars / single_phrase_bar_min / single_phrase_bar_max / cross_phrase_bar_min / cross_phrase_bar_max / long_context_bar_min / long_context_bar_max`
- 把原 line 27-28 注释修改为：
```yaml
  # NEXT 窗口由结构对齐的随机采样路径生成，乐句信息已下沉为 PHRASE token 由模型自学
```

- [ ] **Step 10：冒烟跑训练入口（不开 GPU）**

Run: `python -c "from src.training import train_base; train_base.build_arg_parser().parse_args(['--steps', '1'])"`
Expected: 无异常退出（仅验证 CLI 解析正常）

---

## Task 13: 修正 test_music_analysis.py

**Files:**
- Modify: `tests/test_music_analysis.py`

- [ ] **Step 1：删除已废弃符号的 import**

把：
```python
from src.music_analysis import (
    KeyAnalysisConfig,
    PhraseAnalysisConfig,
    PhraseWindowPolicy,
    analyze_key_timeline,
    analyze_phrase_candidates,
    extract_phrase,
    sample_phrase_window,
)
from src.tokenizer.midi_codec import inject_key_tokens
from src.training.train_base import PhraseSamplingConfig, TokenBinDataset
from src.utils.eval_windows import sample_bar_aligned_subsequence
```

改为：
```python
from src.music_analysis import (
    KeyAnalysisConfig,
    PhraseAnalysisConfig,
    PhraseBoundary,
    analyze_key_timeline,
    analyze_phrase_candidates,
)
from src.tokenizer.midi_codec import inject_key_tokens
from src.utils.eval_windows import sample_phrase_aligned_subsequence
```

- [ ] **Step 2：删除依赖已删函数的测试**

删除以下方法：
- `test_extract_phrase_rebuilds_single_tempo_view`
- `test_extract_phrase_keeps_only_window_start_key_token`
- `test_sample_phrase_window_supports_cross_boundary_and_long_context`
- `test_phrase_fim_builder_falls_back_to_generic_structure`
- `test_phrase_fim_builder_uses_phrase_hole_on_rich_window`
- `test_phrase_fim_builder_keeps_phrase_hole_when_reappending_eos`

- [ ] **Step 3：把原 `test_eval_window_keeps_only_window_start_tempo` 与 `test_eval_window_keeps_only_window_start_key` 调用点改名**

```python
window = sample_phrase_aligned_subsequence(tokens, max_core_tokens=48, min_core_tokens=12, rng=rng)
```
（line 216 同理）

- [ ] **Step 4：新增 boundary 一致性测试**

```python
def test_analyze_phrase_candidates_returns_boundaries(self) -> None:
    analysis = analyze_phrase_candidates(_phrase_source_tokens())
    self.assertTrue(analysis.boundaries)
    first_content_bar = next(i for i, bar in enumerate(analysis.bars) if bar.note_count > 0)
    self.assertEqual(analysis.boundaries[0], PhraseBoundary(first_content_bar, 0))

def test_phrase_spans_align_with_boundaries(self) -> None:
    analysis = analyze_phrase_candidates(_phrase_source_tokens())
    expected_starts = tuple(b.bar_index for b in analysis.boundaries)
    actual_starts = tuple(span.start_bar for span in analysis.phrase_spans)
    self.assertEqual(actual_starts, expected_starts)

def test_mid_bar_anchor_when_rest_threshold_met(self) -> None:
    # Construct: bar 0 dense, bar 1 starts at POS_8 (>= 8 default threshold) with a clear lead-in rest
    tokens = ["BOS", "TEMPO_120"]
    tokens.extend(_bar((0, 60, 4), (4, 62, 4), (8, 64, 4), (12, 66, 4)))
    tokens.extend(_bar((8, 72, 4)))
    tokens.extend(_bar((0, 67, 4)))
    tokens.append("EOS")
    analysis = analyze_phrase_candidates(tokens)
    mid_bar_anchors = [b for b in analysis.boundaries if b.anchor_pos > 0]
    # At least one anchor should land on bar 1 with POS_8
    self.assertTrue(any(b.bar_index == 1 and b.anchor_pos == 8 for b in mid_bar_anchors))
```

- [ ] **Step 5：跑整个测试文件**

Run: `python -m pytest tests/test_music_analysis.py -v`
Expected: PASS

---

## Task 14: 跑完整测试集 + 文档更新

**Files:**
- Modify: `docs/todo.md`

- [ ] **Step 1：跑全部 pytest**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 2：把 `docs/todo.md` 第 1 项「Phrase 结构显式建模」追加完成标注**

在「## 1. Phrase 结构显式建模」section 起始处增加一段：
```markdown
> **2026-05-17 更新**：本 PR 已落地「PHRASE 边界 token 引入」与「动态 window / 启发式切句机制下沉」。剩余子项（句类型 / 句关系 / 句尾收束 / 呼吸 / 重音 / 动机角色等分类头 / 句级辅助任务头 / 基于 PHRASE 的训练采样重加权）保留为后续工作。
```

- [ ] **Step 3：跑一次 lint / import 完整性冒烟**

Run: `python -c "import src.training.train_base; import src.music_analysis; import src.tokenizer.midi_codec; import src.utils.eval_windows; import src.decoding.grammar_fsm; print('ok')"`
Expected: `ok`

---

## Task 15: 一次大 commit

**Files:**
- 所有上述修改

- [ ] **Step 1：检查 git 状态**

Run: `git status`
Expected: 列出全部上述新增 / 修改文件

- [ ] **Step 2：分批 add（避免误带）**

Run:
```
git add src/music_analysis/phrase_analysis.py src/music_analysis/__init__.py
git add src/tokenizer/midi_codec.py src/tokenizer/tokenize_dataset.py
git add src/decoding/grammar_fsm.py
git add src/utils/eval_windows.py src/utils/benchmarking.py
git add src/training/train_base.py
git add configs/train/train_base_run_full.yaml configs/train/train_base_run_small.yaml
git add tests/test_tokenizer_midi_codec.py tests/test_music_analysis.py tests/test_decoding_grammar_fsm.py
git add docs/todo.md docs/superpowers/plans/2026-05-17-phrase-token.md
```

- [ ] **Step 3：提交**

Run:
```
git commit -m "$(cat <<'EOF'
feat: 引入 PHRASE 边界 token 与训练管线下沉

把"乐句开始"作为显式 token PHRASE 写入数据，删除训练侧基于启发式的
phrase-aware 窗口采样与 phrase-aware FIM，评测窗口升级为 PHRASE 优先、
BAR 兜底。词表偏移，落地后必须从 scratch 重训。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4：再次跑全测确认 commit 后状态干净**

Run: `git status && python -m pytest tests/ -v`
Expected: working tree clean, 全测 PASS

---

## Self-Review

Spec 覆盖：
- §2.1 词表 PHRASE → Task 4
- §2.2 文法 → Task 5
- §2.3 FSM AFTER_PHRASE → Task 10
- §2.4 validate_token_order → Task 5
- §3.1 phrase_analysis boundary-first → Task 1, 2
- §3.2 inject_phrase_tokens → Task 6
- §3.3 tokenize_midi → Task 7
- §3.4 build_vocab → Task 4
- §3.5 tokens_to_midi → Task 8
- §3.6 tokenize_dataset 统计 → Task 9
- §4.1 训练管线删除 → Task 12 step 1-2, 8
- §4.2 PHRASE 不构成 cut 边界 + phrase_event 单元 → Task 12 step 6, 7
- §4.3 采样路径简化 → Task 12 step 3-5
- §4.4 训练配置 → Task 12 step 9
- §5 评测窗口 → Task 11
- §6 测试 → Task 5/6/8/10/11/13
- §9 文档 → Task 14 step 2

Type 一致性：`PhraseBoundary(bar_index, anchor_pos)` 字段在 Task 1 / 2 / 13 中一致；`sample_phrase_aligned_subsequence` 在 Task 11 / 13 一致；`phrase_event` 单元 tuple `(idx, idx+6, "phrase_event", group_id)` 在 Task 12 step 7 内部一致。

无 placeholder。
