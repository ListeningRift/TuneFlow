# 分层乐句边界重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前乐句划分逻辑升级为“完整单元优先、核心重复修正、发展回应可识别”的三层边界分析流程，并移除首句强制乐句边界。

**Architecture:** 保留现有 `analyze_phrase_candidates()` 作为统一入口，但在 note-level 特征与三层边界评分之间新增“候选结构单元”和“单元关系分类”两层中间语义。最终输出中新增 `analysis_start` 作为分析起点元数据，`boundaries` 仅承载真实结构边界，`inject_phrase_tokens()` 只消费真实 `phrase` 边界。

**Tech Stack:** Python 3.11、标准库 dataclasses/typing、现有 TuneFlow tokenizer、annotation review 工具、pytest/unittest

---

## 文件结构

### 核心实现

- Modify: `src/music_analysis/phrase_analysis.py`
  负责分析起点、候选结构单元、单元关系分类、三层边界评分、后处理与最终边界输出。

### 对外导出

- Modify: `src/music_analysis/__init__.py`
  暴露 `AnalysisAnchor` 等新增结构。

### 训练兼容接入

- Modify: `src/tokenizer/midi_codec.py`
  让 `inject_phrase_tokens()` 只消费真实 `phrase` 边界，不再自动写入起始位置。

### 评审输出

- Modify: `src/utils/annotation_review.py`
  序列化 `analysis_start`、单元完整性、单元关系等新字段，并区分分析起点与真实结构边界。

- Modify: `tools/annotation_review_viewer.js`
  展示 `analysis_start`、单元关系与新的中文原因标签。

### 测试

- Modify: `tests/test_music_analysis.py`
  覆盖起点语义、完整单元、核心重复修正、发展回应、非对应局部相似、首句无强证据等行为。

- Modify: `tests/test_annotation_review.py`
  覆盖新字段序列化、viewer 依赖字段与中文来源标签。

---

### Task 1: 先把分析起点与真实结构边界彻底分离

**Files:**
- Modify: `src/music_analysis/phrase_analysis.py`
- Modify: `src/music_analysis/__init__.py`
- Modify: `tests/test_music_analysis.py`

- [ ] **Step 1: 写失败测试，约束 `PhraseAnalysis` 必须暴露独立的 `analysis_start`**

在 `tests/test_music_analysis.py` 新增：

```python
def test_phrase_analysis_exposes_analysis_start_separately(self) -> None:
    analysis = analyze_phrase_candidates(_phrase_source_tokens(), config=PhraseAnalysisConfig())
    self.assertIsNotNone(analysis.analysis_start)
    self.assertEqual(analysis.analysis_start.bar_index, 0)
    self.assertEqual(analysis.analysis_start.anchor_pos, 0)
```

```python
def test_analysis_start_is_not_forced_into_boundaries(self) -> None:
    analysis = analyze_phrase_candidates(_phrase_source_tokens(), config=PhraseAnalysisConfig())
    if analysis.boundaries:
        self.assertNotEqual(
            (analysis.boundaries[0].bar_index, analysis.boundaries[0].anchor_pos),
            (analysis.analysis_start.bar_index, analysis.analysis_start.anchor_pos),
        )
```

- [ ] **Step 2: 运行测试确认当前实现失败**

Run: `python -m pytest tests/test_music_analysis.py -k "analysis_start" -v`
Expected: FAIL，提示 `PhraseAnalysis` 缺少 `analysis_start`，或起点仍被强制写入 `boundaries`

- [ ] **Step 3: 新增 `AnalysisAnchor` 数据结构并扩展 `PhraseAnalysis`**

在 `src/music_analysis/phrase_analysis.py` 新增：

```python
@dataclass(frozen=True)
class AnalysisAnchor:
    """表示分析起点，仅用于说明分析从哪里开始，不表示真实结构边界。"""

    bar_index: int
    anchor_pos: int
```

并把 `PhraseAnalysis` 改为：

```python
@dataclass(frozen=True)
class PhraseAnalysis:
    """单条 token 序列的乐句分析结果。"""

    bars: tuple[BarInfo, ...]
    notes: tuple[NoteInfo, ...]
    boundary_features: tuple[BoundaryFeature, ...]
    boundary_scores: tuple[HierarchicalBoundaryScore, ...]
    boundaries: tuple[PhraseBoundary, ...]
    phrase_spans: tuple[PhraseSpan, ...]
    analysis_start: AnalysisAnchor | None
```

- [ ] **Step 4: 实现 `_resolve_analysis_start()`，从首个有内容位置生成分析起点**

在 `src/music_analysis/phrase_analysis.py` 新增：

```python
def _resolve_analysis_start(
    notes: Sequence[NoteInfo],
) -> AnalysisAnchor | None:
    """解析分析起点，仅表示分析起始位置，不表示真实结构边界。"""

    if not notes:
        return None
    first = notes[0]
    return AnalysisAnchor(
        bar_index=first.bar_index,
        anchor_pos=first.pos_in_bar,
    )
```

- [ ] **Step 5: 改写最终边界装配逻辑，不再强制把起点写入 `boundaries`**

把 `analyze_phrase_candidates()` 中的返回路径调整为：

```python
analysis_start = _resolve_analysis_start(notes)
boundaries = _assemble_phrase_boundaries_from_scores(notes, boundary_scores, config)
phrase_spans = _derive_phrase_spans(bars, boundaries, analysis_start)
```

并把 `_assemble_phrase_boundaries_from_scores()` 的起始强制边界移除，改为只从真实评分结果收集：

```python
def _assemble_phrase_boundaries_from_scores(
    notes: Sequence[NoteInfo],
    scores: Sequence[HierarchicalBoundaryScore],
    config: PhraseAnalysisConfig,
) -> tuple[PhraseBoundary, ...]:
    """从最终评分中提取真实结构边界，不再注入分析起点。"""

    del notes
    del config
    boundary_map: dict[tuple[int, int], PhraseBoundary] = {}
    for score in scores:
        if score.boundary_type != "phrase":
            continue
        key = (score.bar_index, score.anchor_pos)
        boundary_map[key] = PhraseBoundary(
            bar_index=score.bar_index,
            anchor_pos=score.anchor_pos,
        )
    return tuple(sorted(boundary_map.values(), key=lambda item: (item.bar_index, item.anchor_pos)))
```

- [ ] **Step 6: 更新 `src/music_analysis/__init__.py` 导出新增结构**

把导出补成：

```python
from .phrase_analysis import (
    AnalysisAnchor,
    BarInfo,
    BoundaryFeature,
    BoundaryScore,
    HierarchicalBoundaryScore,
    NoteInfo,
    PhraseAnalysis,
    PhraseAnalysisConfig,
    PhraseBoundary,
    PhraseSpan,
    analyze_phrase_candidates,
)
```

- [ ] **Step 7: 重新运行测试确认起点语义骨架通过**

Run: `python -m pytest tests/test_music_analysis.py -k "analysis_start" -v`
Expected: PASS

---

### Task 2: 用测试驱动实现“首句无强证据时允许没有正式边界”

**Files:**
- Modify: `src/music_analysis/phrase_analysis.py`
- Modify: `tests/test_music_analysis.py`

- [ ] **Step 1: 写失败测试，约束没有强证据时不能再补首句边界**

在 `tests/test_music_analysis.py` 新增：

```python
def _no_clear_first_phrase_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 4), (4, 62, 4), (8, 64, 4), (12, 65, 4)],
            [(0, 67, 4), (4, 69, 4), (8, 71, 4), (12, 72, 4)],
            [(0, 74, 4), (4, 76, 4), (8, 77, 4), (12, 79, 4)],
        ]
    )
```

```python
def test_no_forced_first_phrase_boundary_when_no_strong_evidence(self) -> None:
    analysis = analyze_phrase_candidates(_no_clear_first_phrase_tokens(), config=PhraseAnalysisConfig())
    self.assertEqual(tuple(analysis.boundaries), tuple())
```

- [ ] **Step 2: 运行测试确认当前实现失败**

Run: `python -m pytest tests/test_music_analysis.py -k "no_forced_first_phrase_boundary" -v`
Expected: FAIL，当前实现仍会给出首句起点或过早边界

- [ ] **Step 3: 调整 `_derive_phrase_spans()`，允许只有分析起点而没有真实边界**

把 `_derive_phrase_spans()` 改为：

```python
def _derive_phrase_spans(
    bars: Sequence[BarInfo],
    boundaries: Sequence[PhraseBoundary],
    analysis_start: AnalysisAnchor | None,
) -> tuple[PhraseSpan, ...]:
    """根据分析起点与真实结构边界推导乐句跨度。"""

    if not bars or analysis_start is None:
        return tuple()
    sorted_boundaries = sorted(boundaries, key=lambda item: (item.bar_index, item.anchor_pos))
    cut_bars = [analysis_start.bar_index, *(item.bar_index for item in sorted_boundaries)]
    unique_cut_bars: list[int] = []
    for bar_index in cut_bars:
        if not unique_cut_bars or unique_cut_bars[-1] != bar_index:
            unique_cut_bars.append(bar_index)
    if unique_cut_bars[-1] != len(bars):
        unique_cut_bars.append(len(bars))
    spans: list[PhraseSpan] = []
    for start_bar, end_bar in zip(unique_cut_bars, unique_cut_bars[1:], strict=True):
        if end_bar <= start_bar:
            continue
        spans.append(
            PhraseSpan(
                start_bar=start_bar,
                end_bar=end_bar,
                start_token=bars[start_bar].start_token,
                end_token=bars[end_bar - 1].end_token,
                tempo_token=bars[start_bar].effective_tempo_token,
                key_token=bars[start_bar].effective_key_token,
                tokens=tuple(),
                source_kind="single_phrase",
            )
        )
    return tuple(spans)
```

- [ ] **Step 4: 调整起点相关断言，确保 `phrase_spans` 不依赖起点边界**

在 `tests/test_music_analysis.py` 把依赖“第一个 boundary 是起点”的断言改成：

```python
def test_phrase_spans_can_be_derived_from_analysis_start_without_boundaries(self) -> None:
    analysis = analyze_phrase_candidates(_no_clear_first_phrase_tokens(), config=PhraseAnalysisConfig())
    self.assertIsNotNone(analysis.analysis_start)
    self.assertEqual(len(analysis.boundaries), 0)
    self.assertEqual(len(analysis.phrase_spans), 1)
    self.assertEqual(analysis.phrase_spans[0].start_bar, analysis.analysis_start.bar_index)
```

- [ ] **Step 5: 运行测试确认无强证据时不再强制边界**

Run: `python -m pytest tests/test_music_analysis.py -k "no_forced_first_phrase_boundary or phrase_spans_can_be_derived_from_analysis_start_without_boundaries" -v`
Expected: PASS

---

### Task 3: 建立候选结构单元与单元完整性评分骨架

**Files:**
- Modify: `src/music_analysis/phrase_analysis.py`
- Modify: `tests/test_music_analysis.py`

- [ ] **Step 1: 写失败测试，约束 `BoundaryFeature` 必须携带单元层语义**

在 `tests/test_music_analysis.py` 新增：

```python
def test_boundary_feature_exposes_unit_level_fields(self) -> None:
    analysis = analyze_phrase_candidates(_phrase_source_tokens(), config=PhraseAnalysisConfig())
    first = analysis.boundary_features[0]
    self.assertTrue(hasattr(first, "unit_completeness_score"))
    self.assertTrue(hasattr(first, "unit_relation_type"))
    self.assertTrue(hasattr(first, "unit_relation_score"))
    self.assertTrue(hasattr(first, "core_overlap_ratio"))
    self.assertTrue(hasattr(first, "developmental_similarity_score"))
```

- [ ] **Step 2: 运行测试确认当前实现失败**

Run: `python -m pytest tests/test_music_analysis.py -k "unit_level_fields" -v`
Expected: FAIL，当前 `BoundaryFeature` 尚未包含这些字段

- [ ] **Step 3: 扩展 `BoundaryFeature` 数据结构**

在 `src/music_analysis/phrase_analysis.py` 中把 `BoundaryFeature` 扩展为：

```python
@dataclass(frozen=True)
class BoundaryFeature:
    """相邻音符之间的边界特征，包含局部特征和结构单元语义。"""

    note_index: int
    left_end_unit: int
    right_start_unit: int
    bar_index: int
    anchor_pos: int
    gap: int
    local_gap_mean: float
    local_duration_mean: float
    gap_break_score: float
    duration_release_score: float
    cadence_score: float
    motive_end_score: float
    repeat_start_score: float
    repeat_end_score: float
    sequence_stop_score: float
    continuity_penalty: float
    bar_hint_score: float
    sequence_role: str
    unit_completeness_score: float
    unit_relation_type: str
    unit_relation_score: float
    core_overlap_ratio: float
    developmental_similarity_score: float
    reasons: tuple[str, ...]
```

- [ ] **Step 4: 新增候选结构单元辅助结构与生成函数**

在 `src/music_analysis/phrase_analysis.py` 新增：

```python
@dataclass(frozen=True)
class StructuralUnit:
    """表示用于结构判断的候选完整单元。"""

    start_note_index: int
    end_note_index: int
    start_unit: int
    end_unit: int
    note_count: int
    start_bar_index: int
    end_bar_index: int
```

```python
def _build_structural_units(
    notes: Sequence[NoteInfo],
    boundary_features: Sequence[BoundaryFeature],
) -> tuple[StructuralUnit, ...]:
    """根据弱边界信号组织候选结构单元。"""

    del boundary_features
    if not notes:
        return tuple()
    return tuple(
        StructuralUnit(
            start_note_index=index,
            end_note_index=min(len(notes) - 1, index + 3),
            start_unit=notes[index].start_unit,
            end_unit=notes[min(len(notes) - 1, index + 3)].end_unit,
            note_count=(min(len(notes) - 1, index + 3) - index) + 1,
            start_bar_index=notes[index].bar_index,
            end_bar_index=notes[min(len(notes) - 1, index + 3)].bar_index,
        )
        for index in range(max(0, len(notes) - 1))
    )
```

- [ ] **Step 5: 新增单元完整性评分函数，并先给出最小可运行实现**

在 `src/music_analysis/phrase_analysis.py` 新增：

```python
def _score_unit_completeness(
    notes: Sequence[NoteInfo],
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> float:
    """估计边界前后两个单元的整体完整性。"""

    del notes
    left_span = max(1, left_end - left_start + 1)
    right_span = max(1, right_end - right_start + 1)
    shorter = min(left_span, right_span)
    if shorter >= 4:
        return 0.75
    if shorter == 3:
        return 0.60
    return 0.35
```

- [ ] **Step 6: 在 `_build_default_boundary_features()` 中补齐新增字段默认值**

先把构造逻辑补成：

```python
unit_completeness_score=0.0,
unit_relation_type="none",
unit_relation_score=0.0,
core_overlap_ratio=0.0,
developmental_similarity_score=0.0,
```

- [ ] **Step 7: 运行测试确认骨架通过**

Run: `python -m pytest tests/test_music_analysis.py -k "unit_level_fields" -v`
Expected: PASS

---

### Task 4: 实现“完整单元优先”的单元关系分类

**Files:**
- Modify: `src/music_analysis/phrase_analysis.py`
- Modify: `tests/test_music_analysis.py`

- [ ] **Step 1: 写失败测试，覆盖严格重复、核心重复、发展回应和仅局部相似**

在 `tests/test_music_analysis.py` 新增：

```python
def _core_repeat_with_small_non_core_edges_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 58, 2), (2, 60, 4), (6, 62, 4), (10, 64, 4), (14, 65, 2)],
            [(0, 72, 2), (2, 60, 4), (6, 62, 4), (10, 64, 4), (14, 74, 2)],
        ]
    )
```

```python
def _developmental_response_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 4), (4, 62, 4), (8, 64, 4), (12, 67, 4)],
            [(0, 60, 4), (4, 62, 4), (8, 65, 4), (12, 69, 4)],
        ]
    )
```

```python
def test_core_repeat_allows_small_non_core_edges(self) -> None:
    analysis = analyze_phrase_candidates(_core_repeat_with_small_non_core_edges_tokens(), config=PhraseAnalysisConfig())
    self.assertTrue(any(feature.unit_relation_type == "core_repeat" for feature in analysis.boundary_features))
    self.assertTrue(any(feature.core_overlap_ratio >= 0.60 for feature in analysis.boundary_features))
```

```python
def test_developmental_response_is_not_downgraded_to_local_similarity_only(self) -> None:
    analysis = analyze_phrase_candidates(_developmental_response_tokens(), config=PhraseAnalysisConfig())
    self.assertTrue(any(feature.unit_relation_type == "developmental_response" for feature in analysis.boundary_features))
    self.assertFalse(all(feature.unit_relation_type == "local_similarity_only" for feature in analysis.boundary_features))
```

```python
def test_same_contour_but_not_repeat_is_local_similarity_only(self) -> None:
    analysis = analyze_phrase_candidates(_same_contour_but_not_repeat_tokens(), config=PhraseAnalysisConfig())
    self.assertTrue(any(feature.unit_relation_type in {"local_similarity_only", "none"} for feature in analysis.boundary_features))
    self.assertFalse(any(feature.unit_relation_type in {"exact_repeat", "core_repeat"} for feature in analysis.boundary_features))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_music_analysis.py -k "core_repeat_allows_small_non_core_edges or developmental_response or local_similarity_only" -v`
Expected: FAIL，当前还没有单元关系分类

- [ ] **Step 3: 新增单元关系分类辅助函数**

在 `src/music_analysis/phrase_analysis.py` 新增：

```python
def _find_core_overlap_ratio(
    left: Sequence[NoteInfo],
    right: Sequence[NoteInfo],
) -> float:
    """估计两个完整单元在裁掉少量首尾非核心成分后的核心重叠比例。"""

    if len(left) < 3 or len(right) < 3:
        return 0.0
    best_ratio = 0.0
    for left_trim_start in range(0, 2):
        for left_trim_end in range(0, 2):
            core_left = left[left_trim_start : len(left) - left_trim_end]
            if len(core_left) < 3:
                continue
            for right_trim_start in range(0, 2):
                for right_trim_end in range(0, 2):
                    core_right = right[right_trim_start : len(right) - right_trim_end]
                    if len(core_right) != len(core_left) or len(core_right) < 3:
                        continue
                    if _is_structurally_similar_fragment(core_left, core_right):
                        ratio = float(len(core_left) / max(len(left), len(right)))
                        best_ratio = max(best_ratio, ratio)
    return best_ratio
```

```python
def _developmental_similarity_score(
    left: Sequence[NoteInfo],
    right: Sequence[NoteInfo],
) -> float:
    """估计两个单元是否共享起始动机并在后续形成发展关系。"""

    if len(left) < 3 or len(right) < 3:
        return 0.0
    start_span = min(3, len(left), len(right))
    start_similarity = _fragment_similarity_score(left[:start_span], right[:start_span])
    contour_similarity = _contour_similarity(left, right)
    pitch_shape_similarity = _pitch_shape_similarity(left, right, tolerance=3)
    return float(
        (0.45 * start_similarity)
        + (0.30 * contour_similarity)
        + (0.25 * pitch_shape_similarity)
    )
```

```python
def _classify_unit_relation(
    left: Sequence[NoteInfo],
    right: Sequence[NoteInfo],
) -> tuple[str, float, float, float]:
    """把两个候选完整单元分类为严格重复、核心重复、发展回应或局部相似。"""

    exact_score = _fragment_similarity_score(left, right)
    if exact_score >= 0.85:
        return "exact_repeat", exact_score, 1.0, 0.0
    core_overlap_ratio = _find_core_overlap_ratio(left, right)
    if core_overlap_ratio >= 0.60:
        return "core_repeat", 0.75, core_overlap_ratio, 0.0
    developmental_score = _developmental_similarity_score(left, right)
    if developmental_score >= 0.62:
        return "developmental_response", developmental_score, 0.0, developmental_score
    if developmental_score >= 0.35 or exact_score >= 0.35:
        return "local_similarity_only", max(exact_score, developmental_score), 0.0, developmental_score
    return "none", 0.0, 0.0, 0.0
```

- [ ] **Step 4: 在边界特征构造里注入单元关系结果**

把 `_build_default_boundary_features()` 改造成：

```python
def _build_default_boundary_features(
    notes: Sequence[NoteInfo],
    config: PhraseAnalysisConfig,
) -> tuple[BoundaryFeature, ...]:
    """构造 note-level 边界特征，并补充完整单元与单元关系语义。"""

    units = _build_structural_units(notes, tuple())
    ...
    left_unit = units[max(0, idx - 1)] if units else None
    right_unit = units[min(len(units) - 1, idx)] if units else None
    relation_type = "none"
    relation_score = 0.0
    core_overlap_ratio = 0.0
    developmental_similarity_score = 0.0
    unit_completeness_score = 0.0
    if left_unit is not None and right_unit is not None:
        left_notes = notes[left_unit.start_note_index : left_unit.end_note_index + 1]
        right_notes = notes[right_unit.start_note_index : right_unit.end_note_index + 1]
        (
            relation_type,
            relation_score,
            core_overlap_ratio,
            developmental_similarity_score,
        ) = _classify_unit_relation(left_notes, right_notes)
        unit_completeness_score = _score_unit_completeness(
            notes,
            left_unit.start_note_index,
            left_unit.end_note_index,
            right_unit.start_note_index,
            right_unit.end_note_index,
        )
```

- [ ] **Step 5: 运行测试确认关系分类通过**

Run: `python -m pytest tests/test_music_analysis.py -k "core_repeat_allows_small_non_core_edges or developmental_response or local_similarity_only" -v`
Expected: PASS

---

### Task 5: 把完整单元与关系分类整合进三层边界评分

**Files:**
- Modify: `src/music_analysis/phrase_analysis.py`
- Modify: `tests/test_music_analysis.py`

- [ ] **Step 1: 写失败测试，约束正式乐句边界不能只靠局部相似抬升**

在 `tests/test_music_analysis.py` 新增：

```python
def test_local_similarity_only_cannot_be_promoted_to_phrase(self) -> None:
    analysis = analyze_phrase_candidates(_same_contour_but_not_repeat_tokens(), config=PhraseAnalysisConfig())
    flagged = [
        score
        for score, feature in zip(analysis.boundary_scores, analysis.boundary_features, strict=True)
        if feature.unit_relation_type == "local_similarity_only"
    ]
    self.assertTrue(flagged)
    self.assertTrue(all(score.boundary_type != "phrase" for score in flagged))
```

```python
def test_core_repeat_or_developmental_response_can_raise_subphrase_or_phrase(self) -> None:
    analysis = analyze_phrase_candidates(_developmental_response_tokens(), config=PhraseAnalysisConfig())
    self.assertTrue(
        any(
            feature.unit_relation_type in {"core_repeat", "developmental_response"}
            and score.boundary_type in {"subphrase", "phrase"}
            for feature, score in zip(analysis.boundary_features, analysis.boundary_scores, strict=True)
        )
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_music_analysis.py -k "local_similarity_only_cannot_be_promoted_to_phrase or core_repeat_or_developmental_response_can_raise_subphrase_or_phrase" -v`
Expected: FAIL，当前评分还未吸收单元关系

- [ ] **Step 3: 在 `_score_boundary_features()` 中引入单元级加权项**

把评分逻辑更新为：

```python
relation_phrase_bonus = 0.0
relation_subphrase_bonus = 0.0
if feature.unit_relation_type == "exact_repeat":
    relation_subphrase_bonus = 0.24 * feature.unit_relation_score
    relation_phrase_bonus = 0.20 * feature.unit_relation_score
elif feature.unit_relation_type == "core_repeat":
    relation_subphrase_bonus = 0.22 * feature.unit_relation_score
    relation_phrase_bonus = 0.18 * feature.core_overlap_ratio
elif feature.unit_relation_type == "developmental_response":
    relation_subphrase_bonus = 0.18 * feature.unit_relation_score
    relation_phrase_bonus = 0.14 * feature.developmental_similarity_score
elif feature.unit_relation_type == "local_similarity_only":
    relation_subphrase_bonus = 0.04 * feature.unit_relation_score
    relation_phrase_bonus = -0.10
```

并把三层分数更新为：

```python
motif_score = (
    ...
    + 0.08 * feature.unit_relation_score
    - 0.14 * (1.0 if feature.unit_relation_type == "local_similarity_only" else 0.0)
)
```

```python
subphrase_score = (
    ...
    + 0.22 * feature.unit_completeness_score
    + relation_subphrase_bonus
)
```

```python
phrase_score = (
    ...
    + 0.26 * feature.unit_completeness_score
    + relation_phrase_bonus
    - 0.18 * (1.0 if feature.unit_relation_type == "local_similarity_only" else 0.0)
)
```

- [ ] **Step 4: 为 `phrase` 判定新增结构关系前置条件**

把分类条件改为：

```python
supports_phrase_relation = feature.unit_relation_type in {
    "exact_repeat",
    "core_repeat",
    "developmental_response",
}
if (
    phrase_score >= config.phrase_threshold
    and feature.sequence_role != "sequence_inside"
    and feature.unit_completeness_score >= 0.55
    and supports_phrase_relation
):
    boundary_type = "phrase"
elif subphrase_score >= config.subphrase_threshold:
    boundary_type = "subphrase"
elif motif_score >= config.motif_threshold:
    boundary_type = "motif"
```

- [ ] **Step 5: 运行测试确认评分整合生效**

Run: `python -m pytest tests/test_music_analysis.py -k "local_similarity_only_cannot_be_promoted_to_phrase or core_repeat_or_developmental_response_can_raise_subphrase_or_phrase" -v`
Expected: PASS

---

### Task 6: 让 `inject_phrase_tokens()` 只消费真实结构边界

**Files:**
- Modify: `src/tokenizer/midi_codec.py`
- Modify: `tests/test_music_analysis.py`
- Modify: `tests/test_tokenizer_midi_codec.py`

- [ ] **Step 1: 写失败测试，约束起点不再自动注入 `PHRASE`**

在 `tests/test_music_analysis.py` 新增：

```python
def test_phrase_token_injection_ignores_analysis_start(self) -> None:
    from src.tokenizer.midi_codec import inject_phrase_tokens

    with_key = inject_key_tokens(_no_clear_first_phrase_tokens())
    with_phrase = inject_phrase_tokens(with_key)
    self.assertNotIn("PHRASE", with_phrase)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_music_analysis.py -k "ignores_analysis_start" -v`
Expected: FAIL，当前逻辑仍会因为起点写入 `PHRASE`

- [ ] **Step 3: 调整 `inject_phrase_tokens()`，仅使用真实 `phrase` 边界**

在 `src/tokenizer/midi_codec.py` 中把相关逻辑改成：

```python
analysis = analyze_phrase_candidates(tokens, config=phrase_config)
boundary_positions = {
    (boundary.bar_index, boundary.anchor_pos)
    for boundary in analysis.boundaries
}
```

删除任何基于“首个有内容小节”或“起点强制边界”的补偿插入逻辑。

- [ ] **Step 4: 补充 tokenizer 回归测试**

在 `tests/test_tokenizer_midi_codec.py` 新增：

```python
def test_inject_phrase_tokens_does_not_insert_start_boundary_without_real_phrase() -> None:
    tokens = inject_key_tokens(_no_clear_first_phrase_tokens())
    with_phrase = inject_phrase_tokens(tokens)
    assert "PHRASE" not in with_phrase
```

- [ ] **Step 5: 运行测试确认训练注入兼容更新完成**

Run: `python -m pytest tests/test_music_analysis.py tests/test_tokenizer_midi_codec.py -k "analysis_start or phrase_tokens" -v`
Expected: PASS

---

### Task 7: 扩展 review 序列化与前端展示

**Files:**
- Modify: `src/utils/annotation_review.py`
- Modify: `tools/annotation_review_viewer.js`
- Modify: `tests/test_annotation_review.py`

- [ ] **Step 1: 写失败测试，约束 review 输出必须包含分析起点与单元关系**

在 `tests/test_annotation_review.py` 中补充：

```python
self.assertIn("analysis_start", case["phrase_analysis"])
self.assertIn("unit_completeness_score", case["phrase_analysis"]["boundary_features"][0])
self.assertIn("unit_relation_type", case["phrase_analysis"]["boundary_features"][0])
self.assertIn("unit_relation_score", case["phrase_analysis"]["boundary_features"][0])
self.assertIn("core_overlap_ratio", case["phrase_analysis"]["boundary_features"][0])
self.assertIn("developmental_similarity_score", case["phrase_analysis"]["boundary_features"][0])
```

并补充静态检查：

```python
self.assertIn("analysis_start", js_text)
self.assertIn("unit_relation_type", js_text)
self.assertIn("unit_completeness_score", js_text)
self.assertIn("core_overlap_ratio", js_text)
self.assertIn("developmental_similarity_score", js_text)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_annotation_review.py -k "phrase_analysis or fixed_viewer_files_exist" -v`
Expected: FAIL，当前 review 还没有这些字段

- [ ] **Step 3: 扩展 `serialize_phrase_analysis()`**

在 `src/utils/annotation_review.py` 中补充：

```python
analysis_start = (
    {
        "bar_index": int(analysis.analysis_start.bar_index),
        "anchor_pos": int(analysis.analysis_start.anchor_pos),
        "source_rule": "analysis_start",
        "source_label": "分析起点",
        "source_reasons": ["分析起点"],
    }
    if analysis.analysis_start is not None
    else None
)
```

并在 `boundary_features` 序列化中补齐：

```python
"unit_completeness_score": float(item.unit_completeness_score),
"unit_relation_type": str(item.unit_relation_type),
"unit_relation_score": float(item.unit_relation_score),
"core_overlap_ratio": float(item.core_overlap_ratio),
"developmental_similarity_score": float(item.developmental_similarity_score),
```

最后把返回体改成：

```python
return {
    "bars": bars,
    "notes": notes,
    "analysis_start": analysis_start,
    "boundary_features": feature_rows,
    "boundary_scores": boundary_scores,
    "boundaries": boundaries,
    "phrase_spans": phrase_spans,
    "mean_phrase_bars": float(mean_phrase_bars),
}
```

- [ ] **Step 4: 更新 viewer，区分分析起点与真实结构边界**

在 `tools/annotation_review_viewer.js` 中新增：

```javascript
const analysisStart = detail.phrase_analysis?.analysis_start || null;
```

并把边界说明扩展为：

```javascript
const analysisStartHtml = analysisStart ? `
  <div class="boundary-label boundary-label--analysis-start">
    <strong>分析起点</strong>
    <code>bar=${escapeHtml(analysisStart.bar_index)}</code>
    <code>pos=${escapeHtml(analysisStart.anchor_pos)}</code>
  </div>
` : "";
```

在边界表格中增加：

```javascript
<th>完整性</th>
<th>关系类型</th>
<th>关系强度</th>
<th>核心重叠</th>
<th>发展相似度</th>
```

并在行渲染中增加：

```javascript
<td>${Number(feature.unit_completeness_score || 0).toFixed(3)}</td>
<td><code>${escapeHtml(feature.unit_relation_type || "-")}</code></td>
<td>${Number(feature.unit_relation_score || 0).toFixed(3)}</td>
<td>${Number(feature.core_overlap_ratio || 0).toFixed(3)}</td>
<td>${Number(feature.developmental_similarity_score || 0).toFixed(3)}</td>
```

- [ ] **Step 5: 运行测试确认 review 输出兼容**

Run: `python -m pytest tests/test_annotation_review.py -v`
Expected: PASS

---

### Task 8: 全量回归验证

**Files:**
- Modify: 无
- Verify: `src/music_analysis/phrase_analysis.py`, `src/music_analysis/__init__.py`, `src/tokenizer/midi_codec.py`, `src/utils/annotation_review.py`, `tools/annotation_review_viewer.js`, `tests/*`

- [ ] **Step 1: 跑乐句分析相关测试**

Run: `python -m pytest tests/test_music_analysis.py -v`
Expected: PASS

- [ ] **Step 2: 跑 review 与 tokenizer 相关测试**

Run: `python -m pytest tests/test_annotation_review.py tests/test_tokenizer_midi_codec.py -v`
Expected: PASS

- [ ] **Step 3: 跑全量测试**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: 做一次 import 冒烟**

Run: `python -c "from src.music_analysis import AnalysisAnchor, analyze_phrase_candidates; from src.utils.annotation_review import serialize_phrase_analysis; print('ok')"`
Expected: 输出 `ok`

---

## Self-Review

### Spec 覆盖

- 分析起点与真实边界分离：Task 1、Task 2、Task 6、Task 7
- 首句无强证据时允许无正式边界：Task 2
- 完整单元优先的候选结构单元：Task 3
- 核心重复修正：Task 4、Task 5
- 发展回应关系：Task 4、Task 5
- 局部相似不能误升格：Task 4、Task 5
- `PHRASE` 注入只消费真实边界：Task 6
- review 输出新字段与中文标签：Task 7

### Placeholder 扫描

- 本文档不包含未定字段、回填提示、延后实现说明或跨任务引用式空泛描述。

### 类型一致性

- `AnalysisAnchor` 在实现、导出、序列化、测试中统一命名。
- `BoundaryFeature` 的新增字段在实现、序列化、viewer 与测试中保持一致：
  - `unit_completeness_score`
  - `unit_relation_type`
  - `unit_relation_score`
  - `core_overlap_ratio`
  - `developmental_similarity_score`
- `analysis.boundaries` 在所有任务中都只表示真实结构边界，不再混入分析起点。
