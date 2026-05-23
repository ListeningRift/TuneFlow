# 结构性中心优先调性检测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前调性检测改造成“结构性中心优先”的保守发布方案，降低短时离调误报，同时保留对中等长度真实转调的识别能力。

**Architecture:** 保留现有滑窗、逐 key 打分和 HMM 作为前处理层；新增逐帧支持图与结构中心状态机，统一由状态机决定 `KeyFrame.best_key`、`KeySegment` 和 `ModulationPoint`。测试分成公开行为回归和内部状态机规则两组，先锁行为，再集成实现。

**Tech Stack:** Python 3.10、`unittest`、现有 `src/music_analysis/key_analysis.py` 调性分析流水线

---

## 文件结构

- 修改：`src/music_analysis/key_analysis.py:18-586`
  - 新增保守型配置项
  - 新增前处理帧数据结构
  - 新增初始稳定调判定与状态机 helper
  - 让 `analyze_key_timeline()` 改为 `raw frames -> HMM 前处理 -> 状态机 -> segments/modulation`
- 修改：`tests/test_music_analysis.py:111-165`
  - 新增更贴近真实音乐语义的 token 构造器
  - 新增内部 helper 用于拼装状态机输入帧
- 修改：`tests/test_music_analysis.py:1005-1057`
  - 新增公开 API 回归测试
  - 新增内部状态机规则测试

### Task 1: 写公开行为回归测试

**Files:**
- Modify: `tests/test_music_analysis.py:111-165`
- Modify: `tests/test_music_analysis.py:1005-1057`
- Test: `tests/test_music_analysis.py`

- [ ] **Step 1: 写失败测试用的 token 构造器**

```python
def _c_major_with_secondary_dominant_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 62, 12), (8, 66, 8), (16, 69, 12)],
            [(0, 67, 12), (8, 71, 8), (16, 74, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 65, 12), (8, 69, 8), (16, 72, 12)],
        ]
    )


def _c_major_with_modal_mixture_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 12), (8, 63, 8), (16, 67, 12)],
            [(0, 58, 12), (8, 63, 8), (16, 67, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 65, 12), (8, 69, 8), (16, 72, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
        ]
    )


def _c_major_with_brief_dominant_excursion_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 67, 12), (8, 71, 8), (16, 74, 12)],
            [(0, 64, 12), (8, 67, 8), (16, 71, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
        ]
    )


def _c_major_with_confirmed_g_major_modulation_tokens() -> list[str]:
    return _bars_to_tokens(
        [
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 57, 12), (8, 60, 8), (16, 64, 12)],
            [(0, 65, 12), (8, 69, 8), (16, 72, 12)],
            [(0, 60, 12), (8, 64, 8), (16, 67, 12)],
            [(0, 67, 12), (8, 71, 8), (16, 74, 12)],
            [(0, 64, 12), (8, 67, 8), (16, 71, 12)],
            [(0, 62, 12), (8, 66, 8), (16, 69, 12)],
            [(0, 67, 12), (8, 71, 8), (16, 74, 12)],
            [(0, 64, 12), (8, 67, 8), (16, 71, 12)],
            [(0, 67, 12), (8, 71, 8), (16, 74, 12)],
        ]
    )
```

- [ ] **Step 2: 写公开 API 的失败测试**

```python
def test_key_timeline_ignores_secondary_dominant_excursion(self) -> None:
    analysis = analyze_key_timeline(_c_major_with_secondary_dominant_tokens())
    self.assertEqual(analysis.initial_key, "C:maj")
    self.assertEqual([segment.key for segment in analysis.segments], ["C:maj"])
    self.assertEqual(len(analysis.modulation_points), 0)


def test_key_timeline_ignores_modal_mixture(self) -> None:
    analysis = analyze_key_timeline(_c_major_with_modal_mixture_tokens())
    self.assertEqual([segment.key for segment in analysis.segments], ["C:maj"])
    self.assertEqual(len(analysis.modulation_points), 0)


def test_key_timeline_ignores_brief_dominant_excursion(self) -> None:
    analysis = analyze_key_timeline(_c_major_with_brief_dominant_excursion_tokens())
    self.assertEqual([segment.key for segment in analysis.segments], ["C:maj"])
    self.assertEqual(len(analysis.modulation_points), 0)


def test_key_timeline_detects_confirmed_modulation_without_backfill(self) -> None:
    analysis = analyze_key_timeline(_c_major_with_confirmed_g_major_modulation_tokens())
    self.assertEqual([segment.key for segment in analysis.segments], ["C:maj", "G:maj"])
    self.assertEqual(len(analysis.modulation_points), 1)
    point = analysis.modulation_points[0]
    self.assertEqual((point.bar_index, point.pos_in_bar), (analysis.segments[1].start_bar, analysis.segments[1].start_pos))
    self.assertGreaterEqual(point.bar_index, 4)
```

- [ ] **Step 3: 运行公开回归测试，确认它们先失败**

Run: `pytest tests/test_music_analysis.py -k "secondary_dominant or modal_mixture or brief_dominant_excursion or without_backfill" -v`

Expected: FAIL，失败原因是当前实现仍会把短时偏离解释成转调，且还没有“不回填”的发布规则。

- [ ] **Step 4: 提交测试基线**

```bash
git add tests/test_music_analysis.py
git commit -m "test: add structural key stability regressions"
```

### Task 2: 写内部状态机规则测试

**Files:**
- Modify: `tests/test_music_analysis.py:111-165`
- Modify: `tests/test_music_analysis.py:1005-1057`
- Test: `tests/test_music_analysis.py`

- [ ] **Step 1: 在测试文件里增加内部帧构造 helper**

```python
from src.music_analysis.key_analysis import (
    KeyAnalysisConfig,
    _PreprocessedFrame,
    _initialize_stable_key,
    _resolve_structural_frames,
)


def _preprocessed_frame(
    frame_index: int,
    support_map: dict[str, float],
    *,
    raw_key: str,
    hmm_key: str | None = None,
    is_uncertain: bool = False,
) -> _PreprocessedFrame:
    ranked = sorted(support_map.items(), key=lambda item: (-float(item[1]), str(item[0])))
    best_score = float(ranked[0][1])
    second_score = float(ranked[1][1]) if len(ranked) > 1 else float("-inf")
    return _PreprocessedFrame(
        start_unit=frame_index * 16,
        end_unit=(frame_index + 1) * 16,
        raw_key=raw_key,
        hmm_key=raw_key if hmm_key is None else hmm_key,
        best_score=best_score,
        margin_to_second=0.0 if second_score == float("-inf") else max(0.0, best_score - second_score),
        is_uncertain=is_uncertain,
        support_by_key=tuple((key_name, float(score)) for key_name, score in ranked),
    )
```

- [ ] **Step 2: 写初始化、冻结和 challenger 重置规则的失败测试**

```python
def test_initialize_stable_key_skips_uncertain_opening_frames(self) -> None:
    config = KeyAnalysisConfig(initial_stable_window_frames=2)
    frames = (
        _preprocessed_frame(0, {"G:maj": 0.56, "C:maj": 0.55}, raw_key="G:maj", is_uncertain=True),
        _preprocessed_frame(1, {"C:maj": 0.72, "G:maj": 0.30}, raw_key="C:maj"),
        _preprocessed_frame(2, {"C:maj": 0.69, "G:maj": 0.35}, raw_key="C:maj"),
    )
    self.assertEqual(_initialize_stable_key(frames, config), "C:maj")


def test_structural_frames_freeze_on_uncertain_frame(self) -> None:
    config = KeyAnalysisConfig(
        stable_key_min_support=0.30,
        challenger_min_lead=0.12,
        modulation_min_run_frames=3,
        modulation_min_accumulated_lead=0.45,
        modulation_min_newkey_support=0.45,
    )
    frames = (
        _preprocessed_frame(0, {"C:maj": 0.74, "G:maj": 0.18}, raw_key="C:maj"),
        _preprocessed_frame(1, {"G:maj": 0.70, "C:maj": 0.46}, raw_key="G:maj"),
        _preprocessed_frame(2, {"G:maj": 0.52, "C:maj": 0.50}, raw_key="G:maj", is_uncertain=True),
        _preprocessed_frame(3, {"C:maj": 0.73, "G:maj": 0.28}, raw_key="C:maj"),
    )
    resolved = _resolve_structural_frames(frames, config)
    self.assertTrue(all(frame.best_key == "C:maj" for frame in resolved))


def test_structural_frames_reset_when_challenger_changes(self) -> None:
    config = KeyAnalysisConfig(
        stable_key_min_support=0.30,
        challenger_min_lead=0.12,
        modulation_min_run_frames=3,
        modulation_min_accumulated_lead=0.45,
        modulation_min_newkey_support=0.45,
    )
    frames = (
        _preprocessed_frame(0, {"C:maj": 0.75, "G:maj": 0.15}, raw_key="C:maj"),
        _preprocessed_frame(1, {"G:maj": 0.72, "C:maj": 0.44}, raw_key="G:maj"),
        _preprocessed_frame(2, {"D:maj": 0.74, "C:maj": 0.43, "G:maj": 0.42}, raw_key="D:maj"),
        _preprocessed_frame(3, {"D:maj": 0.73, "C:maj": 0.45}, raw_key="D:maj"),
    )
    resolved = _resolve_structural_frames(frames, config)
    self.assertTrue(all(frame.best_key == "C:maj" for frame in resolved))
```

- [ ] **Step 3: 运行内部规则测试，确认它们先失败**

Run: `pytest tests/test_music_analysis.py -k "initialize_stable_key or freeze_on_uncertain or challenger_changes" -v`

Expected: FAIL，失败原因是 `_PreprocessedFrame`、`_initialize_stable_key()`、`_resolve_structural_frames()` 还不存在。

- [ ] **Step 4: 提交内部测试基线**

```bash
git add tests/test_music_analysis.py
git commit -m "test: add structural key state machine rules"
```

### Task 3: 实现前处理支持图和初始稳定调逻辑

**Files:**
- Modify: `src/music_analysis/key_analysis.py:18-444`
- Test: `tests/test_music_analysis.py`

- [ ] **Step 1: 给配置和内部数据结构补齐 V1 需要的字段**

```python
@dataclass(frozen=True)
class KeyAnalysisConfig:
    """加权局部调性分析、HMM 前处理与结构中心状态机配置。"""

    positions_per_bar: int = 32
    window_bars: float = 1.0
    hop_bars: float = 0.5
    bar_start_weight: float = 1.60
    strong_beat_weight: float = 1.30
    weak_beat_weight: float = 1.00
    strong_beat_stride: int = 8
    min_best_score: float = 0.30
    min_score_margin: float = 0.10
    neighborhood_radius_frames: int = 3
    neighborhood_decay: float = 0.65
    modulation_confirmation_frames: int = 2
    global_key_bias: float = 0.18
    key_change_penalty: float = 0.45
    stable_key_min_support: float = 0.30
    challenger_min_lead: float = 0.12
    initial_stable_window_frames: int = 2
    modulation_min_run_frames: int = 4
    modulation_min_accumulated_lead: float = 0.60
    modulation_min_newkey_support: float = 0.45
    stable_key_max_decay_frames: int = 3
    modulation_release_frames: int = 2


@dataclass(frozen=True)
class _PreprocessedFrame:
    start_unit: int
    end_unit: int
    raw_key: str
    hmm_key: str
    best_score: float
    margin_to_second: float
    is_uncertain: bool
    support_by_key: tuple[tuple[str, float], ...]
```

- [ ] **Step 2: 实现支持图前处理 helper 和初始稳定调 helper**

```python
def _build_preprocessed_frames(
    raw_frames: Sequence[_RawFrame],
    *,
    global_scores: dict[str, float],
    config: KeyAnalysisConfig,
) -> tuple[_PreprocessedFrame, ...]:
    if not raw_frames:
        return tuple()

    hmm_path = _decode_hmm_key_path(raw_frames, global_scores=global_scores, config=config)
    radius = max(0, int(config.neighborhood_radius_frames))
    items: list[_PreprocessedFrame] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        support_by_key: dict[str, float] = defaultdict(float)
        for neighbor_index in range(max(0, frame_index - radius), min(len(raw_frames), frame_index + radius + 1)):
            neighbor = raw_frames[neighbor_index]
            decay = float(config.neighborhood_decay) ** abs(neighbor_index - frame_index)
            for key_name, score in neighbor.score_by_key:
                support_by_key[key_name] += max(0.0, float(score)) * decay
        for key_name in _ALL_KEY_NAMES:
            support_by_key[key_name] += float(config.global_key_bias) * max(0.0, float(global_scores.get(key_name, 0.0)))
        ranked = sorted(support_by_key.items(), key=lambda item: (-float(item[1]), str(item[0])))
        items.append(
            _PreprocessedFrame(
                start_unit=raw_frame.start_unit,
                end_unit=raw_frame.end_unit,
                raw_key=raw_frame.raw_key,
                hmm_key=hmm_path[frame_index] if frame_index < len(hmm_path) else raw_frame.raw_key,
                best_score=raw_frame.best_score,
                margin_to_second=raw_frame.margin_to_second,
                is_uncertain=raw_frame.is_uncertain,
                support_by_key=tuple((key_name, float(score)) for key_name, score in ranked),
            )
        )
    return tuple(items)


def _initialize_stable_key(
    frames: Sequence[_PreprocessedFrame],
    config: KeyAnalysisConfig,
) -> str | None:
    start_index = next((index for index, frame in enumerate(frames) if not _is_low_confidence_frame(frame, config)), None)
    if start_index is None:
        return None
    end_index = min(len(frames), start_index + max(1, int(config.initial_stable_window_frames)))
    support_totals: dict[str, float] = defaultdict(float)
    for frame in frames[start_index:end_index]:
        if _is_low_confidence_frame(frame, config):
            continue
        for key_name, score in frame.support_by_key:
            support_totals[str(key_name)] += max(0.0, float(score))
    if not support_totals:
        return None
    return min(support_totals, key=lambda key_name: (-float(support_totals[key_name]), str(key_name)))
```

- [ ] **Step 3: 运行内部初始化测试，确认先变绿**

Run: `pytest tests/test_music_analysis.py -k "initialize_stable_key" -v`

Expected: PASS

- [ ] **Step 4: 提交前处理与初始化实现**

```bash
git add src/music_analysis/key_analysis.py tests/test_music_analysis.py
git commit -m "feat: add structural key preprocessing helpers"
```

### Task 4: 实现结构状态机并接管最终发布

**Files:**
- Modify: `src/music_analysis/key_analysis.py:398-586`
- Test: `tests/test_music_analysis.py`

- [ ] **Step 1: 实现低置信冻结、challenger 重置和最终帧归属**

```python
def _resolve_structural_frames(
    frames: Sequence[_PreprocessedFrame],
    config: KeyAnalysisConfig,
) -> tuple[KeyFrame, ...]:
    if not frames:
        return tuple()

    stable_key = _initialize_stable_key(frames, config)
    if stable_key is None:
        return tuple(
            KeyFrame(
                start_bar=0,
                start_pos=0,
                end_bar=0,
                end_pos=0,
                best_key=_UNCERTAIN_KEY,
                best_score=float(frame.best_score),
                margin_to_second=float(frame.margin_to_second),
                is_uncertain=True,
                raw_key=frame.raw_key,
                smoothed_support=0.0,
            )
            for frame in frames
        )

    challenger_key: str | None = None
    challenger_run_frames = 0
    challenger_accumulated_lead = 0.0
    stable_key_decay_frames = 0
    published_keys: list[str] = []

    for frame in frames:
        if _is_low_confidence_frame(frame, config):
            published_keys.append(stable_key)
            continue

        support_map = dict(frame.support_by_key)
        top_key = str(frame.support_by_key[0][0])
        stable_support = max(0.0, float(support_map.get(stable_key, 0.0)))
        top_support = max(0.0, float(frame.support_by_key[0][1]))
        lead = top_support - stable_support

        if top_key == stable_key or lead < float(config.challenger_min_lead):
            challenger_key = None
            challenger_run_frames = 0
            challenger_accumulated_lead = 0.0
            stable_key_decay_frames = 0
            published_keys.append(stable_key)
            continue

        if challenger_key != top_key:
            challenger_key = top_key
            challenger_run_frames = 1
            challenger_accumulated_lead = lead
            stable_key_decay_frames = 1
        else:
            challenger_run_frames += 1
            challenger_accumulated_lead += lead
            if stable_support < float(config.stable_key_min_support):
                stable_key_decay_frames += 1

        if (
            challenger_run_frames >= int(config.modulation_min_run_frames)
            and challenger_accumulated_lead >= float(config.modulation_min_accumulated_lead)
            and stable_key_decay_frames >= int(config.stable_key_max_decay_frames)
            and top_support >= float(config.modulation_min_newkey_support)
        ):
            stable_key = challenger_key
            challenger_key = None
            challenger_run_frames = 0
            challenger_accumulated_lead = 0.0
            stable_key_decay_frames = 0

        published_keys.append(stable_key)

    return _publish_structural_key_frames(frames, published_keys, config)
```

- [ ] **Step 2: 让 `analyze_key_timeline()` 改用新流水线，并保持 `modulation_point` 不回填**

```python
def analyze_key_timeline(
    tokens: Sequence[str],
    config: KeyAnalysisConfig | None = None,
) -> KeyTimelineAnalysis:
    """分析单条 token 序列的稳定调性时间线与稀疏转调点。"""
    config = KeyAnalysisConfig() if config is None else config
    parsed = _parse_token_events(tokens, config)
    raw_frames = _build_raw_frames(parsed, config)
    global_scores = _score_lookup(_global_ranked_scores(parsed, config))
    preprocessed_frames = _build_preprocessed_frames(raw_frames, global_scores=global_scores, config=config)
    frames = _resolve_structural_frames(preprocessed_frames, config)
    segments = _build_segments(frames, total_units=parsed.total_units, config=config)
    modulation_points = _build_modulation_points(frames, segments, config)
    initial_key = segments[0].key if segments else _UNCERTAIN_KEY
    return KeyTimelineAnalysis(
        frames=frames,
        segments=segments,
        modulation_points=modulation_points,
        initial_key=initial_key,
    )
```

- [ ] **Step 3: 运行本任务相关测试，确认都变绿**

Run: `pytest tests/test_music_analysis.py -k "key_timeline or structural_frames" -v`

Expected: PASS

- [ ] **Step 4: 提交状态机集成**

```bash
git add src/music_analysis/key_analysis.py tests/test_music_analysis.py
git commit -m "feat: publish structural key centers conservatively"
```

### Task 5: 运行完整验证并清理

**Files:**
- Modify: `src/music_analysis/key_analysis.py:18-586`
- Modify: `tests/test_music_analysis.py:111-1057`
- Test: `tests/test_music_analysis.py`

- [ ] **Step 1: 跑聚焦回归，确认没有漏掉关键场景**

Run: `pytest tests/test_music_analysis.py -k "single_major_key or single_minor_key or misleading_bar or ambiguous_sequence or confirmed_modulation or secondary_dominant or modal_mixture or brief_dominant_excursion" -v`

Expected: PASS

- [ ] **Step 2: 跑完整音乐分析测试**

Run: `pytest tests/test_music_analysis.py -v`

Expected: PASS

- [ ] **Step 3: 如有必要，补最小注释与命名清理**

```python
def _is_low_confidence_frame(frame: _PreprocessedFrame, config: KeyAnalysisConfig) -> bool:
    """判断当前前处理帧是否应在状态机中冻结处理。"""


def _publish_structural_key_frames(
    frames: Sequence[_PreprocessedFrame],
    published_keys: Sequence[str],
    config: KeyAnalysisConfig,
) -> tuple[KeyFrame, ...]:
    """把状态机结果映射回对外发布的 KeyFrame。"""
```

- [ ] **Step 4: 提交最终实现**

```bash
git add src/music_analysis/key_analysis.py tests/test_music_analysis.py
git commit -m "refactor: stabilize key analysis around structural centers"
```

## Self-Review

- Spec coverage：`support/lead` 定义、`challenger` 更换即重置、低置信冻结、初始 `stable_key` 初始化、`modulation_point` 不回填、HMM 仅作前处理，都分别落在 Task 2 到 Task 4。
- Placeholder scan：计划中没有 `TODO`、`TBD`、`implement later` 之类占位项；所有测试步骤都给了明确的测试名和命令。
- Type consistency：统一使用 `_PreprocessedFrame`、`_initialize_stable_key()`、`_resolve_structural_frames()` 这组名称；后续步骤不再改名。
