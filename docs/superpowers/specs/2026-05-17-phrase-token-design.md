# PHRASE Token 引入设计

- 日期：2026-05-17
- 状态：待用户审阅
- 关联 TODO：`docs/todo.md` 第 1 项「Phrase 结构显式建模」

## 1. 背景与目标

当前 TuneFlow 的「乐句」能力完全是运行期启发式：`src/music_analysis/phrase_analysis.py` 在 token 流上跑 bar 级 boundary scoring，训练侧 `src/training/train_base.py` 用启发式结果做 `single_phrase / cross_boundary / long_context` 三类窗口采样以及 phrase-aware FIM hole 选取。乐句结构没有进入 token 体系，模型也没有显式监督。

本设计的目标：

- 把"乐句开始"作为一类显式 token（`PHRASE`）写入数据，让模型可以直接监督到。
- 拿掉训练侧基于启发式的乐句感知采样、phrase-aware FIM；启发式仅保留为编码时注入 PHRASE 的依据。
- 评测窗口切分升级为 PHRASE 优先、BAR 兜底。
- 不动评测代理指标 `phrase_coherence_score`、不动 KEY / 移调增强 / velocity / tempo 任何已有机制。

非目标（留给后续 PR）：

- 句类型 / 句关系 / 句尾收束 / 呼吸点 / 重音 / 动机角色等更细标签
- 句级辅助任务头
- 基于 PHRASE token 的训练采样重加权

## 2. Token 体系与文法

### 2.1 新增 token

新增单一无参 token `PHRASE`，语义"新乐句开始"。词表里加在 `BAR` 后、`POS_*` 前。引入后 token id 全表会发生偏移，**必须重新生成 vocab / `.tok` / `.bin` / `.idx`，并重新训练**。

### 2.2 合法序列文法

```
seq         = "BOS" [TEMPO] [KEY] body "EOS"
body        = bar*
bar         = "BAR" [TEMPO] [KEY] [PHRASE] event_block
event_block = ( event | "PHRASE" event )*
event       = POS INST PITCH DUR VEL
```

约束：

- 句首位置：`BAR [TEMPO] [KEY] PHRASE POS ...`（PHRASE 在 BAR 头部 TEMPO/KEY 之后、首个 POS 之前）。
- 句中位置：`... VEL PHRASE POS ...`（PHRASE 紧跟下一个完整 event 之前）。
- 不允许两个 PHRASE 相邻：FSM（§2.3）规定 PHRASE 后必须出现 POS；编码侧再做一道幂等去重兜底。
- 首句强制：含 event 的序列在「首个含 event 的 bar」位置一定插入 PHRASE。
- PHRASE 不允许出现在 `BOS` 直后、`EOS` 直前、无 event 的 BAR 头部之后立刻接 BAR/EOS 的位置。

注：上面 BNF 是描述性表达，允许 `[PHRASE]` 后紧接空 `event_block`。实际由 §2.3 的 FSM 进一步收紧：PHRASE 之后必须紧跟 event，不可直接转移到 BAR 或 EOS。FSM 与 `validate_token_order` 一并对外承担合法性边界。

### 2.3 FSM 改动（`src/decoding/grammar_fsm.py`）

新增状态 `AFTER_PHRASE`。新增转移：

- `AFTER_BAR / AFTER_BAR_TEMPO / AFTER_BAR_KEY / AFTER_BAR_TEMPO_KEY` 接收 `PHRASE` → `AFTER_PHRASE`
- `AFTER_VEL` 接收 `PHRASE` → `AFTER_PHRASE`
- `AFTER_PHRASE` 仅接收 `POS_*` → `AFTER_POS`
- 不引入 `AFTER_BOS → PHRASE`：第一个 PHRASE 永远在第一个含 event 的 bar 内

`compatible_states_for_suffix_ids` / `bridgeable_states_for_target_states` 自动覆盖新状态（已有逻辑基于 `_NON_TERMINAL_STATES` + 转移表泛化）。需要把 `AFTER_PHRASE` 加入 `_NON_TERMINAL_STATES` 元组。

### 2.4 `validate_token_order` 改动（`src/tokenizer/midi_codec.py`）

按 §2.2 文法重写解析循环：
- 在 BAR 头部处理完 `TEMPO` / `KEY` 后增加一处可选 `PHRASE` 消费
- 在 event 循环里，每消费完一个 event-5tuple 后，若下一个 token 是 `PHRASE`，必须紧跟一个新的 event-5tuple；连续两个 PHRASE 或 PHRASE 后接 BAR/EOS 视为非法

## 3. 编码端

### 3.1 启发式扩展（`src/music_analysis/phrase_analysis.py`）

保留 `_build_bar_info / _build_boundary_scores / _pick_candidate_boundaries` 这些为编码服务的核心分析逻辑。删除上层窗口采样函数（详见 §4）。

把"最终采纳的边界"提升为一等数据，`PhraseSpan` 退化为基于边界的 derived view。

新增数据结构：

```python
@dataclass(frozen=True)
class PhraseBoundary:
    bar_index: int       # PHRASE 落在哪个 bar
    anchor_pos: int      # 0 = bar-aligned（BAR 头部之后）；>0 = mid-bar（POS 槽前）
```

`PhraseAnalysis` 扩展为：

```python
@dataclass(frozen=True)
class PhraseAnalysis:
    bars: tuple[BarInfo, ...]
    boundary_scores: tuple[BoundaryScore, ...]   # raw 评分，诊断 / 调参用
    boundaries: tuple[PhraseBoundary, ...]       # 最终采纳的边界（含首句强制、含 anchor_pos）
    phrase_spans: tuple[PhraseSpan, ...]         # 从 boundaries 派生
```

需要新增 / 修改：

- `BarInfo.onset_positions: tuple[int, ...]`：bar 内出现 onset 的 POS 槽列表（已在 `_build_bar_info` 内聚合，只是没保留）。
- `PhraseAnalysisConfig.mid_bar_min_rest_pos: int`：新配置项，默认 `positions_per_bar // 4`（4/4 拍 32 ppb 下为 8 个 POS 槽，约 1 拍）。
- `_pick_in_bar_anchor(left_bar: BarInfo, right_bar: BarInfo, cfg) -> int`：默认返回 `0`；当 `right_bar.onset_positions[0] >= cfg.mid_bar_min_rest_pos`（即右 bar 起首到第一个 onset 之间至少有 1 拍留白）时，返回 `right_bar.onset_positions[0]`，让 PHRASE 推迟到首个 onset 之前。
- `_assemble_final_boundaries(bars, candidate_boundary_bars, cfg) -> tuple[PhraseBoundary, ...]`：把所有"显式策略"集中在这一步——
  1. 首句强制：插入 `PhraseBoundary(first_content_bar, 0)`，其中 `first_content_bar` 是首个 `BarInfo.note_count > 0` 的 bar 索引。
  2. 对每个候选 boundary（来自 `_pick_candidate_boundaries`），调用 `_pick_in_bar_anchor(bars[bar_index - 1], bars[bar_index], cfg)` 决定 `anchor_pos`，得到 `PhraseBoundary(bar_index, anchor_pos)`。
  3. 按 `(bar_index, anchor_pos)` 排序、去重。
  4. 长 span 切分：相邻 boundary 之间 bar 跨度 > `max_phrase_bars` 时，在 `preferred_phrase_bars` 附近插入合成 boundary（anchor_pos = 0）。
  5. 短 span 合并：相邻 boundary 之间 bar 跨度 < `min_phrase_bars` 时，删除后一个 boundary；但首句强制的 boundary 永远不可被删除。
- `_derive_phrase_spans(bars, boundaries) -> tuple[PhraseSpan, ...]`：两两相邻 boundary 形成一个 `PhraseSpan`，最后一个 boundary 到 `len(bars)` 形成尾段。span 的 `start_bar` 取 `boundary.bar_index`；`tokens` 字段已不再被任何调用方使用，保留为空 tuple 以维持兼容。

`analyze_phrase_candidates` 主流程：

```
bars                  = _build_bar_info(...)
boundary_scores       = _build_boundary_scores(bars, cfg)
candidate_boundary_bars = _pick_candidate_boundaries(boundary_scores, cfg)
boundaries            = _assemble_final_boundaries(bars, candidate_boundary_bars, cfg)
phrase_spans          = _derive_phrase_spans(bars, boundaries)
return PhraseAnalysis(bars, boundary_scores, boundaries, phrase_spans)
```

v1 仅在 `_pick_in_bar_anchor` 里实现"右 bar 推迟到首 onset"；左 bar 提前留到后续。

### 3.2 注入函数 `inject_phrase_tokens(tokens)`（`src/tokenizer/midi_codec.py`）

输入为已经注入 KEY 之后的完整 token 列表。流程：

1. `_iter_bar_slices(tokens)` 拿到每个 bar 的 `(start_token, end_token)` 与头部 TEMPO/KEY 跨度。
2. `analyze_phrase_candidates(tokens)` 得到 `PhraseAnalysis`；**只读 `analysis.boundaries`**，不再消费 `phrase_spans`。
3. 从后往前遍历 `boundaries`，依次在 token 列表里插入 `"PHRASE"`：
   - 防御：若目标 bar 没有任何 event（空 bar），跳过该边界——PHRASE 必须有后继 event 才能合法。这种边界在 `_assemble_final_boundaries` 里也应已被过滤，这里仅作兜底。
   - `anchor_pos == 0`：定位该 bar 的 `BAR [TEMPO] [KEY]` 头部末尾，插入到首个 `POS_*` 之前。
   - `anchor_pos > 0`：在 bar 内寻找 `POS_{anchor_pos}`；找到则插在其前；没有完全匹配的 POS 则退化到 bar 内首个 POS 之前；都找不到则跳过该边界。
4. 相邻 PHRASE 去重：插入完成后扫一遍，若 `tokens[i] == tokens[i+1] == "PHRASE"`，删除后者（FSM 层冗余兜底，正常路径不会触发）。

### 3.3 `tokenize_midi`

`_tokenize_note_events` 的末尾从 `return inject_key_tokens(tokens)` 改为 `return inject_phrase_tokens(inject_key_tokens(tokens))`。

### 3.4 `build_vocab`

在 `vocab.append("BAR")` 之后追加 `vocab.append("PHRASE")`。

### 3.5 `tokens_to_midi`

PHRASE 视为可忽略结构 token：

- 主循环里 BAR 头部消费完 `TEMPO` / `KEY` 之后再加一段 `if normalized[idx] == "PHRASE": idx += 1`。
- event 循环改为 `while idx < ... and (normalized[idx].startswith("POS_") or normalized[idx] == "PHRASE")`；遇到 PHRASE 单步 skip，不生成 MIDI 消息。

`_validate_complete_sequence` 走更新后的 `validate_token_order`，PHRASE 在合法位置通过校验。

### 3.6 `tokenize_dataset` 统计

`split_stats` / `total stats` 中新增：

- `phrase_token_total`
- `mean_phrases_per_sequence`
- `mid_bar_phrase_ratio`：anchor_pos > 0 的 PHRASE 占比
- `mean_phrase_bar_span`：相邻两个 PHRASE 之间（或最后一个 PHRASE 到 EOS 之间）所跨的 BAR token 数均值；mid-bar PHRASE 也按它所在 bar 计入起点

## 4. 训练管线（`src/training/train_base.py`）

### 4.1 移除（全部删除）

- `PhraseSamplingConfig` dataclass
- `TokenBinDataset._sample_phrase_window`
- `TokenBinDataset._pick_phrase_policy_kind / _phrase_policy_fallback_order / _phrase_policy_for_kind`
- `TokenBinDataset._choose_phrase_hole_bar_span`
- `TokenBinDataset._build_phrase_aware_fim_example`
- `TokenBinDataset._build_phrase_or_fallback_fim_example`
- `sample_batch` / `sample_mixed_batch` / `_evaluate` 的 `phrase_sampling` 参数及对应分支
- `sample_stats` 字段：`single_phrase_examples / cross_boundary_examples / long_context_examples / phrase_fim_examples / phrase_fim_fallback_examples`
- 训练日志与 `metrics.jsonl` 中以上字段，及 `run_start` 中 `use_phrase_window_sampling / single_phrase_sample_ratio / cross_phrase_sample_ratio / long_context_sample_ratio`
- CLI 参数：`--use-phrase-window-sampling`, `--single-phrase-sample-ratio`, `--cross-phrase-sample-ratio`, `--long-context-sample-ratio`, `--phrase-min-bars`, `--phrase-max-bars`, `--single-phrase-bar-min/max`, `--cross-phrase-bar-min/max`, `--long-context-bar-min/max`
- `main()` 中相关校验与打印（含 `total_phrase_ratio` 求和校验）
- `train_base.py` 顶部 `from ..music_analysis import` 里删除 `PhraseAnalysisConfig / PhraseWindowPolicy / analyze_phrase_candidates / sample_phrase_window`

`src/music_analysis/phrase_analysis.py`：

- 删除 `PhraseWindowPolicy / SampledWindow` 数据类
- 删除 `_choose_single_phrase_window / _choose_cross_boundary_window / _choose_long_context_window / sample_phrase_window`
- 删除 `extract_phrase`
- 删除 `_phrase_boundaries_from_spans / _count_phrase_boundaries / _build_phrase_view_tokens / _build_phrase_span / _normalized_bar_tokens`（这些只服务于已被删除的窗口采样路径）
- 重写 `_build_phrase_spans`：拆为 `_assemble_final_boundaries(bars, candidate_boundary_bars, cfg) -> tuple[PhraseBoundary, ...]` 与 `_derive_phrase_spans(bars, boundaries) -> tuple[PhraseSpan, ...]`，详见 §3.1。删除 `_merge_short_spans / _find_best_split` 中只服务于旧 spans-driven 流水线的逻辑（其归一化策略迁移到 `_assemble_final_boundaries`）。
- `PhraseSpan.tokens` 字段不再被使用，但为减少外部 API 震动暂保留为空 tuple；后续 PR 可一并清理。

`src/music_analysis/__init__.py`：

- 移除 `PhraseWindowPolicy / SampledWindow / sample_phrase_window / extract_phrase` 的 re-export
- 保留 `BarInfo / BoundaryScore / PhraseAnalysis / PhraseAnalysisConfig / PhraseSpan / analyze_phrase_candidates`

### 4.2 保留并增强

`TokenBinDataset._collect_window_cut_positions`：

- PHRASE 不构成独立 cut 边界。扫描器遇到 `PHRASE` 时强制 lookahead 5 个 token 验证是否紧跟完整 event；若是，把 PHRASE 与 event 一起作为一个长度 6 的整体推进，cut 位置只在该整体前后。
- 结果：窗口可以以 `PHRASE POS INST PITCH DUR VEL ...` 开头，但不可能以 PHRASE 单 token 收尾。

`TokenBinDataset._collect_fim_maskable_units`：单元类型清单更新为：

- `bar`：长度 1（不变）
- `event`：长度 5，`POS INST PITCH DUR VEL`（不变）
- `phrase_event`：长度 6，`PHRASE POS INST PITCH DUR VEL`（新增）

扫描器遇到 `PHRASE` 时 lookahead 验证后接 event；若是，发射 `(idx, idx+6, "phrase_event", group_id)` 并前进 6 步；若不是（理论上不会发生于合法编码后的数据），按未识别 token 处理并切断 group_id 序列。

效果：FIM hole 永远不会只 mask PHRASE 单 token；任何包含 PHRASE 的 hole 一定同时包含紧随的 event。

### 4.3 采样路径简化

- NEXT：仅 `_sample_aligned_window`（随机锚点）。
- FIM：仅 `_build_fim_example`（基于结构单元的 generic hole）。
- `sample_batch / sample_mixed_batch / _evaluate` 签名移除 `phrase_sampling`。

### 4.4 训练配置

`configs/train/train_base_run_full.yaml` 与 `configs/train/train_base_run_small.yaml`：

- 删除：`use_phrase_window_sampling` / 三个 `*_sample_ratio` / `phrase_min_bars` / `phrase_max_bars` / 三个 bar range 段
- 修订原 line 27-28 的注释为「NEXT 窗口由结构对齐的随机采样路径生成，乐句信息已下沉为 PHRASE token 由模型自学」

`configs/train/model_base.yaml`：vocab 重建后 `special_token_ids` / `eos_token_id` 由训练数据构建流程自动刷新，不需要本 PR 手改。

## 5. 评测窗口（`src/utils/eval_windows.py`）

### 5.1 函数改名

`sample_bar_aligned_subsequence` → `sample_phrase_aligned_subsequence`，反映「PHRASE 优先、bar fallback」的新语义。

同步更新调用点（共 4 处）：

- `src/utils/benchmarking.py:13`（import）
- `src/utils/benchmarking.py:241`
- `src/utils/benchmarking.py:301`
- `tests/test_music_analysis.py:17`（import）
- `tests/test_music_analysis.py:206`
- `tests/test_music_analysis.py:216`

### 5.2 新实现

1. `_iter_bar_slices` 拿到 bars；同时扫一遍 source tokens 提取所有 PHRASE token 位置 `phrase_positions`。
2. 构建切点列表：合并 PHRASE 位置 + 各 bar 起点，按 token_index 排序，每项标注类型 `phrase` / `bar`。
3. 选起点：先在 `phrase` 切点里随机抽满足 `min_core_tokens <= body_len <= max_core_tokens` 的；抽不到降级到 `bar` 切点；都不行返回 `None`。
4. 选终点：同样优先 `phrase`，再 `bar`。
5. 拼装窗口：
   - 头部：`BOS` + 起点所在 bar 的 effective `TEMPO` + effective `KEY`（按现有规则）
   - 主体：从起点切点沿原序复制到终点切点；BAR 头部规范化扩展为「保留首个 PHRASE，裁掉冗余 TEMPO/KEY」
   - 尾部：`EOS`

### 5.3 不变量

- 起点为 phrase 切点时，body 第一个非头部 token 是 PHRASE。
- 起点为 bar 切点时，body 第一个非头部 token 是 BAR（与当前实现一致）。
- 沿途所有 PHRASE 原样保留。

## 6. 测试

### 6.1 新增 / 修改

`tests/test_tokenizer_midi_codec.py`：

- `PHRASE` 在词表中、id 稳定
- 编码端 round-trip：`tokens_to_midi(tokens_with_phrase)` 与 `tokens_to_midi(tokens_without_phrase)` 产出等价 MIDI
- `validate_token_order` 接受合法 PHRASE 位置（bar-aligned / mid-bar）
- `validate_token_order` 拒绝非法位置（PHRASE 后非 POS、连续 PHRASE、BOS 直后 PHRASE）
- `inject_phrase_tokens` 行为：首句强制、相邻去重、mid-bar anchor 在触发阈值附近的边界行为

`tests/test_music_analysis.py`：

- 删除：`test_extract_phrase_*`、`test_sample_phrase_window_*`、`test_phrase_fim_builder_*` 三组测试（对应函数都被删）
- 更新：`test_analyze_phrase_candidates_*` 系列断言 `analysis.boundaries` 字段就位、首句强制 `PhraseBoundary(first_content_bar, 0)` 在内、`phrase_spans` 与 `boundaries` 派生关系一致
- 新增：mid-bar anchor 触发用例——构造右 bar 在 POS_8 起首、前置 ≥1 拍空白的输入，断言对应 `PhraseBoundary.anchor_pos != 0`
- 更新：调用点改名 `sample_bar_aligned_subsequence` → `sample_phrase_aligned_subsequence`

`tests/test_decoding_grammar_fsm.py`（如不存在则新建）：

- PHRASE 在 `AFTER_BAR / AFTER_BAR_TEMPO / AFTER_BAR_KEY / AFTER_BAR_TEMPO_KEY / AFTER_VEL` 接收后转移到 `AFTER_PHRASE`
- `AFTER_PHRASE` 仅接收 `POS_*`，其他 token 全部拒绝
- `compatible_states_for_suffix_ids` 与 `bridgeable_states_for_target_states` 在含 PHRASE 的目标 suffix 下正确传播

### 6.2 不动

- `tests/test_absolute_benchmark_scoring.py`、`tests/test_checkpoint_selection.py`、`tests/test_benchmarking.py`、`tests/test_training_metrics.py` 不需要为本 PR 修改（`phrase_coherence_score` 是合成代理指标，不依赖 `analyze_phrase_candidates`）。

## 7. 迁移步骤

PR 落地后用户执行顺序：

1. `python -m src.tokenizer.tokenize_dataset --config configs/tokenizer/tokenizer.yaml`：生成新 `.tok` 与新 `tokenizer_vocab.json`
2. `python -m scripts.data.build_training_data --config configs/data/build_training.yaml`：生成新 `.bin` / `.idx.json`
3. 重新训练（旧 checkpoint 因 vocab id 偏移不兼容，必须从 scratch）

## 8. 不变更范围

- 评测代理指标 `phrase_coherence_score` 计算口径
- `phrase_coherence_score` 在 checkpoint selection 中的权重
- KEY analysis / `KEY_*` token 机制
- 移调增强 / velocity 分桶 / tempo 分桶
- FIM ratio / FIM span 配置
- 推理 / 生成路径（`src/inference/generation.py`）：PHRASE 已通过 FSM 状态机自然纳入生成约束，不需要单独的代码改动

## 9. 文档

`docs/todo.md` 第 1 项：标注本 PR 已完成「PHRASE 边界 token 引入」与「动态 window / 启发式切句机制下沉」；剩余子项（句类型 / 句关系 / 句尾收束 / 呼吸 / 重音 / 动机角色等分类头）保留为后续工作。
