# TuneFlow Benchmark 任务型能力重构设计

- 日期：2026-05-29
- 状态：待审阅
- 关联文件：
  - `src/utils/benchmarking.py`
  - `src/utils/absolute_benchmark_scoring.py`
  - `src/utils/checkpoint_selection.py`
  - `scripts/eval/benchmark_runner.py`
  - `docs/benchmark_metrics.md`

## 1. 背景

当前 benchmark 仍然是“统计型诊断面板主导”的混合体系：结构合法、补全完整、停止成功等底线能力和大量 pitch / rhythm / repetition / training 指标混在一起，导致主分很难稳定回答“哪个模型真的更会写音乐”。

本次重构要把主分改成真正的任务型能力分，优先服务模型能力对比，同时尽量兼顾听感一致性。

## 2. 设计目标

1. 主分主要反映任务完成能力，而不是代理统计分布。
2. 每类任务都要有明确输入、输出、成功标准。
3. 每类任务都要拆成“结构命中”和“音乐实现质量”两层。
4. 避免同一现象在多个维度重复计分。
5. 训练健康类指标不再参与主排序。
6. 排序辅助改为相对基线模型的对比或 win-rate。
7. 每类任务至少要有一条硬约束证据和一条音乐实现证据，尽量不要让同一个启发式分析器同时完成任务构造和任务判分。

## 3. 总体结构

新的 benchmark 由三层组成：

1. 任务层
   - 结构控制能力
   - 局部发展能力
   - 长程连贯能力
   - 补全一致性能力

2. 评分层
   - 主分：`task_capability_score`
   - 辅助分：`task_control_score`、`task_realization_score`
   - 对战分：`vs_baseline_win_rate`

3. 诊断层
   - pitch / rhythm / repetition / 空小节 / 训练健康等仅用于解释和报警

## 4. 任务定义

### 4.1 结构控制能力

目标：检查模型是否真的会利用结构提示或结构 token。

输入：
- 前文 token
- 结构条件 token 或结构标签
- 边界类型标签

输出：
- continuation 结果
- 边界位置和句内结构结果

成功标准：
- `control_hit`：是否在正确位置起新句、是否遵守边界约束、是否延续句内结构
- `music_realization`：命中边界后，局部首段是否自然，前后过渡是否合理

建议子指标：
- `boundary_type_hit_rate`
- `illegal_boundary_rate`
- `boundary_timing_hit_rate`
- `post_boundary_realization_score`

### 4.2 局部发展能力

目标：检查模型会不会做合理重复、模进、变形，而不是机械复制或完全跑偏。

输入：
- 含局部动机的前文
- 发展条件标签：`repeat` / `develop` / `transform`

输出：
- 局部 continuation

成功标准：
- `control_hit`：是否满足给定发展关系
- `music_realization`：变化是否仍然像音乐发展，而不是 token 级死抄或断裂

建议子指标：
- `motif_relation_hit_rate`
- `copy_overuse_penalty`
- `unrelated_drift_penalty`
- `development_quality_score`

### 4.3 长程连贯能力

目标：检查模型在多个小节范围内是否保持主题、组织感和结构连续性。

输入：
- 较长前文窗口
- continuation 长度目标
- 可选段落状态标签

输出：
- 多小节 continuation

成功标准：
- `control_hit`：长度范围内不明显失控，不频繁空转，不早衰
- `music_realization`：主题线索可追踪，段落之间有持续创作思路

建议子指标：
- `long_horizon_completion_rate`
- `theme_retention_score`
- `section_continuity_score`
- `degeneration_penalty`

### 4.4 补全一致性能力

目标：检查 infilling 是否与前后文形成合理连接。

输入：
- prefix tokens
- suffix tokens
- hole 结构标签：`inside_phrase` / `across_boundary`

输出：
- 中间补全部分

成功标准：
- `control_hit`：与前后文可拼接，边界合法，结构位置合理
- `music_realization`：节奏、音高、结构位置与上下文一致

建议子指标：
- `bridge_validity_rate`
- `boundary_compatibility_hit_rate`
- `rhythmic_connection_score`
- `pitch_connection_score`
- `structural_fit_score`

## 5. 主分与排序

### 5.1 主分公式

建议主分直接替换为 `task_capability_score`：

```text
task_capability_score =
  0.30 * structure_control_score
  + 0.25 * local_development_score
  + 0.25 * long_context_coherence_score
  + 0.20 * infilling_consistency_score
```

每个一级任务内部再拆为：

```text
task_score = 0.40 * control_hit_score + 0.60 * music_realization_score
```

### 5.2 辅助排序

- `task_capability_score` 负责主排序
- `vs_baseline_win_rate` 负责对战对比
- 旧的 `balanced_score` 不再作为主排序依据
- 旧的 `absolute_score` 不再作为主能力分主轴

## 6. 指标分层

### 6.1 建议保留在主分里的指标

仅保留真正直接支撑任务完成的指标：

- `continuation_structural_validity_rate`
- `continuation_time_order_validity_rate`
- `continuation_stop_success_rate`
- `infilling_structural_validity_rate`
- `infilling_time_order_validity_rate`
- `continuation_first_event_hit_rate`
- `duration_bin_l1_distance`
- `onset_position_l1_distance`

说明：
- 这些指标不能继续各自单独主导总分
- 它们只应作为对应任务里的局部子项

### 6.2 建议降级为 gate 的指标

这些指标更适合做底线报警，不适合主导能力排序：

- `continuation_budget_stop_rate`
- `continuation_missing_eos_rate`
- `continuation_syntax_invalid_rate`
- `infilling_syntax_invalid_rate`
- `continuation_empty_bar_rate`
- `low_density_bar_rate`
- `multi_empty_bar_run_rate`

### 6.3 建议保留为诊断项的指标

这些指标有价值，但更适合解释失败模式：

- `most_common_pitch_ratio`
- `longest_same_pitch_run_ratio`
- `pitch_diversity_score`
- `duration_diversity_score`
- `rhythm_diversity_score`
- `event_ngram_repeat_ratio`
- `rhythm_ngram_repeat_ratio`
- `pitch_analysis_coverage`
- `rhythm_analysis_coverage`
- `repetition_analysis_coverage`
- `append_eos_recoverable_rate`
- `same_pitch_overlap_rate`
- `pitch_span_delta_mean`
- `generated_event_delta_mean`
- `generated_bar_delta_mean`

### 6.4 建议移出主分的指标

这些是训练健康指标，只保留在训练看板或特殊排查中：

- `valid_loss_from_training`
- `best_valid_loss_so_far`
- `train_loss_ema`
- `overfit_gap`

## 7. 旧维度重构

### 7.1 `phrase_coherence`

建议拆除这个一级维度名。

原因：
- 语义过宽
- 混入了结构、分布和重复统计
- 容易重复计分

去向：
- 边界和句首/句内控制，迁入结构控制能力
- 局部材料延续与变化，迁入局部发展能力
- 简单分布贴近项，仅保留为诊断或子项

### 7.2 `musical_expression`

建议拆除这个一级维度名。

原因：
- 更像统计桶，不像单一能力
- 容易把“多样性”误当成“表达能力”

去向：
- 局部组织，迁入局部发展能力
- 多小节持续组织，迁入长程连贯能力
- pitch / rhythm / repetition 的大部分统计项降为诊断

### 7.3 `long_context_stability`

建议重构为 `长程连贯能力`。

原因：
- `stability` 太偏防退化
- 目标应是持续创作思路，而不是“不坏掉”

去向：
- 主题保持
- 段落承接
- 长窗口组织性

## 8. 重复计分清理原则

1. 同一现象最多进入一个主任务分和一个诊断项。
2. `结构合法 / 时间顺序 / 停止成功` 只算底线，不再反复加权。
3. `pitch` 重复、`n-gram` 重复、空小节退化只做报警，不再主导排序。
4. `append_eos_recoverable_rate` 只保留为特殊诊断，不进主分。
5. 训练损失类指标彻底退出 benchmark 主排序。

## 9. 分阶段实施顺序

### 阶段 1

- 直接替换主排序框架
- 把主分改成任务型总分
- 先用现有数据和现有解析器搭出 `v1`
- 训练健康类指标移出主分

### 阶段 2

- 补齐结构控制任务
- 显式区分 `start_new_phrase` / `continue_inside_phrase`
- 为未来 `MOTIF / SUBPHRASE / PHRASE` 拆分预留接口

### 阶段 3

- 补齐局部发展任务
- 补齐长程连贯任务
- 降低对单一统计代理的依赖

### 阶段 4

- 加入 `vs_baseline_win_rate`
- 重写报告页和 summary 文案
- 逐步移除旧的主排序字段

## 10. 非目标

- 不先改训练代码
- 不先改 tokenizer
- 不先改 phrase 标签方案本身
- 不把统计项重新包装成主分

## 11. 结论

新的 benchmark 以“任务完成情况”为主轴，旧的统计型诊断面板退居辅助层。这样可以更稳定地比较不同模型/不同方案的真实能力差距，同时尽量保留对听感一致性的解释能力。
