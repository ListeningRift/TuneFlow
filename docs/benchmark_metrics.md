# Benchmark 指标说明

这份文档说明 TuneFlow 当前 benchmark 的任务型评分体系，以及报告里各类指标应该怎么读。

## 1. 先看什么

当前 benchmark 的主阅读顺序是：

1. `task_capability_score`
   这是主分，也是 fast / formal 排序与最终推荐的主键。
2. `vs_baseline_win_rate`
   这是相对基线的辅助排序字段，但它不是 case-level 胜率，而是任务维度对战率代理值。
3. `task_control_score` 与 `task_realization_score`
   用来拆开看“任务控制是否稳”和“音乐实现是否好”。
4. 4 个一级任务分
   - `structure_control_score`
   - `local_development_score`
   - `long_context_coherence_score`
   - `infilling_consistency_score`

如果你只想回答“这一轮应该推荐哪个 checkpoint”，优先看前两层即可。

## 2. 指标分层

TuneFlow benchmark 现在分为 3 层：

1. 任务型主分层
   - `task_capability_score`
   - `vs_baseline_win_rate`
2. 任务能力拆解层
   - `task_control_score`
   - `task_realization_score`
   - 4 个一级任务分
3. 诊断 / gate 层
   - 结构合法性、时序合法性、停止行为
   - pitch / rhythm / repetition 退化指标
   - training health 指标
   - 旧 `balanced_score` / `absolute_score` 兼容字段

其中第 3 层不再承担主推荐叙事，主要用于解释“为什么这个 checkpoint 被挡掉”或“为什么虽然主分高，但行为仍有风险”。

## 3. 主分定义

### `task_capability_score`

- 含义：任务型 benchmark 的主分
- 用途：作为 checkpoint selection 的主排序键
- 解读：分数越高，说明该 checkpoint 在任务控制与音乐实现的综合任务表现越强

### `vs_baseline_win_rate`

- 含义：当前 checkpoint 相对基线 checkpoint 的任务维度对战率代理值
- 用途：作为主分相近时的辅助排序信息
- 计算方式：
  - 综合 benchmark（`task_scope=all`）会对以下 7 个字段逐项与基线比较：
    - `task_capability_score`
    - `task_control_score`
    - `task_realization_score`
    - `structure_control_score`
    - `local_development_score`
    - `long_context_coherence_score`
    - `infilling_consistency_score`
  - scope-specific benchmark（`continuation` / `infilling`）只会在当前实际可比较的任务字段上做同样比较
  - 当前值大于基线记 `1`
  - 当前值小于基线记 `0`
  - 当前值等于基线记 `0.5`
  - 对所有可比较字段取均值，得到最终 `vs_baseline_win_rate`
- 默认基线：
  - 如果显式传了 `--baseline-checkpoint`，就使用该 checkpoint
  - 否则 fast 和 formal 各自使用本阶段结果集合里 `step` 最小的 checkpoint
- 约束：
  - 如果显式指定的 baseline 不在当前阶段结果集合里，会直接报错，不会静默写成 `None`

## 4. 任务能力拆解

### `task_control_score`

反映模型是否能按任务约束稳定完成结构与控制要求。

### `task_realization_score`

反映模型在音乐实现层面的完成度，包括内容展开、长程连贯和补全衔接等表现。

### 4 个一级任务分

#### `structure_control_score`

看结构控制是否稳定，重点关注边界类型、边界时机和边界后的实现质量。

#### `local_development_score`

看局部动机发展是否自然，是否既能延续材料又不过度抄写或无关漂移。

#### `long_context_coherence_score`

看长程主题保持、段落连续性和长上下文退化控制。

#### `infilling_consistency_score`

看补全段是否能在边界、节奏、音高和结构上与上下文接住。

## 5. gate 与诊断指标的定位

这些指标仍然重要，但定位已经下降为“诊断 / gate”：

- `continuation_stop_success_rate`
- `continuation_budget_stop_rate`
- `continuation_time_order_validity_rate`
- `infilling_structural_validity_rate`
- pitch collapse / rhythm / repetition 指标
- training health 指标

它们现在主要回答两类问题：

1. 这个 checkpoint 是否应该直接被 gate 掉
2. 主分变化背后，具体是哪里变好了或变坏了

不要再把这些统计型指标当成主推荐文案。

## 6. 旧字段现在怎么用

### `balanced_score`

- 保留原因：兼容旧报告、旧选择链路和历史排查
- 当前定位：辅助兼容字段，不再是 `benchmark_overall` 的主叙述

### `absolute_score`

- 保留原因：历史报告兼容，以及少量旧输出结构需要过渡
- 当前定位：诊断型兼容字段，不再主导推荐结论

如果任务型主分与旧字段结论不一致，应优先相信：

1. `task_capability_score`
2. `vs_baseline_win_rate`
3. 4 个一级任务分

旧字段只用于帮助解释差异，不再反向覆盖主推荐。

## 7. 读报告建议

推荐按下面顺序读 summary：

1. 看“最终推荐”里的 `task_capability_score`、`vs_baseline_win_rate`
2. 看 `task_control_score`、`task_realization_score`
3. 看 4 个一级任务分，判断是结构、局部发展、长程连贯还是补全一致性拖了后腿
4. 最后再看 gate、训练健康度和退化指标，确认有没有隐藏风险

如果你是在比较不同 run 的长期趋势，也优先看任务型主分和一级任务分的变化，不要再用 `balanced_score` / `absolute_score` 做主叙述。
