# Benchmark 指标说明

这份文档面向长期使用 TuneFlow benchmark 的人。目标不是罗列术语，而是帮助你在看 `benchmark_report.json`、`benchmark_summary.md`、leaderboard 和图表时，快速判断模型到底是“更强了”，还是只是“在当前候选里更靠前”。

## 1. 指标分层

TuneFlow benchmark 现在把指标分成 5 层：

1. `raw metrics`
   直接由单样本 decode 结果或训练期 `metrics.jsonl` 得到的原始指标，例如：
   - `continuation_time_order_validity_rate`
   - `infilling_structural_validity_rate`
   - `most_common_pitch_ratio`
   - `valid_loss_from_training`

2. `aggregate metrics`
   对一批 benchmark case 聚合后的 checkpoint 级指标，例如 continuation / infilling 的均值、比例、覆盖率。

3. 相对分数 `balanced_score`
   这是当前旧体系保留的相对分数。
   它依赖“同一批候选 checkpoint 的相对排序”，适合当前批次内选点，不适合跨 run 长期追踪。

4. 绝对分数 `absolute_score`
   新增的固定量纲绝对分数，范围 `0-100`。
   它由固定维度、固定阈值、固定映射函数得到，目标是跨 run、跨阶段、跨轮次观察模型能力变化。

5. `gate / threshold`
   gate 仍然用于原有推荐逻辑中的“硬性安全检查”，例如 stop 行为或结构合法率。
   gate 不是长期能力分数，不能替代能力面板。

一句话理解：

- `balanced_score`：回答“这一批 checkpoint 里谁更适合被选出来”
- `absolute_score`：回答“这个 checkpoint 本身的能力大概在什么水平”

## 2. 看报告时先看什么

如果你的目标是“选本轮最佳 checkpoint”，先看：

- `gate_passed`
- `balanced_score`
- `balanced_rank`

如果你的目标是“判断模型是不是比上个 run 更强”，先看：

- `absolute_score`
- capability dimension scores
- pitch collapse 指标
- `absolute_score_coverage`

如果 `absolute_score` 上升，但 `balanced_score` 下降，通常说明：

- 当前候选池更强了，导致相对名次变差
- 但模型本身的固定尺度能力并没有退步，甚至可能更强

## 3. 核心 raw metrics 的定义、含义与方向

### continuation 相关

#### `continuation_stop_success_rate`
- 计算对象：continuation 样本
- 方向：越高越好
- 含义：模型是否能在结构合法的前提下自然结束
- 异常时通常意味着：容易拖到预算上限、结束感差
- 它不能反映：内容是否好听

#### `continuation_budget_stop_rate`
- 计算对象：continuation 样本
- 方向：越低越好
- 含义：有多少样本是因为 token 预算耗尽才停下
- 异常时通常意味着：模型不会收尾，或者 EOS 行为弱
- 它不能反映：停下来的内容是否合理

#### `continuation_structural_validity_rate`
- 计算对象：continuation 样本
- 方向：越高越好
- 含义：结构上是否合法，是否接近预期生成片段
- 异常时通常意味着：结构约束不稳，续写容易坏
- 它不能反映：音乐是否自然

#### `continuation_time_order_validity_rate`
- 计算对象：continuation 样本
- 方向：越高越好
- 含义：同一 bar 内事件时间顺序是否正确
- 异常时通常意味着：模型出现倒序事件，结构稳定性差
- 它不能反映：旋律质量

#### `continuation_empty_bar_rate`
- 计算对象：continuation 样本
- 方向：越低越好
- 含义：生成结果中空 bar 的比例
- 异常时通常意味着：模型在长上下文下容易“空掉”
- 它不能反映：非空 bar 是否有音乐性

#### `continuation_syntax_invalid_rate`
- 计算对象：continuation 样本
- 方向：越低越好
- 含义：语法/结构非法的比例
- 异常时通常意味着：decode 输出稳定性不够
- 它不能反映：合法样本内部的内容质量

#### `continuation_missing_eos_rate`
- 计算对象：continuation 样本
- 方向：越低越好
- 含义：应该结束但没有显式 EOS 的比例
- 异常时通常意味着：closure 能力弱
- 它不能反映：即便补上 EOS 后内容是否合理

#### `append_eos_recoverable_rate`
- 计算对象：continuation 样本
- 方向：越高越好
- 含义：如果只是缺 EOS，补一个 EOS 是否可以恢复合法结构
- 异常时通常意味着：问题不只是“忘记结束”，而是整体结构已坏
- 它不能反映：内容是否优美

### infilling 相关

#### `infilling_structural_validity_rate`
- 计算对象：infilling 样本
- 方向：越高越好
- 含义：补全过程整体是否结构合法
- 异常时通常意味着：补全任务基础能力不足
- 它不能反映：补出来的内容是否自然

#### `infilling_time_order_validity_rate`
- 计算对象：infilling 样本
- 方向：越高越好
- 含义：补全结果的时间顺序是否合法
- 异常时通常意味着：中间补全的时序控制差
- 它不能反映：补全内容是否贴合上下文

#### `infilling_syntax_invalid_rate`
- 计算对象：infilling 样本
- 方向：越低越好
- 含义：补全结果语法/结构非法的比例
- 异常时通常意味着：补全 decode 稳定性问题
- 它不能反映：合法补全是否好听

### 训练健康相关

#### `valid_loss_from_training`
- 计算对象：checkpoint 对齐的训练期 eval
- 方向：越低越好
- 含义：该 checkpoint 在训练期验证集上的 loss
- 异常时通常意味着：泛化不足或训练还没到位
- 它不能直接反映：benchmark 音乐行为是否更自然

#### `best_valid_loss_so_far`
- 计算对象：checkpoint 对齐的训练期 eval 历史
- 方向：越低越好
- 含义：到当前 step 为止出现过的最佳验证 loss
- 用途：判断当前 checkpoint 是否仍处在健康下降趋势上

#### `overfit_gap`
- 计算对象：`valid_loss - train_loss_ema`
- 方向：不是简单的“越低越好”或“越高越好”
- 含义：训练 loss 和验证 loss 的差距
- 一般解读：
  - 略微负值或接近 0：通常比较健康
  - 明显正值：过拟合风险上升
  - 过度负值：也可能意味着训练/验证状态不稳定，不能机械乐观解读

## 4. 新增 pitch collapse 指标

这次新增了 3 个专门抓“结构合法但内容塌缩”的指标。三者都保留，因为单独看一个很容易漏判。

### `most_common_pitch_ratio`
- 方向：越低越好
- 定义：生成结果中，最高频 pitch 占全部 pitch event 的比例
- 抓什么问题：全段大量重复同一个音高
- 典型异常：结构没坏，但旋律几乎都在同一个 pitch 上打转
- 不能单独用的原因：
  - 交替两三个音高时，这个指标可能没那么高
  - 但内容仍可能非常机械

### `longest_same_pitch_run_ratio`
- 方向：越低越好
- 定义：最长连续相同 pitch run 占全部 pitch event 的比例
- 抓什么问题：长连续同 pitch 拉平，像“从头到尾一直按住同一个音”
- 典型异常：局部出现很长一段单音连续重复
- 不能单独用的原因：
  - 如果重复是分段出现而不是连续出现，这个指标会低估风险

### `pitch_diversity_score`
- 方向：越高越好
- 定义：一个 `0-1` 的可解释多样性分数，当前 v1 由 normalized entropy 和 unique pitch 数量共同构成
- 抓什么问题：整体音高分布是否过于单一
- 典型异常：虽然不是单一音高长串，但整体 pitch 集合非常窄
- 不能单独用的原因：
  - 某些重复模式可以有多个音高，entropy 不一定极低

### 为什么必须保留三个

- `most_common_pitch_ratio` 擅长抓“全局单音占比过高”
- `longest_same_pitch_run_ratio` 擅长抓“局部长连续 run”
- `pitch_diversity_score` 擅长抓“整体音高分布贫乏”

三者合看时，才能更稳地抓到 pitch collapse。

### 长度保护

短样本天然 pitch 事件少，容易误伤。当前实现对 pitch event 太少的样本不直接给 collapse 分数，而是：

- 将该样本视为 pitch 指标 coverage 不足
- 在 aggregate 时单独输出 coverage
- 避免极短片段把均值拉歪

如果你看到 `continuation_pitch_collapse_coverage` 或 `infilling_pitch_collapse_coverage` 很低，说明这批样本中可用于 pitch collapse 判断的片段偏少，结论要更保守。

## 5. absolute capability panel

`absolute_score` 由 6 个固定能力维度组成：

### `Continuation Closure`
- 关注是否会合理结束
- 主要使用：
  - `continuation_stop_success_rate`
  - `continuation_budget_stop_rate`
  - `continuation_missing_eos_rate`
  - `append_eos_recoverable_rate`

### `Continuation Structure`
- 关注 continuation 结构与时间顺序是否稳定
- 主要使用：
  - `continuation_structural_validity_rate`
  - `continuation_time_order_validity_rate`
  - `continuation_empty_bar_rate`
  - `low_density_bar_rate`
  - `continuation_syntax_invalid_rate`

### `Infilling Integrity`
- 关注 infilling 的结构合法性与时序合法性
- 主要使用：
  - `infilling_structural_validity_rate`
  - `infilling_time_order_validity_rate`
  - `infilling_syntax_invalid_rate`
  - `infilling_pitch_collapse_coverage`

### `Phrase Coherence`
- 当前是 `v1 proxy`
- 目的：先给“乐句相关能力”一个工程上可解释的首版观察面板
- 当前主要使用：
  - `continuation_first_event_hit_rate`
  - `duration_bin_l1_distance`
  - continuation / infilling 的 `pitch_diversity_score`
  - continuation / infilling 的 `most_common_pitch_ratio`

### `Long-Context Stability`
- 当前也是 `v1 proxy`
- 目的：先用长上下文下的发散/重复/空转信号做稳定性代理
- 当前主要使用：
  - `multi_empty_bar_run_rate`
  - continuation / infilling 的 `longest_same_pitch_run_ratio`
  - `overall_most_common_pitch_ratio`
  - `overall_pitch_diversity_score`
  - `continuation_time_order_validity_rate`

### `Training Health`
- 关注训练期健康度
- 当前主要使用：
  - `valid_loss_from_training`
  - `best_valid_loss_so_far`
  - `train_loss_ema`
  - `overfit_gap`

## 6. raw metric -> subscore -> dimension -> total score

当前 absolute score 的计算路径是：

1. 读取 raw metric
2. 用固定阈值和固定映射函数，把 raw metric 映射到 `0-100`
3. 按权重汇总成维度分
4. 再按维度权重汇总成 `absolute_score`

当前映射函数特点：

- 使用固定阈值，不依赖候选集合
- 使用平滑映射，不是硬阈值一步跳
- 每个维度和总分都带 coverage

当前不会因为“同批候选换了一组”而改变同一个 checkpoint 的 `absolute_score`。

## 7. 指标使用建议

如果你想看“模型是否更稳”，优先看：

- `Continuation Structure`
- `Infilling Integrity`
- `Long-Context Stability`
- `continuation_time_order_validity_rate`
- `multi_empty_bar_run_rate`
- `longest_same_pitch_run_ratio`

如果你想看“模型是否更自然，不再那么塌缩”，优先看：

- `Phrase Coherence`
- `pitch_diversity_score`
- `most_common_pitch_ratio`
- `duration_bin_l1_distance`

如果你想做“本轮 checkpoint 排名”，优先看：

- `gate_passed`
- `balanced_score`
- `balanced_rank`

如果你想做“长期能力追踪”，优先看：

- `absolute_score`
- 6 个 capability dimension scores
- `absolute_score_coverage`

## 8. 读数时的常见误区

### 结构合法不等于音乐质量高

模型可以：

- token 结构完全合法
- 时间顺序完全正确
- 但旋律严重塌缩，几乎全是单一 pitch

这正是本次 pitch collapse 指标补进来的原因。

### `balanced_score` 高，不代表跨 run 更强

它只说明在当前候选集合里相对更优。
如果换一批候选，`balanced_score` 可能变。

### `absolute_score` 也不是人工听感替代品

它比相对分更适合长期追踪，但仍然只是 benchmark 指标体系。
最终音乐质量仍然不能完全脱离人工听感和主观审美判断。

### proxy 维度不要过度解释

目前这两个维度明确还是首版代理：

- `Phrase Coherence`
- `Long-Context Stability`

它们很有用，但当前更适合做趋势观察，不适合做过度细粒度的学术式结论。

## 9. 实战建议

一个比较稳妥的看报告顺序是：

1. 先看 `balanced_score` 和 gate，确认这轮推荐逻辑有没有异常
2. 再看 `absolute_score` 和 capability panel，判断模型是否真的变强
3. 最后看 pitch collapse 三指标，确认有没有“结构合法但内容塌缩”的隐藏退化

如果出现下面这种情况，要高度警惕：

- `continuation_time_order_validity_rate` 很高
- `continuation_structural_validity_rate` 不差
- 但 `most_common_pitch_ratio` 很高、`pitch_diversity_score` 很低

这通常不是“模型变稳了”，而是“模型用重复 pitch 伪装稳定”。
