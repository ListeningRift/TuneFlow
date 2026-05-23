# 结构性中心优先的稳定调性检测设计

- 日期：2026-05-23
- 状态：设计已确认，待评审
- 目标模块：`src/music_analysis/key_analysis.py`
- 关联模块：
  - `src/music_analysis/__init__.py`
  - `src/utils/annotation_review.py`
  - `tests/test_music_analysis.py`
  - `tests/test_annotation_review.py`

## 1. 背景

当前调性检测实现基于加权 pitch-class histogram、Krumhansl-Schmuckler 风格相关性打分、全曲先验以及 HMM 平滑。它已经能够输出：

- 局部帧级调性时间线
- 平滑后的稳定调性段
- 稀疏转调点

但当前系统仍然偏向“这一瞬间最像哪个调”，在以下场景中容易产生不够稳定的转调判断：

- 单小节或短时离调材料
- 借和弦、平行大小调混合
- 次属和弦或属功能扩展
- 半音经过音、辅助音、装饰性变化
- 短暂接近属调或关系调但很快回归主中心的片段

这些情况在音乐结构上往往仍然服从原有中心，只是局部颜色发生变化。当前实现虽然使用了 HMM 惩罚频繁切换，但本质上仍然是对局部窗口结果做平滑，因此仍可能把短时异色材料误判成转调。

## 2. 设计目标

本次设计要把调性检测从“瞬时拟合最优”调整为“结构性中心识别优先”，满足以下目标：

1. 输出结果更像“这段音乐的结构性中心在哪里”，而不是“这一帧最像哪个调”。
2. 明显偏保守，宁可少报真实短转调，也尽量避免被临时离调、借和弦、次属和弦、半音装饰带偏。
3. 对中等长度以上、确实建立了新中心的真实转调，仍保留识别能力。
4. 最终对外发布的 `segments` 与 `modulation_points` 应比帧级结果更稳定、更稀疏。
5. 尽量复用现有滑窗与打分框架，在当前实现基础上增量改造，而不是重写为全新的分段系统。

## 3. 非目标

本次不处理以下内容：

- 不引入新的和声功能分析器或和弦级分析器。
- 不建立完整的乐段级全局分段后再反推调性。
- 不追求极短转调、瞬时 tonicization 或瞬时局部色彩变化的高召回。
- 不修改 `KEY_*` token 的注入协议。
- 不改变现有调性名称集合，仍使用 24 个大小调 key 状态。

## 4. 总体方案

采用“结构中心优先 + 证据累积转调”的两层方案：

### 4.1 局部证据层

保留现有的滑窗 pitch-class histogram 和 key 打分流程，让每个 frame 继续产出：

- 当前窗口对 24 个 key 的相对支持分数
- 当前窗口的瞬时最优 key
- 窗口是否存在较高不确定性

这一层只负责表达“局部看起来像什么”，不直接决定是否转调。

### 4.2 结构中心层

在帧级证据之上新增“结构中心跟踪”状态机。它不直接跟随瞬时最优 key，而是维护：

- `stable_key`：当前结构性中心
- `challenger_key`：当前最有可能取代 `stable_key` 的挑战者 key
- `challenger_run_frames`：挑战者连续成立的帧数
- `challenger_accumulated_lead`：挑战者相对稳定调累计领先的优势
- `stable_key_decay_frames`：稳定调持续失去支持的帧数

只有当挑战者经过持续、稳定、累计的证据确认后，才真正发布转调。

V1 明确保留 HMM，但只把它视为前处理层：

- HMM 负责抑制局部窗的高频抖动，并提供前处理后的帧级候选信息。
- 状态机负责最终帧归属、段落发布和转调点发布。
- `frames`、`segments`、`modulation_points` 的最终语义都以状态机结果为准，而不是以 HMM 解码路径为准。

## 5. 核心判定语义

### 5.1 稳定调优先保留

只要当前 `stable_key` 在某一帧仍有中等以上支持，就优先保持原结构中心，不因为瞬时最优 key 变化而立即切换。

这意味着以下情况默认优先解释为“原调内张力”而不是“转调”：

- 临时离调音
- 次属和弦
- 借和弦
- 半音装饰
- 短时接近属调或关系调

### 5.2 挑战者先累计，不能直接上位

某个新 key 即使成为局部最优，也只是“挑战者”。它必须满足以下趋势，才进入转调候选：

- 多帧连续成为最强候选
- 相对 `stable_key` 的领先不是偶然的极小差值
- 领先优势能持续累积，而不是一两帧后立即消失

### 5.3 真正转调的定义是“新中心建立”

本次不再把“瞬时领先”视为转调，而把以下组合作为结构性转调的成立条件：

1. 新 key 持续领先达到中等长度以上。
2. 旧 `stable_key` 在这段期间持续失去支撑。
3. 新 key 的优势来自整体材料，而不是少量异音或装饰性偏离。
4. 切换后新 key 还能继续稳定维持，而不是马上回退。

### 5.4 允许漏报短转调

如果某段材料既可以解释为“短时真实转调”，也可以解释为“原调内临时离调”，系统优先选择后者。只有在新中心建立证据明显时才发布调性切换。

## 6. 算法落地

### 6.1 保留现有帧级打分

以下流程保留：

- `_parse_token_events()`
- `_weighted_pitch_class_histogram()`
- `_rank_key_scores()`
- `_build_raw_frames()`

现有相关性打分仍然作为基础输入，不在本次设计中推翻。

### 6.2 V1 的 HMM 定位

现有 HMM 路径解码更像“带切换惩罚的逐帧路径选择”，对抑制抖动有效，但仍不足以表达“结构性中心保持”。

V1 明确保留 HMM，但只作为前处理层，职责固定为：

- 对局部帧级结果做一次温和去抖；
- 为每一帧提供前处理后的候选 key 信息；
- 不直接决定最终 `best_key`、`segments` 或 `modulation_points`。

V1 不再保留“由 HMM 直接输出最终时间线”的双重语义，统一改为：

`raw frame 分数 -> HMM 前处理 -> 结构中心状态机 -> 最终发布`

这样可以在保留现有实现骨架的前提下，把最终调性语义统一收口到状态机层。

### 6.3 新增结构中心跟踪状态机

对于每个 frame，围绕当前 `stable_key` 和候选 `challenger_key` 更新内部状态：

1. 读取该帧对 `stable_key` 的支持度。
2. 读取该帧瞬时最优 key 及其相对 `stable_key` 的领先值。
3. 如果瞬时最优 key 不是 `stable_key`，但领先幅度不足，或 `stable_key` 仍有足够支持，则忽略本帧挑战。
4. 如果同一挑战者连续多帧稳定领先，则累计：
   - 连续帧数
   - 累计优势
   - 稳定调衰减计数
5. 若挑战中断、挑战者切换、或 `stable_key` 恢复足够支持，则清空或显著回退挑战状态。

#### 6.3.1 支持度与领先值的精确定义

V1 中状态机不直接使用原始相关性排名，而使用“前处理后的每帧逐 key 支持图”。对每个 frame `t` 和 key `k`，定义：

- `local_score(t, k)`：该帧原始局部窗口对 key `k` 的相关性分数；
- `global_bonus(k) = global_key_bias * max(0.0, global_score(k))`；
- `neighbor_bonus(t, k) = Σ(max(0.0, local_score(n, k)) * neighborhood_decay^abs(n - t))`，其中 `n` 遍历 `t` 的邻域帧；
- `preprocessed_score(t, k) = local_score(t, k) + global_bonus(k) + neighbor_bonus(t, k)`；
- `support(t, k) = max(0.0, preprocessed_score(t, k))`

其中：

- `preprocessed_score(t, k)` 表示经过 V1 前处理后的逐 key 分数；
- HMM 在 V1 中只负责对帧级候选做前处理去抖，不直接覆盖这张逐 key 支持图的定义；
- 状态机只消费这个逐 key 支持图，不直接消费 HMM 最终路径标签。

对当前稳定调 `stable_key` 与某个候选 key `candidate_key`，定义：

- `stable_support(t) = support(t, stable_key)`
- `candidate_support(t) = support(t, candidate_key)`
- `lead(t, candidate_key, stable_key) = candidate_support(t) - stable_support(t)`

对单帧而言，只有当以下条件同时成立，该帧才算“有效挑战帧”：

- `candidate_key` 是该帧支持度最高的 key；
- `candidate_key != stable_key`；
- `lead(t, candidate_key, stable_key) >= challenger_min_lead`。

#### 6.3.2 低置信帧规则

V1 明确规定：低置信帧既不推进挑战，也不推进稳定调衰减。

低置信帧定义为以下任一情况成立：

- 该帧 `is_uncertain == True`；
- 该帧最高支持 key 的支持度 `< stable_key_min_support`；
- 该帧最高 key 与次高 key 的支持差 `< challenger_min_lead`。

对低置信帧的处理统一为：

- 不增加 `challenger_run_frames`；
- 不增加 `challenger_accumulated_lead`；
- 不增加 `stable_key_decay_frames`；
- 不更换 `stable_key`；
- 若当前已有挑战者，则挑战状态保持冻结，不在该帧上前进也不在该帧上清零。

这样可以避免模糊帧既误推挑战，又误伤原调。

#### 6.3.3 `stable_key` 的初始化规则

V1 的 `stable_key` 初始化按以下规则执行：

1. 从头跳过所有低置信帧。
2. 找到第一个高置信帧后，向后收集一个初始化窗口，长度由 `initial_stable_window_frames` 决定。
3. 在该窗口内，对每个 key 累加其 `support(t, k)`。
4. 取累计支持度最高的 key 作为初始 `stable_key`。
5. 如果从头到尾都没有足够的高置信帧，则整个结果保持 `uncertain`，不初始化 `stable_key`。

这条规则的目的，是避免开头一两帧偶然偏色直接把整段音乐定到错误起调。

#### 6.3.4 挑战者更换即重置

V1 明确规定：如果当前存在 `challenger_key = A`，但在后续某一高置信有效挑战帧上，新的有效挑战者变为 `B`，且 `B != A`，则立即执行完整重置：

- `challenger_key` 改为 `B`
- `challenger_run_frames` 重置为 1
- `challenger_accumulated_lead` 重置为该帧对 `B` 的 `lead`
- `stable_key_decay_frames` 重置为仅从该帧开始累计

不允许把不同 challenger 的领先优势串接累计。V1 只承认“同一个挑战者持续建立新中心”的证据链。

### 6.4 双门槛转调确认

为了避免“看起来有点像新调”就切换，转调确认至少同时满足以下门槛：

- `连续时长门槛`
  挑战者连续成立达到最小帧数，对应中等长度材料。
- `累计优势门槛`
  挑战者累计领先幅度达到最小值，而不是每帧只略高一点。
- `旧中心衰减门槛`
  `stable_key` 持续失去支持，而不是仍然保持中等稳定。
- `新中心支持门槛`
  挑战者自身的平均支持度达到最低要求。

只有四类条件共同成立，才允许从 `stable_key` 切换到 `challenger_key`。

### 6.5 切换后的落地确认

即便前述条件满足，也不应立即将切换点作为正式转调点发布。建议在切换后再增加一个“落地确认”阶段：

- 新 key 在后续若干帧继续稳定存在，则正式认定为结构性转调。
- 若切换后很快退回旧调，则撤销这次切换，视为一次失败挑战。

这一步的目的，是进一步消除“短时看起来像新调，但没有真正站住”的误报。

#### 6.5.1 `modulation_point` 是否回填

V1 明确规定：`modulation_point` 不回填。

也就是说：

- 当挑战者刚开始出现时，不立即发布转调点；
- 当挑战者满足确认条件并完成落地确认后，才把“正式承认新中心的那一帧”发布为 `modulation_point`；
- 不把 `modulation_point` 回填到挑战首次出现的帧，也不回填到挑战累计开始的帧。

对应的发布语义是：

- `modulation_point` 表示“系统正式承认新中心”的位置；
- 它允许晚于音乐学上最早可能发生转调的时间；
- `KeySegment` 的新段起点与该 `modulation_point` 保持一致，V1 不做事后追溯式改写。

### 6.6 发布层只输出结构中心

最终对外输出时遵循以下原则：

- `frames` 可以保留“瞬时最优”和“最终结构中心”之间的差异。
- `segments` 只根据结构中心变化切段。
- `modulation_points` 只发布已经通过确认的新中心建立点。
- 短时离调材料不应单独形成新的稳定调性段。

## 7. 配置项调整

建议在 `KeyAnalysisConfig` 中新增以下保守型配置项：

- `stable_key_min_support`
  - 含义：当前结构中心只要支持度高于该阈值，就优先保留。
- `challenger_min_lead`
  - 含义：挑战者单帧相对稳定调至少领先多少，才算有效挑战。
- `initial_stable_window_frames`
  - 含义：初始化 `stable_key` 时用于累计支持度的起始窗口长度。
- `modulation_min_run_frames`
  - 含义：挑战者至少连续多少帧，才具备被确认的资格。
- `modulation_min_accumulated_lead`
  - 含义：挑战者累计领先稳定调的总优势阈值。
- `modulation_min_newkey_support`
  - 含义：挑战者自身平均支持度下限。
- `stable_key_max_decay_frames`
  - 含义：稳定调连续失去支持的最大容忍时长，超过后更容易触发切换。
- `modulation_release_frames`
  - 含义：切换后新 key 还需继续稳定多少帧，才正式发布调性切换。

现有以下参数需要重新定位或复核：

- `modulation_confirmation_frames`
  - 从“发布段落前确认连续帧数”调整为更偏向结构中心状态机内部确认。
- `key_change_penalty`
  - 若 HMM 保留，应视为局部平滑参数，而不是主要转调判定手段。
- `global_key_bias`
  - 继续保留，但要防止其过强导致真实中段转调被压制。

## 8. 输出语义调整

### 8.1 `KeyFrame`

建议保留现有字段，并允许以下语义：

- `raw_key`：该帧瞬时最优 key
- `best_key`：该帧经结构中心判定后归属的稳定 key

这样可以明确区分“局部像什么”与“最终认为属于哪个结构中心”。

### 8.2 `KeySegment`

`KeySegment` 应只代表结构上已经稳定成立的调性段，因此：

- 段长会整体变长
- 段数会减少
- 更适合做结构分析、token 注入和 review 展示

### 8.3 `ModulationPoint`

`ModulationPoint` 应理解为“新中心开始被正式承认的位置”，而不是音乐学上精确到单拍的真实转调瞬间。

V1 明确不做回填，因此它通常会晚于局部证据最早出现的位置。这是本次保守化策略的预期行为，而不是误差。

## 9. 测试策略

### 9.1 原调稳定性回归

新增样例，验证以下情况不应触发转调：

- 单小节离调材料
- 次属和弦造成的临时升降音
- 借和弦或平行大小调混合
- 半音经过与装饰音密集出现
- 短暂属调或关系调偏移但很快回归

预期：

- `segments` 仍只有 1 段
- `initial_key` 保持原调
- `modulation_points` 为空

### 9.2 真实转调保留能力

新增样例，验证：

- 持续数小节的新中心仍能被识别
- 转调点可以比现有实现更晚，但不能完全丢失

预期：

- 至少出现 2 段稳定 `segments`
- 新段 key 正确
- `modulation_points` 落在保守确认后的合理位置

### 9.3 短暂挑战失败

新增样例，专门覆盖状态机行为：

- 挑战者连续出现 1 到 2 帧但很快失败
- 挑战者曾领先，但累计优势不足
- 挑战者领先期间旧中心仍未真正失效
- challenger 中途更换，且新 challenger 不能继承旧 challenger 的累计值
- 低置信帧夹在挑战过程中，但不应推进挑战，也不应推进衰减

预期：

- 不切段
- 不发布 `modulation_points`
- 最终仍保留原 `stable_key`

### 9.4 发布层稳定性

新增样例，验证最终输出语义：

- 允许 `raw_key != best_key`
- `segments` 比 `frames` 更保守、更长
- `modulation_points` 数量少于易抖动方案

### 9.5 建议新增测试构造器

建议在 `tests/test_music_analysis.py` 中新增类似以下 token 构造器：

- `_c_major_with_secondary_dominant_tokens()`
- `_c_major_with_modal_mixture_tokens()`
- `_c_major_with_chromatic_neighbor_tokens()`
- `_c_major_with_brief_dominant_excursion_tokens()`
- `_c_major_with_confirmed_g_major_modulation_tokens()`

## 10. 风险与控制

### 10.1 主要风险

- 过于保守，导致部分中短篇幅真实转调漏检。
- 状态机参数过多，调参复杂度上升。
- 若 `global_key_bias` 过强，可能压制中段真实转调。
- 若挑战者累计逻辑设计不清晰，可能产生迟滞过大或边界不稳定的问题。

### 10.2 控制策略

- 通过 TDD 先固定“短偏离不转、长建立才转”的测试基线，再写实现。
- 把“连续时长”“累计优势”“旧中心衰减”“落地确认”分成独立可解释参数，避免隐式耦合。
- 保留帧级调试信息，方便 review 为什么某次挑战未被确认。
- 在 annotation review 层继续展示段级与帧级差异，降低后续调参成本。

## 11. 成功标准

满足以下条件即可认为本次设计落地成功：

1. 调性输出明显更保守，频繁小幅摆动显著减少。
2. 临时离调、借和弦、次属和弦、半音装饰不再轻易触发转调。
3. 中等长度以上、确实建立新中心的转调仍能被识别。
4. `segments` 与 `modulation_points` 更接近结构分析语义，而不是瞬时拟合语义。
5. 新增回归测试能够稳定约束“结构中心优先”的行为。
