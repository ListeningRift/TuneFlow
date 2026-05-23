# 分层乐句边界重构设计

- 日期：2026-05-20
- 状态：已完成设计确认，待审阅
- 目标模块：`src/music_analysis/phrase_analysis.py`
- 关联模块：
  - `src/music_analysis/__init__.py`
  - `src/tokenizer/midi_codec.py`
  - `src/utils/annotation_review.py`
  - `tools/annotation_review_viewer.js`
  - `tests/test_music_analysis.py`
  - `tests/test_annotation_review.py`

## 1. 背景

当前乐句划分已经从小节级推进到音符级，但核心判断仍偏向局部片段相似、局部重复信号和若干边界启发式。这个版本已经能够输出 `motif / subphrase / phrase` 三层候选，也能兼容 `PHRASE` token 注入，但仍存在以下问题：

- 重复判断仍可能从零散局部出发，而不是先识别相对完整的结构单元。
- 当前对“重复”的偏好强于“发展、回应、模进、延展”等结构关系，容易漏掉相关但不严格重复的乐句。
- 仅凭局部相似、同头、轮廓接近就可能过早抬高边界等级。
- 段落起始位置仍被当作强制乐句边界参与最终结果，不符合“分析起点不等于结构边界”的原则。

本次重构的目标不是继续加重局部重复检测，而是把乐句划分的主轴调整为：

- 先识别相对完整的候选结构单元。
- 再判断单元之间是否构成重复、核心重复、并列回应或发展关系。
- 最后把这些结构关系回写到 `motif / subphrase / phrase` 三层边界评分。

## 2. 本次设计确认的关键决策

### 2.1 核心重复修正策略

采用“中等修正”策略：

- 允许裁掉候选单元首尾少量非核心成分。
- 但保留下来的核心重复段必须占原单元主体部分，目标阈值按约 `60%` 设计。
- 不允许为了凑相似而只保留过短局部。

### 2.2 段落起始位置的落地方式

采用语义更干净的方案：

- 段落起始位置不再进入 `analysis.boundaries`。
- 段落起始位置也不再参与 `PHRASE` token 注入。
- 段落起始位置单独作为分析元数据记录，只表示分析起点，不表示结构边界。

### 2.3 首句缺少强证据时的处理方式

采用非强制策略：

- 如果第一句整体都没有足够强的重复、收束、停顿或结构转换证据，允许暂时不给出首句结束边界。
- 不为“首句必须有边界”额外设计兜底规则。

## 3. 设计目标

本次设计要满足以下结构原则：

1. 乐句识别优先以完整结构单元作为判断窗口，而不是为寻找相似性任意截取零散局部片段。
2. 若完整单元前后仅带有少量非核心成分，不能因为逐音不完全匹配就否定重复关系，应允许剔除首尾少量非核心部分后比较核心内容。
3. 两个单元即使不构成高相似度严格重复，只要共享起始动机，并在后续形成发展、变形、模进、延展或回应关系，且各自完整，也允许视作彼此关联的乐句。
4. 仅有局部音高接近、偶然同头、短小片段相似，不足以认定为对应乐句。
5. 乐段开头是分析起点，不自动构成具有独立结构证据的乐句边界。
6. 最终乐句划分仍以结构完整性、听感组织、边界感和单位间呼应关系为核心，而不是机械依赖逐音全重复或固定位置切分。

## 4. 非目标

本次不处理以下内容：

- 不新增训练词表中的新结构 token，训练层仍只保留 `PHRASE`。
- 不把 `motif` 或 `subphrase` 直接写回训练 token 序列。
- 不引入全局动态规划式的最优分段搜索。
- 不做和声级功能分析、终止式分类或问答句类型分类。
- 不改动 `KEY_*` token 注入机制本身。

## 5. 总体方案

本次采用“完整单元优先”的两阶段判断框架：

### 第一阶段：生成候选结构单元

- 继续使用现有 note-level 特征，生成弱候选边界点。
- 这些点不直接等于最终边界，只作为切分参考。
- 在弱候选边界之间组织出若干相对完整的候选结构单元。

### 第二阶段：判断单元之间的结构关系

- 比较候选单元之间是否构成严格重复、核心重复、发展回应或仅局部相似。
- 将单元关系与边界完整性证据回写到三层分数：
  - `motif_score`
  - `subphrase_score`
  - `phrase_score`

### 方案选择理由

不采用“只强化重复匹配”的轻量方案，因为它无法完整吸收本次提出的结构原则；也不采用全局最优分段方案，因为它改动过重、风险高。本次选择在当前 note-level 框架之上插入“候选单元层”，既能保留已有实现，又能把结构语义补齐。

## 6. 数据结构调整

### 6.1 保留的主体结构

以下结构继续保留：

- `NoteInfo`
- `BoundaryFeature`
- `HierarchicalBoundaryScore`
- `PhraseAnalysis`
- `PhraseSpan`
- `PhraseBoundary`

### 6.2 新增分析起点元数据

需要为 `PhraseAnalysis` 增加单独的分析起点字段，例如：

```python
@dataclass(frozen=True)
class AnalysisAnchor:
    """表示分析起点，仅用于说明结构分析从哪里开始，不表示真实乐句边界。"""

    bar_index: int
    anchor_pos: int
```

`PhraseAnalysis` 扩展为：

```python
@dataclass(frozen=True)
class PhraseAnalysis:
    bars: tuple[BarInfo, ...]
    notes: tuple[NoteInfo, ...]
    boundary_features: tuple[BoundaryFeature, ...]
    boundary_scores: tuple[HierarchicalBoundaryScore, ...]
    boundaries: tuple[PhraseBoundary, ...]
    phrase_spans: tuple[PhraseSpan, ...]
    analysis_start: AnalysisAnchor | None
```

语义约束如下：

- `analysis_start` 表示分析起点。
- `boundaries` 只包含真实结构边界。
- `boundaries` 不再强制包含首个有内容小节的起始点。

### 6.3 新增单元级关系字段

建议在 `BoundaryFeature` 或单独中间结构中增加以下字段：

- `unit_completeness_score`
- `unit_relation_type`
- `unit_relation_score`
- `core_overlap_ratio`
- `developmental_similarity_score`

若继续沿用 `BoundaryFeature` 承载，可扩展为：

```python
@dataclass(frozen=True)
class BoundaryFeature:
    """相邻音符之间的边界特征，既包含局部特征，也包含单元关系特征。"""

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

## 7. 候选结构单元模型

### 7.1 基本原则

候选结构单元不是任意长度的滑动片段，而是“听感上可能构成相对完整单位”的窗口。生成时应优先考虑：

- 相对清晰的起始动机。
- 内部延续性。
- 末端的停顿感、收束感、时值拉长、节拍稳定、音级稳定、局部密度变化或语气转折。

### 7.2 单元生成方式

建议沿用现有 note-level 候选点，但不把它们直接当成最终边界，而是作为切分参考：

- 先收集局部强信号点：
  - `gap_break`
  - `duration_release`
  - `cadence`
  - `sequence_stop`
  - 较强的 `motive_end / repeat_end`
- 用这些点生成多个候选单元窗口：
  - 前后相邻窗口
  - 部分重叠窗口
  - 跨越短尾饰、引入音的扩展窗口

### 7.3 单元完整性评分

为每个边界前后的单元计算 `unit_completeness_score`，重点考虑：

- 起点是否自然。
- 末端是否具有收束或停顿感。
- 单元内部是否连贯。
- 单元是否过短到只剩局部。
- 单元是否因为截取过度而失去完整性。

完整性不足时，即使局部相似度高，也不应直接抬高为正式乐句边界。

## 8. 单元关系分类

### 8.1 关系类型

单元之间的关系分为以下几类：

- `exact_repeat`
  - 完整单元在节奏骨架、相对音高骨架、轮廓、时值组织上高度一致。
- `core_repeat`
  - 完整单元整体不完全一致，但剔除首尾少量非核心成分后，中间核心段明显重复。
- `developmental_response`
  - 不构成高相似度严格重复，但共享相同或相近的起始动机，后续形成发展、变形、模进、延展或回应关系。
- `local_similarity_only`
  - 仅存在局部相似、偶然同头、短小片段接近，不构成结构对应。
- `none`
  - 不存在有意义的单元关系。

### 8.2 核心重复修正

对 `core_repeat` 的判断规则如下：

- 允许裁掉首尾少量非核心成分。
- 核心重复段必须保留在单元中部或主体位置。
- 核心重复段长度必须达到单元主体比例阈值，目标值约为 `60%`。
- 若裁剪后仅剩短小局部，则降为 `local_similarity_only` 或 `none`。

### 8.3 发展回应关系

对 `developmental_response` 的判断需要同时满足：

- 起始动机相同或相近。
- 后续材料虽然变化，但保持可识别的延展逻辑，例如：
  - 模进
  - 节奏扩展
  - 音型变形
  - 结构回应
- 前后两个单元各自都相对完整。

这类关系的目标是解决“不是严格重复，但显然是并列、回应或发展”的乐句。

### 8.4 仅局部相似的降权规则

若仅满足以下弱关系，不应认定为正式对应乐句：

- 局部音高接近
- 偶然同头
- 极短片段相似
- 只有轮廓相似但后续发展完全不同

这类情况最多提供弱 `motif` 提示，不直接支持 `phrase`。

## 9. 与现有三层评分的整合

### 9.1 保留现有局部特征

以下局部特征继续保留，并作为第一层证据：

- `gap_break_score`
- `duration_release_score`
- `cadence_score`
- `motive_end_score`
- `repeat_start_score`
- `repeat_end_score`
- `sequence_stop_score`
- `continuity_penalty`
- `bar_hint_score`
- `sequence_role`

### 9.2 新增单元级证据

新增单元级证据作为第二层证据：

- `unit_completeness_score`
- `unit_relation_type`
- `unit_relation_score`
- `core_overlap_ratio`
- `developmental_similarity_score`

### 9.3 三层评分语义

三层边界分数的职责进一步明确：

- `motif_score`
  - 更关注局部动机收束、弱边界和提示性对应。
- `subphrase_score`
  - 明显吸收“完整单元 + 单元关系成立”的证据。
- `phrase_score`
  - 只有在完整性、边界感和结构关系三者同时成立时才应显著升高。

### 9.4 映射原则

建议采用以下映射规则：

- `exact_repeat`
  - 强增强 `subphrase_score` 与 `phrase_score`
- `core_repeat`
  - 中强增强 `subphrase_score` 与 `phrase_score`
- `developmental_response`
  - 中度增强 `subphrase_score`，在完整性高且边界感明确时可增强 `phrase_score`
- `local_similarity_only`
  - 只允许形成弱 `motif` 或低等级 `subphrase` 提示
- `none`
  - 不提供结构关系加分

### 9.5 Phrase 判定约束

`phrase_score` 的成立不再仅仅依赖局部分数升高，而需要满足以下组合倾向：

- 存在相对完整的前后结构单元。
- 存在明确边界感，例如停顿、收束、时值释放、节拍稳定、音级稳定或语气转换。
- 两个单元之间存在并列、回应、发展或重复中的至少一种结构关系。

## 10. 后处理规则

### 10.1 保留 sequence 约束

继续保留以下约束：

- `sequence_inside` 不能直接升格为 `phrase`
- `sequence_stop` 可以强化 `subphrase`
- 只有同时叠加更强边界证据时，`sequence_stop` 才能支撑 `phrase`

### 10.2 局部相似降权

若某个候选命中的是 `local_similarity_only`：

- 即使局部相似度较高，也不得直接升为正式 `phrase`
- 只能保留为 `motif` 或弱 `subphrase`

### 10.3 单元不完整降权

若候选边界前后单元完整性不足：

- 即使出现局部重复，也要降低其最终层级
- 防止把“只在截取后才相似”的零散局部误判成对应乐句

### 10.4 第一真实结构边界不强制补出

不再保留“首句强制边界”规则：

- 前段允许只存在分析起点而没有真实结构边界
- 直到第一次出现足够强的结构转换证据才产生首个正式边界

## 11. 输出语义调整

### 11.1 `analysis.boundaries`

新语义：

- 只包含真实结构边界
- 不再自动包含首个有内容小节

### 11.2 `analysis_start`

新语义：

- 表示分析起点
- 用于 review 展示、span 推导和解释分析起始位置
- 不属于结构边界集合

### 11.3 `phrase_spans`

`phrase_spans` 不再假设第一个边界就是起点，改为：

- 从 `analysis_start` 开始，到第一个真实边界之前，形成第一个候选区间
- 后续区间由相邻真实边界切分
- 若整段都没有真实结构边界，则允许只形成一个从 `analysis_start` 到结尾的整体区间

### 11.4 `inject_phrase_tokens()`

调整为：

- 只依据真实 `phrase` 边界注入 `PHRASE`
- 不因为分析起点自动注入

## 12. Review 与可视化调整

### 12.1 序列化内容

`serialize_phrase_analysis()` 需要新增或同步输出：

- `analysis_start`
- `unit_completeness_score`
- `unit_relation_type`
- `unit_relation_score`
- `core_overlap_ratio`
- `developmental_similarity_score`

### 12.2 中文说明标签

需要为 review 层提供更直观的中文解释标签，例如：

- `完整单元重复`
- `核心重复`
- `起始动机呼应并发展`
- `仅局部相似，未升格`
- `停顿收束成立`
- `分析起点`

### 12.3 Viewer 展示

viewer 需要明确区分：

- 分析起点
- 真实结构边界

边界表格中需要展示：

- 边界三层分数
- 单元完整性
- 单元关系类型
- 核心重叠比例
- 发展相似度
- 命中原因

## 13. 测试策略

### 13.1 单元关系测试

需要补充以下测试：

- 完整重复单元可提升正式边界
- 首尾带少量装饰音的重复仍可被识别为 `core_repeat`
- 仅有局部相似或偶然同头时不会被误判为正式对应乐句
- 共享起始动机且后续发展明显的单元可识别为 `developmental_response`

### 13.2 起点语义测试

需要补充以下测试：

- `analysis_start` 单独存在
- `analysis.boundaries` 不再强制包含起点
- `inject_phrase_tokens()` 不再在起点自动注入 `PHRASE`

### 13.3 首句无强证据测试

需要补充以下测试：

- 第一段没有强结构证据时，可以不存在首个正式边界
- 直到后续首次出现显著结构转换后才形成第一个边界

### 13.4 Review 输出测试

需要补充以下测试：

- `analysis_start` 已序列化
- 单元关系字段已序列化
- viewer 依赖的新字段存在
- 中文来源标签与原因映射正确

## 14. 风险与控制

### 14.1 主要风险

- 候选单元生成过宽，导致比较成本上升或误配增多。
- `developmental_response` 定义过松，导致非对应片段被误升格。
- 去掉首句强制边界后，某些数据可能在前半段长时间没有正式边界。

### 14.2 控制方式

- 保留现有局部特征体系作为约束，不完全放弃 note-level 证据。
- 对 `core_repeat` 和 `developmental_response` 增加严格最小主体比例与完整性阈值。
- 在 review 中完整展示结构关系与原因，方便人工回看与调参。
- 通过新增测试明确约束“无强证据时允许无边界”，避免后续逻辑偷偷恢复首句强制规则。

## 15. 成功标准

满足以下条件即可认为本次设计落地成功：

- 乐句判断从“零散局部优先”切换为“完整单元优先”。
- 核心重复能够容忍少量首尾非核心成分，不再机械要求整段逐音一致。
- 发展、模进、回应类乐句关系可以被识别为正式结构关系，而不再依赖严格重复。
- 仅有表面局部相似的片段不会被误判为正式对应乐句。
- 段落起始位置被明确降级为分析起点，不再伪装成真实乐句边界。
- 最终 `PHRASE` 注入只反映真实结构边界。
