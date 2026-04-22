# TODO 模块

本文档用于集中维护当前项目中已确认、但尚未执行的中期工作项。各项内容以“背景、待完成事项、启动条件、实施范围”的格式记录，便于后续持续扩展与跟踪。

## 1. Phrase Metadata 下沉至 Tokenizer

### 背景说明

当前乐句相关能力仍保留在训练阶段的动态 window view 中，暂不下沉到 tokenizer 静态产物。这样处理的主要原因是：乐句边界 heuristic、窗口采样比例以及 FIM hole 策略仍处于迭代阶段，若过早固化到底层 tokenized 数据格式，会显著放大后续规则调整的数据重建成本。

### 待完成事项

- 评估是否需要在 tokenizer 阶段额外输出 phrase metadata / phrase index；
- 明确 metadata 的最小必要字段，例如 bar 索引、候选边界、phrase span、phrase-start tempo；
- 设计训练、评估、前端共同复用的 metadata 读取方式；
- 确保 tokenizer 仍然保留原始全曲 token 序列，不将 phrase window 直接预展开为完全独立的静态样本；
- 在完成下沉前，为 `src/music_analysis/` 保持唯一的乐句边界与提取逻辑来源，避免多套规则漂移。

### 启动条件

在满足以下条件之前，不建议启动 tokenizer 下沉工作：

- 连续 1 到 2 次 `full` 训练与 benchmark 结果保持稳定，`40/40/20` 的单句 / 跨句 / 长窗分布不再频繁调整；
- 乐句视图协议基本冻结，即维持 `BOS + 当前乐句 tempo + BAR ... + EOS` 的表示方式，且不再频繁修改 tempo、bar、phrase span 的构造规则；
- 乐句切分结果不再仅服务训练，而是同时被评估流程、数据处理流程或前端产品复用。

### 实施范围

后续若执行下沉，范围建议限定为以下内容：

- tokenizer 继续输出原始全曲 token 序列；
- 额外产出可复用的 phrase metadata / phrase index；
- 训练、评估、前端统一复用 `src/music_analysis/` 中的乐句边界与提取逻辑；
- 保留训练阶段的动态 window 构造能力，不将所有 phrase window 固化为静态训练样本。

### 建议实施节点

1. 完成当前一轮 `full` 训练与评估。
2. 基于结果进行 1 次小幅调参。
3. 若第二轮结束后乐句规则基本稳定，再启动 phrase metadata 下沉至 tokenizer 阶段的工作。

## 2. Chord Token 引入评估与分阶段落地

### 背景说明

当前 token 体系仍以音符事件为主，核心结构是 `BAR + POS + INST + PITCH + DUR + VEL`，其中 `INST` 现阶段固定为 `PIANO`。这套表示在结构合法性上已经可用，但对显式和声信息几乎没有建模入口，模型只能从局部音高共现里隐式猜测 harmony。

从现有 benchmark 看，当前薄弱项主要集中在“乐句连贯性”和“长上下文稳定性”，而不是基础可训练性或补全完整性。因此，引入 chord token 更适合作为“和声条件增强”与“长程约束增强”，而不应被视为解决语法合法率问题的主手段。

结合当前数据现状，chord token 具备可行性，但建议采用“低频、稀疏、可回退”的设计：

- 不把 chord 拆成新的主事件流，不按 note 级别高频插入；
- 优先采用每小节或每次和声变化插入 1 个 chord token 的稀疏方案；
- 先限制 chord 词表规模，例如 `CHORD_{ROOT}_{QUALITY}` + `CHORD_NC`，暂不在首版引入复杂 slash/bass 扩展；
- 保持原始 note token 序列不变，使 chord token 更像可选控制层，而不是替换底层音符表达。

### 待完成事项

- 设计 chord 表示协议，优先评估以下候选：
  - `BAR [TEMPO_x] [CHORD_x] POS ...`
  - 仅在和声变化处插入 `CHORD_x`
  - 保留 `CHORD_NC` / `CHORD_UNKNOWN` / `CHORD_AMBIG` 兜底类别
- 实现离线 chord 标注与统计脚本，至少输出：
  - chord 覆盖率
  - chord 置信度分布
  - 每首平均 chord 数
  - chord 变化频率
  - 高频类别分布与长尾占比
- 评估 chord 提取所需输入是否需要扩展：
  - 当前 `NoteEvent` 仅保留起止时间、pitch、velocity，若 chord heuristic 需要 track/program 线索，再决定是否补充；
  - 若仅依赖聚合后的 simultaneity / pitch-class 集合即可稳定提取，则优先不改底层数据结构。
- 为 tokenizer / 训练 / 评测同步补齐 chord 感知能力：
  - tokenizer 词表构建与顺序校验
  - `tokens_to_midi` 对 chord token 的忽略式反编译
  - 训练窗口切分、FIM maskable unit、benchmark continuation/infilling 切分规则
  - 诊断指标中新增 chord 对齐类统计
- 先做 shadow 产物而非直接替换主训练集：
  - 在不进入正式训练的前提下，先导出带 chord 的 `.tok` 或 sidecar metadata；
  - 先验证 token 增幅、标注噪声和 roundtrip 兼容性，再决定是否正式进训练。

### 启动条件

在满足以下条件前，不建议直接把 chord token 合入默认训练主线：

- 当前 benchmark 的主要短板仍然是和声连贯性，而不是语法合法率；若后续瓶颈转回 EOS / syntax / decode 停机，则 chord 优先级应后移；
- chord 提取在代表性样本上具备可接受稳定性，不能大量产出错误或高歧义标签；
- 稀疏插入方案的 token 增幅可控，不能明显恶化现有训练吞吐与数据覆盖；
- benchmark 与导出 MIDI 链路已经明确 chord token 为“可忽略控制信息”，不会破坏现有 roundtrip 与试听流程；
- 至少完成一轮小规模 A/B，对照无 chord 基线，确认收益主要落在和声与长程指标，而非只带来 vocab 膨胀。

### 实施范围

若后续启动 chord token 工作，建议范围限定为以下内容：

- 首版只做稀疏 chord token，不改 note event 五元组主体；
- 首版只支持有限 chord 类别集合，不做复杂转位、slash chord、借和弦细分爆炸；
- chord token 仅作为附加条件，不要求 `tokens_to_midi` 反编译出独立伴奏轨；
- 默认保持与无 chord 配置兼容，可通过 tokenizer 配置开关启用或关闭；
- benchmark 新增 chord-aware 指标，但保留当前结构合法率、时间顺序合法率等主指标不变，避免评测体系被单一新特性绑架。

### 建议实施节点

1. 先完成 chord heuristic 的离线分析与 shadow 数据统计，不进入正式训练。
2. 若覆盖率、歧义率、token 增幅都在可接受范围内，再建立一版小规模 chord vocab 与 tokenizer 原型。
3. 用 `small` 训练做 A/B，对比以下方向是否改善：
   - continuation 首事件命中率
   - 时值分桶 L1 距离
   - 音高多样性 / pitch collapse 指标
   - 生成事件数偏差均值
4. 若小规模 A/B 有稳定正收益，再进入 `full` 训练验证。
5. 只有在 `full` benchmark 证明收益稳定后，才将 chord token 纳入默认训练路线。

## 3. 丰富性导向训练目标评估与分阶段落地

### 背景说明

当前 `train_base` 仍是标准 teacher-forcing next-token cross entropy，训练期直接优化的是“给定真实前缀时，下一个 token 的条件概率”。这对结构合法性、收敛稳定性和基础续写能力是合理起点，但它天然更偏向学习“平均意义上最安全的局部延续”，不直接约束自由生成时的内容丰富性。

从现有实现看，benchmark 已经补进了 `pitch collapse`、`low_density_bar_rate`、`longest_same_pitch_run_ratio` 等“结构没坏但内容塌缩”的诊断信号；但训练目标本身还没有显式惩罚：

- 音高分布过窄，只围绕少数 pitch / 和弦反复打转；
- 节奏过于保守，过多落在 bar 起点、整拍或低密度位置；
- 局部 pattern 机械重复，例如 pitch / duration / onset 的短 n-gram 重复到底。

这说明“丰富性”方向具备工程价值，但不建议直接把 decode 后的 Pitch Collapse Penalty、Rhythm Complexity Loss、n-gram Repetition Loss 一次性并入默认主 loss。主要原因是：

- 这些信号大多是序列级、decode 后信号，而当前训练是 teacher-forcing；若每个 batch 先自由生成再算惩罚，训练开销和不稳定性都会显著上升；
- 离散 token decode 后再做惩罚不可微，若不引入 RL / sequence-level estimator，只能走近似或 surrogate loss；
- 音乐里存在合法重复，例如 ostinato、pedal point、稳定节奏型；若惩罚设计不做 target-relative 约束，容易把“有风格的重复”错杀成退化。

因此，这个方向“可行”，但更适合按“先补齐诊断与基线，再做轻量辅助 loss”的方式推进。

### 待完成事项

- 先补齐 richness 方向的 benchmark / shadow 指标，而不是先改主 loss：
  - 节奏丰富性：例如 onset position entropy、强拍占比、整拍占比、duration diversity；
  - 重复度：例如 pitch n-gram / duration n-gram / onset-position n-gram 重复率；
  - 若后续有 chord metadata，再补 chord change rate / harmonic repetition 指标。
- 对训练集、验证集、benchmark case 先做离线统计，建立“目标分布基线”：
  - pitch 多样性分布；
  - duration / onset 分布；
  - bar 内密度分布；
  - 高频重复 pattern 的自然出现频率。
- 优先评估不改 loss 或轻改 loss 的高性价比方案：
  - 基于现有 phrase / density 特征做训练采样重加权，而不是直接上复杂序列奖励；
  - 提高高密度、跨句、长上下文、节奏更活跃窗口的采样概率；
  - 对明显塌缩片段降低采样权重，但不直接删除，避免风格分布失真。
- 若采样重加权收益不够，再引入轻量辅助 loss，优先考虑：
  - anti-repetition unlikelihood：仅对“不在 target 中出现的重复 token / n-gram”施加惩罚；
  - target-relative pitch diversity surrogate：仅在目标片段本身具有较高 pitch 多样性时，抑制模型把分布塌到单一 pitch；
  - target-relative rhythm surrogate：对 `POS_*` / `DUR_*` 的预测分布与目标窗口的 onset / duration 统计做轻量对齐，而不是机械鼓励“越复杂越好”。
- 暂不把以下方案作为 v1 默认路线：
  - 每步训练都先自由生成再算 sequence reward；
  - 依赖 RL / policy gradient 的非稳定训练；
  - 未经 chord metadata 支撑的“和弦重复惩罚”。
- 训练日志与评估报告补充 richness 相关分项，避免只看总 loss：
  - `aux_pitch_loss`
  - `aux_rhythm_loss`
  - `aux_repetition_loss`
  - A/B 配置差异与吞吐影响

### 启动条件

在满足以下条件前，不建议把丰富性辅助 loss 合入默认训练主线：

- 先确认当前问题确实主要表现为“内容安全但塌缩”，而不是 decode 约束、EOS、时序合法率或数据分布问题；
- 至少先有一版稳定的 richness benchmark / shadow 指标，否则训练会优化一个尚未量化的目标；
- v1 辅助 loss 必须能在 teacher-forcing 下直接计算，不依赖每 batch 额外 rollout；
- 训练吞吐下降需控制在可接受范围内，优先目标是不明显恶化当前 `small/full` 训练节奏；
- 小规模 A/B 需要同时证明 richness 指标改善且结构指标不回退，例如：
  - `pitch_diversity_score` 上升；
  - `most_common_pitch_ratio` / `longest_same_pitch_run_ratio` 下降；
  - 新增 rhythm / repetition 指标改善；
  - continuation / infilling structural validity 不明显变差。

### 实施范围

若后续启动丰富性训练目标工作，建议范围限定为以下内容：

- v1 仍以 CE 为主损失，辅助 loss 只做小权重加和，不替代主目标；
- v1 只做 teacher-forcing 可计算的 surrogate loss，不引入 rollout-based sequence reward；
- anti-repetition 首选 unlikelihood 或 target-conditioned repetition penalty，不做生硬全局禁重复；
- rhythm 丰富性首选 target-relative 分布约束，不把“复杂”误当成绝对更优；
- harmonic repetition 相关工作优先依赖 chord metadata / chord token 的 shadow 能力，暂不从当前 note token 序列硬推高噪声和弦标签；
- 所有辅助项都通过配置开关启用，默认保持与当前训练配置兼容。

### 建议实施节点

1. 先在 benchmark / 离线统计层补齐 rhythm richness 与 repetition 指标，不改训练。
2. 基于现有 `phrase_analysis` 的密度与边界特征，先做一轮“采样重加权” A/B，验证是否已经能缓解安全塌缩。
3. 若采样方案收益有限，先上单一辅助项，优先推荐 `anti-repetition unlikelihood`，不要一次同时引入 pitch / rhythm / repetition 三个 loss。
4. 第二阶段再评估 target-relative 的 pitch / rhythm surrogate loss，并记录吞吐、收敛和 benchmark 影响。
5. 只有在 `small` 与 `full` 都证明 richness 改善稳定且结构指标不退化后，才考虑把丰富性辅助 loss 纳入默认训练路线。
