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
