# MIDI Infilling Transformer 设计文档（v1.3）

## 1. 项目目标
- 构建基于 Transformer 的 Symbolic MIDI 生成模型。
- 长期支持三种生成模式：中间补全（Infilling）、续写到结尾（Continuation）、从头生成（Free Generation）。
- 当前阶段采用 `NEXT + FIM` 混合训练：NEXT 作为主任务，FIM 作为中间编辑辅助任务。
- 首版先不启用 `STYLE_x`，待风格标注数据就绪后再引入风格控制。
- 建立客观评估体系，主决策不依赖主观听感。

## 2. 非目标
- 非目标：
  - 不做音频生成（仅符号级 MIDI）
  - 不做完整 DAW 插件（先离线推理）
  - 不做过度兼容（先支持主流 MIDI 事件）

## 3. 总体架构
```
MIDI Dataset
  -> 清洗与切分
  -> Tokenizer（结构化事件序列）
  -> 乐器归类（生成 `INST_x`，先于 style）
  -> Base Model（阶段1: NEXT + FIM 混合训练）
  -> Base Model（阶段2: 多任务扩展训练）
  -> Style LoRA 微调
  -> Inference（补全 / 续写 / 直接生成）
```

### 3.1 基座模型实现策略（明确约束）
- 采用 Qwen 架构实现：`Qwen2ForCausalLM`（或 `Qwen2.5ForCausalLM`）。
- 仅复用“模型结构与实现方式”，不使用官方预训练权重。
- 初始化方式：使用配置随机初始化（如 `from_config`），不使用 `from_pretrained`。
- 代码落地方式：将对应源码复制到本地仓库 `src/model/` 维护，后续可按项目需求修改。
- 目标：保留未来在注意力、位置编码、多轨建模等方向上的自主改造能力。

### 3.2 工程化定义（输入/输出/验收）
- 输入：
  - 上游参考实现（Qwen2/Qwen2.5 CausalLM 源码）
  - 本地模型配置 `configs/train/model_base*.yaml`
- 输出：
  - 本地模型源码目录 `src/model/`（可直接 import 训练）
  - 初始化脚本与最小前向测试
- 验收标准：
  - 代码中不出现 `from_pretrained(...)` 作为训练入口
  - 随机初始化可完成前向与反向传播
  - 本地源码修改不受外部包升级阻断

## 4. Tokenizer 设计

### 4.1 Token 类型（首版）
- 结构 token：
  - `BAR`
  - `POS_0..31`
  - `INST_{PIANO|GUITAR|BASS|STRINGS|LEAD|PAD}`
  - `PITCH_21..108`
  - `DUR_{1,2,3,4,6,8,12,16,24,32}`
  - `VEL_0..15`（16 档，非线性分桶：中间更细、两端更稀）
- 控制 token：
  - `TEMPO_x`（特殊控制 token）
- 特殊 token（阶段1）：`BOS` `EOS` `FIM_HOLE` `FIM_MID`

阶段1Token 约束：
- `TEMPO_x` 只允许在两种位置出现：`BOS` 后的开头位置，或 `BAR` 后（仅当发生速度变化时）。
- 音符事件统一为四元组顺序：`INST_x PITCH_x DUR_x VEL_x`。
- `INST_x` 首版仅启用 6 类：`PIANO` `GUITAR` `BASS` `STRINGS` `LEAD` `PAD`。
- `STYLE_x` 暂不启用；阶段2后按数据集可用性引入，并作为 `BOS` 后的可选单次控制 token。
- `DUR` 在数据处理时采用“就近映射”到上述 10 个常用档位（含 triplet：`3/6/12/24`）。
- `VEL` 使用“中心对称 μ-law 压扩 + 16 档均匀量化”（中间更细、两端更稀），固定参数：
  - `μ = 8`，`c = 64`（中心），`r = 63`（半幅）
  - 编码（MIDI velocity `v in [1,127]` -> `VEL_k`）：
    - `x = (clip(v,1,127) - c) / r`
    - `s = sign(x) * ln(1 + μ*abs(x)) / ln(1 + μ)`
    - `k = clip(round(((s + 1) / 2) * 15), 0, 15)`
  - 解码（`VEL_k` -> velocity）：
    - `s_hat = 2*k/15 - 1`
    - `x_hat = sign(s_hat) * (((1 + μ)^abs(s_hat) - 1) / μ)`
    - `v_hat = clip(round(c + r*x_hat), 1, 127)`
  - 约束：数据中若出现 `velocity=0`（常见于 note-off 事件）不参与音符力度建模；若被写入音符力度字段则先夹到 `1` 再编码。

阶段2扩展（按需引入）：
- 任务 token：`TASK_INFILL` `TASK_GEN`

### 4.2 序列格式
```
BOS TEMPO_120
BAR POS_0 INST_PIANO PITCH_60 DUR_4 VEL_10
POS_4 INST_PIANO PITCH_62 DUR_3 VEL_9
BAR TEMPO_128
POS_0 INST_LEAD PITCH_67 DUR_8 VEL_11
EOS
```

### 4.3 工程化定义（输入/输出/验收）
- 输入：
  - 清洗后的标准 MIDI（单文件）
  - `tokenizer.yaml`（网格、力度分桶、pitch 范围）
- 输出：
  - `*.tok`（token 序列）
  - `tokenizer_vocab.json`
  - `token_stats.json`（长度分布、OOV=0）
- 验收标准：
  - 100% 样本可编码；解码回 MIDI 不报错
  - 词表规模在 `200~320`
  - token 非法顺序率 `< 1%`

### 4.4 乐器 Token 引入顺序（先于 Style）
- `INST_x` 的构建放在 style 处理之前，先保证“乐器语义正确”，再做 `STYLE_x`。
- 首阶段清洗/切分可先聚焦主轨（如 Piano/Lead）保证数据干净，后续再扩展多轨并映射到 6 类乐器。
- 在 `STYLE_x` 引入前，`tokenize_dataset.py` 先完成 `INST_x` 的词表与样本写入。

## 5. 数据工程

### 5.1 数据来源
- Lakh MIDI Dataset
- GiantMIDI
- Groove MIDI Dataset
- BitMIDI（补充）

### 5.2 清洗规则
- 音符数 `>= 100`
- 时长 `>= 10s`
- 至少 `1` 条有效旋律轨
- BPM 在 `40~220`
- 去重：标准化事件序列哈希去重

### 5.3 数据切分
- `train/valid/test = 90/5/5`
- 同曲目或同哈希族不跨集合

### 5.4 工程化定义（输入/输出/验收）
- 输入：
  - `data/raw/**/*.mid`
  - `cleaning.yaml`
- 输出：
  - `data/clean/`
  - `data/base/{train,valid,test}.jsonl`
  - `data/style/rnb_train.jsonl`
  - `data/eval/fixed_eval.jsonl`
- 验收标准：
  - 过滤率统计齐全（含被过滤原因分布）
  - `eval` 固定且不可被训练脚本读取
  - 不存在重复哈希跨集合泄漏

## 6. Pipeline 流程

### 6.1 流程图
```
raw MIDI
  -> clean_dataset.py
  -> split_dataset.py
  -> tokenize_dataset.py（含 INST_x 归类）
  -> build_training_data.py
  -> train.bin / train.idx
```

### 6.2 分步 I/O 与验收
| 步骤 | 脚本                     | 输入                        | 输出                                        | 验收标准                     |
| ---- | ------------------------ | --------------------------- | ------------------------------------------- | ---------------------------- |
| 1    | `clean_dataset.py`       | `data/raw`                  | `data/clean` + `clean_report.json`          | 可用样本率、过滤原因统计完整 |
| 2    | `split_dataset.py`       | `data/clean`                | `data/base/style/eval` 切分清单             | 无跨集合泄漏                 |
| 3    | `tokenize_dataset.py`    | 切分清单 + `tokenizer.yaml` | `data/tokenized/*.tok` + `token_stats.json` | OOV=0，非法顺序率达标        |
| 4    | `build_training_data.py` | `data/tokenized/*.tok`      | `data/tokenized/train.bin/.idx` 等          | 可被训练脚本直接加载         |

### 6.3 失败处理
- 任一步骤失败即停止，不串行跳过。
- 失败重试前必须保留 `outputs/logs/<step>/<run_id>.log`。
- 中间产物带 `run_id`，避免污染上次结果。

## 7. 任务发展规划（分阶段）

### 7.1 阶段1（当前）：NEXT 主任务 + FIM 辅助
- NEXT（主任务）：
  - 输入：`<PREFIX>`
  - 输出：`<NEXT_TOKENS>`
- FIM（辅助任务）：
  - 输入：`<PREFIX> + FIM_HOLE + <SUFFIX>`
  - 输出：`<MIDDLE>`
- 训练目标：优先稳定续写能力，同时补齐中间编辑能力。
- 对应评估：
  - `scripts/eval/eval_all.py`：统一串行执行阶段1所需的两类评估
  - `scripts/eval/eval_continuation.py`：覆盖 NEXT 主任务的续写行为
  - `scripts/eval/eval_infilling.py`：覆盖 FIM 的中间编辑行为

训练样本格式（阶段1）：
NEXT：
```
BOS [TEMPO_120] <SEQUENCE> EOS
```

FIM：
```
BOS [TEMPO_120] <PREFIX> FIM_HOLE <SUFFIX> FIM_MID <MIDDLE> EOS
```

### 7.2 阶段2（后续）：扩展 Free Generation / 条件控制
- Continuation 行为仍由阶段1的 NEXT 统一承担，不单独新增 `TASK_CONT` 训练任务。
- Free Generation：
  - 输入：`BOS`（可选 `STYLE_x`，可选 `TEMPO_x`）
  - 输出：`<FULL_SEQUENCE_TO_EOS>`

训练样本格式（阶段2，建议，当前未启用）：
Infilling：
```
TASK_INFILL BOS [STYLE_x] [TEMPO_x] <PREFIX> FIM_HOLE <SUFFIX> FIM_MID <MIDDLE> EOS
```

Free Generation：
```
TASK_GEN BOS [STYLE_x] [TEMPO_x] <FULL_SEQUENCE_TO_EOS> EOS
```

### 7.3 工程化定义（输入/输出/验收）
- 输入：
  - `train.bin/.idx`（阶段1用于 NEXT + FIM 混合采样）
  - `configs/train/train_base_run_small.yaml` / `configs/train/train_base_run_full.yaml`
- 输出：
  - `outputs/checkpoints/base/<run_id>/`
  - `outputs/metrics/base/<run_id>.json`
- 验收标准：
  - 训练过程无 NaN/Inf
  - `task_loss` 稳定下降
  - 阶段1：每个 checkpoint 可自动评估并产出 `valid_loss` / `ppl` / `structural_validity_rate`
  - 阶段1：`regression_check.py` 可一键通过（采样训练 + eval + save/resume）
  - 阶段2：三类任务都能在固定样本上输出可解析结果

### 7.4 `STYLE_x` 引入计划
- 阶段1：不使用 `STYLE_x`（当前数据集无稳定 style 标注）。
- 阶段2：先保持无 `STYLE_x` 的多任务训练，优先验证任务扩展稳定性。
- 阶段3（风格数据就绪后）：引入 `STYLE_x`，位置固定为 `BOS` 后可选单次，并同步更新 `tokenizer_vocab.json` 与样本构造逻辑。

## 8. Mask 策略

### 8.1 策略优先级
1. `Bar Mask`（主策略，整小节）
2. `Phrase Mask`（1~2 小节）
3. `Span Mask`（补充，约 15% token）

说明：
- 阶段1：Mask 策略仅用于 FIM 子样本构造，NEXT 子样本不使用 hole mask。
- 阶段2：`TASK_GEN` 不使用 hole mask；Continuation 若保留为 NEXT 形式，同样不使用 hole mask。

### 8.2 参数约束（建议）
- `bar_mask_prob = 0.6`
- `phrase_mask_prob = 0.3`
- `span_mask_prob = 0.1`
- `max_mask_ratio <= 0.4`

### 8.3 工程化定义（输入/输出/验收）
- 输入：
  - token 序列
  - `masking.yaml`
- 输出：
  - 带 mask 的训练样本
  - `mask_stats.json`（mask 长度、位置分布）
- 验收标准：
  - 不以单 token mask 为主策略
  - 各策略命中分布与配置偏差 `< 5%`
  - 掩码后样本仍可被语法校验器通过

## 9. 训练节奏

### 9.1 基线训练（v0）
- 模型：50M Decoder-only
- 长度：1024
- 优化器：AdamW
- 学习率：3e-4（warmup + cosine）
- 精度：bf16/fp16（按硬件）
- `NEXT + FIM` 混合训练（NEXT 主任务，FIM 辅助任务）
- 配比：`NEXT:FIM = 7:3`（或按 `fim_ratio` 网格实验

### 9.2 单轮实验循环
```
仅改 1 个变量
  -> 训练 10M tokens
  -> 固定验证集评估
  -> 记录结论（保留/回滚）
```

### 9.3 工程化定义（输入/输出/验收）
- 输入：
  - 训练配置（`configs/train/*.yaml`）
  - 固定数据快照 ID
- 输出：
  - `outputs/checkpoints/`、`outputs/metrics/`、`outputs/logs/`
  - `experiments.csv`（配置+结果）
- 验收标准：
  - 每轮实验可复现（同 seed 偏差在可接受范围）
  - 每轮有明确结论：保留/回滚/继续观察

## 10. 验证体系设计

### 10.1 指标
- 当前自动评估闭环指标（已落地）：
  - `valid_loss`
  - `ppl`
  - `structural_validity_rate`

- `Infilling Success Rate`：合法补全比例
- `Constraint Violation Rate`：非法 token/顺序比例
- `Masked Perplexity`：仅补全区间（Infilling）
- 结构指标：小节完整率、节拍覆盖率、Pitch 分布偏差

阶段2扩展指标：
- `Continuation Success Rate`：合法续写比例
- `Generation Success Rate`：合法整段生成比例
- `Continuation Perplexity`：续写区间
- `Generation Perplexity`：整段生成

### 10.2 v0 验收门槛
- Infilling 合法率 `>= 98%`
- 约束违规率 `<= 1%`
- 相对随机/检索基线有显著提升

阶段2门槛（建议）：
- Continuation 合法率 `>= 98%`
- Generation 合法率 `>= 97%`

### 10.3 工程化定义（输入/输出/验收）
- 输入：
  - `data/eval/fixed_eval.jsonl`
  - 待评估 checkpoint
- 输出：
  - `outputs/reports/eval/<run_id>.json`
  - `outputs/reports/eval_continuation/<run_id>.json`
- 验收标准：
  - 每次评估自动生成同结构 JSON 报告
  - 支持统一入口 `scripts/eval/eval_all.py` 一条命令跑完整阶段1评估
  - 默认支持按 run 下所有 checkpoint 逐个评估
  - 指标可追溯到 checkpoint、配置、数据版本

## 11. 模型发展路线
1. `v0`（50M）：打通端到端 pipeline，达到基础验收门槛
2. `v1`（100M）：提升稳定性与补全质量
3. `v2`（150M，可选）：评估规模收益与成本拐点

工程化验收：
- 每次升规模前，先完成上一版本复盘文档（收益/风险/资源成本）。

## 12. 风格模型（LoRA）

### 12.1 架构
```
Base Model + LoRA(R&B)
```

### 12.2 数据与训练
- 数据：约 2000 首 R&B MIDI（约 `20M~30M` tokens）
- 训练目标：保持结构合法前提下提升风格一致性
- 训练时间：单卡约 `6~10` 小时

### 12.3 工程化定义（输入/输出/验收）
- 输入：
  - Base checkpoint
  - R&B 风格数据切分
  - `train_lora_rnb.yaml`
- 输出：
  - `outputs/checkpoints/lora_rnb/<run_id>/adapter.safetensors`
  - `outputs/metrics/lora_rnb/<run_id>.json`
- 验收标准：
  - 结构合法性不低于 Base 的 99%
  - 风格分类器或风格相似度指标有提升

## 13. 项目结构
```
TuneFlow/
  configs/
    tokenizer/
    data/
    train/
    eval/
  scripts/
    data/
      clean_dataset.py
      split_dataset.py
      tokenize_dataset.py
      build_training_data.py
      build_data.py
      validate_data_outputs.py
    train/
      train_base.py
      train_base_from_config.py
      regression_check.py
      train_lora.py
    eval/
      eval_all.py
      eval_infilling.py
      eval_continuation.py
  src/
    model/
      modeling.py
      configuration.py
    tokenizer/
    training/
    utils/
  data/
    raw/
    clean/
    base/
    style/
    eval/
    tokenized/
  outputs/
    checkpoints/
      base/
      lora_rnb/
    logs/
    metrics/
    reports/
```

目录验收：
- 训练脚本只读 `data/eval`，不可写入。
- `outputs/checkpoints`、`outputs/logs`、`outputs/metrics` 按 `run_id` 隔离。

## 14. 工程化规范（必须执行）

### 14.1 配置管理
- 统一使用 YAML 配置，不在代码里硬编码关键超参。
- 配置变更必须进入实验记录。
- Base 训练默认配置路径：`configs/train/train_base_run_small.yaml`。
- 完整规模训练模板路径：`configs/train/train_base_run_full.yaml`。
- 训练入口统一为：`scripts/train/train_base_from_config.py`。

### 14.2 实验记录
- 必填字段：
  - `run_id`
  - `git_commit`（如有）
  - `data_snapshot`
  - `config_path`
  - `core_metrics`
  - `decision`

### 14.3 完成定义（DoD）
- 一个实验任务完成必须同时满足：
  - 有 checkpoint
  - 有结构化指标报告（至少包含 infilling 与 continuation 两类评估）
  - 有结论（保留/回滚）
  - 有可复现配置
  - 可通过自动回归检查：`scripts/train/regression_check.py`

### 14.4 回滚策略
- 指标未达门槛且无新发现，默认回滚到上个稳定配置。
- 不允许“指标下降但继续叠加改动”。

## 15. 里程碑（建议）
1. M1（1 周）：Tokenizer + 数据 Pipeline 跑通，产出固定 eval 集
2. M2（1~2 周）：v0 50M 完成 `NEXT + FIM` 基线并通过基础验收
3. M3（1 周）：在 M2 稳定后扩展 Continuation / Free Generation
4. M4（1 周）：R&B LoRA 首版可控生成
5. M5（持续）：v1 扩模与指标优化

## 16. 总结
这是一个“音乐版 Copilot”系统：
- 用 Transformer 学 MIDI 语法
- 采用“先 `NEXT + FIM` 稳定基线、后多任务扩展”的演进路线
- 用 LoRA 做风格控制

工程上以“输入/输出/验收标准”驱动执行，先保证可复现，再追求更高质量。

