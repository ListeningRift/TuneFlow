# TuneFlow

TuneFlow 是一个面向 Symbolic MIDI 的生成项目，目标是做成编曲场景里的 Copilot。
当前重点能力包括：
- 中间补全（Infilling）
- 前缀续写（Continuation）
- 从头生成（Free Generation）

## 项目定位
- 任务形态：MIDI Infilling + Continuation + Free Generation
- 能力方向：结构正确、可控、可评估的符号级音乐生成
- 使用场景：旋律续写、片段补全、整段草稿生成、风格化生成

## Roadmap
- M1：打通数据与 Tokenizer 流程
- M2：基础模型达到首版可用
- M3：风格化微调首版可用
- M4：持续优化规模、稳定性与质量

## 文档
- 设计文档：[design.md](./design.md)

## 环境准备
```bash
conda create -n tune-flow python=3.10 -y
conda activate tune-flow
python -m pip install -r requirements.txt
```

## 数据构建
如果已经有完整的 `data/tokenized/*.bin`、`data/tokenized/*.idx.json`、`data/tokenized/tokenizer_vocab.json`，可以直接跳到“配置化训练”。

1. 一键跑完整流程（clean -> split -> tokenize -> build -> validate）
```bash
python scripts/data/build_data.py
```

2. 冒烟模式（只处理少量样本）
```bash
python scripts/data/build_data.py --clean-limit 200 --split-limit 200 --tokenize-limit-per-split 100
```

3. 从中间步骤继续（例如从 tokenize 开始）
```bash
python scripts/data/build_data.py --start-from tokenize --stop-after validate
```

4. 也可分步执行
```bash
python scripts/data/clean_dataset.py --config configs/data/cleaning.yaml
python scripts/data/split_dataset.py --config configs/data/split.yaml
python scripts/data/tokenize_dataset.py --config configs/tokenizer/tokenizer.yaml
python scripts/data/build_training_data.py --config configs/data/build_training.yaml
python scripts/data/validate_data_outputs.py
```

关键产物：
- `data/base/{train,valid,test}.jsonl`
- `data/eval/fixed_eval.jsonl`
- `data/tokenized/{train,valid,test,eval}.tok`
- `data/tokenized/tokenizer_vocab.json`
- `data/tokenized/{train,valid,test,eval}.bin`
- `data/tokenized/{train,valid,test,eval}.idx.json`
- `outputs/reports/data/validate_data_report.json`

## 配置化训练
训练参数统一放到 YAML，并拆成两档：
- `configs/train/train_base_run_small.yaml`：小规模正式训练
- `configs/train/train_base_run_full.yaml`：完整规模训练

推荐按下面顺序执行。

1. 先检查小规模 YAML 是否能正常展开为训练参数：
```bash
python scripts/train/train_base_from_config.py --preset small --dry-run
```

2. 启动训练：
```bash
python scripts/train/train_base_from_config.py --preset small
python scripts/train/train_base_from_config.py --preset full
```

如果你更习惯显式写配置路径，也可以继续使用：
```bash
python scripts/train/train_base_from_config.py --config configs/train/train_base_run_small.yaml
python scripts/train/train_base_from_config.py --config configs/train/train_base_run_full.yaml
```

3. 训练结束后，对该 run 下所有 checkpoint 做自动评估：
```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/<run_id> --run-id <run_id>
```
这条命令会自动顺序执行：
- `eval_infilling.py`
- `eval_continuation.py`

4. 开长训前或改动训练代码后，建议先跑一遍最小回归检查：
```bash
python scripts/train/regression_check.py --device cpu --precision fp32 --seq-len 64 --batch-size 1
```

评估报告默认写到：
- 报告路径：`outputs/reports/eval/<run_id>.json`
- 图表路径：`outputs/reports/eval/<run_id>.png`
- 重点关注字段：`valid_loss`、`ppl`、`structural_validity_rate`
- 报告路径：`outputs/reports/eval_continuation/<run_id>.json`
- 图表路径：`outputs/reports/eval_continuation/<run_id>.png`
- 重点关注字段：`valid_loss`、`ppl`、`structural_validity_rate`、`first_token_accuracy`

## 评估最小闭环（按 checkpoint）
`eval_all.py` 是统一评估入口，会按顺序跑完 infilling 与 continuation 两类评估：
```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/<run_id> --run-id <run_id>
```

`eval_infilling.py` 会对 run 目录下每个 checkpoint 逐个评估，并输出：
- `valid_loss`
- `ppl`
- `structural_validity_rate`

示例：
```bash
python scripts/eval/eval_infilling.py --checkpoint-dir outputs/checkpoints/base/<run_id> --run-id <run_id>
```

评估报告默认写入：
- `outputs/reports/eval/<run_id>.json`
- `outputs/reports/eval/<run_id>.png`

`eval_continuation.py` 会对同一批 checkpoint 逐个评估续写能力，并输出：
- `valid_loss`
- `ppl`
- `structural_validity_rate`
- `first_token_accuracy`

示例：
```bash
python scripts/eval/eval_continuation.py --checkpoint-dir outputs/checkpoints/base/<run_id> --run-id <run_id>
```

评估报告默认写入：
- `outputs/reports/eval_continuation/<run_id>.json`
- `outputs/reports/eval_continuation/<run_id>.png`

## 自动回归检查
一条命令覆盖 train + save + resume + eval 的最小链路：
```bash
python scripts/train/regression_check.py
```

该检查会自动执行：
- 真实数据采样训练 1 步并保存 checkpoint
- 从 `latest.pt` 恢复并继续到第 2 步
- 运行 `eval_infilling.py`
- 运行 `eval_continuation.py`
- 校验 `outputs/reports/eval/<run_id>.json`、`outputs/reports/eval_continuation/<run_id>.json` 以及对应 `.png` 图表

## 当前训练策略（NEXT + FIM）
`train_base` 当前采用混合训练：
- NEXT：保持前缀续写能力（主任务）
- FIM：补充中间编辑能力（辅助任务）

可在 `configs/train/train_base_run_small.yaml` 或 `configs/train/train_base_run_full.yaml` 调整：
- `fim_ratio`：每个 batch 中 FIM 样本比例（0~1）
- `fim_min_span`：FIM 挖洞最小 token 长度
- `fim_max_span`：FIM 挖洞最大 token 长度

## 代码结构
- `scripts/data/`：数据清洗、切分、分词、打包、校验
- `scripts/train/`：训练入口与训练链路工具
- `scripts/eval/`：评估入口与评估脚本
- `src/tokenizer/`：分词核心实现
- `src/training/`：训练核心实现

## 许可与合规
- 许可证：Apache License 2.0（见 [LICENSE](./LICENSE)）
- 数据使用：请遵守各数据集许可与使用条款
