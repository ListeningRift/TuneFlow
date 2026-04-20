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
- Benchmark 指标说明：[docs/benchmark_metrics.md](./docs/benchmark_metrics.md)

## 环境准备
项目现在使用 `conda` 做环境管理，使用 `uv` 做依赖管理和命令管理。

先创建并激活环境：

```bash
conda create -n tune-flow python=3.10 -y
conda activate tune-flow
```

然后用 `uv` 安装依赖并使用命令入口：

```bash
uv sync
uv run data-clean --help
```

## 数据构建
如果已经有完整的 `data/tokenized/*.bin`、`data/tokenized/*.idx.json`、`data/tokenized/tokenizer_vocab.json`，可以直接跳到“配置化训练”。

1. 一键跑完整流程（clean -> split -> tokenize -> build -> validate）
```bash
uv run data-pipeline
```

2. 冒烟模式（只处理少量样本）
```bash
uv run data-pipeline --clean-limit 200 --split-limit 200 --tokenize-limit-per-split 100
```

3. 从中间步骤继续（例如从 tokenize 开始）
```bash
uv run data-pipeline --start-from tokenize --stop-after validate
```

4. 也可分步执行
```bash
uv run data-clean --config configs/data/cleaning.yaml
uv run data-split --config configs/data/split.yaml
uv run data-tokenize --config configs/tokenizer/tokenizer.yaml
uv run data-build-training --config configs/data/build_training.yaml
uv run data-validate
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
uv run train-base --preset small --dry-run
```

2. 启动训练：
```bash
uv run train-base --preset small
uv run train-base --preset full
```

如果你想一条命令从训练直接跑到完整 benchmark 评估，可以使用：
```bash
uv run train-pipeline --preset small
uv run train-pipeline --preset full
```

如果你更习惯显式写配置路径，也可以继续使用：
```bash
uv run train-base --config configs/train/train_base_run_small.yaml
uv run train-base --config configs/train/train_base_run_full.yaml
```

当前内置 preset 的默认 run 名称分别是 `base_small` 和 `base_full`。
配置文件名暂时仍保留为 `train_base_run_small.yaml` 和 `train_base_run_full.yaml`，只是默认输出目录与 run 名称已经精简。

3. 训练结束后，使用统一 benchmark 入口做完整评估：
```bash
uv run eval-all --preset small
uv run eval-all --preset full
```

如果你使用的是自定义训练配置：
```bash
uv run eval-all --config configs/train/train_base_run_small.yaml
```

如果只想单独优化某一个任务，也可以直接运行：
```bash
uv run eval-infilling --preset small
uv run eval-continuation --preset small
```

这套评估会直接从训练配置里的 `output_dir` 读取 checkpoint 与 `metrics.jsonl`，不再需要手动传 `--checkpoint-dir`。
完整 benchmark 会按顺序执行：
- `fast benchmark` 扫全量 checkpoint
- `formal benchmark` 复评前 3 名
- 导出 best checkpoint、leaderboard 和样本摘要

4. 开长训前或改动训练代码后，建议先跑一遍最小回归检查：
```bash
uv run train-regression-check --device cpu --precision fp32 --seq-len 64 --batch-size 1
```

评估产物默认写到：
- `outputs/benchmark/<run_id>/benchmark_report.json`
- `outputs/benchmark/<run_id>/benchmark_summary.md`
- `outputs/benchmark/<run_id>/benchmark_infilling_report.json`
- `outputs/benchmark/<run_id>/benchmark_continuation_report.json`
- `outputs/benchmark/<run_id>/samples/<checkpoint>/continuation.json`
- `outputs/benchmark/<run_id>/samples/<checkpoint>/infilling.json`

如果你想把某条 benchmark sample 直接反编译成 MIDI 来听结果，可以使用：

```bash
uv run eval-export-midi --input-json outputs/benchmark/base_full/samples/final_top3/step_100000/continuation.json --output outputs/debug/sample_case_all
```

不传 `--case-index` 时，会把 JSON 里的所有 case 都导出到目标目录下：
- continuation 样本会生成 `0_full.mid`、`0_continuation.mid`
- infilling 样本会生成 `0_full.mid`、`0_infilling.mid`
- 另外还会生成 `0_target.mid` 和 `0_reference_full.mid`，用于和原始真值对比

如果只想导出某一条 case：

```bash
uv run eval-export-midi --input-json outputs/benchmark/base_full/samples/final_top3/step_100000/continuation.json --case-index 0 --output outputs/debug/sample_case_0.mid
```

这时会同时生成：
- `sample_case_0.mid` 作为完整结果
- `sample_case_0_continuation.mid` 或 `sample_case_0_infilling.mid` 作为模型新增部分
- `sample_case_0_target.mid` 作为原始真值片段
- `sample_case_0_reference_full.mid` 作为原始真值完整拼接结果

默认会导出 `fsm_reconstructed_tokens`；如果要对比未加约束的完整结果，可额外传：

```bash
uv run eval-export-midi --input-json outputs/benchmark/base_full/samples/final_top3/step_100000/continuation.json --case-index 0 --token-field raw_reconstructed_tokens --output outputs/debug/sample_case_0_raw.mid
```

如果当前结果值得长期保留，可在下次训练/评估前先归档一份快照：
## 归档命令

```bash
uv run tools-archive --preset small
uv run tools-archive --config configs/train/train_base_run_small.yaml --tag baseline_v1
uv run tools-archive --preset small --dry-run
```

归档脚本会默认：
- 优先读取 benchmark 报告里的 `recommended_checkpoint`
- 复制该 checkpoint、`metrics.jsonl`、训练配置，以及整个 `outputs/benchmark/<run_id>/`
- 写入 `outputs/archive/<run_id>/<run_id>__<checkpoint>__YYYYMMDD_HHMMSS[__tag]/`

更详细的评估参数与使用方式见：[docs/eval_guide.md](./docs/eval_guide.md)

## 当前训练策略（NEXT + FIM）
`train_base` 当前采用混合训练：
- NEXT：保持前缀续写能力（主任务）
- FIM：补充中间编辑能力（辅助任务）

可在 `configs/train/train_base_run_small.yaml` 或 `configs/train/train_base_run_full.yaml` 调整：
- `fim_ratio`：每个 batch 中 FIM 样本比例（0~1）
- `fim_min_span`：FIM 挖洞最小 token 长度
- `fim_max_span`：FIM 挖洞最大 token 长度

## TODO 模块

当前待办事项统一维护在文档中：

- [docs/todo.md](./docs/todo.md)

## 代码结构
- `scripts/data/`：数据清洗、切分、分词、打包、校验
- `scripts/train/`：训练入口与训练链路工具
- `scripts/eval/`：评估入口与评估脚本
- `src/tokenizer/`：分词核心实现
- `src/training/`：训练核心实现

## 许可与合规
- 许可证：Apache License 2.0（见 [LICENSE](./LICENSE)）
- 数据使用：请遵守各数据集许可与使用条款
