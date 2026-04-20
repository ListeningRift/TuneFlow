# Eval Guide

TuneFlow 现在保留三种评估入口，但它们共用同一套 benchmark 核心逻辑：

- `scripts/eval/eval_all.py`
- `scripts/eval/eval_infilling.py`
- `scripts/eval/eval_continuation.py`

训练期指标统一来自 checkpoint 目录下的 `metrics.jsonl`，benchmark 阶段不再重算 `valid_loss`。

指标定义、absolute score、pitch collapse 检测与使用建议见：[benchmark_metrics.md](./benchmark_metrics.md)

## 入口区别

### `eval_all.py`

完整 benchmark 入口：

1. 读取所有 checkpoint
2. 跑 `fast benchmark`
3. 选出前 3 名
4. 对前 3 名跑 `formal benchmark`
5. 输出 leaderboard、推荐 checkpoint、样本文件和 Markdown 摘要

适合：

- 正式选 best checkpoint
- 做完整实验对比
- 导出最终样本

### `eval_infilling.py`

只跑 infilling benchmark，选点与样本导出都只基于 infilling 指标。

适合：

- 单独优化补全能力
- 调整 FIM 比例、洞长、补全行为

### `eval_continuation.py`

只跑 continuation benchmark，选点与样本导出都只基于 continuation 指标。

适合：

- 单独优化续写能力
- 调整停止行为、时间顺序、空 bar 等问题

## 推荐用法

使用内置训练 preset：

```bash
uv run eval-all --preset small
uv run eval-infilling --preset small
uv run eval-continuation --preset small
```

使用自定义训练配置：

```bash
uv run eval-all --config configs/train/train_base_run_small.yaml
uv run eval-infilling --config configs/train/train_base_run_small.yaml
uv run eval-continuation --config configs/train/train_base_run_small.yaml
```

三个脚本都会从训练配置里的 `output_dir` 自动定位 checkpoint 目录，不再需要手动传 `--checkpoint-dir`。
当前内置 preset 默认对应的 run 名称分别是 `base_small` 和 `base_full`。
配置文件名暂时仍保留为 `train_base_run_small.yaml` 和 `train_base_run_full.yaml`，只是默认输出目录与 run 名称已经精简。

## 核心指标

完整 benchmark 的硬门槛：

- `continuation_stop_success_rate >= 0.20`
- `continuation_budget_stop_rate <= 0.75`
- `continuation_time_order_validity_rate >= 0.85`
- `infilling_structural_validity_rate >= 0.60`

完整 benchmark 的主排序指标：

- `continuation_stop_success_rate`
- `continuation_budget_stop_rate`
- `continuation_structural_validity_rate`
- `continuation_time_order_validity_rate`
- `continuation_empty_bar_rate`
- `infilling_structural_validity_rate`
- `infilling_time_order_validity_rate`
- `valid_loss_from_training`

单任务 benchmark 会只使用对应任务的 raw 指标做排序，FSM 结果仍然保留为诊断项。

## 输出目录

默认输出到：

```text
outputs/benchmark/<run_id>/
```

主要产物：

- `benchmark_report.json`
- `benchmark_summary.md`
- `benchmark_infilling_report.json`
- `benchmark_infilling_summary.md`
- `benchmark_continuation_report.json`
- `benchmark_continuation_summary.md`
- `benchmark_fast_manifest.json`
- `benchmark_formal_manifest.json`
- `benchmark_infilling_fast_manifest.json`
- `benchmark_infilling_formal_manifest.json`
- `benchmark_continuation_fast_manifest.json`
- `benchmark_continuation_formal_manifest.json`
- `samples/<checkpoint>/continuation.json`
- `samples/<checkpoint>/infilling.json`

如果你想把 sample JSON 里的完整序列直接导出为 MIDI 来听效果，可以使用：

```bash
uv run eval-export-midi --input-json outputs/benchmark/base_full/samples/final_top3/step_100000/continuation.json --output outputs/debug/sample_case_all
```

不传 `--case-index` 时，会把 JSON 里的所有 case 都导出到目标目录：
- continuation 样本会生成 `0_full.mid`、`0_continuation.mid`
- infilling 样本会生成 `0_full.mid`、`0_infilling.mid`
- 另外还会生成 `0_target.mid` 和 `0_reference_full.mid`，方便对照原始真值

如果只想导出单条：

```bash
uv run eval-export-midi --input-json outputs/benchmark/base_full/samples/final_top3/step_100000/continuation.json --case-index 0 --output outputs/debug/sample_case_0.mid
```

这时会同时生成：
- `sample_case_0.mid` 作为完整结果
- `sample_case_0_continuation.mid` 或 `sample_case_0_infilling.mid` 作为模型新增部分
- `sample_case_0_target.mid` 作为原始真值片段
- `sample_case_0_reference_full.mid` 作为原始真值完整拼接结果

默认导出 `fsm_reconstructed_tokens`。如果想对比未加约束的完整重建结果：

```bash
uv run eval-export-midi \
  --input-json outputs/benchmark/base_full/samples/final_top3/step_100000/continuation.json \
  --case-index 0 \
  --token-field raw_reconstructed_tokens \
  --output outputs/debug/sample_case_0_raw.mid
```

## 常用参数

- `--config <train-yaml>`
  作用：指定训练配置文件，从里面自动读取 `output_dir` 找到对应 checkpoint 目录。
  例子：`uv run eval-all --config configs/train/train_base_run_small.yaml`
- `--preset {small,full}`
  作用：直接使用内置训练预设，省掉手写配置路径。
  例子：`uv run eval-all --preset small`
- `--device {auto,cpu,cuda}`
  作用：指定评估在哪个设备上跑。`auto` 会自动优先 GPU。
  例子：`uv run eval-all --preset small --device cpu`
- `--precision {auto,fp32,bf16,fp16}`
  作用：指定推理精度。GPU 上通常 `bf16/fp16` 更快，CPU 上一般用 `fp32`。
  例子：`uv run eval-all --preset small --device cuda --precision bf16`
- `--max-new-tokens N`
  作用：限制单条样本最多能生成多少个新 token。值越大越慢，但也越不容易被长度截断。
  例子：`uv run eval-all --preset small --max-new-tokens 96`
- `--limit-checkpoints N`
  作用：只评估前 N 个 checkpoint，适合 smoke 测试。
  例子：`uv run eval-all --preset small --limit-checkpoints 2`
- `--checkpoint-policy {all,sampled}`
  作用：决定 fast 阶段跑全部 checkpoint 还是均匀抽样。
  例子：`uv run eval-all --preset small --checkpoint-policy sampled`
- `--sample-count N`
  作用：当 `--checkpoint-policy sampled` 时，指定要保留多少个 step checkpoint。
  例子：`uv run eval-all --preset small --checkpoint-policy sampled --sample-count 6`
- `--include-alias-checkpoints`
  作用：把 `best.pt`、`last.pt`、`latest.pt` 这类别名 checkpoint 也加入 fast 扫描。默认不加。
  例子：`uv run eval-all --preset small --include-alias-checkpoints`
- `--prefilter-top-k-by-valid-loss N`
  作用：fast benchmark 之前先按训练期 `valid_loss` 只保留 top K 个 checkpoint。设成 `0` 表示关闭。
  例子：`uv run eval-all --preset small --prefilter-top-k-by-valid-loss 8`
- `--prefilter-preserve-earliest N`
  作用：在按 `valid_loss` 预筛时，额外保留最早的 N 个 eval 对齐 checkpoint，避免早期峰值被误删。
  例子：`uv run eval-all --preset small --prefilter-top-k-by-valid-loss 8 --prefilter-preserve-earliest 4`
- `--fast-config <yaml>`
  作用：指定 fast benchmark 的配置文件，比如样本抽样数量、样本导出数量。
  例子：`uv run eval-all --preset small --fast-config configs/eval/benchmark_fast.yaml`
- `--formal-config <yaml>`
  作用：指定 formal benchmark 的配置文件，一般控制全量复评集。
  例子：`uv run eval-all --preset small --formal-config configs/eval/benchmark_formal.yaml`

最常用的几个命令组合：

```bash
# 直接跑默认 small 综合 benchmark
uv run eval-all --preset small

# 只做一次很快的 smoke 检查
uv run eval-all --preset small --limit-checkpoints 2 --max-new-tokens 64

# 用 sampled 模式快速看趋势
uv run eval-all --preset small --checkpoint-policy sampled --sample-count 6

# 关闭 valid_loss 预筛，强制跑全部 step checkpoint
uv run eval-all --preset small --prefilter-top-k-by-valid-loss 0
```

## Benchmark 配置

默认配置：

- `configs/eval/benchmark_fast.yaml`
- `configs/eval/benchmark_formal.yaml`

其中约定了：

- 分层抽样数量
- 每个 bucket 的上限
- continuation prefix 比例
- infilling hole 比例
- 样本导出数量

## 回归检查

项目里的 smoke 回归默认验证完整 benchmark：

```bash
uv run train-regression-check --device cpu --precision fp32 --seq-len 64 --batch-size 1
```

它会训练 2 步、恢复一次训练、然后跑缩小版 benchmark，最后校验报告和样本导出是否完整。
