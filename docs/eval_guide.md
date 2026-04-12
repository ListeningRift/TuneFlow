# Eval Guide

这份文档整理了 TuneFlow 当前 3 个评估入口的用途、全部命令行参数，以及几组可以直接复制运行的常用命令：

- `scripts/eval/eval_infilling.py`
- `scripts/eval/eval_continuation.py`
- `scripts/eval/eval_all.py`

## 三个脚本分别做什么

### `eval_infilling.py`

对一个训练 run 目录下的多个 checkpoint 做 infilling 评估，核心关注：

- `valid_loss`
- `ppl`
- `structural_validity_rate`
- `fsm_structural_validity_rate`

同时会统计：

- 是否成功生成到 `EOS`
- 语法是否合法
- 非法样本原因分布
- FSM 约束介入次数、非法 top-1 比例等

默认输出：

- `outputs/reports/eval_infilling/train_base_run_small.json`
- `outputs/reports/eval_infilling/train_base_run_small.png`
- `outputs/reports/eval_infilling/train_base_run_full.json`
- `outputs/reports/eval_infilling/train_base_run_full.png`

如果开启 `--debug-invalid-samples > 0`，还会额外输出：

- `outputs/reports/eval_infilling/train_base_run_small.invalid_samples.json`
- `outputs/reports/eval_infilling/train_base_run_full.invalid_samples.json`

### `eval_continuation.py`

对一个训练 run 目录下的多个 checkpoint 做 continuation 评估，核心关注：

- `valid_loss`
- `ppl`
- `structural_validity_rate`
- `fsm_structural_validity_rate`
- `first_token_accuracy`
- `fsm_first_token_accuracy`

同时会统计：

- 是否成功生成到 `EOS`
- 是否被 `max_new_tokens` 截断
- 追加 `EOS` 后是否本可闭合
- FSM 自动闭合、EOS bias、安全边界停止等行为

默认输出：

- `outputs/reports/eval_continuation/train_base_run_small.json`
- `outputs/reports/eval_continuation/train_base_run_small.png`
- `outputs/reports/eval_continuation/train_base_run_full.json`
- `outputs/reports/eval_continuation/train_base_run_full.png`

如果开启 `--debug-continuation-samples > 0`，还会额外输出：

- `outputs/reports/eval_continuation/train_base_run_small.samples.json`
- `outputs/reports/eval_continuation/train_base_run_full.samples.json`

### `eval_all.py`

统一入口，会顺序执行：

1. `eval_infilling.py`
2. `eval_continuation.py`

适合在训练结束后做一轮完整回看，或者做快速抽样复评。

需要注意两点：

- `eval_all.py` 自己不接收 `--output-path`，两个子评估仍然写默认报告路径
- 它的默认策略和单独跑子脚本不同，更偏向“快速模式”

默认策略：

- `--checkpoint-policy sampled`
- `--valid-loss-source metrics`

而单独运行 `eval_infilling.py` / `eval_continuation.py` 时默认是：

- `--checkpoint-policy all`
- `--valid-loss-source recompute`

## 参数总览

## 公共参数

下面这些参数在 `eval_infilling.py` 和 `eval_continuation.py` 中都支持；`eval_all.py` 也支持其中绝大多数，并会透传给两个子脚本。

| 参数                  | `infilling` 默认值                              | `continuation` 默认值                           | `all` 默认值                          | 说明                                                                                  |
| --------------------- | ----------------------------------------------- | ----------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------- |
| `--checkpoint-dir`    | `outputs/checkpoints/base/train_base_run_small` | `outputs/checkpoints/base/train_base_run_small` | 必填                                  | 待评估 run 的 checkpoint 目录                                                         |
| `--run-id`            | `None`                                          | `None`                                          | `None`                                | 报告名里的 run 标识；不传时默认用 checkpoint 目录名                                   |
| `--model-config`      | `configs/train/model_base.yaml`                 | `configs/train/model_base.yaml`                 | `configs/train/model_base.yaml`       | 当 checkpoint 中未保存 `model_config` 时使用的兜底配置                                |
| `--valid-idx`         | `data/tokenized/valid.idx.json`                 | `data/tokenized/valid.idx.json`                 | `data/tokenized/valid.idx.json`       | 用于重算 `valid_loss` / `ppl` 的验证集索引                                            |
| `--valid-bin`         | `None`                                          | `None`                                          | `None`                                | 可选覆盖验证集 `.bin` 路径                                                            |
| `--eval-tok`          | `data/tokenized/eval.tok`                       | `data/tokenized/eval.tok`                       | `data/tokenized/eval.tok`             | 用于构造 infilling / continuation prompt 的评估 token 文件                            |
| `--vocab-path`        | `data/tokenized/tokenizer_vocab.json`           | `data/tokenized/tokenizer_vocab.json`           | `data/tokenized/tokenizer_vocab.json` | tokenizer 词表路径                                                                    |
| `--device`            | `auto`                                          | `auto`                                          | `auto`                                | 评估设备，可选 `auto` / `cpu` / `cuda`                                                |
| `--precision`         | `auto`                                          | `auto`                                          | `auto`                                | 精度模式，可选 `auto` / `fp32` / `bf16` / `fp16`                                      |
| `--seq-len`           | `256`                                           | `256`                                           | `256`                                 | 重算 `valid_loss` 时采样窗口长度                                                      |
| `--batch-size`        | `2`                                             | `2`                                             | `2`                                   | 验证集 loss 评估时的 micro-batch 大小                                                 |
| `--eval-batches`      | `10`                                            | `10`                                            | `10`                                  | 每个 checkpoint 上用于计算 `valid_loss` 的 batch 数                                   |
| `--max-new-tokens`    | `128`                                           | `128`                                           | `128`                                 | 每条 prompt 最多生成多少 token                                                        |
| `--limit-checkpoints` | `None`                                          | `None`                                          | `None`                                | 只评估前 N 个 checkpoint，适合冒烟                                                    |
| `--checkpoint-policy` | `all`                                           | `all`                                           | `sampled`                             | `all`=全量；`sampled`=只抽样 `step_*.pt`，并保留 `best/last/latest`                   |
| `--sample-count`      | `6`                                             | `6`                                             | `6`                                   | `checkpoint-policy=sampled` 时，从 `step_*.pt` 中均匀抽样的数量                       |
| `--valid-loss-source` | `recompute`                                     | `recompute`                                     | `metrics`                             | `recompute`=重算；`metrics`=复用训练期 `metrics.jsonl`；`auto`=优先复用，缺失时再重算 |
| `--metrics-path`      | `None`                                          | `None`                                          | `None`                                | 显式指定训练期 `metrics.jsonl` 路径；不传时默认尝试 `<checkpoint-dir>/metrics.jsonl`  |
| `--seed`              | `42`                                            | `42`                                            | `42`                                  | 随机种子                                                                              |

### 公共参数使用建议

- 想看最准确、可横向对比的 `valid_loss`：用 `--valid-loss-source recompute`
- 想快速扫很多 checkpoint：用 `--checkpoint-policy sampled --valid-loss-source metrics`
- 只想确认链路是否通：加 `--limit-checkpoints 1`
- 显存紧张时：优先把 `--batch-size` 降到 `1`，必要时再改 `--seq-len`

## `eval_infilling.py` 独有参数

| 参数                      | 默认值 | 说明                                                                                                     |
| ------------------------- | ------ | -------------------------------------------------------------------------------------------------------- |
| `--output-path`           | `None` | 自定义报告 JSON 输出路径；不传时默认写到 `outputs/reports/eval_infilling/<checkpoint_dir_name>.json`     |
| `--num-infilling-samples` | `32`   | 每个 checkpoint 上抽多少条 infilling 样本做结构评估                                                      |
| `--debug-invalid-samples` | `0`    | 每个 checkpoint 额外保留多少条非法 infilling 样本；大于 0 时会打印摘要并额外写出 `.invalid_samples.json` |
| `--debug-preview-tokens`  | `16`   | 控制台打印 debug 样本时，每段 token 预览的长度                                                           |

### 什么时候改这些参数

- 想让结构合法率更稳定：增大 `--num-infilling-samples`
- 想分析为什么 infilling 不合法：设置 `--debug-invalid-samples 10`
- 控制台输出太长：减小 `--debug-preview-tokens`
- 想把报告写到实验目录：使用 `--output-path`

## `eval_continuation.py` 独有参数

| 参数                           | 默认值 | 说明                                                                                          |
| ------------------------------ | ------ | --------------------------------------------------------------------------------------------- |
| `--output-path`                | `None` | 自定义报告 JSON 输出路径；不传时默认写到 `outputs/reports/eval_continuation/<checkpoint_dir_name>.json` |
| `--num-continuation-samples`   | `32`   | 每个 checkpoint 上抽多少条 continuation 样本做生成评估                                        |
| `--min-prefix-tokens`          | `16`   | 构造 continuation prompt 时保留的最小 prefix 长度                                             |
| `--debug-continuation-samples` | `0`    | 每个 checkpoint 额外保留多少条 continuation 实际生成样本；大于 0 时会额外写出 `.samples.json` |
| `--debug-preview-tokens`       | `16`   | 控制台打印 debug 样本时，每段 token 预览的长度                                                |

### 什么时候改这些参数

- continuation 太短，前缀不够稳定：适当增大 `--min-prefix-tokens`
- 想看模型到底生成了什么：设置 `--debug-continuation-samples 10`
- 想更快出结果：减小 `--num-continuation-samples`
- 想把报告写到特定目录：使用 `--output-path`

## `eval_all.py` 独有参数

| 参数                           | 默认值                | 说明                                 |
| ------------------------------ | --------------------- | ------------------------------------ |
| `--num-infilling-samples`      | `32`                  | 透传给 `eval_infilling.py`           |
| `--num-continuation-samples`   | `32`                  | 透传给 `eval_continuation.py`        |
| `--min-prefix-tokens`          | `16`                  | 透传给 `eval_continuation.py`        |
| `--python-exec`                | 当前 `sys.executable` | 用哪个 Python 去启动两个子脚本       |
| `--dry-run`                    | `False`               | 只打印将要执行的两条命令，不实际评估 |
| `--debug-invalid-samples`      | `0`                   | 透传给 `eval_infilling.py`           |
| `--debug-preview-tokens`       | `16`                  | 透传给两个子脚本                     |
| `--debug-continuation-samples` | `0`                   | 透传给 `eval_continuation.py`        |

### `eval_all.py` 的推荐使用方式

- 训练结束后统一跑一次：用它
- 想先确认透传参数是否正确：加 `--dry-run`
- 想快速扫全 run：直接用默认值
- 想做严格重评：显式加 `--checkpoint-policy all --valid-loss-source recompute`

## 常用可复制命令

下面命令直接列出 `train_base_run_small` 和 `train_base_run_full` 两条可复制版本。

### 1. 训练后做一轮默认快速评估

这是最常用的一条。会抽样 checkpoint，并优先复用训练时记录的 `valid_loss`。

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small
```

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full
```

适用场景：

- 刚训完一版，先快速看趋势
- checkpoint 较多，不想每个都重算验证集 loss

### 2. 对一个 run 做全量精评

适合做正式对比实验，或者准备挑 best checkpoint 时使用。

```bash
python scripts/eval/eval_all.py  --checkpoint-dir outputs/checkpoints/base/train_base_run_small  --run-id train_base_run_small  --checkpoint-policy all  --valid-loss-source recompute
```

```bash
python scripts/eval/eval_all.py  --checkpoint-dir outputs/checkpoints/base/train_base_run_full  --run-id train_base_run_full  --checkpoint-policy all  --valid-loss-source recompute
```

适用场景：

- 论文式/实验记录式对比
- 不想依赖训练阶段的 `metrics.jsonl`

### 3. 只跑 infilling 评估

当你只关心补全能力时，用这一条更直接。

```bash
python scripts/eval/eval_infilling.py  --checkpoint-dir outputs/checkpoints/base/train_base_run_small  --run-id train_base_run_small
```

```bash
python scripts/eval/eval_infilling.py  --checkpoint-dir outputs/checkpoints/base/train_base_run_full  --run-id train_base_run_full
```

适用场景：

- 最近只改了 FIM / infilling 相关逻辑
- 不想花时间跑 continuation

### 4. 只跑 continuation 评估

```bash
python scripts/eval/eval_continuation.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small
```

```bash
python scripts/eval/eval_continuation.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full
```

适用场景：

- 最近只改了续写逻辑
- 重点关注 `first_token_accuracy` 和 `EOS` 收尾

### 5. 快速冒烟，只看少量 checkpoint

如果你只是想确认脚本没有坏，先限制 checkpoint 数量即可。

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small --limit-checkpoints 1 --eval-batches 1 --num-infilling-samples 4 --num-continuation-samples 4
```

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full --limit-checkpoints 1 --eval-batches 1 --num-infilling-samples 4 --num-continuation-samples 4
```

适用场景：

- 改完评估代码后的最小回归
- 资源紧张时先探路

### 6. 只抽样部分 checkpoint，但重新计算 valid loss

这一条适合“我想快一点，但又不想直接信训练期 metrics”。

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small --checkpoint-policy sampled --sample-count 6 --valid-loss-source recompute
```

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full --checkpoint-policy sampled --sample-count 6 --valid-loss-source recompute
```

适用场景：

- checkpoint 很多
- 仍希望 `valid_loss` 来自当前评估流程

### 7. 调试 infilling 非法样本

```bash
python scripts/eval/eval_infilling.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small --debug-invalid-samples 10 --debug-preview-tokens 24
```

```bash
python scripts/eval/eval_infilling.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full --debug-invalid-samples 10 --debug-preview-tokens 24
```

运行后重点看：

- 控制台里每条非法样本的摘要
- `outputs/reports/eval_infilling/train_base_run_small.invalid_samples.json`
- `outputs/reports/eval_infilling/train_base_run_full.invalid_samples.json`

适用场景：

- 想知道非法样本是缺 `EOS`、结构断裂，还是别的原因
- 想分析 raw 解码和 FSM 解码的差异

### 8. 调试 continuation 实际生成样本

```bash
python scripts/eval/eval_continuation.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small --debug-continuation-samples 10 --debug-preview-tokens 24
```

```bash
python scripts/eval/eval_continuation.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full --debug-continuation-samples 10 --debug-preview-tokens 24
```

运行后重点看：

- 控制台中的 prompt / target / raw / fsm 预览
- `outputs/reports/eval_continuation/train_base_run_small.samples.json`
- `outputs/reports/eval_continuation/train_base_run_full.samples.json`

适用场景：

- 想确认模型是不是“快停不下来”或者“过早结束”
- 想看 FSM 自动闭合对结果的影响

### 9. continuation 评估时提高前缀长度

```bash
python scripts/eval/eval_continuation.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small --min-prefix-tokens 32
```

```bash
python scripts/eval/eval_continuation.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full --min-prefix-tokens 32
```

适用场景：

- 评估样本的前缀太短，导致 continuation 太不稳定
- 想让续写起点更“有上下文”

### 10. 在 CPU 上跑一版稳定复现

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small --device cpu --precision fp32
```

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full --device cpu --precision fp32
```

适用场景：

- 没有可用 GPU
- 想做更保守的复现检查

### 11. 显式指定别的验证集 / 词表 / eval 集

如果你有另一套 tokenized 数据，或者在做迁移实验，这条很实用。

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small --valid-idx data/tokenized_alt/valid.idx.json --valid-bin data/tokenized_alt/valid.bin --eval-tok data/tokenized_alt/eval.tok --vocab-path data/tokenized_alt/tokenizer_vocab.json
```

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full --valid-idx data/tokenized_alt/valid.idx.json --valid-bin data/tokenized_alt/valid.bin --eval-tok data/tokenized_alt/eval.tok --vocab-path data/tokenized_alt/tokenizer_vocab.json
```

适用场景：

- 切换到另一套 tokenized 数据
- 做数据版本对比

### 12. 先看 `eval_all.py` 最终会执行哪两条命令

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_small --run-id train_base_run_small --dry-run
```

```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/train_base_run_full --run-id train_base_run_full --dry-run
```

适用场景：

- 想确认参数透传是否符合预期
- 改过脚本后先做命令级自检

## 怎么选脚本

- 想“一条命令全跑”：用 `eval_all.py`
- 只关心补全质量：用 `eval_infilling.py`
- 只关心续写质量：用 `eval_continuation.py`
- 想省时间：优先 `eval_all.py` 默认配置
- 想做严格实验：显式指定 `--checkpoint-policy all --valid-loss-source recompute`

## 结果文件怎么看

### `eval_infilling.py`

报告 JSON 里建议优先看：

- `summary.best_valid_loss`
- `summary.best_structural_validity_rate`
- `summary.best_fsm_structural_validity_rate`
- `results[*].invalid_reason_counts`
- `results[*].fsm_illegal_top1_rate`

### `eval_continuation.py`

报告 JSON 里建议优先看：

- `summary.best_valid_loss`
- `summary.best_structural_validity_rate`
- `summary.best_first_token_accuracy`
- `summary.best_fsm_structural_validity_rate`
- `summary.best_fsm_first_token_accuracy`
- `summary.total_fsm_auto_close_count`
- `summary.total_fsm_eos_bias_step_count`

### 图表 PNG

两个脚本都会自动生成折线图，便于看 checkpoint 维度的趋势。通常建议把下面几项一起看：

- `valid_loss`
- `ppl`
- raw 指标
- FSM 指标

## 一个简单的经验法则

- 日常开发：先跑 `eval_all.py` 默认配置
- 要写实验结论：再跑一遍 `--checkpoint-policy all --valid-loss-source recompute`
- 发现结果怪：先打开各自的 debug 输出，再看具体样本
