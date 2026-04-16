# 训练策略说明：NEXT + FIM

## 背景

TuneFlow 当前采用混合训练：

- `NEXT`：保留前缀续写能力
- `FIM`：增强中间补全能力

因此 `train_base` 会在训练中混合采样普通 continuation 样本和 FIM 样本。

## 关键实现

- [train_base.py](/d:/Project/TuneFlow/src/training/train_base.py)
- [train_base_from_config.py](/d:/Project/TuneFlow/scripts/train/train_base_from_config.py)

训练期会持续写入：

- `loss`
- `valid_loss`
- `fim_ratio_in_batch`
- `tokens_seen`
- `train_loss_ema`
- `best_valid_loss_so_far`
- `overfit_gap`

这些指标会被 benchmark 直接读取，不再在评估阶段重算 `valid_loss`。

## 关键参数

训练配置中重点关注：

- `fim_ratio`
- `fim_min_span`
- `fim_max_span`
- `eval_every`
- `save_every`
- `output_dir`

默认训练配置：

- `configs/train/train_base_run_small.yaml`
- `configs/train/train_base_run_full.yaml`

## 训练命令

```bash
python scripts/train/train_base_from_config.py --preset small
python scripts/train/train_base_from_config.py --preset full
```

或：

```bash
python scripts/train/train_base_from_config.py --config configs/train/train_base_run_small.yaml
python scripts/train/train_base_from_config.py --config configs/train/train_base_run_full.yaml
```

## 评估闭环

训练完成后，可以按需求选择 benchmark 入口：

```bash
python scripts/eval/eval_all.py --preset small
python scripts/eval/eval_infilling.py --preset small
python scripts/eval/eval_continuation.py --preset small
```

如果训练使用自定义 YAML：

```bash
python scripts/eval/eval_all.py --config configs/train/train_base_run_small.yaml
python scripts/eval/eval_infilling.py --config configs/train/train_base_run_small.yaml
python scripts/eval/eval_continuation.py --config configs/train/train_base_run_small.yaml
```

三个入口都会自动从训练配置里的 `output_dir` 读取 checkpoint，不再需要手动指定 checkpoint 目录。
当前内置 preset 默认对应的 run 名称分别是 `base_small` 和 `base_full`。
配置文件名暂时仍保留为 `train_base_run_small.yaml` 和 `train_base_run_full.yaml`，只是默认输出目录与 run 名称已经精简。

## 回归检查

回归脚本会固定开启 `fim_ratio=1.0`，确保 FIM 分支被覆盖：

```bash
python scripts/train/regression_check.py --device cpu --precision fp32 --seq-len 64 --batch-size 1
```

它会验证：

1. 训练
2. 保存 checkpoint
3. 从 `latest.pt` 恢复
4. 跑缩小版 benchmark
5. 校验 benchmark 报告与样本导出
