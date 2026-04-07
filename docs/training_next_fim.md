# 训练策略说明：NEXT + FIM 混合训练

## 背景
- 项目主场景是 Copilot 式续写（`prefix -> continuation`）。
- 同时需要支持中间编辑（Infilling）。

因此，当前 `train_base` 采用：
- `NEXT` 作为主训练目标（保证续写能力）
- `FIM` 作为辅助训练目标（补充中间编辑能力）

## 代码实现
- [src/training/train_base.py](/d:/Project/TuneFlow/src/training/train_base.py)
  - `TokenBinDataset.sample_mixed_batch(...)`：混合采样入口
  - `TokenBinDataset._build_fim_example(...)`：构造 FIM 样本
  - 训练循环使用 `sample_mixed_batch(...)`
  - metrics 增加：
    - `fim_examples`
    - `fim_ratio_in_batch`

## 关键参数
在 `train_base` 中新增参数：
- `--fim-ratio`：每个 batch 中 FIM 样本比例，范围 `[0, 1]`
- `--fim-min-span`：FIM 挖洞最小长度（token）
- `--fim-max-span`：FIM 挖洞最大长度（token）

默认值写在：
- `configs/train/train_base_run.yaml`

## 训练入口（配置化）
```bash
python scripts/train/train_base_from_config.py --config configs/train/train_base_run.yaml
```

## 回归链路同步
- [scripts/train/regression_check.py](/d:/Project/TuneFlow/scripts/train/regression_check.py) 中固定开启 `fim_ratio=1.0`
- 目的是在冒烟回归中强制覆盖 FIM 分支，避免后续改动导致分支失效

## 评估闭环
- [scripts/eval/eval_infilling.py](/d:/Project/TuneFlow/scripts/eval/eval_infilling.py)：评估中间编辑能力，输出 `valid_loss`、`ppl`、`structural_validity_rate`
- [scripts/eval/eval_continuation.py](/d:/Project/TuneFlow/scripts/eval/eval_continuation.py)：评估 NEXT 主任务对应的续写能力，输出 `valid_loss`、`ppl`、`structural_validity_rate`、`first_token_accuracy`
- [scripts/train/regression_check.py](/d:/Project/TuneFlow/scripts/train/regression_check.py) 会在最小链路中同时跑这两个评估脚本
