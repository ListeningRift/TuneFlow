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
- `configs/train/train_base_run_small.yaml`
- `configs/train/train_base_run_full.yaml`

## 训练入口（配置化）
```bash
python scripts/train/train_base_from_config.py --preset small
python scripts/train/train_base_from_config.py --preset full
```

## 回归链路同步
- [scripts/train/regression_check.py](/d:/Project/TuneFlow/scripts/train/regression_check.py) 中固定开启 `fim_ratio=1.0`
- 目的是在冒烟回归中强制覆盖 FIM 分支，避免后续改动导致分支失效

## 评估闭环
- [scripts/eval/eval_all.py](/d:/Project/TuneFlow/scripts/eval/eval_all.py)：统一评估入口，一条命令顺序执行 infilling 与 continuation 两类评估
- [scripts/eval/eval_infilling.py](/d:/Project/TuneFlow/scripts/eval/eval_infilling.py)：评估中间编辑能力，同时输出原始解码与 FSM 约束解码两套结果，重点字段包括 `valid_loss`、`ppl`、`structural_validity_rate`、`fsm_structural_validity_rate`
- [scripts/eval/eval_continuation.py](/d:/Project/TuneFlow/scripts/eval/eval_continuation.py)：评估 NEXT 主任务对应的续写能力，同时输出原始解码与 FSM 约束解码两套结果，重点字段包括 `valid_loss`、`ppl`、`structural_validity_rate`、`fsm_structural_validity_rate`、`first_token_accuracy`、`fsm_first_token_accuracy`
- [scripts/train/regression_check.py](/d:/Project/TuneFlow/scripts/train/regression_check.py) 会在最小链路中同时跑这两个评估脚本

统一入口示例：
```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/<run_id> --run-id <run_id>
```

默认优化策略：
- 只抽样部分 `step_*.pt` 做结构评估，同时保留 `best.pt`、`last.pt`、`latest.pt`
- `valid_loss` 默认优先复用训练期 `metrics.jsonl`，避免每个 checkpoint 重新扫验证集

如需全量精评，可切换为：
```bash
python scripts/eval/eval_all.py --checkpoint-dir outputs/checkpoints/base/<run_id> --run-id <run_id> --checkpoint-policy all --valid-loss-source recompute
```
