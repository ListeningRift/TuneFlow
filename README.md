# TuneFlow

TuneFlow 是一个面向 Symbolic MIDI 的音乐生成项目，目标是成为创作者的“音乐版 Copilot”：支持中间补全、续写到结尾、从头生成，并逐步实现风格可控生成。


## 项目定位
- 任务：MIDI Infilling + Continuation + Free Generation
- 方向：结构正确、可控、可评估的符号级音乐生成
- 场景：旋律续写、编曲补洞、整段草稿生成、风格化生成

## Roadmap
- M1：数据与 Tokenizer 流程打通
- M2：基础模型达到首版验收门槛
- M3：风格化微调首版可用
- M4：规模与质量持续优化

## 愿景
TuneFlow 希望把“补全”从一次性生成能力，升级为创作链路中的长期协作能力。  
我们关注的不只是生成结果，更是创作者是否更快进入心流。

## 文档
- 项目设计文档：[`design.md`](./design.md)

## 数据构建
当前阶段（不处理风格化）建议使用一键脚本串行执行：

1. 创建并激活环境（首次）
```bash
conda create -n tune-flow python=3.10 -y
conda activate tune-flow
```

2. 安装项目依赖（首次）
```bash
python -m pip install -r requirements.txt
```

3. 一键执行全流程（clean -> split -> tokenize -> build -> validate）
```bash
python scripts/data/build_data.py
```

4. 冒烟测试（只跑少量样本）
```bash
python scripts/data/build_data.py --clean-limit 200 --split-limit 200 --tokenize-limit-per-split 100
```

5. 从中间步骤续跑（例如从 tokenize 开始）
```bash
python scripts/data/build_data.py --start-from tokenize --stop-after validate
```

可选：仍可分步手动执行
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

## 脚本结构
按功能拆分为三类目录：

- `scripts/data/`：数据相关（清洗、切分、分词、打包、验收）
- `scripts/train/`：训练相关（`train_base.py`、`train_lora.py`）
- `scripts/eval/`：评估相关（`eval_infilling.py`）

说明（当前分层约定）：
- `scripts/` 仅保留命令行入口与参数转发（薄封装）。
- `src/tokenizer/` 存放分词核心实现（由 `scripts/data/tokenize_dataset.py` 调用）。
- `src/training/` 存放训练核心实现（由 `scripts/train/*.py` 调用）。

## 许可证与数据合规
- 许可证：Apache License 2.0（见 [`LICENSE`](./LICENSE)）
- 数据使用：请遵守各数据集许可证与使用条款

## 回归冒烟检查
一条命令执行最小端到端回归检查：

```bash
python scripts/train/regression_check.py
```

该命令会自动执行：
- 真实数据采样训练 1 步
- 保存 checkpoint，并从 `latest.pt` 恢复到第 2 步
- 运行 `eval_infilling.py`
- 写出评估报告到 `outputs/reports/eval/<run_id>.json`
