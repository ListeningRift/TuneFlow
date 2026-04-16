"""评估报告图表导出工具。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

# 强制使用无界面的后端，避免在服务器或终端环境里弹图形窗口。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from src.utils.training_metrics import summarize_training_metrics, training_metrics_history


def _build_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    """把评估结果整理成适合画图的 DataFrame。"""
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("report['results'] 不能为空，无法导出图表")

    frame = pd.DataFrame(results).copy()
    if "step" in frame.columns:
        frame["step"] = pd.to_numeric(frame["step"], errors="coerce")

    raw_labels: list[str] = []
    for index, row in frame.iterrows():
        checkpoint_name = row.get("checkpoint_name")
        if isinstance(checkpoint_name, str) and checkpoint_name:
            checkpoint_stem = Path(checkpoint_name).stem
            if not checkpoint_stem.startswith("step_"):
                raw_labels.append(checkpoint_stem)
                continue
        step_value = row.get("step")
        if pd.notna(step_value) and float(step_value) >= 0:
            raw_labels.append(f"step_{int(step_value)}")
            continue
        if isinstance(checkpoint_name, str) and checkpoint_name:
            raw_labels.append(Path(checkpoint_name).stem)
            continue
        raw_labels.append(f"ckpt_{index + 1}")

    seen_labels: dict[str, int] = {}
    x_labels: list[str] = []
    for label in raw_labels:
        count = seen_labels.get(label, 0) + 1
        seen_labels[label] = count
        x_labels.append(label if count == 1 else f"{label}#{count}")
    frame["x_label"] = x_labels
    return frame


def _format_metric_value(value: float | None, percent: bool) -> str:
    """格式化图中标注文字。"""
    if value is None:
        return "NA"
    if percent:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def _coerce_metric_value(series: pd.Series) -> tuple[pd.Series, float | None, float | None]:
    """把列安全转成数值，并提取首尾有限值。"""
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[numeric.notna()]
    if finite.empty:
        return numeric, None, None
    return numeric, float(finite.iloc[-1]), float(finite.iloc[0])


def write_eval_report_plot(
    report_path: Path,
    report: dict[str, Any],
    title: str,
    metric_specs: list[dict[str, Any]],
    *,
    chart_path: Path | None = None,
) -> Path:
    """
    使用 matplotlib/pandas 把评估报告导出为 PNG 图表。

    输出特点：
    - 路径与 JSON 同名，仅扩展名改为 `.png`
    - 每个指标单独一个子图，便于直接比较 checkpoint 趋势
    - 可以在 README、PR、实验记录里直接预览
    """
    chart_path = chart_path if chart_path is not None else report_path.with_suffix(".png")
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    frame = _build_dataframe(report)
    x_positions = list(range(len(frame)))
    x_labels = frame["x_label"].tolist()
    label_stride = max(1, (len(x_labels) + 9) // 10)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figure_height = max(4.0, 3.2 * len(metric_specs) + 1.2)
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(12, figure_height), dpi=160, constrained_layout=True)
    if hasattr(axes, "tolist"):
        axes = axes.tolist()
    if not isinstance(axes, list):
        axes = [axes]

    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(
        f"{title}\nrun_id={report.get('run_id', 'unknown')} | checkpoints={len(frame)}",
        fontsize=15,
        fontweight="bold",
    )

    summary = report.get("summary", {})
    summary_lines: list[str] = []
    if isinstance(summary, dict):
        for key in (
            "best_valid_loss",
            "best_structural_validity_rate",
            "best_fsm_structural_validity_rate",
            "best_first_token_accuracy",
            "best_fsm_first_token_accuracy",
            "elapsed_sec",
        ):
            if key not in summary:
                continue
            value = pd.to_numeric(pd.Series([summary.get(key)]), errors="coerce").iloc[0]
            if pd.isna(value):
                rendered = "NA"
            elif key.endswith("_rate") or key.endswith("_accuracy"):
                rendered = f"{float(value) * 100:.2f}%"
            elif key == "elapsed_sec":
                rendered = f"{float(value):.2f}s"
            else:
                rendered = f"{float(value):.4f}"
            summary_lines.append(f"{key}={rendered}")
    if summary_lines:
        fig.text(0.01, 0.01, " | ".join(summary_lines), fontsize=9, color="#4b5563")

    for axis, metric_spec in zip(axes, metric_specs):
        metric_key = str(metric_spec["key"])
        metric_label = str(metric_spec["label"])
        color = str(metric_spec.get("color", "#2563eb"))
        percent = bool(metric_spec.get("percent", False))
        goal = str(metric_spec.get("goal", "max" if percent else "min"))

        numeric, latest_value, first_value = _coerce_metric_value(
            frame[metric_key] if metric_key in frame else pd.Series(dtype="float64")
        )
        finite = numeric[numeric.notna()]

        axis.set_facecolor("#ffffff")
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.28)
        axis.plot(x_positions, numeric.tolist(), color=color, marker="o", linewidth=2.2, markersize=5)

        if percent:
            axis.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            if not finite.empty:
                axis.set_ylim(bottom=min(0.0, float(finite.min()) * 0.95), top=max(1.0, float(finite.max()) * 1.05))
            else:
                axis.set_ylim(0.0, 1.0)

        best_value: float | None = None
        if not finite.empty:
            best_value = float(finite.max()) if goal == "max" else float(finite.min())

        axis.set_title(metric_label, loc="left", fontsize=12, fontweight="bold")
        axis.text(
            0.01,
            0.96,
            f"latest={_format_metric_value(latest_value, percent)} | "
            f"best={_format_metric_value(best_value, percent)} | "
            f"first={_format_metric_value(first_value, percent)}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#475569",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f8fafc", "edgecolor": "#dbe4ee"},
        )

        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [label if (idx % label_stride == 0 or idx == len(x_labels) - 1) else "" for idx, label in enumerate(x_labels)],
            rotation=25,
            ha="right",
            fontsize=8,
        )

    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if not frame.empty and "step" in frame.columns:
        frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    return frame


def _series_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _annotate_axis(axis: Any, *, text: str) -> None:
    axis.text(
        0.01,
        0.96,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#475569",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f8fafc", "edgecolor": "#dbe4ee"},
    )


def _set_axis_style(axis: Any) -> None:
    axis.set_facecolor("#ffffff")
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.28)


def _plot_step_series(
    axis: Any,
    *,
    frame: pd.DataFrame,
    column: str,
    label: str,
    color: str,
    linewidth: float = 2.2,
    alpha: float = 1.0,
    marker: str | None = None,
) -> pd.Series:
    numeric = _series_or_empty(frame, column)
    if frame.empty or numeric.empty:
        return numeric
    axis.plot(
        frame["step"].tolist(),
        numeric.tolist(),
        color=color,
        label=label,
        linewidth=linewidth,
        alpha=alpha,
        marker=marker,
        markersize=4 if marker else 0,
    )
    return numeric


def _latest_finite_value(series: pd.Series) -> float | None:
    finite = series[series.notna()]
    if finite.empty:
        return None
    return float(finite.iloc[-1])


def write_training_metrics_dashboard(
    chart_path: Path,
    metrics_payload: dict[str, Any],
    *,
    run_id: str,
) -> Path:
    """把训练过程中的关键健康指标导出成单张 dashboard。"""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    history = training_metrics_history(metrics_payload)
    summary = summarize_training_metrics(metrics_payload)
    train_frame = _to_frame(history["train"])
    eval_frame = _to_frame(history["eval"])

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(4, 1, figsize=(13, 15), dpi=160, constrained_layout=True)
    axes = axes.tolist() if hasattr(axes, "tolist") else [axes]
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle(f"TuneFlow 训练健康度\nrun_id={run_id}", fontsize=16, fontweight="bold")

    loss_axis = axes[0]
    _set_axis_style(loss_axis)
    raw_loss = _plot_step_series(
        loss_axis,
        frame=train_frame,
        column="loss",
        label="训练 loss",
        color="#94a3b8",
        linewidth=1.5,
        alpha=0.65,
    )
    train_ema = _plot_step_series(
        loss_axis,
        frame=train_frame,
        column="train_loss_ema",
        label="训练 loss EMA",
        color="#2563eb",
    )
    valid_loss = _plot_step_series(
        loss_axis,
        frame=eval_frame,
        column="valid_loss",
        label="验证 loss",
        color="#dc2626",
        marker="o",
    )
    best_valid = _plot_step_series(
        loss_axis,
        frame=eval_frame,
        column="best_valid_loss_so_far",
        label="历史最优验证 loss",
        color="#16a34a",
        marker="o",
    )
    loss_axis.set_title("Loss 收敛轨迹", loc="left", fontsize=12, fontweight="bold")
    loss_axis.set_ylabel("loss")
    loss_axis.legend(loc="upper right", fontsize=9)
    _annotate_axis(
        loss_axis,
        text=(
            f"最新 train EMA={_format_metric_value(_latest_finite_value(train_ema), False)} | "
            f"最新 valid={_format_metric_value(_latest_finite_value(valid_loss), False)} | "
            f"历史最优 valid={_format_metric_value(_latest_finite_value(best_valid), False)}"
        ),
    )

    gap_axis = axes[1]
    _set_axis_style(gap_axis)
    overfit_gap = _plot_step_series(
        gap_axis,
        frame=eval_frame,
        column="overfit_gap",
        label="overfit gap = valid - train_ema",
        color="#7c3aed",
        marker="o",
    )
    gap_axis.axhline(0.0, color="#94a3b8", linewidth=1.0, linestyle="--")
    gap_axis.set_title("泛化间隙", loc="left", fontsize=12, fontweight="bold")
    gap_axis.set_ylabel("gap")
    gap_axis.legend(loc="upper right", fontsize=9)
    _annotate_axis(
        gap_axis,
        text=(
            f"最新 overfit gap={_format_metric_value(_latest_finite_value(overfit_gap), False)} | "
            f"连续未刷新最佳 eval 次数={summary.get('plateau_eval_streak', 0)}"
        ),
    )

    throughput_axis = axes[2]
    _set_axis_style(throughput_axis)
    tok_per_sec = _plot_step_series(
        throughput_axis,
        frame=train_frame,
        column="tok_per_sec",
        label="tokens / sec",
        color="#ea580c",
    )
    throughput_axis.set_title("吞吐与训练稳定性", loc="left", fontsize=12, fontweight="bold")
    throughput_axis.set_ylabel("tok/s")
    throughput_axis.legend(loc="upper right", fontsize=9)
    _annotate_axis(
        throughput_axis,
        text=(
            f"均值={_format_metric_value(summary.get('tok_per_sec_mean'), False)} | "
            f"中位数={_format_metric_value(summary.get('tok_per_sec_median'), False)} | "
            f"最新={_format_metric_value(_latest_finite_value(tok_per_sec), False)}"
        ),
    )

    fim_axis = axes[3]
    _set_axis_style(fim_axis)
    fim_ratio = _plot_step_series(
        fim_axis,
        frame=train_frame,
        column="fim_ratio_in_batch",
        label="batch 内 FIM 比例",
        color="#0891b2",
    )
    fim_axis.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fim_axis.set_ylim(0.0, max(1.0, float(_series_or_empty(train_frame, "fim_ratio_in_batch").max(skipna=True) or 0.0) * 1.1))
    fim_axis.set_title("训练批次组成", loc="left", fontsize=12, fontweight="bold")
    fim_axis.set_xlabel("step")
    fim_axis.set_ylabel("比例")
    fim_axis.legend(loc="upper right", fontsize=9)
    _annotate_axis(
        fim_axis,
        text=(
            f"均值={_format_metric_value(summary.get('fim_ratio_mean'), True)} | "
            f"波动={_format_metric_value(summary.get('fim_ratio_std'), True)} | "
            f"最新={_format_metric_value(_latest_finite_value(fim_ratio), True)}"
        ),
    )

    footer_lines = [
        f"最后训练 step={summary.get('last_train_step')} | 最后验证 step={summary.get('last_eval_step')}",
        (
            f"tokens_seen={summary.get('tokens_seen_last')} | "
            f"best_valid_step={summary.get('best_valid_step')} | "
            f"latest_valid_delta={_format_metric_value(summary.get('latest_valid_loss_delta'), False)}"
        ),
    ]
    fig.text(0.01, 0.01, " | ".join(footer_lines), fontsize=9, color="#4b5563")
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)
    return chart_path
