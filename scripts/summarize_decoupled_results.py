"""Create paper-oriented figures and an evidence report from a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY = {
    "training_mean": "Training mean",
    "direct_pretrained": "Direct physics pretraining",
    "mean_common_diff_scratch": "Decoupled, scratch",
    "mean_common_diff_pretrained": "Decoupled, pretrained",
    "mean_common_diff_shrunk_pretrained": "Risk-controlled transfer",
    "mean_common_diff_constrained": "Physics-constrained transfer",
}


def interval_statement(stats: dict[str, float]) -> str:
    low = stats["ci_2_5_percent_db"]
    high = stats["ci_97_5_percent_db"]
    if high < 0:
        return "第一种方法显著更优"
    if low > 0:
        return "第一种方法显著更差"
    return "差异未达到统计显著"


def make_joint_plot(metrics: pd.DataFrame, output: Path) -> None:
    methods = list(DISPLAY)
    indexed = metrics.set_index("method").loc[methods]
    colors = ["#808080", "#d95f02", "#7570b3", "#1b9e77", "#e6ab02", "#66a61e"]
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    positions = np.arange(len(methods))
    bars = ax.bar(
        positions,
        indexed["mean_channel_rmse_db"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    baseline = float(indexed.loc["training_mean", "mean_channel_rmse_db"])
    ax.axhline(baseline, color="black", linestyle="--", linewidth=1, label="Training mean")
    ax.set_xticks(positions, [DISPLAY[m] for m in methods], rotation=24, ha="right")
    ax.set_ylabel("Mean of HH and VV grouped-OOF RMSE (dB)")
    ax.set_title("Unseen-field performance: selective transfer controls negative transfer")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for bar, value in zip(bars, indexed["mean_channel_rmse_db"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.savefig(output, dpi=200)
    plt.close(fig)


def make_selection_plot(summary: dict[str, object], output: Path) -> None:
    families = [
        ("Risk-control weight", summary["selected_shrinkage_weight_frequency"]),
        ("Constraint weight", summary["selected_differential_weight_frequency"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, (title, records) in zip(axes, families):
        candidates = [str(record["candidate"]) for record in records]
        fractions = [float(record["fraction"]) for record in records]
        bars = ax.bar(candidates, fractions, color="#4c78a8")
        ax.set(title=title, xlabel="Inner-CV selected weight", ylabel="Fraction of outer folds")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.22)
        for bar, fraction in zip(bars, fractions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                fraction + 0.025,
                f"{fraction:.0%}",
                ha="center",
                fontsize=9,
            )
    fig.savefig(output, dpi=200)
    plt.close(fig)


def evidence_rows(summary: dict[str, object]) -> pd.DataFrame:
    comparisons = summary["paired_field_bootstrap"]["joint_channels"]
    descriptions = {
        "pretraining_vs_scratch": "解耦物理预训练 vs 解耦从零训练",
        "pretraining_vs_training_mean": "解耦物理预训练 vs 训练均值",
        "risk_control_vs_pretrained": "风险控制迁移 vs 未收缩预训练",
        "risk_control_vs_training_mean": "风险控制迁移 vs 训练均值",
        "risk_control_vs_direct_pretrained": "风险控制迁移 vs 直接双输出预训练",
        "constraint_vs_pretrained": "物理约束 vs 仅预训练",
    }
    rows = []
    for key, description in descriptions.items():
        stats = comparisons[key]
        rows.append(
            {
                "comparison": key,
                "description": description,
                "mean_channel_rmse_delta_db": stats["mean_channel_rmse_delta_db"],
                "ci_2_5_percent_db": stats["ci_2_5_percent_db"],
                "ci_97_5_percent_db": stats["ci_97_5_percent_db"],
                "probability_first_better": stats["probability_first_better"],
                "interpretation": interval_statement(stats),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    summary: dict[str, object], metrics: pd.DataFrame, evidence: pd.DataFrame, output: Path
) -> None:
    indexed = metrics.set_index("method")
    baseline = indexed.loc["training_mean"]
    risk = indexed.loc["mean_common_diff_shrunk_pretrained"]
    pretrain = indexed.loc["mean_common_diff_pretrained"]
    direct = indexed.loc["direct_pretrained"]
    scratch = indexed.loc["mean_common_diff_scratch"]
    comparison = evidence.set_index("comparison")

    def evidence_line(key: str) -> str:
        row = comparison.loc[key]
        return (
            f"ΔRMSE={row.mean_channel_rmse_delta_db:+.4f} dB，"
            f"95% CI [{row.ci_2_5_percent_db:+.4f}, {row.ci_97_5_percent_db:+.4f}]，"
            f"{row.interpretation}"
        )

    shrinkage = ", ".join(
        f"λ={record['candidate']}: {record['fraction']:.0%}"
        for record in summary["selected_shrinkage_weight_frequency"]
    )
    text = f"""# 解耦物理迁移实验：论文证据总结

## 实验范围

- 有效样本：{summary['samples']}；地块：{summary['fields']}；日期：{summary['dates']}。
- 外层按地块分组验证，内层按地块选择超参数；5 次神经网络随机重复；4000 次地块块自助法。
- 主指标：HH 与 VV 的 RMSE 均值。该指标同时保留两种极化，避免选择性报告单通道收益。

## 核心数值

| 方法 | 联合 RMSE (dB) | HH RMSE | VV RMSE |
|---|---:|---:|---:|
| 训练均值 | {baseline.mean_channel_rmse_db:.4f} | {baseline.hh_rmse_db:.4f} | {baseline.vv_rmse_db:.4f} |
| 直接双输出物理预训练 | {direct.mean_channel_rmse_db:.4f} | {direct.hh_rmse_db:.4f} | {direct.vv_rmse_db:.4f} |
| 解耦从零训练 | {scratch.mean_channel_rmse_db:.4f} | {scratch.hh_rmse_db:.4f} | {scratch.vv_rmse_db:.4f} |
| 解耦物理预训练 | {pretrain.mean_channel_rmse_db:.4f} | {pretrain.hh_rmse_db:.4f} | {pretrain.vv_rmse_db:.4f} |
| 风险控制选择性迁移 | {risk.mean_channel_rmse_db:.4f} | {risk.hh_rmse_db:.4f} | {risk.vv_rmse_db:.4f} |

## 可成立的结论

1. 物理预训练显著优于结构相同的从零训练：{evidence_line('pretraining_vs_scratch')}。这支持“低保真物理预训练能稳定小样本极化差异学习”。
2. 解耦且带风险控制的迁移显著优于直接双输出预训练：{evidence_line('risk_control_vs_direct_pretrained')}。这支持“物理知识应选择性注入，而不是无差别施加到所有响应分量”。
3. 风险控制显著修复未收缩预训练的负迁移：{evidence_line('risk_control_vs_pretrained')}。内层选择频率为 {shrinkage}，表明当前数据中可迁移信号具有明显地块依赖性。

## 尚不能宣称的结论

1. 风险控制方法尚未优于训练均值：{evidence_line('risk_control_vs_training_mean')}。因此不能写成“提高未见地块绝对预测精度”。
2. 未收缩预训练反而略差于训练均值：{evidence_line('pretraining_vs_training_mean')}。这说明低保真 SPM 与真实地块之间仍存在域差异。
3. 本实验来自单一观测计划，不能宣称跨区域、跨土壤类型或跨传感器泛化。

## 第一篇论文的合理定位

建议将论文定位为“面向小样本粗糙地表极化后向散射的风险控制选择性物理迁移”，而不是宣称提出通用高精度代理模型。方法贡献由三部分构成：

1. 将共同响应与极化差异响应完全解耦，阻断低保真教师对共同响应的负迁移；
2. 用 SPM 参数化样本只预训练极化差异分支；
3. 用嵌套未见地块验证选择迁移强度，使方法在物理信号不足时自动回退到稳健基线。

当前证据足以形成方法消融和负迁移分析，但尚不足以单独支撑完整投稿。投稿前最少需要新增一个独立站点/年份，或加入 IEM/AIEM 等较高保真教师并在完全相同协议下验证；主张成立的门槛应是风险控制方法在独立域上显著优于训练均值，并保持对直接预训练的显著优势。
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a completed decoupled experiment")
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    input_dir = args.input.resolve()
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(input_dir / "joint_channel_metrics.csv")
    evidence = evidence_rows(summary)
    evidence.to_csv(input_dir / "paper_evidence_table.csv", index=False)
    make_joint_plot(metrics, input_dir / "05_paper_joint_rmse.png")
    make_selection_plot(summary, input_dir / "06_selected_transfer_weights.png")
    write_report(summary, metrics, evidence, input_dir / "RESULTS_INTERPRETATION.md")
    print(f"Paper evidence artifacts saved to: {input_dir}")


if __name__ == "__main__":
    main()
