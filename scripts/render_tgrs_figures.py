"""从冻结结果重绘TGRS论文图。

本程序只读取现有CSV和JSON：
1. 不训练模型；
2. 不选择模型；
3. 不修改冻结结果；
4. --submission 按投稿栏宽覆盖既有16幅PDF，不新增清单或PNG；
5. 历史输出模式保留PDF、PNG和来源清单。
投稿模式重算既有SPM角度敏感性，I²EM损耗面板只使用已保存最大值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


# ============================================================
# 统一风格
# ============================================================

COLORS = {
    "mean": "#333333",
    "scratch": "#7F7F7F",
    "spm": "#0072B2",
    "i2em": "#E69F00",
    "sequential": "#D55E00",
    "residual_ridge": "#009E73",
    "residual_rbf": "#CC79A7",
    "risk_control": "#56B4E9",
    "source_ridge": "#6A3D9A",
}

METHOD_LABELS = {
    "spm_teacher": "SPM teacher",
    "i2em_train_mean": "I²EM train mean",
    "i2em_only": "I²EM only",
    "spm_only": "SPM only",
    "spm_to_i2em": "SPM→I²EM",
    "risk_spm_to_i2em": "Risk-controlled",
    "scratch": "Scratch",
    "two_mean": "Two-channel mean",
    "two_channel_means": "Two-channel mean",
    "common_offset_source_mean": "Common offset",
    "target_ridge_fixed_alpha1": "Target Ridge",
    "residual_ridge": "Residual Ridge",
    "residual_rbf": "Residual RBF",
    "direct_ridge": "Direct Ridge",
    "direct_rbf": "Direct RBF",
}

METHOD_COLORS = {
    "spm_teacher": COLORS["spm"],
    "i2em_train_mean": COLORS["i2em"],
    "i2em_only": COLORS["i2em"],
    "spm_only": COLORS["spm"],
    "spm_to_i2em": COLORS["sequential"],
    "risk_spm_to_i2em": COLORS["risk_control"],
    "scratch": COLORS["scratch"],
    "two_mean": COLORS["mean"],
    "two_channel_means": COLORS["mean"],
    "common_offset_source_mean": "#4D4D4D",
    "target_ridge_fixed_alpha1": COLORS["source_ridge"],
    "residual_ridge": COLORS["residual_ridge"],
    "residual_rbf": COLORS["residual_rbf"],
    "direct_ridge": "#8C6D31",
    "direct_rbf": "#17BECF",
}

METHOD_MARKERS = {
    "spm_teacher": "^",
    "i2em_train_mean": "D",
    "i2em_only": "s",
    "spm_only": "^",
    "spm_to_i2em": "o",
    "risk_spm_to_i2em": "P",
    "scratch": "v",
    "two_mean": "X",
    "two_channel_means": "X",
    "common_offset_source_mean": "d",
    "target_ridge_fixed_alpha1": "<",
    "residual_ridge": "*",
    "residual_rbf": "h",
    "direct_ridge": ">",
    "direct_rbf": "p",
}

METHOD_LINESTYLES = {
    "spm_teacher": ":",
    "i2em_train_mean": ":",
    "i2em_only": "--",
    "spm_only": "-.",
    "spm_to_i2em": "-",
    "risk_spm_to_i2em": "-",
    "scratch": "--",
    "two_mean": ":",
    "two_channel_means": ":",
    "common_offset_source_mean": "--",
    "target_ridge_fixed_alpha1": "-.",
    "residual_ridge": "--",
    "residual_rbf": "-",
    "direct_ridge": "--",
    "direct_rbf": "-.",
}


def apply_tgrs_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def format_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(
        axis=grid_axis,
        color="#D9D9D9",
        linewidth=0.5,
        alpha=0.75,
        zorder=0,
    )
    ax.tick_params(length=3)


def save_figure(
    fig,
    output_without_suffix: Path,
    figure_records: dict,
) -> None:
    output_without_suffix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = output_without_suffix.with_suffix(".pdf")
    png_path = output_without_suffix.with_suffix(".png")

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    figure_records[pdf_path.name] = {
        "pdf": str(pdf_path),
        "png": str(png_path),
        "pdf_sha256": sha256(pdf_path),
        "png_sha256": sha256(png_path),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_source(path: Path, source_records: dict) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    source_records[str(path)] = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def actual_fraction_percent(value: float) -> float:
    mapping = {
        0.05: 6.6666666667,
        0.10: 10.0,
        0.20: 20.0,
    }

    for nominal, actual in mapping.items():
        if np.isclose(value, nominal):
            return actual

    raise ValueError(f"未知少样本比例：{value}")


# ============================================================
# Fig01
# ============================================================

def plot_fig01(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "multifidelity_pretraining_20260910_v1"
        / "evaluation/metrics_summary.csv"
    )
    register_source(path, sources)

    frame = pd.read_csv(path)
    frame = frame[
        frame["response"].isin(["HH", "VV"])
    ].copy()

    order = [
        "spm_teacher",
        "i2em_only",
        "spm_only",
        "spm_to_i2em",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.8),
        sharey=True,
        constrained_layout=True,
    )

    for ax, response, panel in zip(
        axes,
        ["HH", "VV"],
        ["(a)", "(b)"],
    ):
        part = (
            frame[frame["response"].eq(response)]
            .set_index("method")
            .loc[order]
        )

        x = np.arange(len(order))
        y = part["rmse_mean_db"].to_numpy(float)
        yerr = (
            part["rmse_std_db"]
            .fillna(0.0)
            .to_numpy(float)
        )

        for index, method in enumerate(order):
            ax.errorbar(
                x[index],
                y[index],
                yerr=yerr[index],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=5,
                capsize=3,
                linestyle="none",
                elinewidth=0.9,
                zorder=3,
            )

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([
            "SPM\nteacher",
            "I²EM\nonly",
            "SPM\nonly",
            "SPM→\nI²EM",
        ])
        ax.set_xlabel("Surrogate or teacher")
        ax.set_title(f"{response} polarization")
        add_panel_label(ax, panel)
        format_axis(ax)

    axes[0].set_ylabel(
        "RMSE against independent I²EM test set (dB)"
    )

    save_figure(
        fig,
        output / "main/Fig01_synthetic_teacher_test",
        figures,
    )


# ============================================================
# Fig02
# ============================================================

def plot_fig02(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "multifidelity_sample_efficiency_20260910_v1"
        / "metrics_summary.csv"
    )
    register_source(path, sources)

    frame = pd.read_csv(path)
    frame = frame[
        frame["response"].isin(["HH", "VV"])
        & frame["method"].isin([
            "i2em_only",
            "spm_to_i2em",
        ])
    ].copy()

    budgets = [32, 64, 128, 256]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.8),
        sharey=True,
        constrained_layout=True,
    )

    for ax, response, panel in zip(
        axes,
        ["HH", "VV"],
        ["(a)", "(b)"],
    ):
        part = frame[
            frame["response"].eq(response)
        ]

        for method in [
            "i2em_only",
            "spm_to_i2em",
        ]:
            curve = (
                part[part["method"].eq(method)]
                .sort_values("i2em_train_samples")
            )

            ax.errorbar(
                curve["i2em_train_samples"],
                curve["rmse_mean_db"],
                yerr=curve["rmse_std_db"].fillna(0.0),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                capsize=3,
                label=METHOD_LABELS[method],
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(v) for v in budgets])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Number of I²EM training samples")
        ax.set_title(f"{response} polarization")
        add_panel_label(ax, panel)
        format_axis(ax)

    axes[0].set_ylabel(
        "RMSE against independent I²EM test set (dB)"
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
    )

    save_figure(
        fig,
        output / "main/Fig02_i2em_sample_efficiency",
        figures,
    )


# ============================================================
# Fig03
# ============================================================

def plot_fig03(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "residual_baselines_v1"
        / "summary.csv"
    )
    register_source(path, sources)

    frame = pd.read_csv(path)

    methods = [
        "i2em_only",
        "spm_to_i2em",
        "residual_ridge",
        "residual_rbf",
    ]

    frame = frame[
        frame["method"].isin(methods)
    ].copy()

    fig, ax = plt.subplots(
        figsize=(3.5, 2.85),
        constrained_layout=True,
    )

    for method in methods:
        curve = (
            frame[frame["method"].eq(method)]
            .sort_values("size")
        )

        ax.errorbar(
            curve["size"],
            curve["rmse_mean_db"],
            yerr=curve["rmse_std_db"].fillna(0.0),
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            capsize=2.5,
            label=METHOD_LABELS[method],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([32, 64, 128, 256])
    ax.set_xticklabels(["32", "64", "128", "256"])
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("Number of I²EM training samples")
    ax.set_ylabel("Mean HH/VV RMSE (dB)")
    format_axis(ax)

    ax.legend(
        loc="upper right",
        ncol=1,
    )

    save_figure(
        fig,
        output / "main/Fig03_residual_baselines",
        figures,
    )


# ============================================================
# Fig04
# ============================================================

def plot_fig04(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "paper_submission_closeout_20260911_v1"
        / "paper_audit_supplement_v1/summary.json"
    )
    register_source(path, sources)

    content = json.loads(
        path.read_text(encoding="utf-8")
    )
    results = content["results"]

    fractions = np.array([
        row["actual_fraction"] * 100.0
        for row in results
    ])

    methods = [
        "scratch",
        "spm_only",
        "spm_to_i2em",
        "two_mean",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.9),
        constrained_layout=True,
    )

    for method in methods:
        values = np.array([
            row["rmse_db"][method]
            for row in results
        ])

        axes[0].plot(
            fractions,
            values,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )

    axes[0].set_ylabel("Mean HH/VV RMSE (dB)")
    axes[0].set_xlabel("Target fields used for adaptation")
    add_panel_label(axes[0], "(a)")
    format_axis(axes[0])

    contrasts = [
        row["contrasts"]["spm_only_minus_two_mean"]
        for row in results
    ]

    delta = np.array([
        row["delta_db"]
        for row in contrasts
    ])
    intervals = np.array([
        row["conditional_field_reweighting_interval"]
        for row in contrasts
    ])

    lower = delta - intervals[:, 0]
    upper = intervals[:, 1] - delta

    axes[1].errorbar(
        fractions,
        delta,
        yerr=np.vstack([lower, upper]),
        color=COLORS["spm"],
        marker="o",
        linestyle="none",
        capsize=4,
        elinewidth=1.0,
    )
    axes[1].axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1].set_ylabel(
        "SPM minus two-channel mean RMSE (dB)"
    )
    axes[1].set_xlabel(
        "Target fields used for adaptation"
    )
    add_panel_label(axes[1], "(b)")
    format_axis(axes[1])

    tick_labels = [
        "6.67%\n(2 fields)",
        "10%\n(3 fields)",
        "20%\n(6 fields)",
    ]

    for ax in axes:
        ax.set_xticks(fractions)
        ax.set_xticklabels(tick_labels)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
    )

    save_figure(
        fig,
        output / "main/Fig04_calibration_and_contrasts",
        figures,
    )


# ============================================================
# Fig05
# ============================================================

def plot_fig05(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "paper_submission_closeout_20260911_v1"
        / "response_learning_v1/response_summary.csv"
    )
    register_source(path, sources)

    frame = pd.read_csv(path)
    frame = frame[
        frame["component"].eq("differential")
    ].copy()

    methods = [
        "scratch",
        "spm_only",
        "spm_to_i2em",
        "risk_spm_to_i2em",
        "two_mean",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.9),
        constrained_layout=True,
    )

    for method in methods:
        part = (
            frame[frame["method"].eq(method)]
            .sort_values("fraction")
        )

        x = np.array([
            actual_fraction_percent(value)
            for value in part["fraction"]
        ])

        axes[0].plot(
            x,
            part["centered_skill"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )

        axes[1].plot(
            x,
            part["variance_ratio"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
        )

    axes[0].axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )
    axes[0].set_ylabel(
        "Centered skill of differential response"
    )
    axes[0].set_xlabel(
        "Target fields used for adaptation (%)"
    )
    add_panel_label(axes[0], "(a)")
    format_axis(axes[0])

    axes[1].axhline(
        1.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel(
        "Predicted-to-observed variance ratio"
    )
    axes[1].set_xlabel(
        "Target fields used for adaptation (%)"
    )
    add_panel_label(axes[1], "(b)")
    format_axis(axes[1])

    inset = inset_axes(
        axes[1],
        width="48%",
        height="46%",
        loc="upper right",
        borderpad=1.0,
    )

    for method in methods:
        part = (
            frame[frame["method"].eq(method)]
            .sort_values("fraction")
        )
        x = np.array([
            actual_fraction_percent(value)
            for value in part["fraction"]
        ])

        inset.plot(
            x,
            part["variance_ratio"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=0.9,
            markersize=3,
        )

    inset.set_ylim(-0.005, 0.22)
    inset.set_xticks([6.67, 10, 20])
    inset.set_xticklabels(["6.67", "10", "20"])
    inset.tick_params(labelsize=5.5, length=2)
    inset.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.4,
    )

    for ax in axes:
        ax.set_xticks([6.67, 10, 20])
        ax.set_xticklabels(["6.67", "10", "20"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
    )

    save_figure(
        fig,
        output / "main/Fig05_centered_response",
        figures,
    )


# ============================================================
# Fig06
# ============================================================

def prepare_campaign_components(
    frame: pd.DataFrame,
    angle_column: str,
) -> pd.DataFrame:
    result = pd.DataFrame({
        "common": (
            frame["sigma0_hh_db"].to_numpy(float)
            + frame["sigma0_vv_db"].to_numpy(float)
        ) / 2.0,
        "differential": (
            frame["sigma0_vv_db"].to_numpy(float)
            - frame["sigma0_hh_db"].to_numpy(float)
        ),
        "angle": frame[angle_column].to_numpy(float),
    })
    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()


def plot_fig06(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    source_path = (
        base
        / "portable_source_v1"
        / "smapvex12_portable_source.csv"
    )
    target_path = (
        base
        / "smex02_external_final_20260911_v1"
        / "smex02_field_day_model_ready.csv"
    )

    register_source(source_path, sources)
    register_source(target_path, sources)

    source = pd.read_csv(source_path)
    target = pd.read_csv(target_path)

    source_values = prepare_campaign_components(
        source,
        "nominal_incidence_angle_deg",
    )
    target_values = prepare_campaign_components(
        target,
        "incidence_angle_deg",
    )

    configurations = [
        (
            "common",
            "Common response (dB)",
            "(a)",
        ),
        (
            "differential",
            "Differential response VV-HH (dB)",
            "(b)",
        ),
        (
            "angle",
            "Incidence angle (degree)",
            "(c)",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.6),
        constrained_layout=True,
    )

    for ax, (column, xlabel, panel) in zip(
        axes,
        configurations,
    ):
        combined = np.concatenate([
            source_values[column].to_numpy(float),
            target_values[column].to_numpy(float),
        ])

        edges = np.histogram_bin_edges(
            combined,
            bins="auto",
        )

        ax.hist(
            source_values[column],
            bins=edges,
            density=True,
            histtype="stepfilled",
            alpha=0.28,
            color=COLORS["spm"],
            edgecolor=COLORS["spm"],
            linewidth=1.0,
            label=f"SMAPVEX12 (n={len(source_values)})",
        )
        ax.hist(
            target_values[column],
            bins=edges,
            density=True,
            histtype="stepfilled",
            alpha=0.25,
            color=COLORS["sequential"],
            edgecolor=COLORS["sequential"],
            linewidth=1.0,
            label=f"SMEX02 (n={len(target_values)})",
        )

        ax.axvline(
            np.median(source_values[column]),
            color=COLORS["spm"],
            linestyle="--",
            linewidth=0.9,
        )
        ax.axvline(
            np.median(target_values[column]),
            color=COLORS["sequential"],
            linestyle=":",
            linewidth=0.9,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        add_panel_label(ax, panel)
        format_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
    )

    save_figure(
        fig,
        output / "main/figS07_campaign_distributions",
        figures,
    )


# ============================================================
# FigS01
# ============================================================

def plot_figs01(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "paper_submission_closeout_20260911_v1"
        / "response_learning_v1"
        / "spm_controlled_angle_scan.csv"
    )
    register_source(path, sources)

    frame = pd.read_csv(path)

    # 文件保存的是共同和差分分量，在此还原HH和VV。
    frame["hh"] = (
        frame["common"]
        - frame["differential"] / 2.0
    )
    frame["vv"] = (
        frame["common"]
        + frame["differential"] / 2.0
    )

    grouped = (
        frame.groupby(
            ["angle", "loss_tangent"],
            as_index=False,
        )[["hh", "vv"]]
        .median()
    )

    loss_values = sorted(
        grouped["loss_tangent"].unique()
    )

    loss_colors = [
        "#333333",
        "#0072B2",
        "#E69F00",
        "#D55E00",
    ]
    loss_markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.8),
        sharey=True,
        constrained_layout=True,
    )

    for ax, response, panel in zip(
        axes,
        ["hh", "vv"],
        ["(a)", "(b)"],
    ):
        for index, loss in enumerate(loss_values):
            part = (
                grouped[
                    np.isclose(
                        grouped["loss_tangent"],
                        loss,
                    )
                ]
                .sort_values("angle")
            )

            ax.plot(
                part["angle"],
                part[response],
                color=loss_colors[index],
                marker=loss_markers[index],
                linestyle=[
                    "-",
                    "--",
                    "-.",
                    ":",
                ][index],
                label=rf"$\tan\delta={loss:g}$",
            )

        ax.axvline(
            40.0,
            color="#666666",
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_xlabel("Incidence angle (degree)")
        ax.set_title(response.upper())
        add_panel_label(ax, panel)
        format_axis(ax)

    axes[0].set_ylabel(
        "Median SPM backscatter (dB)"
    )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(loss_values),
    )

    save_figure(
        fig,
        output / "supplementary/figS01_angle_loss",
        figures,
    )


# ============================================================
# FigS02
# ============================================================

def plot_figs02(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "paper_submission_closeout_20260911_v1"
        / "response_learning_v1/response_summary.csv"
    )
    register_source(path, sources)

    frame = pd.read_csv(path)

    methods = [
        "scratch",
        "spm_only",
        "spm_to_i2em",
        "risk_spm_to_i2em",
        "two_mean",
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.16, 5.1),
        constrained_layout=True,
    )

    configurations = [
        (
            "common",
            "centered_skill",
            "Common response centered skill",
            "(a)",
        ),
        (
            "common",
            "variance_ratio",
            "Common response variance ratio",
            "(b)",
        ),
        (
            "differential",
            "centered_skill",
            "Differential response centered skill",
            "(c)",
        ),
        (
            "differential",
            "variance_ratio",
            "Differential response variance ratio",
            "(d)",
        ),
    ]

    for ax, (
        component,
        metric,
        ylabel,
        panel,
    ) in zip(axes.ravel(), configurations):
        for method in methods:
            part = (
                frame[
                    frame["component"].eq(component)
                    & frame["method"].eq(method)
                ]
                .sort_values("fraction")
            )

            x = [
                actual_fraction_percent(value)
                for value in part["fraction"]
            ]

            ax.plot(
                x,
                part[metric],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method],
                label=METHOD_LABELS[method],
            )

        reference = 0.0 if metric == "centered_skill" else 1.0
        ax.axhline(
            reference,
            color="#333333",
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(
            "Target fields used for adaptation (%)"
        )
        ax.set_xticks([6.67, 10, 20])
        ax.set_xticklabels(["6.67", "10", "20"])
        add_panel_label(ax, panel)
        format_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
    )

    save_figure(
        fig,
        output / "supplementary/FigS02_all_centered_controls",
        figures,
    )


# ============================================================
# FigS03
# ============================================================

def plot_figs03(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "paper_submission_closeout_20260911_v1"
        / "review_20260914_v1"
        / "dielectric_sensitivity_v1/scores.json"
    )
    register_source(path, sources)

    frame = pd.DataFrame(
        json.loads(path.read_text(encoding="utf-8"))
    )

    methods = [
        "spm_only",
        "spm_to_i2em",
        "two_channel_means",
    ]

    selected = frame[
        frame["method"].isin(methods)
    ].copy()

    index_columns = ["fraction", "method"]

    unified = (
        selected[selected["arm"].eq("unified")]
        .set_index(index_columns)
    )
    original = (
        selected[selected["arm"].eq("original")]
        .set_index(index_columns)
    )

    if not unified.index.equals(original.index):
        original = original.reindex(unified.index)

    delta = pd.DataFrame({
        "fraction": [
            item[0]
            for item in unified.index
        ],
        "method": [
            item[1]
            for item in unified.index
        ],
        "rmse_delta": (
            original["rmse_mean_hh_vv_db"]
            - unified["rmse_mean_hh_vv_db"]
        ).to_numpy(),
        "skill_delta": (
            original["centered_skill_differential"]
            - unified["centered_skill_differential"]
        ).to_numpy(),
    })

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.9),
        constrained_layout=True,
    )

    for method in methods:
        part = (
            delta[delta["method"].eq(method)]
            .sort_values("fraction")
        )

        x = [
            actual_fraction_percent(value)
            for value in part["fraction"]
        ]

        axes[0].plot(
            x,
            part["rmse_delta"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            x,
            part["skill_delta"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
        )

    axes[0].set_ylabel(
        "Original minus unified-input RMSE (dB)"
    )
    axes[1].set_ylabel(
        "Original minus unified-input centered skill"
    )

    for ax, panel in zip(
        axes,
        ["(a)", "(b)"],
    ):
        ax.axhline(
            0.0,
            color="#333333",
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_xlabel(
            "Target fields used for adaptation (%)"
        )
        ax.set_xticks([6.67, 10, 20])
        ax.set_xticklabels(["6.67", "10", "20"])
        add_panel_label(ax, panel)
        format_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
    )

    save_figure(
        fig,
        output / "supplementary/figS03_dielectric_sensitivity",
        figures,
    )


# ============================================================
# FigS04
# ============================================================

def plot_figs04(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "controlled_mismatch_v1"
        / "summary.csv"
    )
    register_source(path, sources)

    frame = pd.read_csv(path)
    frame = frame[
        np.isclose(frame["noise_db"], 0.0)
        & frame["size"].eq(32)
    ].copy()

    scenarios = [
        "none",
        "offset",
        "gain",
        "nonlinear",
    ]

    methods = [
        "unadapted",
        "adaptation_mean",
        "offset_calibrated",
        "oracle_reference",
    ]

    method_labels = {
        "unadapted": "Unadapted",
        "adaptation_mean": "Adaptation mean",
        "offset_calibrated": "Offset calibrated",
        "oracle_reference": "Oracle reference",
    }

    method_colors = {
        "unadapted": COLORS["scratch"],
        "adaptation_mean": COLORS["mean"],
        "offset_calibrated": COLORS["spm"],
        "oracle_reference": COLORS["i2em"],
    }

    method_markers = {
        "unadapted": "v",
        "adaptation_mean": "X",
        "offset_calibrated": "o",
        "oracle_reference": "D",
    }

    scenario_labels = {
        "none": "None",
        "offset": "Offset",
        "gain": "Gain",
        "nonlinear": "Nonlinear",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.9),
        constrained_layout=True,
    )

    x = np.arange(len(scenarios))

    for method_index, method in enumerate(methods):
        part = (
            frame[frame["method"].eq(method)]
            .set_index("scenario")
            .reindex(scenarios)
        )

        offset = (
            method_index - (len(methods) - 1) / 2
        ) * 0.08

        fill = (
            "none"
            if method == "oracle_reference"
            else method_colors[method]
        )

        axes[0].scatter(
            x + offset,
            part["rmse_db"],
            color=method_colors[method],
            facecolors=fill,
            marker=method_markers[method],
            s=28,
            linewidth=1.0,
            label=method_labels[method],
            zorder=3,
        )

        axes[1].scatter(
            x + offset,
            part["centered_rmse_db"],
            color=method_colors[method],
            facecolors=fill,
            marker=method_markers[method],
            s=28,
            linewidth=1.0,
            zorder=3,
        )

    axes[0].set_ylabel("Mean HH/VV RMSE (dB)")
    axes[1].set_ylabel("Centered RMSE (dB)")

    for ax, panel in zip(
        axes,
        ["(a)", "(b)"],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([
            scenario_labels[value]
            for value in scenarios
        ])
        ax.set_xlabel("Controlled mismatch scenario")
        add_panel_label(ax, panel)
        format_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
    )

    save_figure(
        fig,
        output / "supplementary/figS04_controlled_mismatch",
        figures,
    )


# ============================================================
# FigS05
# ============================================================

def plot_figs05(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "smex02_common_cohort_20260921_v1"
        / "common_cohort_metrics.csv"
    )

    if not path.exists():
        warnings.warn(
            "未发现138条共同样本结果，跳过FigS05。"
            "请先运行postprocess_smex02_common_cohort.py。"
        )
        return

    register_source(path, sources)
    frame = pd.read_csv(path)

    methods = [
        "Source mean",
        "SPM",
        "I2EM",
        "SPM only",
        "I2EM only",
        "SPM to I2EM",
        "Risk-controlled transfer",
    ]

    colors = {
        "Source mean": COLORS["mean"],
        "SPM": COLORS["spm"],
        "I2EM": COLORS["i2em"],
        "SPM only": COLORS["spm"],
        "I2EM only": COLORS["i2em"],
        "SPM to I2EM": COLORS["sequential"],
        "Risk-controlled transfer": COLORS["risk_control"],
    }

    markers = {
        "Source mean": "X",
        "SPM": "^",
        "I2EM": "s",
        "SPM only": "v",
        "I2EM only": "D",
        "SPM to I2EM": "o",
        "Risk-controlled transfer": "P",
    }

    labels = [
        "Source\nmean",
        "SPM",
        "I²EM",
        "SPM\nonly",
        "I²EM\nonly",
        "SPM→\nI²EM",
        "Risk-\ncontrolled",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.9),
        constrained_layout=True,
    )

    x = np.arange(len(methods))

    for component, offset, marker in [
        ("common", -0.11, "o"),
        ("differential", 0.11, "s"),
    ]:
        part = (
            frame[frame["component"].eq(component)]
            .set_index("method")
            .reindex(methods)
        )

        axes[0].scatter(
            x + offset,
            part["centered_skill"],
            c=[colors[m] for m in methods],
            marker=marker,
            s=28,
            edgecolors="#333333",
            linewidths=0.35,
            label=component.capitalize(),
            zorder=3,
        )

        axes[1].scatter(
            x + offset,
            part["bias_db"],
            c=[colors[m] for m in methods],
            marker=marker,
            s=28,
            edgecolors="#333333",
            linewidths=0.35,
            zorder=3,
        )

    axes[0].axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1].axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )

    axes[0].set_ylabel("Centered skill")
    axes[1].set_ylabel("Prediction bias (dB)")

    for ax, panel in zip(
        axes,
        ["(a)", "(b)"],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        add_panel_label(ax, panel)
        format_axis(ax)

    axes[0].legend(
        loc="lower left",
        ncol=2,
    )

    save_figure(
        fig,
        output / "supplementary/FigS05_smex02_common_cohort",
        figures,
    )


# ============================================================
# FigS06
# ============================================================

def binned_median(
    x: np.ndarray,
    y: np.ndarray,
    number_of_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )
    edges = np.unique(
        np.quantile(x, quantiles)
    )

    x_values = []
    y_values = []

    for lower, upper in zip(
        edges[:-1],
        edges[1:],
    ):
        if upper == edges[-1]:
            mask = (x >= lower) & (x <= upper)
        else:
            mask = (x >= lower) & (x < upper)

        if mask.sum() == 0:
            continue

        x_values.append(float(np.median(x[mask])))
        y_values.append(float(np.median(y[mask])))

    return np.array(x_values), np.array(y_values)


def plot_figs06(
    base: Path,
    output: Path,
    sources: dict,
    figures: dict,
) -> None:
    path = (
        base
        / "multifidelity_pretraining_20260910_v1"
        / "evaluation/paired_teacher_disagreement.csv"
    )
    register_source(path, sources)

    frame = pd.read_csv(path)

    if "split" in frame.columns:
        frame = frame[
            frame["split"].eq("test")
        ].copy()

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[
            "i2em_validity_utilization",
            "teacher_absolute_delta_hh_db",
            "teacher_absolute_delta_vv_db",
            "validity_stratum",
        ]
    )

    stratum_colors = {
        "interior_le_0.4": "#0072B2",
        "middle_0.4_to_0.7": "#E69F00",
        "nearer_boundary_gt_0.7": "#D55E00",
    }

    stratum_labels = {
        "interior_le_0.4": "Interior",
        "middle_0.4_to_0.7": "Middle",
        "nearer_boundary_gt_0.7": "Near boundary",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.85),
        sharex=True,
        constrained_layout=True,
    )

    for ax, response, column, panel in [
        (
            axes[0],
            "HH",
            "teacher_absolute_delta_hh_db",
            "(a)",
        ),
        (
            axes[1],
            "VV",
            "teacher_absolute_delta_vv_db",
            "(b)",
        ),
    ]:
        for stratum in [
            "interior_le_0.4",
            "middle_0.4_to_0.7",
            "nearer_boundary_gt_0.7",
        ]:
            part = frame[
                frame["validity_stratum"].eq(stratum)
            ]

            ax.scatter(
                part["i2em_validity_utilization"],
                part[column],
                s=15,
                alpha=0.58,
                color=stratum_colors[stratum],
                edgecolors="none",
                label=stratum_labels[stratum],
            )

        x = frame[
            "i2em_validity_utilization"
        ].to_numpy(float)
        y = frame[column].to_numpy(float)

        x_median, y_median = binned_median(
            x,
            y,
            number_of_bins=5,
        )

        ax.plot(
            x_median,
            y_median,
            color="#222222",
            marker="o",
            linewidth=1.1,
            label="Binned median",
            zorder=4,
        )

        ax.set_xlabel("I²EM validity-domain utilization")
        ax.set_title(f"{response} polarization")
        add_panel_label(ax, panel)
        format_axis(ax)

    axes[0].set_ylabel(
        "Absolute I²EM-SPM disagreement (dB)"
    )

    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
    )

    save_figure(
        fig,
        output / "supplementary/figS06_teacher_domain",
        figures,
    )


# ============================================================
# 主程序
# ============================================================


def render_compact_revision(root: Path, output: Path) -> None:
    """Render the two revised assets from the included compact result bundle."""
    data = root / 'reproducibility/results'
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(data/'residual_baselines/summary.csv')
    fig = plt.figure(figsize=(7.16, 3.6), layout='constrained')
    ax = fig.add_subplot(111)
    series = [('i2em_only', 'I$^2$EM-only neural', 'o', '-'),
              ('spm_to_i2em', 'Sequential neural', 's', '-'),
              ('spm_teacher', 'SPM (zero I$^2$EM labels)', None, '--'),
              ('direct_rbf', 'Direct RBF', '^', ':'),
              ('residual_ridge', 'Residual Ridge', 'D', '-.'),
              ('residual_rbf', 'Residual RBF', 'v', '-')]
    for method, label, marker, line in series:
        curve = summary[summary.method.eq(method)].sort_values('size')
        ax.errorbar(curve['size'], curve.rmse_mean_db, yerr=curve.rmse_std_db,
                    label=label, marker=marker, linestyle=line, capsize=3, linewidth=1.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([32, 64, 128, 256], ['32', '64', '128', '256'])
    ax.set_xlabel('Number of I$^2$EM training labels')
    ax.set_ylabel('Mean HH/VV RMSE against I$^2$EM (dB)')
    ax.tick_params(labelsize=9)
    ax.legend(ncols=2, fontsize=8, loc='lower left')
    ax.grid(axis='y', alpha=0.25)
    fig.savefig(output/'fig03_sample_efficiency.pdf', bbox_inches='tight')
    plt.close(fig)

    table = pd.read_csv(data/'stage_diagnostics/stage_metrics.csv').set_index('method')
    order = ['i2em_fixed40_teacher', 'pretraining_only_surrogate',
             'source_finetuned_surrogate', 'source_reset_adam_003',
             'pretraining_source_offset']
    labels = [r'I$^2$EM teacher (40$^\circ$)', 'Pretraining only',
              'Source fine-tuning (inherited Adam)', 'Source fine-tuning (reset Adam)',
              'Pretraining + source intercept']
    values = table.loc[order]
    fig = plt.figure(figsize=(7.16, 3.0), layout='constrained')
    ax = fig.add_subplot(111)
    for j, method in enumerate(order):
        row = values.loc[method]
        point = row.centered_skill
        ax.errorbar(point, j, xerr=[[point-row.centered_skill_ci_low],
                                   [row.centered_skill_ci_high-point]],
                    fmt='D' if method == 'pretraining_source_offset' else ('s' if method == 'source_reset_adam_003' else 'o'), capsize=3, markersize=6)
    ax.axvline(0, linestyle='--', linewidth=0.9)
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Centered differential skill (95% field-block interval)')
    ax.tick_params(labelsize=9)
    ax.set_xlim(-0.77, 0.29)
    ax.grid(axis='x', alpha=0.25)
    fig.savefig(output/'fig05_stage_retention.pdf', bbox_inches='tight')
    plt.close(fig)


def render_submission(root: Path, output: Path) -> None:
    """Redraw the 16 existing PDF assets at their final IEEE column widths.

    Read the bundled summaries without changing predictions or fitted models.
    The SPM angle/loss panel repeats the existing deterministic source scan;
    the I2EM loss panel shows the recorded maxima (per-loss rows are not bundled).
    This mode writes PDF assets only, and may overwrite those same asset names.
    """
    import sys
    from matplotlib.ticker import MaxNLocator

    results = root / "reproducibility/results"
    output.mkdir(parents=True, exist_ok=True)
    apply_tgrs_style()
    plt.rcParams.update({
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
        "axes.linewidth": 0.7, "lines.linewidth": 1.2, "lines.markersize": 5,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    labels = dict(METHOD_LABELS, spm_to_i2em="Sequential", scratch="Random weights",
                  spm_only="SPM pretraining", i2em_only="Direct neural",
                  risk_spm_to_i2em="Archived shrinkage")
    generated = []

    def read(relative: str) -> pd.DataFrame:
        path = results / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required bundled plotting input: {path}")
        return pd.read_csv(path)

    def style(ax, panel: str = ""):
        ax.grid(axis="y", color="0.87", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.7, length=3, pad=3)
        if panel:
            ax.set_title(f"({panel})", loc="left", pad=7, fontweight="bold")

    def line(ax, x, y, method, **kwargs):
        return ax.plot(x, y, label=labels.get(method, method),
                       color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                       linestyle=METHOD_LINESTYLES[method], markerfacecolor="white",
                       markeredgewidth=1.0, **kwargs)

    def fields(ax):
        ax.set_xticks([2, 3, 6]); ax.set_xlim(1.65, 6.35)
        ax.set_xlabel("Adaptation fields (of 30)")

    def legend(fig, ax, ncol=3, y=1.0):
        handles, names = ax.get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, y),
                   ncol=ncol, frameon=False, handlelength=2.3, columnspacing=1.2)

    def save(fig, name):
        fig.savefig(output / name, format="pdf", bbox_inches=None,
                    metadata={"Creator": "Bundled TGRS figure renderer", "CreationDate": None})
        plt.close(fig); generated.append(name)

    # Teacher comparison: the fixed test cohort and the recorded loss maxima.
    paired = read("multifidelity/paired_teacher_disagreement.csv")
    test = paired.loc[paired["split"].eq("test")].copy()
    if len(test) != 128:
        raise ValueError("The teacher figure requires the fixed 128-sample test cohort.")
    formal = json.loads((results / "paper_summaries/phase1_summary.json").read_text())["formal_multifidelity_test"]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.25))
    fig.subplots_adjust(left=.10, right=.98, bottom=.16, top=.94, wspace=.34, hspace=.54)
    for ax, pol, panel in zip(axes[0], ["hh", "vv"], ["a", "b"]):
        x, y = test[f"spm_{pol}_db"], test[f"i2em_{pol}_db"]
        limits = [min(x.min(), y.min()) - .15, max(x.max(), y.max()) + .15]
        ax.plot(limits, limits, "--", color=COLORS["mean"], linewidth=.9)
        ax.scatter(x, y, s=16, color=COLORS["spm"], edgecolors="none")
        ax.set(xlim=limits, ylim=limits, xlabel=f"SPM {pol.upper()} (dB)",
               ylabel=f"I²EM {pol.upper()} (dB)")
        ax.text(.03, .94, f"MAE = {np.mean(np.abs(x-y)):.3f} dB", transform=ax.transAxes, va="top")
        style(ax, panel)
    for stratum, marker, color, title in [
        ("interior_le_0.4", "o", COLORS["spm"], "Interior (101)"),
        ("middle_0.4_to_0.7", "s", COLORS["i2em"], "Intermediate (24)"),
        ("nearer_boundary_gt_0.7", "^", COLORS["sequential"], "Near boundary (3)")]:
        part = test.loc[test.validity_stratum.eq(stratum)]
        axes[1, 0].scatter(part.i2em_validity_utilization,
                           (part.teacher_absolute_delta_hh_db + part.teacher_absolute_delta_vv_db)/2,
                           s=19, marker=marker, color=color, label=title)
    u = test.i2em_validity_utilization.to_numpy()
    delta = (test.teacher_absolute_delta_hh_db + test.teacher_absolute_delta_vv_db).to_numpy()/2
    centers, values = [], []
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        mask = (u > lo) & (u <= hi)
        if mask.any(): centers.append(np.median(u[mask])); values.append(np.median(delta[mask]))
    axes[1, 0].plot(centers, values, "k--", linewidth=1, label="Binned median")
    axes[1, 0].set(xlabel="Domain utilization, u", ylabel="Mean |I²EM − SPM| (dB)", xlim=(0, 1))
    handles, names = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .004),
               ncol=4, frameon=False, fontsize=9, handlelength=1.4, columnspacing=1.0)
    style(axes[1, 0], "c")
    ax = axes[1, 1]
    vals = [formal["maximum_loss_change_hh_db"], formal["maximum_loss_change_vv_db"]]
    for j, value in enumerate(vals):
        ax.vlines(j, 0, value, color=COLORS["i2em"], linewidth=1.2)
        ax.plot(j, value, "o" if j == 0 else "s", color=COLORS["i2em"])
        ax.annotate(f"{value:.4f}", (j, value), xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set(xticks=[0, 1], xticklabels=["HH", "VV"], xlim=(-.6, 1.6), ylim=(0, .047),
           ylabel="Maximum absolute change (dB)", xlabel="Loss tangent: 0–0.10; 18 anchors")
    style(ax, "d"); save(fig, "fig02_teacher_comparison.pdf")

    # Matched high-fidelity label budgets; error bars are repeat standard deviations.
    frame = read("residual_baselines/summary.csv")
    fig, ax = plt.subplots(figsize=(7.16, 3.65))
    fig.subplots_adjust(left=.105, right=.98, bottom=.17, top=.73)
    for method in ["spm_teacher", "i2em_only", "spm_to_i2em", "direct_ridge", "direct_rbf", "residual_ridge", "residual_rbf"]:
        g = frame.loc[frame.method.eq(method)].sort_values("size")
        mean, sd = g.rmse_mean_db.to_numpy(), g.rmse_std_db.to_numpy()
        if np.any(mean - sd <= 0):
            raise ValueError("A log-scale SD interval extends to a nonpositive value.")
        ax.errorbar(g["size"], mean, yerr=sd, color=METHOD_COLORS[method],
                    marker=METHOD_MARKERS[method], linestyle=METHOD_LINESTYLES[method],
                    markerfacecolor="white", capsize=2.5, label=labels[method])
    ax.set(yscale="log", xticks=[32, 64, 128, 256], xlabel="I²EM training labels",
           ylabel="Average HH/VV RMSE (dB)")
    style(ax); legend(fig, ax, 3); save(fig, "fig03_sample_efficiency.pdf")

    # Source-stage point estimates and paired field-bootstrap intervals.
    stage = read("stage_diagnostics/stage_metrics.csv").set_index("method")
    keys = ["i2em_fixed40_teacher", "pretraining_only_surrogate", "source_finetuned_surrogate", "source_reset_adam_003", "pretraining_source_offset"]
    names = ["I²EM, fixed 40°", "Pretraining only", "Source, inherited Adam", "Source, reset Adam", "Pretraining + source intercept"]
    fig, ax = plt.subplots(figsize=(7.16, 2.80))
    fig.subplots_adjust(left=.34, right=.97, bottom=.21, top=.96)
    for j, (key, color, marker) in enumerate(zip(keys, [COLORS[k] for k in ["i2em", "spm", "sequential", "residual_ridge", "source_ridge"]], ["s", "o", "^", "D", "P"])):
        row = stage.loc[key]; v = row.centered_skill
        ax.errorbar(v, j, xerr=[[v-row.centered_skill_ci_low], [row.centered_skill_ci_high-v]],
                    color=color, marker=marker, capsize=3, linestyle="none")
    ax.set(yticks=range(5), yticklabels=names, xlabel="Differential centered skill", xlim=(-.78, .28), ylim=(4.6, -.6))
    ax.axvline(0, color="0.3", linestyle="--", linewidth=.9); style(ax)
    save(fig, "fig05_stage_retention.pdf")

    # Label-information groups remain separate, each with its own matched mean.
    matched = read("matched_information/metrics_summary.csv")
    for name, methods in [
        ("fig04a_physics_offset.pdf", ["two_channel_mean", "i2em_actual_angle_plus_offset"]),
        ("fig04b_neural_transfer.pdf", ["two_channel_mean_all_adaptation", "spm_only", "spm_to_i2em", "risk_spm_to_i2em"])]:
        fig, ax = plt.subplots(figsize=(3.50, 3.00))
        fig.subplots_adjust(left=.19, right=.97, bottom=.18, top=.75)
        for j, method in enumerate(methods):
            g = matched.loc[matched.method.eq(method)].sort_values("requested_fraction")
            alias = "two_mean" if method.startswith("two_channel_mean") else ("i2em_only" if method.startswith("i2em_actual") else method)
            title = "Matched mean" if alias == "two_mean" else ("I²EM + offsets" if alias == "i2em_only" else labels[alias])
            val = g.mean_channel_rmse_db.to_numpy()
            # Horizontal offsets separate interval caps without changing field counts.
            x = np.array([2., 3., 6.]) + (j-(len(methods)-1)/2)*.055
            ax.errorbar(x, val, yerr=[val-g.conditional_ci_low_db.to_numpy(), g.conditional_ci_high_db.to_numpy()-val],
                        color=METHOD_COLORS[alias], marker=METHOD_MARKERS[alias],
                        linestyle=METHOD_LINESTYLES[alias], capsize=2, markerfacecolor="white", label=title)
        ax.set_ylabel("Average HH/VV RMSE (dB)"); fields(ax); style(ax)
        legend(fig, ax, 1 if len(methods)==2 else 2, 1.0)
        save(fig, name)

    response = read("diagnostics/response_summary.csv")
    calibration = sorted(json.loads((results / "diagnostics/calibration_summary.json").read_text())["results"], key=lambda x: x["actual_fraction"])
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.00))
    fig.subplots_adjust(left=.105, right=.98, bottom=.10, top=.88, wspace=.34, hspace=.58)
    for method in ["scratch", "spm_only", "spm_to_i2em", "two_mean"]:
        line(axes[0,0], [2,3,6], [r["rmse_db"][method] for r in calibration], method)
    contrast = [r["contrasts"]["spm_only_minus_two_mean"] for r in calibration]
    val = np.array([r["delta_db"] for r in contrast]); ci = np.array([r["conditional_field_reweighting_interval"] for r in contrast])
    axes[0,1].errorbar([2,3,6], val, yerr=[val-ci[:,0], ci[:,1]-val], color=COLORS["spm"], marker="^", capsize=3, linestyle="none")
    axes[0,1].axhline(0, color="0.3", linestyle="--", linewidth=.9)
    axes[0,0].set_ylabel("Average HH/VV RMSE (dB)")
    axes[0,1].set_ylabel("SPM − mean RMSE (dB)")
    for ax, metric, scale, label in [(axes[1,0], "centered_skill", 1, "Differential centered skill"), (axes[1,1], "variance_ratio", 100, "Differential variance ratio (%)")]:
        for method in ["spm_only", "spm_to_i2em", "two_mean"]:
            g=response.loc[response.method.eq(method)&response.component.eq("differential")].sort_values("fraction")
            line(ax, [2,3,6], g[metric].to_numpy()*scale, method)
        ax.axhline(0, color="0.3", linestyle="--", linewidth=.8); ax.set_ylabel(label)
    for ax, panel in zip(axes.flat, "abcd"): fields(ax); style(ax, panel)
    legend(fig, axes[0,0], 4, 1.0); save(fig, "fig06_few_shot_transfer.pdf")

    # Standalone paired panels use the same 3.5-inch width as the TeX subfigures.
    for component in ["common", "differential"]:
        for metric, suffix, letter in [("centered_skill", "skill", "a" if component=="common" else "c"),
                                       ("variance_ratio", "variance", "b" if component=="common" else "d")]:
            fig, ax=plt.subplots(figsize=(3.5, 3.3)); fig.subplots_adjust(left=.20, right=.97, bottom=.17, top=.68)
            for method in ["scratch", "spm_only", "spm_to_i2em", "risk_spm_to_i2em", "two_mean"]:
                g=response.loc[response.method.eq(method)&response.component.eq(component)].sort_values("fraction")
                values=g[metric].to_numpy().copy(); values[np.abs(values)<1e-12]=0
                line(ax, [2,3,6], values, method)
            ax.axhline(0, color="0.3", linestyle="--", linewidth=.8)
            ax.set_ylabel(("Centered skill" if metric=="centered_skill" else "Variance ratio"))
            if component=="common": ax.set_ylim((0,.01) if metric=="variance_ratio" else (-.01,.01))
            fields(ax); style(ax); legend(fig,ax,2,1.0)
            save(fig, f"figS02{letter}_{component}_{suffix}.pdf")

    # Deterministic SPM sensitivity from the original source scan, without training.
    sys.path.insert(0, str(root / "src"))
    from research_pilots.scattering.surfaces.spm import spm_backscatter_db
    source = pd.read_csv(root / "reproducibility/data/source/smapvex12_portable_source.csv")
    target = pd.read_csv(root / "reproducibility/data/target/smex02_field_day_model_ready.csv")
    k=2*np.pi*1.26e9/299792458
    valid=source.loc[(k*source.pals_rms_height_cm/100<=.3)&(source.pals_rms_height_cm>0)&(source.pals_correlation_length_cm>0)]
    angles=[35,40,42.5,45,50]
    fig, axes=plt.subplots(1,2,figsize=(7.16,2.80)); fig.subplots_adjust(left=.105,right=.98,bottom=.22,top=.78,wspace=.30)
    for loss,marker,ls,col in zip([0,.02,.05,.1],["o","s","^","D"],["-","--","-.",":"],[COLORS[x] for x in ["spm","i2em","residual_ridge","sequential"]]):
        values=[]
        for angle in angles:
            z=spm_backscatter_db(valid.soil_real_dielectric.to_numpy()*(1-1j*loss),valid.pals_rms_height_cm.to_numpy()/100,valid.pals_correlation_length_cm.to_numpy()/100,1.26e9,angle,spectrum_model="exponential")
            values.append([np.median(z["hh_db"]),np.median(z["vv_db"])])
        values=np.asarray(values)
        for j,ax in enumerate(axes): ax.plot(angles,values[:,j],marker=marker,linestyle=ls,color=col,markerfacecolor="none",label=f"tan δ = {loss:.2f}")
    for ax,panel,pol in zip(axes,"ab",["HH","VV"]):
        ax.axvline(40,color="0.3",linestyle="--",linewidth=.8)
        ax.set(xlabel="Incidence angle (°)",ylabel=f"Median SPM {pol} (dB)",xticks=[35,40,45,50]);style(ax,panel)
    legend(fig,axes[0],4);save(fig,"figS01_angle_loss.pdf")

    # Actual-angle control uses the stored paired field-bootstrap intervals.
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.70));fig.subplots_adjust(left=.105,right=.98,bottom=.23,top=.88,wspace=.34)
    g=stage.loc[["i2em_fixed40_teacher","i2em_actual_angle_control"]]
    for ax,metric,panel,label in zip(axes,["centered_skill","bias_db"],"ab",["Differential centered skill","Differential bias (dB)"]):
        val=g[metric].to_numpy(); ax.errorbar([0,1],val,yerr=[val-g[f"{metric}_ci_low"].to_numpy(),g[f"{metric}_ci_high"].to_numpy()-val],marker="o",linestyle="none",color=COLORS["i2em"],capsize=3)
        ax.axhline(0,color="0.3",linestyle="--",linewidth=.8)
        ax.set(xticks=[0,1],xticklabels=["Fixed 40°","Actual incidence"],xlim=(-.4,1.4),ylabel=label);style(ax,panel)
    save(fig,"figS05_angle_control.pdf")

    # Differences in fixed alternative dielectric-input analyses.
    dielectric=pd.DataFrame(json.loads((results/"diagnostics/dielectric_scores.json").read_text()))
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.90));fig.subplots_adjust(left=.105,right=.98,bottom=.21,top=.77,wspace=.36)
    for method in ["spm_only","spm_to_i2em","two_channel_means"]:
        g=dielectric.loc[dielectric.method.eq(method)]
        a=g.loc[g.arm.eq("original")].set_index("fraction").sort_index();b=g.loc[g.arm.eq("unified")].set_index("fraction").sort_index()
        for ax,metric in zip(axes,["rmse_mean_hh_vv_db","centered_skill_differential"]): line(ax,[2,3,6],a[metric].to_numpy()-b[metric].to_numpy(),method)
    for ax,panel,label in zip(axes,"ab",["RMSE difference (dB)","Differential skill difference"]):
        fields(ax);ax.set_ylabel(label);ax.axhline(0,color="0.3",linestyle="--",linewidth=.8);style(ax,panel)
    legend(fig,axes[0],3);save(fig,"figS03_dielectric_sensitivity.pdf")

    mismatch=read("controlled_mismatch/summary.csv")
    mismatch=mismatch.loc[mismatch.noise_db.eq(0)&mismatch["size"].eq(32)]
    scenarios=["none","offset","gain","nonlinear"]
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.55));fig.subplots_adjust(left=.105,right=.98,bottom=.21,top=.76,wspace=.34)
    for j,(method,title,alias) in enumerate([("unadapted","Unadapted","spm_only"),("offset_calibrated","Offset calibrated","spm_to_i2em"),("adaptation_mean","Adaptation mean","two_mean"),("oracle_reference","Oracle reference","residual_ridge")]):
        g=mismatch.loc[mismatch.method.eq(method)].set_index("scenario").loc[scenarios]
        for ax,metric in zip(axes,["rmse_db","centered_rmse_db"]):
            ax.plot(np.arange(4)+(j-1.5)*.1,g[metric],linestyle="none",marker=METHOD_MARKERS[alias],color=METHOD_COLORS[alias],label=title,markerfacecolor="white")
    for ax,panel,label in zip(axes,"ab",["Average HH/VV RMSE (dB)","Centered RMSE (dB)"]):
        ax.set(xticks=range(4),xticklabels=["None","Offset","Gain","Nonlinear"],ylabel=label,xlabel="Simulated mismatch");style(ax,panel)
    # Keep the legend on one row, clear of the panel labels below it.
    legend(fig,axes[0],4);save(fig,"figS04_controlled_mismatch.pdf")

    fig,axes=plt.subplots(1,2,figsize=(7.16,2.55));fig.subplots_adjust(left=.105,right=.98,bottom=.21,top=.88,wspace=.32)
    for ax,pol,panel in zip(axes,["hh","vv"],"ab"):
        y=test[f"teacher_absolute_delta_{pol}_db"].to_numpy()
        ax.scatter(u,y,s=18,color=COLORS["spm"],edgecolors="none")
        xx,yy=[],[]
        for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            mask=(u>lo)&(u<=hi)
            if mask.any():xx.append(np.median(u[mask]));yy.append(np.median(y[mask]))
        ax.plot(xx,yy,"s--",color=COLORS["sequential"],markersize=4,label="Binned median")
        ax.set(xlabel="Domain utilization, u",ylabel=f"|I²EM − SPM|, {pol.upper()} (dB)",xlim=(0,1));style(ax,panel)
    axes[1].legend(frameon=False,loc="upper left");save(fig,"figS06_teacher_domain.pdf")

    fig,axes=plt.subplots(1,3,figsize=(7.16,2.50));fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.79,wspace=.42)
    for j,(component,label) in enumerate([("common","Common component (dB)"),("differential","Differential component (dB)"),("angle","Incidence angle (°)")]):
        values=[]
        for frame,anglecol in [(source,"nominal_incidence_angle_deg"),(target,"incidence_angle_deg")]:
            if component=="common":v=(frame.sigma0_hh_db+frame.sigma0_vv_db)/2
            elif component=="differential":v=frame.sigma0_vv_db-frame.sigma0_hh_db
            else:v=frame[anglecol]
            values.append(v.to_numpy())
        bins=np.linspace(min(np.min(v) for v in values),max(np.max(v) for v in values),13)
        for v,title,color,ls in zip(values,["SMAPVEX12 (240)","SMEX02 (189)"],[COLORS["spm"],COLORS["sequential"]],["-","--"]):
            axes[j].hist(v,bins=bins,density=True,histtype="step",color=color,linestyle=ls,linewidth=1.2,label=title)
            axes[j].axvline(np.median(v),color=color,linestyle=ls,linewidth=.9)
        axes[j].set(xlabel=label,ylabel="Density");axes[j].xaxis.set_major_locator(MaxNLocator(4));style(axes[j],"abc"[j])
    legend(fig,axes[0],2);save(fig,"figS07_campaign_distributions.pdf")
    print(f"Rendered {len(generated)} vector PDF assets in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render all TGRS manuscript figures "
            "from frozen outputs."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--compact-revision", action="store_true",
                        help="Render the revised sample-efficiency and stage assets from reproducibility/")
    parser.add_argument("--submission", action="store_true",
                        help="Redraw all 16 existing PDF assets from the packaged results")
    args = parser.parse_args()
    if args.submission:
        render_submission(args.project_root.resolve(), args.output_root.resolve())
        return
    if args.compact_revision:
        render_compact_revision(args.project_root.resolve(), args.output_root.resolve())
        return

    root = args.project_root
    base = (
        root
        / "outputs/scattering/rough_ground"
    )

    output = args.output_root

    if output.exists() and any(output.rglob("*")):
        raise FileExistsError(
            "输出目录已经包含文件。"
            "请使用新的版本目录，例如tgrs_v2。"
        )

    (output / "main").mkdir(
        parents=True,
        exist_ok=True,
    )
    (output / "supplementary").mkdir(
        parents=True,
        exist_ok=True,
    )

    apply_tgrs_style()

    source_records = {}
    figure_records = {}

    plot_fig01(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_fig02(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_fig03(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_fig04(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_fig05(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_fig06(
        base,
        output,
        source_records,
        figure_records,
    )

    plot_figs01(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_figs02(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_figs03(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_figs04(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_figs05(
        base,
        output,
        source_records,
        figure_records,
    )
    plot_figs06(
        base,
        output,
        source_records,
        figure_records,
    )

    manifest = {
        "status": "COMPLETE",
        "scientific_role": (
            "Rendering only from frozen evidence; "
            "no fitting or model selection"
        ),
        "style": {
            "single_column_width_in": 3.5,
            "double_column_width_in": 7.16,
            "vector_format": "PDF",
            "raster_format": "PNG",
            "raster_dpi": 600,
            "font_family": (
                "Arial/Helvetica/DejaVu Sans fallback"
            ),
            "palette": COLORS,
        },
        "sources": source_records,
        "figures": figure_records,
    }

    manifest_path = (
        output / "figure_provenance.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Generated {len(figure_records)} figure groups."
    )
    print(f"Output: {output}")
    print(f"Provenance: {manifest_path}")


if __name__ == "__main__":
    main()
