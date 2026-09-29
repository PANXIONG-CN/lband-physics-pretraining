"""从冻结结果重绘TGRS论文图。

本程序只读取现有CSV和JSON：
1. 不训练模型；
2. 不选择模型；
3. 不修改冻结结果；
4. 同时输出PDF和600 dpi PNG；
5. 生成图件来源与SHA-256清单。
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
    args = parser.parse_args()

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
