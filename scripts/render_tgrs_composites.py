from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

from render_tgrs_figures import (
    apply_tgrs_style,
    add_panel_label,
    format_axis,
    actual_fraction_percent,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/scattering/rough_ground"
OUTPUT = ROOT / "paper/figures/tgrs_v2/main"


def save_figure(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT / f"{name}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT / f"{name}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def render_synthetic_evidence():
    teacher_path = (
        BASE
        / "multifidelity_pretraining_20260910_v1"
        / "evaluation/metrics_summary.csv"
    )
    efficiency_path = (
        BASE
        / "multifidelity_sample_efficiency_20260910_v1"
        / "metrics_summary.csv"
    )

    teacher = pd.read_csv(teacher_path)
    efficiency = pd.read_csv(efficiency_path)

    responses = ["HH", "VV"]
    teacher_order = [
        "spm_teacher",
        "i2em_only",
        "spm_only",
        "spm_to_i2em",
    ]
    budgets = [32, 64, 128, 256]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.16, 4.35),
        constrained_layout=True,
    )

    for column, response in enumerate(responses):
        ax = axes[0, column]

        part = (
            teacher[
                teacher["response"].eq(response)
                & teacher["method"].isin(teacher_order)
            ]
            .set_index("method")
            .loc[teacher_order]
        )

        x = np.arange(len(teacher_order))

        for index, method in enumerate(teacher_order):
            ax.errorbar(
                x[index],
                part.loc[method, "rmse_mean_db"],
                yerr=part.loc[method, "rmse_std_db"],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle="none",
                capsize=3,
            )

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([
            "SPM\nteacher",
            "I²EM\nonly",
            "SPM\nonly",
            "SPM→\nI²EM",
        ])
        ax.set_title(f"{response} polarization")
        format_axis(ax)
        add_panel_label(ax, "(a)" if column == 0 else "(b)")

        if column == 0:
            ax.set_ylabel("Test RMSE (dB)")

        ax = axes[1, column]

        part = efficiency[
            efficiency["response"].eq(response)
            & efficiency["method"].isin([
                "i2em_only",
                "spm_to_i2em",
            ])
        ]

        for method in ["i2em_only", "spm_to_i2em"]:
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
        ax.set_yscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(value) for value in budgets])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Number of I²EM training samples")
        format_axis(ax)
        add_panel_label(ax, "(c)" if column == 0 else "(d)")

        if column == 0:
            ax.set_ylabel("Test RMSE (dB)")

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )

    save_figure(
        fig,
        "Fig02_multifidelity_synthetic_evidence",
    )


def render_target_domain_evidence():
    audit_path = (
        BASE
        / "paper_submission_closeout_20260911_v1"
        / "paper_audit_supplement_v1/summary.json"
    )
    response_path = (
        BASE
        / "paper_submission_closeout_20260911_v1"
        / "response_learning_v1/response_summary.csv"
    )

    content = json.loads(audit_path.read_text(encoding="utf-8"))
    results = content["results"]
    response = pd.read_csv(response_path)
    response = response[
        response["component"].eq("differential")
    ].copy()

    fractions = np.array([
        row["actual_fraction"] * 100.0
        for row in results
    ])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.16, 4.55),
        constrained_layout=True,
    )

    top_methods = [
        "scratch",
        "spm_only",
        "spm_to_i2em",
        "two_mean",
    ]

    for method in top_methods:
        values = np.array([
            row["rmse_db"][method]
            for row in results
        ])

        axes[0, 0].plot(
            fractions,
            values,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )

    axes[0, 0].set_ylabel("Mean HH/VV RMSE (dB)")
    add_panel_label(axes[0, 0], "(a)")
    format_axis(axes[0, 0])

    contrasts = [
        row["contrasts"]["spm_only_minus_two_mean"]
        for row in results
    ]

    delta = np.array([
        row["delta_db"] for row in contrasts
    ])
    intervals = np.array([
        row["conditional_field_reweighting_interval"]
        for row in contrasts
    ])

    axes[0, 1].errorbar(
        fractions,
        delta,
        yerr=np.vstack([
            delta - intervals[:, 0],
            intervals[:, 1] - delta,
        ]),
        color=METHOD_COLORS["spm_only"],
        marker="o",
        linestyle="none",
        capsize=4,
    )
    axes[0, 1].axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )
    axes[0, 1].set_ylabel(
        "SPM minus mean RMSE (dB)"
    )
    add_panel_label(axes[0, 1], "(b)")
    format_axis(axes[0, 1])

    response_methods = [
        "scratch",
        "spm_only",
        "spm_to_i2em",
        "risk_spm_to_i2em",
        "two_mean",
    ]

    for method in response_methods:
        part = (
            response[response["method"].eq(method)]
            .sort_values("fraction")
        )

        x = np.array([
            actual_fraction_percent(value)
            for value in part["fraction"]
        ])

        axes[1, 0].plot(
            x,
            part["centered_skill"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )

        axes[1, 1].plot(
            x,
            part["variance_ratio"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
        )

    axes[1, 0].axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1, 0].set_ylabel(
        "Centered skill of differential response"
    )
    add_panel_label(axes[1, 0], "(c)")
    format_axis(axes[1, 0])

    axes[1, 1].axhline(
        1.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1, 1].set_ylim(-0.02, 1.05)
    axes[1, 1].set_ylabel(
        "Predicted-to-observed variance ratio"
    )
    add_panel_label(axes[1, 1], "(d)")
    format_axis(axes[1, 1])

    inset = inset_axes(
        axes[1, 1],
        width="44%",
        height="42%",
        loc="upper right",
        borderpad=0.8,
    )

    for method in response_methods:
        part = (
            response[response["method"].eq(method)]
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
            linewidth=0.8,
            markersize=2.8,
        )

    inset.set_ylim(-0.005, 0.22)
    inset.tick_params(labelsize=5.5)

    tick_labels = [
        "6.67%\n(2 fields)",
        "10%\n(3 fields)",
        "20%\n(6 fields)",
    ]

    for ax in axes.ravel():
        ax.set_xticks(fractions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Target fields used for adaptation")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
    )

    save_figure(
        fig,
        "Fig04_target_domain_evidence",
    )


def main():
    apply_tgrs_style()
    render_synthetic_evidence()
    render_target_domain_evidence()
    print(f"Figures saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
