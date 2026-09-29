from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUGH = ROOT / "outputs" / "scattering" / "rough_ground"

ZERO_PATH = (
    ROUGH
    / "smex02_prospective_zero_shot_20260911_v1"
    / "summary.json"
)

FEW_PATH = (
    ROUGH
    / "smex02_preregistered_few_shot_20260911_v1"
    / "summary.json"
)

AUDIT_PATH = (
    ROUGH
    / "paper_submission_closeout_20260911_v1"
    / "paper_audit_supplement_v1"
    / "summary.json"
)

RESPONSE_PATH = (
    ROUGH
    / "paper_submission_closeout_20260911_v1"
    / "response_learning_v1"
    / "response_summary.csv"
)

COMMON_PATH = (
    ROUGH
    / "smex02_common_cohort_20260921_v1"
    / "common_cohort_metrics.csv"
)

OUT_DIR = ROUGH / "phase3_cross_domain_evidence_v1"

FIG_DIR = (
    ROOT
    / "paper"
    / "figures"
    / "tgrs_v3"
    / "main"
)

FIG_STEM = FIG_DIR / "Fig_phase3_target_domain_evidence"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def get_response_rows(
    frame: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    selected = frame.loc[
        (frame["method"] == method)
        & (frame["component"] == "differential")
    ].copy()

    selected["fraction"] = pd.to_numeric(
        selected["fraction"]
    )

    selected = selected.sort_values(
        "fraction"
    ).reset_index(drop=True)

    require(
        len(selected) == 3,
        f"Expected three response rows for {method}",
    )

    return selected


def add_panel_label(
    axis: plt.Axes,
    label: str,
) -> None:
    axis.text(
        0.01,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def main() -> None:
    input_paths = [
        ZERO_PATH,
        FEW_PATH,
        AUDIT_PATH,
        RESPONSE_PATH,
        COMMON_PATH,
    ]

    for path in input_paths:
        require(
            path.exists(),
            f"Missing frozen input: {path}",
        )

    zero = read_json(ZERO_PATH)
    few = read_json(FEW_PATH)
    audit = read_json(AUDIT_PATH)

    response = pd.read_csv(RESPONSE_PATH)
    common = pd.read_csv(COMMON_PATH)

    require(
        zero["status"]
        == "PROSPECTIVE_ZERO_SHOT_COMPLETE",
        "Unexpected zero-shot status",
    )

    require(
        few["status"]
        == "PREREGISTERED_FEW_SHOT_COMPLETE",
        "Unexpected few-shot status",
    )

    require(
        zero["samples"]["source_rows"] == 240,
        "Unexpected SMAPVEX12 sample count",
    )

    require(
        zero["samples"]["external_rows"] == 189,
        "Unexpected SMEX02 sample count",
    )

    require(
        zero["samples"]["external_fields"] == 30,
        "Unexpected SMEX02 field count",
    )

    require(
        audit["checks_passed"] == 60,
        "Post-processing checks are incomplete",
    )

    require(
        audit["bootstrap_draws"] == 4000,
        "Unexpected bootstrap count",
    )

    plans = few["adaptation_plan"]

    fractions = np.array(
        [
            float(item["actual_fraction"])
            for item in plans
        ]
    )

    field_counts = np.array(
        [
            int(item["adaptation_fields"])
            for item in plans
        ]
    )

    require(
        np.allclose(
            fractions,
            [2 / 30, 3 / 30, 6 / 30],
        ),
        "Unexpected adaptation fractions",
    )

    require(
        np.array_equal(
            field_counts,
            [2, 3, 6],
        ),
        "Unexpected adaptation field counts",
    )

    audit_results = sorted(
        audit["results"],
        key=lambda item: float(
            item["actual_fraction"]
        ),
    )

    require(
        np.allclose(
            [
                float(item["actual_fraction"])
                for item in audit_results
            ],
            fractions,
        ),
        "Audit fractions do not match the frozen plan",
    )

    method_keys = [
        "scratch",
        "spm_only",
        "spm_to_i2em",
        "two_mean",
    ]

    rmse = {
        method: np.array(
            [
                float(
                    item["rmse_db"][method]
                )
                for item in audit_results
            ]
        )
        for method in method_keys
    }

    contrast = np.array(
        [
            float(
                item["contrasts"]
                ["spm_only_minus_two_mean"]
                ["delta_db"]
            )
            for item in audit_results
        ]
    )

    ci_low = np.array(
        [
            float(
                item["contrasts"]
                ["spm_only_minus_two_mean"]
                ["conditional_field_reweighting_interval"][0]
            )
            for item in audit_results
        ]
    )

    ci_high = np.array(
        [
            float(
                item["contrasts"]
                ["spm_only_minus_two_mean"]
                ["conditional_field_reweighting_interval"][1]
            )
            for item in audit_results
        ]
    )

    spm_response = get_response_rows(
        response,
        "spm_only",
    )

    sequential_response = get_response_rows(
        response,
        "spm_to_i2em",
    )

    mean_response = get_response_rows(
        response,
        "two_mean",
    )

    common_selected = common.loc[
        (common["component"] == "differential")
        & common["method"].isin(
            [
                "SPM",
                "I2EM",
                "SPM to I2EM",
                "Risk-controlled transfer",
            ]
        )
    ].copy()

    require(
        len(common_selected) == 4,
        "Shared-cohort metrics are incomplete",
    )

    require(
        set(
            common_selected["n"].astype(int)
        )
        == {138},
        "Unexpected shared-cohort size",
    )

    zero_primary = zero["primary_comparison"]

    summary = {
        "status":
            "PHASE3_CROSS_DOMAIN_EVIDENCE_COMPLETE",

        "training_performed_by_this_script":
            False,

        "model_selection_performed_by_this_script":
            False,

        "zero_shot": {
            "source_mean_rmse_db":
                zero[
                    "mean_hh_vv_rmse_ranking_db"
                ]["source_mean"],

            "risk_controlled_rmse_db":
                zero[
                    "mean_hh_vv_rmse_ranking_db"
                ]["risk_spm_to_i2em"],

            "risk_minus_source_mean_db":
                zero_primary[
                    "mean_channel_rmse_delta_db"
                ],

            "field_bootstrap_95_ci_db": [
                zero_primary[
                    "ci_2_5_percent_db"
                ],
                zero_primary[
                    "ci_97_5_percent_db"
                ],
            ],

            "claim_verdict":
                zero["claim_verdict"],
        },

        "few_shot": [],
        "centered_response": [],
        "shared_138_row_cohort": {},

        "input_sha256": {
            str(path): sha256(path)
            for path in input_paths
        },
    }

    for index, item in enumerate(
        audit_results
    ):
        summary["few_shot"].append(
            {
                "adaptation_fields":
                    int(field_counts[index]),

                "actual_fraction":
                    float(fractions[index]),

                "scratch_rmse_db":
                    float(
                        item["rmse_db"]["scratch"]
                    ),

                "spm_only_rmse_db":
                    float(
                        item["rmse_db"]["spm_only"]
                    ),

                "spm_to_i2em_rmse_db":
                    float(
                        item[
                            "rmse_db"
                        ]["spm_to_i2em"]
                    ),

                "two_channel_mean_rmse_db":
                    float(
                        item["rmse_db"]["two_mean"]
                    ),

                "spm_minus_two_channel_mean_db":
                    float(contrast[index]),

                "spm_minus_two_channel_mean_interval_db":
                    [
                        float(ci_low[index]),
                        float(ci_high[index]),
                    ],
            }
        )

    for index, fraction in enumerate(
        fractions
    ):
        summary["centered_response"].append(
            {
                "adaptation_fields":
                    int(field_counts[index]),

                "actual_fraction":
                    float(fraction),

                "spm_centered_skill":
                    float(
                        spm_response.iloc[
                            index
                        ]["centered_skill"]
                    ),

                "sequential_centered_skill":
                    float(
                        sequential_response.iloc[
                            index
                        ]["centered_skill"]
                    ),

                "spm_variance_ratio":
                    float(
                        spm_response.iloc[
                            index
                        ]["variance_ratio"]
                    ),

                "sequential_variance_ratio":
                    float(
                        sequential_response.iloc[
                            index
                        ]["variance_ratio"]
                    ),
            }
        )

    for _, row in common_selected.iterrows():
        summary[
            "shared_138_row_cohort"
        ][str(row["method"])] = {
            "centered_skill":
                float(row["centered_skill"]),

            "variance_ratio":
                float(row["variance_ratio"]),
        }

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        OUT_DIR / "phase3_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            summary,
            stream,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(
        summary["few_shot"]
    ).to_csv(
        OUT_DIR / "few_shot_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        summary["centered_response"]
    ).to_csv(
        OUT_DIR
        / "centered_response_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    common_selected.to_csv(
        OUT_DIR
        / "shared_138_row_cohort.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {
        "scratch": "#7F7F7F",
        "spm_only": "#0072B2",
        "spm_to_i2em": "#D55E00",
        "two_mean": "#333333",
    }

    styles = {
        "scratch":
            ("--", "^", "Scratch"),

        "spm_only":
            ("-", "o", "SPM pretraining"),

        "spm_to_i2em":
            ("-.", "s", "SPM→I²EM"),

        "two_mean":
            (":", "D", "Two-channel mean"),
    }

    x = np.arange(3)

    x_labels = [
        "6.67%\n(2 fields)",
        "10%\n(3 fields)",
        "20%\n(6 fields)",
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.16, 4.75),
        constrained_layout=True,
    )

    axis = axes[0, 0]

    for method in method_keys:
        linestyle, marker, label = (
            styles[method]
        )

        axis.plot(
            x,
            rmse[method],
            color=colors[method],
            linestyle=linestyle,
            marker=marker,
            label=label,
        )

    axis.set_xticks(
        x,
        x_labels,
    )

    axis.set_ylabel(
        "Mean HH/VV RMSE (dB)"
    )

    axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.5,
    )

    axis.legend(
        ncol=2,
        loc="best",
    )

    add_panel_label(
        axis,
        "(a)",
    )

    axis = axes[0, 1]

    y_error = np.vstack(
        [
            contrast - ci_low,
            ci_high - contrast,
        ]
    )

    axis.errorbar(
        x,
        contrast,
        yerr=y_error,
        color=colors["spm_only"],
        marker="o",
        linestyle="none",
        capsize=3,
    )

    axis.axhline(
        0.0,
        color="#333333",
        linewidth=0.8,
    )

    axis.set_xticks(
        x,
        x_labels,
    )

    axis.set_ylabel(
        "SPM minus two-channel mean (dB)"
    )

    axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.5,
    )

    add_panel_label(
        axis,
        "(b)",
    )

    axis = axes[1, 0]

    for frame, method in [
        (spm_response, "spm_only"),
        (
            sequential_response,
            "spm_to_i2em",
        ),
        (mean_response, "two_mean"),
    ]:
        linestyle, marker, label = (
            styles[method]
        )

        axis.plot(
            x,
            frame[
                "centered_skill"
            ].to_numpy(float),
            color=colors[method],
            linestyle=linestyle,
            marker=marker,
            label=label,
        )

    axis.axhline(
        0.0,
        color="#333333",
        linewidth=0.8,
    )

    axis.set_xticks(
        x,
        x_labels,
    )

    axis.set_ylabel(
        "Centered skill, differential component"
    )

    axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.5,
    )

    axis.legend(
        ncol=2,
        loc="best",
    )

    add_panel_label(
        axis,
        "(c)",
    )

    axis = axes[1, 1]

    for frame, method in [
        (spm_response, "spm_only"),
        (
            sequential_response,
            "spm_to_i2em",
        ),
    ]:
        linestyle, marker, label = (
            styles[method]
        )

        axis.plot(
            x,
            100.0
            * frame[
                "variance_ratio"
            ].to_numpy(float),
            color=colors[method],
            linestyle=linestyle,
            marker=marker,
            label=label,
        )

    axis.set_xticks(
        x,
        x_labels,
    )

    axis.set_ylabel(
        "Predicted/observed variance (%)"
    )

    axis.set_ylim(
        bottom=0.0,
    )

    axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.5,
    )

    axis.legend(
        loc="best",
    )

    add_panel_label(
        axis,
        "(d)",
    )

    fig.savefig(
        FIG_STEM.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        FIG_STEM.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(
        "Phase 3 cross-domain evidence is complete."
    )

    print(
        "Summary:",
        OUT_DIR / "phase3_summary.json",
    )

    print(
        "Figure:",
        FIG_STEM.with_suffix(".pdf"),
    )

    print(
        "Zero-shot risk-minus-mean delta:",
        f"{zero_primary['mean_channel_rmse_delta_db']:.6f}",
        "dB",
    )


if __name__ == "__main__":
    main()