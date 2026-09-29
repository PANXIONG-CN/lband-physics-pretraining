"""在共同SMEX02有效样本上比较物理模型与冻结神经网络。

该程序只进行结果后处理：
1. 不重新训练模型；
2. 不修改模型参数；
3. 不根据目标域结果选择方法。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_components(hh: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """返回共同分量和差分分量。

    common = (HH + VV) / 2
    differential = VV - HH
    """
    return np.column_stack([
        (hh + vv) / 2.0,
        vv - hh,
    ])


def calculate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    error = predicted - observed
    observed_centered = observed - observed.mean()
    predicted_centered = predicted - predicted.mean()

    observed_variance = np.mean(observed_centered**2)
    predicted_variance = np.mean(predicted_centered**2)
    centered_mse = np.mean(
        (predicted_centered - observed_centered) ** 2
    )

    return {
        "n": int(len(observed)),
        "bias_db": float(np.mean(error)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "centered_rmse_db": float(np.sqrt(centered_mse)),
        "centered_skill": float(
            1.0 - centered_mse / observed_variance
        ),
        "variance_ratio": float(
            predicted_variance / observed_variance
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    root = args.project_root
    output = args.output

    if output.exists():
        raise FileExistsError(
            f"输出目录已存在，请使用新的版本目录：{output}"
        )
    output.mkdir(parents=True)

    physics_path = (
        root
        / "outputs/scattering/rough_ground/advisor_revision_v1"
        / "angle_control/i2em_evaluation/paired_predictions.csv"
    )
    network_path = (
        root
        / "outputs/scattering/rough_ground"
        / "smex02_prospective_zero_shot_20260911_v1"
        / "zero_shot_predictions.csv"
    )

    physics = pd.read_csv(
        physics_path,
        dtype={"field_id": str},
    )
    network = pd.read_csv(
        network_path,
        dtype={"field_id": str},
    )

    valid_flag = (
        physics["i2em_request_valid"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    physics = physics[
        physics["campaign"].eq("SMEX02")
        & physics["scenario"].eq("field_day_angle")
        & valid_flag
    ].copy()

    keys = ["field_id", "acquisition_date"]

    if physics.duplicated(keys).any():
        raise AssertionError("物理模型结果存在重复地块日期键。")

    if network.duplicated(keys).any():
        raise AssertionError("零样本结果存在重复地块日期键。")

    physics_columns = [
        *keys,
        "i2em_hh_db",
        "i2em_vv_db",
        "spm_hh_db",
        "spm_vv_db",
    ]

    merged = network.merge(
        physics[physics_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 138:
        raise AssertionError(
            f"预期138条共同样本，实际得到{len(merged)}条。"
        )

    methods = {
        "I2EM": ("i2em_hh_db", "i2em_vv_db"),
        "SPM": ("spm_hh_db", "spm_vv_db"),
        "Source mean": (
            "hh_source_mean",
            "vv_source_mean",
        ),
        "Source Ridge": (
            "hh_source_ridge",
            "vv_source_ridge",
        ),
        "Scratch": (
            "hh_scratch",
            "vv_scratch",
        ),
        "SPM only": (
            "hh_spm_only",
            "vv_spm_only",
        ),
        "I2EM only": (
            "hh_i2em_only",
            "vv_i2em_only",
        ),
        "SPM to I2EM": (
            "hh_spm_to_i2em",
            "vv_spm_to_i2em",
        ),
        "Risk-controlled transfer": (
            "hh_risk_spm_to_i2em",
            "vv_risk_spm_to_i2em",
        ),
    }

    observed = to_components(
        merged["sigma0_hh_db"].to_numpy(float),
        merged["sigma0_vv_db"].to_numpy(float),
    )

    records = []

    for method, (hh_column, vv_column) in methods.items():
        predicted = to_components(
            merged[hh_column].to_numpy(float),
            merged[vv_column].to_numpy(float),
        )

        for index, component in enumerate([
            "common",
            "differential",
        ]):
            records.append({
                "method": method,
                "component": component,
                **calculate_metrics(
                    observed[:, index],
                    predicted[:, index],
                ),
            })

    metrics = pd.DataFrame(records)
    metrics.to_csv(
        output / "common_cohort_metrics.csv",
        index=False,
    )
    merged.to_csv(
        output / "common_cohort_predictions.csv",
        index=False,
    )

    selected_methods = [
        "Source mean",
        "SPM",
        "I2EM",
        "SPM only",
        "I2EM only",
        "SPM to I2EM",
        "Risk-controlled transfer",
    ]

    plot_data = metrics[
        metrics["method"].isin(selected_methods)
    ].copy()

    colors = {
        "common": "#0072B2",
        "differential": "#D55E00",
    }

    matplotlib.rcParams.update({
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
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.16, 3.2))

    x = np.arange(len(selected_methods))
    offsets = {
        "common": -0.12,
        "differential": 0.12,
    }
    markers = {
        "common": "o",
        "differential": "s",
    }

    for component in ["common", "differential"]:
        part = (
            plot_data[plot_data["component"].eq(component)]
            .set_index("method")
            .loc[selected_methods]
        )

        ax.plot(
            x + offsets[component],
            part["centered_skill"],
            marker=markers[component],
            color=colors[component],
            linestyle="none",
            markersize=5,
            label=component.capitalize(),
        )

    ax.axhline(
        0.0,
        color="#333333",
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_ylabel("Centered skill")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Source\nmean",
            "SPM",
            "I²EM",
            "SPM\nonly",
            "I²EM\nonly",
            "SPM→I²EM",
            "Risk-\ncontrolled",
        ]
    )
    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
    )

    fig.tight_layout()
    fig.savefig(
        output / "FigS05_smex02_common_cohort.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output / "FigS05_smex02_common_cohort.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    manifest = {
        "status": "COMPLETE",
        "scientific_role": (
            "Post-processing comparison on the shared "
            "138-row SMEX02 validity cohort"
        ),
        "training_performed": False,
        "model_selection_performed": False,
        "join_keys": keys,
        "rows": int(len(merged)),
        "fields": int(merged["field_id"].nunique()),
        "physics_input": {
            "path": str(physics_path),
            "sha256": file_hash(physics_path),
        },
        "network_input": {
            "path": str(network_path),
            "sha256": file_hash(network_path),
        },
    }

    (
        output / "manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
