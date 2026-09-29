"""High-fidelity sample-efficiency study for SPM-to-I2EM pretraining.

The independent I2EM test set is fixed.  For each repeat, one permutation of
the I2EM training set creates nested subsets (32 is contained in 64, etc.), so
changes along a curve are attributable to high-fidelity sample count rather
than unrelated resampling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_multifidelity_pretraining as multi  # noqa: E402


def parse_sizes(value: str, available: int) -> list[int]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or sizes[0] < 16:
        raise ValueError("Sample sizes must be comma-separated integers >= 16")
    if sizes[-1] > available:
        raise ValueError(f"Largest sample size {sizes[-1]} exceeds available I2EM rows {available}")
    return sizes


def metric(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    denominator = float(np.sum((reference - np.mean(reference)) ** 2))
    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "r_squared": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
    }


def response_arrays(channels: np.ndarray) -> dict[str, np.ndarray]:
    components = multi.to_components(channels)
    return {
        "HH": channels[:, 0],
        "VV": channels[:, 1],
        "common": components[:, 0],
        "differential": components[:, 1],
    }


def save_plots(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    colors = {"i2em_only": "#F58518", "spm_to_i2em": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, response in zip(axes, ["HH", "VV"]):
        part = summary[summary.response == response]
        for method in ["i2em_only", "spm_to_i2em"]:
            curve = part[part.method == method]
            ax.errorbar(
                curve.i2em_train_samples,
                curve.rmse_mean_db,
                yerr=curve.rmse_std_db.fillna(0),
                marker="o",
                capsize=3,
                color=colors[method],
                label=method,
            )
        ax.set(xscale="log", xlabel="I2EM training samples", ylabel="RMSE against independent I2EM (dB)", title=response)
        ax.set_xticks(sorted(part.i2em_train_samples.unique()), [str(value) for value in sorted(part.i2em_train_samples.unique())])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(output / "01_i2em_sample_efficiency.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for response, marker in [("HH", "o"), ("VV", "s"), ("differential", "^")]:
        part = paired[paired.response == response]
        grouped = part.groupby("i2em_train_samples").rmse_improvement_db.agg(["mean", "std"]).reset_index()
        ax.errorbar(grouped.i2em_train_samples, grouped["mean"], yerr=grouped["std"].fillna(0), marker=marker, capsize=3, label=response)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set(xscale="log", xlabel="I2EM training samples", ylabel="I2EM-only RMSE - SPM→I2EM RMSE (dB)", title="Benefit of dense low-fidelity pretraining")
    ax.set_xticks(sorted(paired.i2em_train_samples.unique()), [str(value) for value in sorted(paired.i2em_train_samples.unique())])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "02_multifidelity_improvement.png", dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="I2EM sample-efficiency sensitivity experiment")
    parser.add_argument("--spm", required=True, type=Path)
    parser.add_argument("--i2em-train-requests", required=True, type=Path)
    parser.add_argument("--i2em-train-results", required=True, type=Path)
    parser.add_argument("--i2em-test-requests", required=True, type=Path)
    parser.add_argument("--i2em-test-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-sizes", default="32,64,128,256")
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if min(args.spm_epochs, args.i2em_epochs, args.repeats) < 1:
        raise ValueError("Epochs and repeats must be positive")
    output = args.output.resolve()
    if (output / "summary.json").exists():
        raise FileExistsError("Output already exists; choose a new directory")
    spm = pd.read_csv(args.spm.resolve())
    train = multi.load_paired(args.i2em_train_requests.resolve(), args.i2em_train_results.resolve())
    test = multi.load_paired(args.i2em_test_requests.resolve(), args.i2em_test_results.resolve())
    sizes = parse_sizes(args.sample_sizes, len(train))
    spm_features = multi.features_from(spm)
    spm_channels = spm[["spm_hh_db", "spm_vv_db"]].to_numpy(dtype=float)
    train_features_all = multi.features_from(train)
    train_channels_all = train[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float)
    test_features = multi.features_from(test)
    test_channels = test[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float)
    references = response_arrays(test_channels)
    rows = []
    paired_rows = []
    for repeat in range(1, args.repeats + 1):
        seed = args.seed + (repeat - 1) * 100_000
        order = np.random.default_rng(seed).permutation(len(train))
        for size in sizes:
            selected = order[:size]
            components = multi.train_component_models(
                spm_features,
                multi.to_components(spm_channels),
                train_features_all[selected],
                multi.to_components(train_channels_all[selected]),
                test_features,
                args.spm_epochs,
                args.i2em_epochs,
                seed + size,
            )
            predictions = {
                method: response_arrays(multi.from_components(values))
                for method, values in components.items()
                if method in ["i2em_only", "spm_to_i2em"]
            }
            method_metrics: dict[tuple[str, str], dict[str, float]] = {}
            for method, arrays in predictions.items():
                for response, predicted in arrays.items():
                    result = metric(references[response], predicted)
                    method_metrics[(method, response)] = result
                    rows.append({"repeat": repeat, "seed": seed, "i2em_train_samples": size, "method": method, "response": response, "n_test": len(test), **result})
            for response in references:
                paired_rows.append(
                    {
                        "repeat": repeat,
                        "i2em_train_samples": size,
                        "response": response,
                        "rmse_improvement_db": method_metrics[("i2em_only", response)]["rmse_db"] - method_metrics[("spm_to_i2em", response)]["rmse_db"],
                    }
                )
        print(f"Completed sample-efficiency repeat {repeat}/{args.repeats}", flush=True)
    metrics = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    summary = metrics.groupby(["i2em_train_samples", "method", "response"], as_index=False).agg(
        rmse_mean_db=("rmse_db", "mean"),
        rmse_std_db=("rmse_db", "std"),
        mae_mean_db=("mae_db", "mean"),
        r_squared_mean=("r_squared", "mean"),
    )
    improvement = paired.groupby(["i2em_train_samples", "response"], as_index=False).agg(
        improvement_mean_db=("rmse_improvement_db", "mean"),
        improvement_std_db=("rmse_improvement_db", "std"),
        fraction_repeats_multifidelity_better=("rmse_improvement_db", lambda values: float(np.mean(np.asarray(values) > 0))),
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics_by_repeat.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    paired.to_csv(output / "paired_improvements_by_repeat.csv", index=False)
    improvement.to_csv(output / "paired_improvement_summary.csv", index=False)
    save_plots(summary, paired, output)
    hh_vv = metrics[metrics.response.isin(["HH", "VV"])].groupby(["repeat", "i2em_train_samples", "method"], as_index=False).rmse_db.mean()
    ranking = hh_vv.groupby(["i2em_train_samples", "method"], as_index=False).agg(mean_hh_vv_rmse_db=("rmse_db", "mean"), std_hh_vv_rmse_db=("rmse_db", "std"))
    ranking.to_csv(output / "mean_channel_ranking.csv", index=False)
    payload = {
        "research_question": "How many expensive I2EM samples are required, and does dense SPM pretraining improve high-fidelity sample efficiency?",
        "sample_sizes": sizes,
        "repeats": args.repeats,
        "nested_subset_design": True,
        "independent_i2em_test_samples": len(test),
        "ranking": ranking.to_dict(orient="records"),
        "decision_rule": "Sample efficiency is supported when SPM-to-I2EM has lower independent-test RMSE than I2EM-only across repeats, especially at 32-128 samples, without selecting a size using the test set.",
        "scope_limit": "This sensitivity concerns approximation of the I2EM teacher, not field-observation accuracy.",
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(ranking.to_string(index=False), flush=True)
    print(f"Outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
