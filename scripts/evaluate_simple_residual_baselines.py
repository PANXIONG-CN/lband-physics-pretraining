"""同I²EM样本预算的直接代理与SPM残差代理比较。"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_multifidelity_pretraining as multi


def channel_rmse(y, prediction):
    return float(np.sqrt(np.mean((y - prediction) ** 2, axis=0)).mean())


SCORER = make_scorer(channel_rmse, greater_is_better=False)


def search_model(kind, x, y, seed):
    if kind == "ridge":
        estimator = Ridge()
        grid = {"alpha": [1e-6, 1e-4, 1e-2, 1.0, 100.0]}
    else:
        estimator = KernelRidge(kernel="rbf")
        grid = {
            "alpha": [1e-6, 1e-4, 1e-2, 1.0],
            "gamma": [0.01, 0.1, 1.0, 10.0],
        }

    model = GridSearchCV(
        estimator,
        grid,
        scoring=SCORER,
        cv=KFold(n_splits=4, shuffle=True, random_state=seed),
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    model.fit(x, y)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("输出目录已存在，请使用新的目录。")
    if args.repeats < 1:
        raise ValueError("repeats必须大于0")

    names = [
        "spm_pretraining.csv",
        "i2em_train_requests.csv",
        "i2em_train_results.csv",
        "i2em_test_requests.csv",
        "i2em_test_results.csv",
    ]
    paths = {name: args.data / name for name in names}

    spm = pd.read_csv(paths["spm_pretraining.csv"])
    train = multi.load_paired(
        paths["i2em_train_requests.csv"],
        paths["i2em_train_results.csv"],
    )
    test = multi.load_paired(
        paths["i2em_test_requests.csv"],
        paths["i2em_test_results.csv"],
    )

    xs = multi.features_from(spm)
    xt = multi.features_from(train)
    xv = multi.features_from(test)

    # 检查独立测试集中没有与训练集完全相同的特征行。
    overlap = set(map(tuple, xt)) & set(map(tuple, xv))
    if overlap:
        raise ValueError("训练与测试特征存在完全重复行，请先核查。")

    sizes = [32, 64, 128, 256]
    if len(train) < max(sizes):
        raise ValueError("有效I²EM训练样本少于256条，停止实验。")

    spm_cols = ["spm_hh_db", "spm_vv_db"]
    high_cols = ["i2em_hh_db", "i2em_vv_db"]

    ys = spm[spm_cols].to_numpy(float)
    yt = train[high_cols].to_numpy(float)
    yv = test[high_cols].to_numpy(float)
    low_train = train[spm_cols].to_numpy(float)
    low_test = test[spm_cols].to_numpy(float)

    for array in [ys, yt, yv, low_train, low_test]:
        if not np.isfinite(array).all():
            raise ValueError("教师输出包含NaN或Inf。")

    # 与已有神经代理一致，使用低保真训练输入确定标准化。
    # 不使用I²EM测试标签，也不使用测试输入拟合标准化。
    scaler = StandardScaler().fit(xs)
    zt = scaler.transform(xt)
    zv = scaler.transform(xv)

    args.output.mkdir(parents=True)
    manifest = {
        "status": "RUNNING",
        "scope": "simulation-only, post-hoc baseline supplement",
        "sizes": sizes,
        "repeats": args.repeats,
        "seed": args.seed,
        "n_test": len(test),
        "budget": "equal high-fidelity labels, not equal total compute",
        "residual_inference": "requires SPM output at each query",
        "files": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    metrics, parameters, subsets = [], [], []

    for repeat in range(1, args.repeats + 1):
        seed = args.seed + (repeat - 1) * 100_000
        order = np.random.default_rng(seed).permutation(len(train))

        for size in sizes:
            selected = order[:size]
            subsets.append({
                "repeat": repeat,
                "size": size,
                "row_indices": selected.tolist(),
                "request_ids": train.iloc[selected]["request_id"].astype(str).tolist(),
            })

            start = time.perf_counter()
            neural_components = multi.train_component_models(
                xs,
                multi.to_components(ys),
                xt[selected],
                multi.to_components(yt[selected]),
                xv,
                200,
                100,
                seed + size,
            )
            neural_seconds = time.perf_counter() - start

            predictions = {
                name: multi.from_components(value)
                for name, value in neural_components.items()
                if name in ["i2em_only", "spm_to_i2em"]
            }
            predictions["spm_teacher"] = low_test.copy()

            for kind in ["ridge", "rbf"]:
                for residual in [False, True]:
                    name = ("residual_" if residual else "direct_") + kind
                    target = yt[selected].copy()
                    if residual:
                        target -= low_train[selected]

                    start = time.perf_counter()
                    model = search_model(kind, zt[selected], target, seed + size)
                    fit_seconds = time.perf_counter() - start

                    start = time.perf_counter()
                    prediction = model.predict(zv)
                    prediction_seconds = time.perf_counter() - start
                    if residual:
                        prediction = prediction + low_test

                    predictions[name] = prediction
                    parameters.append({
                        "repeat": repeat,
                        "size": size,
                        "method": name,
                        "params": model.best_params_,
                        "cv_score": float(model.best_score_),
                        "fit_seconds": fit_seconds,
                        "prediction_seconds_excluding_spm": prediction_seconds,
                    })

            saved = {
                "reference": yv,
                "selected": selected,
                **predictions,
            }
            np.savez_compressed(
                args.output / f"predictions_r{repeat:02d}_n{size}.npz",
                **saved,
            )

            for name, prediction in predictions.items():
                metrics.append({
                    "repeat": repeat,
                    "size": size,
                    "method": name,
                    "rmse_mean_hh_vv_db": channel_rmse(yv, prediction),
                    "rmse_hh_db": float(np.sqrt(np.mean((yv[:, 0] - prediction[:, 0])**2))),
                    "rmse_vv_db": float(np.sqrt(np.mean((yv[:, 1] - prediction[:, 1])**2))),
                })

            parameters.append({
                "repeat": repeat,
                "size": size,
                "method": "neural_joint_training",
                "fit_and_prediction_seconds": neural_seconds,
            })
        print(f"完成重复 {repeat}/{args.repeats}", flush=True)

    frame = pd.DataFrame(metrics)
    frame.to_csv(args.output / "metrics_by_repeat.csv", index=False)
    summary = frame.groupby(["size", "method"], as_index=False).agg(
        rmse_mean_db=("rmse_mean_hh_vv_db", "mean"),
        rmse_std_db=("rmse_mean_hh_vv_db", "std"),
    )
    summary.to_csv(args.output / "summary.csv", index=False)

    paired = frame.pivot(
        index=["repeat", "size"],
        columns="method",
        values="rmse_mean_hh_vv_db",
    )
    for method in ["residual_ridge", "residual_rbf"]:
        paired[f"{method}_minus_sequential"] = (
            paired[method] - paired["spm_to_i2em"]
        )
    paired.to_csv(args.output / "paired_comparisons.csv")

    for name, content in [
        ("selected_subsets.json", subsets),
        ("selected_parameters.json", parameters),
    ]:
        (args.output / name).write_text(
            json.dumps(content, indent=2), encoding="utf-8"
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    for method, group in summary.groupby("method"):
        group = group.sort_values("size")
        ax.errorbar(
            group["size"], group["rmse_mean_db"],
            yerr=group["rmse_std_db"].fillna(0),
            marker="o", capsize=3, label=method,
        )
    ax.set(
        xlabel="I2EM training samples",
        ylabel="Mean HH/VV RMSE (dB)",
    )
    ax.set_xticks(sizes)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "learning_curve.png", dpi=200)
    plt.close(fig)

    manifest["status"] = "COMPLETE"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()