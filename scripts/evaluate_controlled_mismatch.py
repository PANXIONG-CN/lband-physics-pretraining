"""已知失配下的诊断压力测试，不代表新的实测验证。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_multifidelity_pretraining as multi


def diagnostics(y, prediction):
    error = prediction - y
    centered = (
        prediction - prediction.mean(axis=0)
        - (y - y.mean(axis=0))
    )
    variance = np.var(y, axis=0)
    if np.any(variance <= 0):
        raise ValueError("参考响应方差为零，中心化得分未定义。")

    mse = np.mean(error**2, axis=0)
    bias2 = error.mean(axis=0)**2
    centered_mse = np.mean(centered**2, axis=0)
    if not np.allclose(mse, bias2 + centered_mse, atol=1e-10):
        raise AssertionError("误差分解恒等式检查失败")

    yc = multi.to_components(y)
    pc = multi.to_components(prediction)
    ec = pc - pc.mean(0) - (yc - yc.mean(0))
    component_variance = np.var(yc, axis=0)
    if np.any(component_variance <= 0):
        raise ValueError("共同或差分分量方差为零。")

    skill = 1.0 - np.mean(ec**2, axis=0) / component_variance
    return {
        "rmse_db": float(np.sqrt(mse).mean()),
        "centered_rmse_db": float(np.sqrt(centered_mse).mean()),
        "common_skill": float(skill[0]),
        "differential_skill": float(skill[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("请指定新的输出目录。")
    if args.repeats < 1:
        raise ValueError("repeats必须为正数")

    names = [
        "i2em_train_requests.csv", "i2em_train_results.csv",
        "i2em_test_requests.csv", "i2em_test_results.csv",
    ]
    paths = {n: args.data / n for n in names}
    train = multi.load_paired(paths[names[0]], paths[names[1]])
    test = multi.load_paired(paths[names[2]], paths[names[3]])
    cols = ["i2em_hh_db", "i2em_vv_db"]
    a = train[cols].to_numpy(float)
    b = test[cols].to_numpy(float)

    # 情景定义只使用训练池参数，不使用测试输出选择系数。
    mean = a.mean(axis=0)
    moisture_mean = float(train.soil_moisture_m3_m3.mean())
    moisture_std = float(train.soil_moisture_m3_m3.std(ddof=0))
    if moisture_std <= 0:
        raise ValueError("训练含水率没有变化。")

    def transform(frame, y, scenario):
        if scenario == "none":
            return y.copy()
        if scenario == "offset":
            return y + np.array([-2.0, -1.0])
        if scenario == "gain":
            return mean + (y - mean) * np.array([1.4, 0.7])
        z = (
            frame.soil_moisture_m3_m3.to_numpy(float)
            - moisture_mean
        ) / moisture_std
        change = np.tanh(z)[:, None] * np.array([1.0, -1.0])
        return y + change

    rows, assignments = [], []
    args.output.mkdir(parents=True)

    for repeat in range(args.repeats):
        rng = np.random.default_rng(20260915 + repeat)
        order = rng.permutation(len(train))
        noise_train = rng.normal(size=a.shape)
        noise_test = rng.normal(size=b.shape)

        for scenario in ["none", "offset", "gain", "nonlinear"]:
            clean_train = transform(train, a, scenario)
            clean_test = transform(test, b, scenario)

            for noise in [0.0, 0.1, 0.5]:
                observed_train = clean_train + noise * noise_train
                observed_test = clean_test + noise * noise_test

                for size in [8, 16, 32, 64]:
                    if size > len(train):
                        raise ValueError("适应池不足64条。")
                    selected = order[:size]

                    # 只由适应样本估计通道偏移。
                    offset = (
                        observed_train[selected] - a[selected]
                    ).mean(axis=0)

                    predictions = {
                        "unadapted": b,
                        "offset_calibrated": b + offset,
                        "adaptation_mean": np.tile(
                            observed_train[selected].mean(axis=0),
                            (len(b), 1),
                        ),
                        # 已知生成过程参考，不是可部署模型。
                        "oracle_reference": clean_test,
                    }

                    assignments.append({
                        "repeat": repeat + 1,
                        "scenario": scenario,
                        "noise_db": noise,
                        "size": size,
                        "indices": selected.tolist(),
                    })

                    for method, prediction in predictions.items():
                        result = diagnostics(observed_test, prediction)
                        if scenario == "offset" and noise == 0:
                            if method == "offset_calibrated":
                                assert result["rmse_db"] < 1e-10

                        rows.append({
                            "repeat": repeat + 1,
                            "scenario": scenario,
                            "noise_db": noise,
                            "size": size,
                            "method": method,
                            **result,
                        })

        print(f"完成重复 {repeat + 1}/{args.repeats}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "metrics_by_repeat.csv", index=False)
    summary = frame.groupby(
        ["scenario", "noise_db", "size", "method"], as_index=False
    )[["rmse_db", "centered_rmse_db",
       "common_skill", "differential_skill"]].mean()
    summary.to_csv(args.output / "summary.csv", index=False)

    (args.output / "assignments.json").write_text(
        json.dumps(assignments, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "COMPLETE",
        "scope": "controlled diagnostic stress test",
        "adaptation_unit": "synthetic parameter sample, NOT real field",
        "test_role": "already available fixed simulation test points",
        "oracle": "known generating process; not deployable",
        "repeats": args.repeats,
        "files": {
            n: hashlib.sha256(p.read_bytes()).hexdigest()
            for n, p in paths.items()
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    part = summary[
        (summary.noise_db == 0)
        & (summary["size"] == 32)
        & (summary.method == "offset_calibrated")
    ]
    for ax, metric in zip(
        axes, ["rmse_db", "centered_rmse_db"]
    ):
        ax.bar(part.scenario, part[metric])
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(args.output / "controlled_mismatch.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()