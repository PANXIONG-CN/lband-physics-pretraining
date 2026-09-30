"""Diagnose where differential-response skill is lost on the shared SMEX02 cohort.

The script reconstructs the pretraining-only sequential surrogate from frozen
simulation tables and the original seeds.  It does not fit on SMEX02 labels and
does not select a model from target-domain scores.  Source-fine-tuned and
risk-shrunk predictions are read from the frozen prospective evaluation.
"""

from __future__ import annotations

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def centered_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - observed
    observed_centered = observed - observed.mean()
    predicted_centered = predicted - predicted.mean()
    centered_error = predicted_centered - observed_centered
    observed_variance = float(np.mean(observed_centered**2))
    predicted_variance = float(np.mean(predicted_centered**2))
    centered_mse = float(np.mean(centered_error**2))
    return {
        "n": int(len(observed)),
        "bias_db": float(np.mean(error)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "centered_rmse_db": float(np.sqrt(centered_mse)),
        "centered_skill": float(1.0 - centered_mse / observed_variance),
        "variance_ratio": float(predicted_variance / observed_variance),
    }


def field_bootstrap_indices(
    fields: np.ndarray,
    iterations: int,
    seed: int,
) -> list[np.ndarray]:
    unique_fields = np.unique(fields.astype(str))
    members = {
        field: np.flatnonzero(fields.astype(str) == field)
        for field in unique_fields
    }
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(iterations):
        sampled = rng.choice(unique_fields, size=len(unique_fields), replace=True)
        draws.append(np.concatenate([members[field] for field in sampled]))
    return draws


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def save_figure(
    metrics: pd.DataFrame,
    output: Path,
) -> None:
    stage_order = [
        "i2em_fixed40_teacher",
        "pretraining_only_surrogate",
        "source_finetuned_surrogate",
        "risk_shrunk_surrogate",
    ]
    labels = [
        "I$^2$EM\nteacher",
        "Pretraining-only\nsurrogate",
        "Source-fine-tuned\nsurrogate",
        "Risk-shrunk\nsurrogate",
    ]
    colors = ["#E69F00", "#0072B2", "#D55E00", "#56B4E9"]
    selected = metrics.set_index("method").loc[stage_order]
    x = np.arange(len(stage_order))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), constrained_layout=True)

    skill = selected["centered_skill"].to_numpy(float)
    skill_low = selected["centered_skill_ci_low"].to_numpy(float)
    skill_high = selected["centered_skill_ci_high"].to_numpy(float)
    axes[0].errorbar(
        x,
        skill,
        yerr=np.vstack([skill - skill_low, skill_high - skill]),
        fmt="none",
        ecolor="#444444",
        capsize=2.5,
        zorder=1,
    )
    axes[0].scatter(x, skill, c=colors, marker="o", zorder=2)
    axes[0].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Centered differential skill")

    bias = selected["bias_db"].to_numpy(float)
    bias_low = selected["bias_db_ci_low"].to_numpy(float)
    bias_high = selected["bias_db_ci_high"].to_numpy(float)
    axes[1].errorbar(
        x,
        bias,
        yerr=np.vstack([bias - bias_low, bias_high - bias]),
        fmt="none",
        ecolor="#444444",
        capsize=2.5,
        zorder=1,
    )
    axes[1].scatter(x, bias, c=colors, marker="o", zorder=2)
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Mean differential error (dB)")

    for label, axis in zip(["(a)", "(b)"], axes):
        axis.set_xticks(x, labels)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.text(
            0.01,
            0.98,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    stem = output / "Fig_stage_response_retention"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_angle_control_figure(metrics: pd.DataFrame, output: Path) -> None:
    order = ["i2em_fixed40_teacher", "i2em_actual_angle_control"]
    labels = [r"Fixed $40^{\circ}$", "Field-day angle"]
    selected = metrics.set_index("method").loc[order]
    x = np.arange(2)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), constrained_layout=True)
    for axis, metric_name, ylabel in [
        (axes[0], "centered_skill", "Centered differential skill"),
        (axes[1], "bias_db", "Mean differential error (dB)"),
    ]:
        values = selected[metric_name].to_numpy(float)
        low = selected[f"{metric_name}_ci_low"].to_numpy(float)
        high = selected[f"{metric_name}_ci_high"].to_numpy(float)
        axis.errorbar(
            x,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt="o",
            color="#E69F00",
            capsize=2.5,
        )
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[0].text(0.01, 0.98, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")
    axes[1].text(0.01, 0.98, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")

    stem = output / "Fig_i2em_observation_condition_control"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)



def update_frozen_source_offset(root: Path, output: Path, iterations: int,
                                seed: int) -> None:
    """Reproduce the intercept control using saved weights and existing tables.

    Output mirrors existing reproducibility paths. No neural fitting, new data,
    optimizer reset, target-label calibration, or historical-result replacement.
    """
    from recalculate_frozen_results import (
        KEY, YCOL, component_metrics, component_bootstrap, field_weights,
        read_table, rmse_avg, source_offset_predictions, unified_features,
    )
    data = root / 'reproducibility'
    source = read_table(data/'data/source/smapvex12_portable_source.csv', YCOL, KEY)
    zero = read_table(data/'results/cross_domain/zero_shot_predictions.csv', YCOL, KEY)
    stage = read_table(data/'results/stage_diagnostics/stage_predictions.csv', YCOL, KEY)
    if len(source) != 240 or len(zero) != 189 or len(stage) != 138:
        raise ValueError('The intercept control requires the original 240/189/138 cohorts')
    control = source_offset_predictions(data/'results/trajectory_reconstruction',
                                        source, unified_features(zero))
    c = control['common_source_mean_db']
    for method, values in [('pretraining_only', control['target_pretraining']),
                           ('pretraining_source_offset', control['target_offset'])]:
        zero[f'hh_{method}'] = c - values/2
        zero[f'vv_{method}'] = c + values/2
    prediction = zero[KEY].copy()
    prediction['d_pre'] = control['target_pretraining']
    prediction['d_offset'] = control['target_offset']
    linked = stage[KEY].merge(prediction, on=KEY, validate='one_to_one')
    np.testing.assert_allclose(linked.d_pre, stage.differential_pretraining_only_surrogate,
                               atol=1e-11, rtol=0)
    stage['differential_pretraining_source_offset'] = linked.d_offset.to_numpy()
    stage['differential_source_mean'] = control['differential_source_mean_db']

    def write_csv(frame, rel):
        p = output/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False)

    def write_json(value, rel):
        p = output/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    # Retain original rows/columns and append the control under its own method name.
    write_csv(zero, 'results/cross_domain/zero_shot_predictions.csv')
    write_csv(stage, 'results/stage_diagnostics/stage_predictions.csv')
    members = read_table(data/'results/trajectory_reconstruction/predictions_by_member.csv',
                         ['repeat', 'd_pretraining'], ['repeat']+KEY)
    for j in range(5):
        select = members['repeat'].eq(j+1)
        current = members.loc[select, KEY].merge(
            zero[KEY].assign(pre=control['target_pretraining_by_member'][j]),
            on=KEY, validate='one_to_one')
        np.testing.assert_allclose(current.pre, members.loc[select, 'd_pretraining'],
                                   atol=1e-11, rtol=0)
        members.loc[select, 'source_offset_db'] = control['intercepts_by_member_db'][j]
        members.loc[select, 'd_source_offset'] = (current.pre + control['intercepts_by_member_db'][j]).to_numpy()
    write_csv(members, 'results/trajectory_reconstruction/predictions_by_member.csv')

    methods = ['i2em_fixed40_teacher', 'pretraining_only_surrogate',
               'source_finetuned_surrogate', 'risk_shrunk_surrogate',
               'i2em_actual_angle_control', 'pretraining_source_offset']
    if 'differential_source_reset_adam_003' in stage:
        methods.append('source_reset_adam_003')
    methods.append('source_mean')
    y = (stage[YCOL[1]] - stage[YCOL[0]]).to_numpy(float)
    # String ordering reproduces the original stage bootstrap's field universe.
    w, inv, _ = field_weights(stage.field_id.astype(str), iterations, seed)
    boot, points, rows = {}, {}, []
    for method in methods:
        p = stage[f'differential_{method}'].to_numpy(float)
        values = component_metrics(y, p)
        samples = component_bootstrap(y, p, inv, w)
        counts = w @ np.bincount(inv, minlength=w.shape[1])
        samples['rmse'] = np.sqrt((w @ np.bincount(inv, weights=(p-y)**2,
                                                  minlength=w.shape[1]))/counts)
        boot[method], points[method] = samples, values
        rows.append({'method': method, 'n': len(y), **values,
                     'centered_skill_ci_low': interval(samples['skill'])[0],
                     'centered_skill_ci_high': interval(samples['skill'])[1],
                     'centered_skill_probability_above_zero': float(np.mean(samples['skill'] > 0)),
                     'bias_db_ci_low': interval(samples['bias'])[0],
                     'bias_db_ci_high': interval(samples['bias'])[1],
                     'rmse_db_ci_low': interval(samples['rmse'])[0],
                     'rmse_db_ci_high': interval(samples['rmse'])[1]})
    metrics = pd.DataFrame(rows)
    original = pd.read_csv(data/'results/stage_diagnostics/stage_metrics.csv').set_index('method')
    # Historical point estimates stay fixed; published intervals use the archived resampling settings.
    common_cols = [col for col in original.columns if col in metrics.columns]
    if (iterations, seed) != (10000, 27260910):
        common_cols = [col for col in common_cols
                       if '_ci_' not in col and '_probability_' not in col]
    old_methods = [m for m in methods if m not in {'pretraining_source_offset', 'source_mean'} and m in original.index]
    np.testing.assert_allclose(metrics.set_index('method').loc[old_methods, common_cols].to_numpy(float),
                               original.loc[old_methods, common_cols].to_numpy(float), atol=1e-9, rtol=0)
    write_csv(metrics, 'results/stage_diagnostics/stage_metrics.csv')
    comparisons = [('training_stage', methods[0], methods[1]),
                   ('training_stage', methods[1], methods[2]),
                   ('training_stage', methods[2], methods[3]),
                   ('observation_condition_control', methods[0], methods[4]),
                   ('source_intercept_control', methods[1], methods[5]),
                   ('source_intercept_control', methods[2], methods[5])]
    if 'source_reset_adam_003' in methods:
        comparisons.extend([('source_adam_reset_control', first, 'source_reset_adam_003')
                            for first in ['source_finetuned_surrogate', 'pretraining_only_surrogate']])
        comparisons.append(('source_intercept_control', 'source_reset_adam_003', 'pretraining_source_offset'))
    comparisons.append(('source_intercept_control', 'source_mean', 'pretraining_source_offset'))
    pairs = []
    for role, first, second in comparisons:
        a, b = boot[first], boot[second]
        pairs.append({'comparison_type': role, 'first_stage': first, 'second_stage': second,
            'centered_skill_delta_second_minus_first': points[second]['centered_skill']-points[first]['centered_skill'],
            'centered_skill_delta_ci_low': interval(b['skill']-a['skill'])[0],
            'centered_skill_delta_ci_high': interval(b['skill']-a['skill'])[1],
            'bias_delta_second_minus_first_db': points[second]['bias_db']-points[first]['bias_db'],
            'bias_delta_ci_low_db': interval(b['bias']-a['bias'])[0],
            'bias_delta_ci_high_db': interval(b['bias']-a['bias'])[1],
            'rmse_delta_second_minus_first_db': points[second]['rmse_db']-points[first]['rmse_db'],
            'rmse_delta_ci_low_db': interval(b['rmse']-a['rmse'])[0],
            'rmse_delta_ci_high_db': interval(b['rmse']-a['rmse'])[1]})
    write_csv(pd.DataFrame(pairs), 'results/stage_diagnostics/stage_pairwise_deltas.csv')
    joint = pd.read_csv(data/'results/cross_domain/zero_shot_joint_channel_metrics.csv')
    for method in ['pretraining_only', 'pretraining_source_offset']:
        p = zero[[f'hh_{method}', f'vv_{method}']].to_numpy(float)
        channel_mse = np.mean((p-zero[YCOL].to_numpy(float))**2, axis=0)
        joint = joint[joint.method.ne(method)]
        joint = pd.concat([joint, pd.DataFrame([{
            'method': method, 'mean_hh_vv_rmse_db': float(np.sqrt(channel_mse).mean()),
            'pooled_hh_vv_rmse_db': float(np.sqrt(channel_mse.mean())),
            'hh_rmse_db': float(np.sqrt(channel_mse[0])),
            'vv_rmse_db': float(np.sqrt(channel_mse[1]))}])], ignore_index=True)
    write_csv(joint, 'results/cross_domain/zero_shot_joint_channel_metrics.csv')
    comparison = next(p for p in pairs if p['first_stage'] == 'source_finetuned_surrogate'
                       and p['second_stage'] == 'pretraining_source_offset')
    summary = {
        'analysis_role': 'post-hoc source-intercept control',
        'training_performed': False, 'target_labels_used_for_calibration': False,
        'feature_rule': 'Topp dielectric on both campaigns; original NPZ scales fixed',
        'source_rows': len(source), 'source_fields': int(source.field_id.nunique()),
        'common_source_mean_db': c,
        'source_intercept_db': control['source_intercept_db'],
        'member_intercepts_db': control['intercepts_by_member_db'].tolist(),
        'source_calibrated_mean_residual_db': float(np.mean(
             control['source_predictions_by_member'] + control['intercepts_by_member_db'][:, None]
             - (source[YCOL[1]]-source[YCOL[0]]).to_numpy(float)[None, :])),
        'shared_138': {**points['pretraining_source_offset'],
                      'source_mean_differential_rmse_db': component_metrics(
                          y, np.full(len(y), control['differential_source_mean_db']))['rmse_db'],
                      'offset_minus_finetuned': comparison,
                      'paired_contrasts': [p for p in pairs if p['comparison_type'] == 'source_intercept_control']},
        'full_189': {'rows': len(zero), 'fields': int(zero.field_id.nunique()),
                     **component_metrics((zero[YCOL[1]]-zero[YCOL[0]]).to_numpy(float),
                                          control['target_offset']),
                     'mean_hh_vv_rmse_db': float(joint.set_index('method').loc['pretraining_source_offset','mean_hh_vv_rmse_db']),
                     'mean_hh_vv_rmse_delta_vs_source_mean_db': float(
                          joint.set_index('method').loc['pretraining_source_offset','mean_hh_vv_rmse_db']
                          - joint.set_index('method').loc['source_mean','mean_hh_vv_rmse_db'])},
        'bootstrap': {'unit': 'field_id', 'iterations': iterations, 'seed': seed,
                      'source_intercept_fixed': True, 'paired_across_methods': True},
        'original_primary_comparison': 'unchanged; this post-hoc branch is not pooled with original primary results',
    }
    for rel in ['results/cross_domain/zero_shot_summary.json',
                'results/paper_summaries/phase3_summary.json',
                'results/trajectory_reconstruction/summary.json']:
        value = json.loads((data/rel).read_text())
        value['posthoc_source_offset'] = summary
        write_json(value, rel)
    rel = 'results/stage_diagnostics/manifest.json'
    manifest = json.loads((data/rel).read_text())
    manifest['posthoc_source_offset'] = summary
    manifest['source_offset_inputs'] = {
        str(p.relative_to(root)): sha256(p) for p in [
            data/'data/source/smapvex12_portable_source.csv',
            data/'data/target/smex02_field_day_model_ready.csv',
            *[data/f'results/trajectory_reconstruction/rebuilt_member_{j}.npz' for j in range(1, 6)]]}
    write_json(manifest, rel)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-source-offset", action="store_true",
                        help="Forward-only source intercept control from included NPZ weights; no training")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--spm-epochs", type=int, default=200)
    parser.add_argument("--i2em-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    if args.frozen_source_offset:
        update_frozen_source_offset(root, output, args.bootstrap_iterations, args.seed + 7_000_000)
        return

    scripts = root / "scripts"
    source_root = root / "src"
    sys.path[:0] = [str(scripts), str(source_root)]
    import evaluate_decoupled_physics_heads as base
    import evaluate_multifidelity_pretraining as multi

    common_path = root / "reproducibility/results/common_cohort/predictions.csv"
    physics_path = root / "reproducibility/results/stage_diagnostics/stage_predictions.csv"
    spm_path = root / "reproducibility/data/teachers/spm_pretraining.csv"
    i2em_request_path = root / "reproducibility/data/teachers/i2em_train_requests.csv"
    i2em_result_path = root / "reproducibility/data/teachers/i2em_train_results.csv"
    input_paths = [
        common_path,
        physics_path,
        spm_path,
        i2em_request_path,
        i2em_result_path,
    ]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    common = pd.read_csv(common_path, dtype={"field_id": str})
    physics = pd.read_csv(physics_path, dtype={"field_id": str})
    spm = pd.read_csv(spm_path)
    i2em = multi.load_paired(i2em_request_path, i2em_result_path)
    keys = ["field_id", "acquisition_date"]
    columns = [*keys, "i2em_fixed40_hh_db", "i2em_fixed40_vv_db", "i2em_actual_hh_db", "i2em_actual_vv_db"]
    frame = common.merge(physics[columns], on=keys, validate="one_to_one")
    if len(frame) != 138 or frame["field_id"].nunique() != 22:
        raise AssertionError("Expected 138 rows from 22 SMEX02 fields")

    features = frame[list(base.FEATURE_NAMES)].to_numpy(float)
    spm_features = multi.features_from(spm)
    spm_differential = base.to_components(
        spm[["spm_hh_db", "spm_vv_db"]].to_numpy(float)
    )[:, 1]
    i2em_features = multi.features_from(i2em)
    i2em_differential = base.to_components(
        i2em[["i2em_hh_db", "i2em_vv_db"]].to_numpy(float)
    )[:, 1]

    reconstructed = []
    reconstruction_metadata = []
    for repeat in range(1, args.repeats + 1):
        seed = args.seed + (repeat - 1) * 100_000
        spm_bundle = base.pretrain_single_output(
            spm_features, spm_differential, args.spm_epochs, seed
        )
        sequential_model = base.fine_tune_single(
            spm_bundle,
            i2em_features,
            i2em_differential,
            args.i2em_epochs,
            seed + 2,
        )
        reconstructed.append(base.predict_single(spm_bundle, sequential_model, features))
        reconstruction_metadata.append(
            {
                "repeat": repeat,
                "seed": seed,
                "spm_epochs": args.spm_epochs,
                "i2em_epochs": args.i2em_epochs,
                "source_observation_fitting": False,
            }
        )
    frame["differential_pretraining_only"] = np.mean(reconstructed, axis=0)

    observed = frame["sigma0_vv_db"].to_numpy(float) - frame["sigma0_hh_db"].to_numpy(float)
    predictions = {
        "i2em_fixed40_teacher": frame["i2em_fixed40_vv_db"].to_numpy(float)
        - frame["i2em_fixed40_hh_db"].to_numpy(float),
        "pretraining_only_surrogate": frame["differential_pretraining_only"].to_numpy(float),
        "source_finetuned_surrogate": frame["vv_spm_to_i2em"].to_numpy(float)
        - frame["hh_spm_to_i2em"].to_numpy(float),
        "risk_shrunk_surrogate": frame["vv_risk_spm_to_i2em"].to_numpy(float)
        - frame["hh_risk_spm_to_i2em"].to_numpy(float),
        "i2em_actual_angle_control": frame["i2em_actual_vv_db"].to_numpy(float)
        - frame["i2em_actual_hh_db"].to_numpy(float),
    }
    for method, values in predictions.items():
        frame[f"differential_{method}"] = values

    fields = frame["field_id"].astype(str).to_numpy()
    draws = field_bootstrap_indices(fields, args.bootstrap_iterations, args.seed + 7_000_000)
    bootstrap = {
        method: {"centered_skill": [], "bias_db": []}
        for method in predictions
    }
    for index in draws:
        for method, predicted in predictions.items():
            values = centered_metrics(observed[index], predicted[index])
            bootstrap[method]["centered_skill"].append(values["centered_skill"])
            bootstrap[method]["bias_db"].append(values["bias_db"])

    metric_rows = []
    for method, predicted in predictions.items():
        values = centered_metrics(observed, predicted)
        skill_samples = np.asarray(bootstrap[method]["centered_skill"])
        bias_samples = np.asarray(bootstrap[method]["bias_db"])
        metric_rows.append(
            {
                "method": method,
                **values,
                "centered_skill_ci_low": interval(skill_samples)[0],
                "centered_skill_ci_high": interval(skill_samples)[1],
                "centered_skill_probability_above_zero": float(np.mean(skill_samples > 0)),
                "bias_db_ci_low": interval(bias_samples)[0],
                "bias_db_ci_high": interval(bias_samples)[1],
            }
        )
    metrics = pd.DataFrame(metric_rows)

    stage_order = [
        "i2em_fixed40_teacher",
        "pretraining_only_surrogate",
        "source_finetuned_surrogate",
        "risk_shrunk_surrogate",
    ]
    pair_rows = []
    for first, second in zip(stage_order[:-1], stage_order[1:]):
        first_skill = np.asarray(bootstrap[first]["centered_skill"])
        second_skill = np.asarray(bootstrap[second]["centered_skill"])
        first_bias = np.asarray(bootstrap[first]["bias_db"])
        second_bias = np.asarray(bootstrap[second]["bias_db"])
        point_first = metrics.set_index("method").loc[first]
        point_second = metrics.set_index("method").loc[second]
        pair_rows.append(
            {
                "comparison_type": "training_stage",
                "first_stage": first,
                "second_stage": second,
                "centered_skill_delta_second_minus_first": float(
                    point_second["centered_skill"] - point_first["centered_skill"]
                ),
                "centered_skill_delta_ci_low": interval(second_skill - first_skill)[0],
                "centered_skill_delta_ci_high": interval(second_skill - first_skill)[1],
                "bias_delta_second_minus_first_db": float(
                    point_second["bias_db"] - point_first["bias_db"]
                ),
                "bias_delta_ci_low_db": interval(second_bias - first_bias)[0],
                "bias_delta_ci_high_db": interval(second_bias - first_bias)[1],
            }
        )

    first = "i2em_fixed40_teacher"
    second = "i2em_actual_angle_control"
    first_skill = np.asarray(bootstrap[first]["centered_skill"])
    second_skill = np.asarray(bootstrap[second]["centered_skill"])
    first_bias = np.asarray(bootstrap[first]["bias_db"])
    second_bias = np.asarray(bootstrap[second]["bias_db"])
    point_first = metrics.set_index("method").loc[first]
    point_second = metrics.set_index("method").loc[second]
    pair_rows.append(
        {
            "comparison_type": "observation_condition_control",
            "first_stage": first,
            "second_stage": second,
            "centered_skill_delta_second_minus_first": float(
                point_second["centered_skill"] - point_first["centered_skill"]
            ),
            "centered_skill_delta_ci_low": interval(second_skill - first_skill)[0],
            "centered_skill_delta_ci_high": interval(second_skill - first_skill)[1],
            "bias_delta_second_minus_first_db": float(
                point_second["bias_db"] - point_first["bias_db"]
            ),
            "bias_delta_ci_low_db": interval(second_bias - first_bias)[0],
            "bias_delta_ci_high_db": interval(second_bias - first_bias)[1],
        }
    )
    pairwise = pd.DataFrame(pair_rows)

    frame.to_csv(output / "stage_predictions.csv", index=False)
    metrics.to_csv(output / "stage_metrics.csv", index=False)
    pairwise.to_csv(output / "stage_pairwise_deltas.csv", index=False)
    save_figure(metrics, output)
    save_angle_control_figure(metrics, output)

    manifest = {
        "status": "COMPLETE",
        "analysis_role": "post-hoc stage-wise response-retention diagnosis",
        "target_label_use": "metrics only; no target-domain fitting or selection",
        "historical_weight_artifact_available": False,
        "physics_values": "read from the archived stage table, not regenerated by a physical solver",
        "pretraining_only_stage": (
            "deterministically reconstructed from frozen simulation tables, "
            "the original architecture, epochs, and five source seeds"
        ),
        "source_finetuned_and_risk_stages": "read from frozen prospective predictions",
        "cohort": {"rows": len(frame), "fields": int(frame.field_id.nunique())},
        "bootstrap": {
            "unit": "field_id",
            "paired_across_methods": True,
            "iterations": args.bootstrap_iterations,
            "seed": args.seed + 7_000_000,
        },
        "reconstruction": reconstruction_metadata,
        "inputs": {str(path.relative_to(root)): sha256(path) for path in input_paths},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
