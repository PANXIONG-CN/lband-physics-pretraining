"""Reproducible, exploratory polarization diagnostics. Raw inputs are read-only.

This is a diagnostic analysis, not an out-of-sample model evaluation or a causal
decomposition. The two residual coordinates do not uniquely identify mechanisms.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/scattering/rough_ground"
KEYS = ["field_id", "acquisition_date"]
FEATURES = ["soil_moisture_m3_m3", "soil_real_dielectric",
            "pals_rms_height_cm", "pals_correlation_length_cm"]
OBS = ["sigma0_hh_db", "sigma0_vv_db"]
PHYS = ["exponential_spm_hh_raw_db", "exponential_spm_vv_raw_db"]
VWC = "vegetation_water_content_in_situ_kg_m2"
MAP = "vegetation_water_content_map_kg_m2"
VEG_REQUIRED = [VWC, "vegetation_sample_date", "vegetation_time_offset_days"]
VEG_OPTIONAL = ["crop_type", "vegetation_site_count", "canopy_height_cm",
                "vegetation_water_content_in_situ_std_kg_m2", MAP,
                "vwc_map_source", "vwc_map_filename", "vwc_map_pixel_count"]
LABELS = {VWC: "In-situ VWC (kg/m2)", MAP: "Mapped VWC (kg/m2)",
          "soil_moisture_m3_m3": "Soil moisture (m3/m3)",
          "soil_real_dielectric": "Relative permittivity (real part)",
          "pals_rms_height_cm": "RMS height (cm)",
          "pals_correlation_length_cm": "Correlation length (cm)",
          "common_residual_db": "Common residual (dB)",
          "differential_residual_db": "Differential residual (dB)"}


def fingerprint(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def require_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def normalize_keys(frame, name):
    frame = frame.copy()
    require_columns(frame, KEYS, name)
    frame["field_id"] = frame["field_id"].astype("string").str.strip()
    if frame["field_id"].isna().any() or frame["field_id"].eq("").any():
        raise ValueError(f"{name}: empty field_id")
    frame["acquisition_date"] = pd.to_datetime(frame["acquisition_date"], errors="raise").dt.normalize()
    if frame["acquisition_date"].isna().any():
        raise ValueError(f"{name}: missing acquisition_date")
    if frame.duplicated(KEYS).any():
        examples = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head(8)
        raise ValueError(f"{name}: duplicate field-date keys; do not silently aggregate:\n{examples}")
    return frame


def parse_bool(series):
    values = series.astype("string").str.strip().str.lower()
    unknown = values.notna() & ~values.isin(["true", "false", "1", "0"])
    if unknown.any():
        raise ValueError(f"Unrecognized Boolean values: {values[unknown].unique().tolist()}")
    return values.map({"true": True, "false": False, "1": True, "0": False}).fillna(False).astype(bool)


def numeric(frame, columns):
    for column in columns:
        # Reject nonempty malformed strings instead of silently making them missing.
        frame[column] = pd.to_numeric(frame[column], errors="raise")


def counts(frame):
    return {"rows": len(frame), "fields": int(frame.field_id.nunique()),
            "dates": int(frame.acquisition_date.nunique())}


def residuals(frame):
    frame = frame.copy()
    frame["residual_hh_db"] = frame[OBS[0]] - frame[PHYS[0]]
    frame["residual_vv_db"] = frame[OBS[1]] - frame[PHYS[1]]
    frame["observed_polarization_difference_db"] = frame[OBS[1]] - frame[OBS[0]]
    frame["spm_polarization_difference_db"] = frame[PHYS[1]] - frame[PHYS[0]]
    frame["common_residual_db"] = (frame.residual_hh_db + frame.residual_vv_db) / 2
    frame["differential_residual_db"] = frame.residual_vv_db - frame.residual_hh_db
    return frame


def assemble_cohort(base, vegetation, quality_days=2):
    if quality_days < 0:
        raise ValueError("quality_days must be nonnegative")
    base = normalize_keys(base, "SPM source")
    vegetation = normalize_keys(vegetation, "Vegetation source")
    require_columns(base, ["spm_valid", *FEATURES, *OBS, *PHYS], "SPM source")
    require_columns(vegetation, VEG_REQUIRED, "Vegetation source")
    numeric(base, [*FEATURES, *OBS, *PHYS])
    # Verify that the enriched table belongs to the same observations/model run.
    overlap = [c for c in FEATURES + OBS + PHYS if c in vegetation]
    numeric(vegetation, overlap)
    check = base[KEYS + overlap].merge(vegetation[KEYS + overlap], on=KEYS,
                                     suffixes=("_base", "_veg"), validate="one_to_one")
    for col in overlap:
        if not np.allclose(check[col + "_base"], check[col + "_veg"],
                           rtol=1e-9, atol=1e-8, equal_nan=True):
            raise ValueError(f"Conflicting source values for {col}; check dataset versions")
    veg_cols = VEG_REQUIRED + [c for c in VEG_OPTIONAL if c in vegetation]
    for col in (VWC, "vegetation_time_offset_days", MAP):
        if col in vegetation:
            numeric(vegetation, [col])
    collisions = set(veg_cols).intersection(base.columns)
    if collisions:
        raise ValueError(f"Base input must be the original spectrum table, not enriched: {collisions}")
    merged = base.merge(vegetation[KEYS + veg_cols], on=KEYS, how="left",
                        validate="one_to_one", indicator="vegetation_join")
    merged["in_original_model_cohort"] = parse_bool(merged.spm_valid) & np.isfinite(
        merged[FEATURES + OBS + PHYS].to_numpy(dtype=float)).all(axis=1)
    merged["vegetation_sample_date"] = pd.to_datetime(merged.vegetation_sample_date, errors="raise")
    gap = (merged.acquisition_date - merged.vegetation_sample_date).dt.total_seconds().abs() / 86400
    provided_gap = merged.vegetation_time_offset_days
    known = gap.notna() & provided_gap.notna()
    if not np.allclose(gap[known], provided_gap[known], atol=1e-8):
        raise ValueError("Vegetation date and recorded time offset disagree")
    merged["in_situ_gap_days_verified"] = gap
    merged["in_situ_match_class"] = np.select(
        [gap.eq(0), gap.gt(0)], ["same_day", "nearest_date"], default="missing_date")
    available = np.isfinite(merged[VWC]) & merged[VWC].ge(0) & gap.notna()
    merged["in_common_cohort"] = merged.in_original_model_cohort & available
    merged["in_high_quality_cohort"] = merged.in_common_cohort & gap.le(quality_days)
    reasons = []
    for _, row in merged.iterrows():
        reason = []
        if not row.in_original_model_cohort:
            reason.append("outside_original_spm_valid_finite_cohort")
        if row.vegetation_join != "both":
            reason.append("no_vegetation_key_match")
        if not np.isfinite(row[VWC]) or row[VWC] < 0:
            reason.append("missing_or_invalid_in_situ_vwc")
        if pd.isna(row.in_situ_gap_days_verified):
            reason.append("missing_vegetation_date")
        reasons.append(";".join(reason) or "included")
    merged["common_cohort_exclusion_reason"] = reasons
    merged = residuals(merged).sort_values(KEYS).reset_index(drop=True)
    return merged


def build_cohort(base_path, vegetation_path, output, quality_days=2):
    output = Path(output).resolve()
    files = ["cohort_audit.csv", "cohort_base.csv", "cohort_common.csv",
             "cohort_high_quality.csv", "cohort_counts_by_field.csv", "cohort_manifest.json"]
    if any((output / f).exists() for f in files):
        raise FileExistsError("Cohort outputs already exist. Choose a new --output directory.")
    base = pd.read_csv(base_path, dtype={"field_id": "string"})
    vegetation = pd.read_csv(vegetation_path, dtype={"field_id": "string"})
    audit = assemble_cohort(base, vegetation, quality_days)
    if not audit.in_common_cohort.any():
        raise ValueError("No common samples; inspect source keys and missing values")
    parts = {"base": audit[audit.in_original_model_cohort],
             "common": audit[audit.in_common_cohort],
             "high_quality": audit[audit.in_high_quality_cohort]}
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "cohort_audit.csv", index=False, encoding="utf-8-sig")
    for name, part in parts.items():
        part.to_csv(output / f"cohort_{name}.csv", index=False, encoding="utf-8-sig")
    audit.groupby("field_id", observed=True).agg(
        input_rows=("acquisition_date", "size"),
        base_rows=("in_original_model_cohort", "sum"),
        common_rows=("in_common_cohort", "sum"),
        high_quality_rows=("in_high_quality_cohort", "sum")
    ).to_csv(output / "cohort_counts_by_field.csv", encoding="utf-8-sig")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [fingerprint(base_path), fingerprint(vegetation_path)],
        "code": fingerprint(__file__), "input": counts(audit),
        "cohorts": {key: counts(value) for key, value in parts.items()},
        "quality_days": quality_days,
        "in_situ_gap_counts": parts["common"].in_situ_gap_days_verified.value_counts().sort_index().to_dict(),
        "map_source_counts": parts["common"].get("vwc_map_source", pd.Series(dtype=str)).value_counts().to_dict(),
        "missing_columns_optional": [c for c in VEG_OPTIONAL if c not in vegetation],
        "limitations": ["Quality-day cutoff is an analysis choice, not an official quality standard.",
            "In-situ values are nearest-date matches, not interpolated map values.",
            "Map provenance labels are inherited from the existing vegetation audit, not revalidated here.",
            "No missing-value imputation; no model training; no observation or raw-file modification."]}
    write_json(output / "cohort_manifest.json", manifest)
    print(json.dumps(manifest["cohorts"], ensure_ascii=False, indent=2), flush=True)
    return manifest


def pair_blocks(frame, x, y, mode):
    """Each array is one field block; within-field demeaning uses paired finite rows."""
    valid = frame[["field_id", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    blocks = []
    for _, group in valid.groupby("field_id", sort=True, observed=True):
        values = group[[x, y]].to_numpy(dtype=float)
        if mode == "between_field":
            values = values.mean(axis=0, keepdims=True)
        elif mode == "within_field":
            if len(values) < 2 or np.ptp(values[:, 0]) < 1e-10:
                continue
            values = values - values.mean(axis=0)
        elif mode != "pooled":
            raise ValueError(f"Unknown mode {mode}")
        blocks.append(values)
    return blocks


def rho(values):
    if len(values) < 3 or np.any(np.ptp(values, axis=0) < 1e-10):
        return float("nan")
    ranks = np.column_stack([rankdata(values[:, 0]), rankdata(values[:, 1])])
    return float(np.corrcoef(ranks.T)[0, 1])


def association(frame, x, y, mode, n_boot, seed):
    blocks = pair_blocks(frame, x, y, mode)
    values = np.concatenate(blocks) if blocks else np.empty((0, 2))
    estimate = rho(values)
    result = {"variable": x, "residual": y, "mode": mode, "n_rows_used": len(values),
              "n_fields_used": len(blocks), "rho": estimate,
              "ci_low": np.nan, "ci_high": np.nan, "bootstrap_valid": 0,
              "lofo_rho_min": np.nan, "lofo_rho_max": np.nan,
              "status": "insufficient_or_constant"}
    if len(blocks) < 4 or len(values) < 8 or not np.isfinite(estimate):
        return result
    rng = np.random.default_rng(seed)
    draws = np.array([rho(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]))
                      for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    result["bootstrap_valid"] = len(draws)
    if len(draws) >= 0.9 * n_boot:
        result["ci_low"], result["ci_high"] = np.quantile(draws, [0.025, 0.975]).tolist()
        result["status"] = "exploratory" if len(blocks) >= 10 else "few_fields_exploratory"
    leave_one = [rho(np.concatenate([b for j, b in enumerate(blocks) if j != i]))
                 for i in range(len(blocks))]
    finite = [v for v in leave_one if np.isfinite(v)]
    if finite:
        result["lofo_rho_min"], result["lofo_rho_max"] = min(finite), max(finite)
    return result


def draw_figures(frame, statistics, output):
    # Configure before importing pyplot. Cache and figures remain on the D-drive project.
    import os
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    color = "#216B8D"

    def save(fig, name):
        fig.savefig(output / name, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def scatter(ax, x, y, title, identity=False):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        ax.scatter(x, y, s=23, alpha=0.65, color=color, edgecolors="none")
        if len(x) and identity:
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
        elif len(x):
            ax.axhline(0, color="0.5", ls="--", lw=0.8)
        estimate = rho(np.column_stack([x, y])) if len(x) else np.nan
        annotation = f"n={len(x)}; Spearman rho={estimate:.2f}" if np.isfinite(estimate) else f"n={len(x)}; rho unavailable"
        ax.set_title(title + "\n" + annotation, fontsize=11)
        ax.grid(alpha=0.15)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    scatter(axes[0], frame.observed_polarization_difference_db,
            frame.spm_polarization_difference_db, "HH/VV contrast", identity=True)
    axes[0].set(xlabel="Observed VV - HH (dB)", ylabel="Raw SPM VV - HH (dB)")
    scatter(axes[1], (frame[OBS[0]] + frame[OBS[1]]) / 2,
            (frame[PHYS[0]] + frame[PHYS[1]]) / 2, "Common intensity", identity=True)
    axes[1].set(xlabel="Observed (HH + VV) / 2 (dB)", ylabel="Raw SPM (HH + VV) / 2 (dB)")
    fig.suptitle(f"Matched cohort: {len(frame)} field-days, {frame.field_id.nunique()} fields")
    save(fig, "01_polarization_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, x in zip(axes.flat, [VWC, FEATURES[1], FEATURES[2], FEATURES[3]]):
        scatter(ax, frame[x], frame.common_residual_db, "Pooled field-days")
        ax.set(xlabel=LABELS[x], ylabel=LABELS["common_residual_db"])
    fig.suptitle("Common residual: observed minus raw exponential SPM")
    save(fig, "02_common_residual_environment.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for i, mode in enumerate(["pooled", "within_field"]):
        for j, x in enumerate([VWC, FEATURES[1]]):
            blocks = pair_blocks(frame, x, "differential_residual_db", mode)
            values = np.concatenate(blocks) if blocks else np.empty((0, 2))
            ax = axes[i, j]
            scatter(ax, values[:, 0], values[:, 1], f"{mode.replace('_', ' ')}; {len(blocks)} fields")
            suffix = " (field-centered)" if i else ""
            ax.set(xlabel=LABELS[x] + suffix, ylabel="Differential residual (dB)" + suffix)
    fig.suptitle("Polarization discrepancy: associations are not causal attribution")
    save(fig, "03_differential_residual_pooled_within.png")

    subset_names = ["common", "same_day", "gap_le_2d", "gap_le_4d"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for i, y in enumerate(["common_residual_db", "differential_residual_db"]):
        for j, mode in enumerate(["pooled", "within_field"]):
            ax = axes[i, j]
            labels = []
            for pos, subset in enumerate(subset_names):
                row = statistics[(statistics.subset == subset) & (statistics.variable == VWC)
                                 & (statistics.residual == y) & (statistics["mode"] == mode)].iloc[0]
                labels.append(f"{subset}\nn={int(row.n_rows_used)}, fields={int(row.n_fields_used)}")
                if np.isfinite(row.rho):
                    ax.plot(row.rho, pos, "o", color=color)
                    if np.isfinite(row.ci_low):
                        ax.plot([row.ci_low, row.ci_high], [pos, pos], color=color, lw=2)
                else:
                    ax.text(0, pos, "unavailable", ha="center", fontsize=9)
            ax.set_yticks(range(len(labels)), labels)
            ax.set(xlim=(-1.05, 1.05), ylim=(-0.6, len(labels) - 0.4),
                   xlabel="Spearman rho and pointwise 95% field-bootstrap CI",
                   title=f"{LABELS[y]} / {mode.replace('_', ' ')}")
            ax.axvline(0, color="0.5", ls="--", lw=1)
            ax.invert_yaxis()
    fig.suptitle("In-situ date-match sensitivity (different subsets, not paired improvement tests)")
    save(fig, "04_in_situ_quality_sensitivity.png")


def analyze(cohort_path, output, n_boot=2000, seed=20260908):
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap draws (2000+ recommended)")
    frame = normalize_keys(pd.read_csv(cohort_path, dtype={"field_id": "string"}), "cohort")
    require_columns(frame, [*FEATURES, *OBS, *PHYS, VWC, "in_situ_gap_days_verified"], "cohort")
    numeric(frame, FEATURES + OBS + PHYS + [VWC, "in_situ_gap_days_verified"])
    if not np.isfinite(frame[FEATURES + OBS + PHYS + [VWC]].to_numpy(dtype=float)).all():
        raise ValueError("Analysis requires the finite common cohort, not cohort_audit.csv")
    frame = residuals(frame)
    if frame.empty:
        raise ValueError("Empty cohort")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output is nonempty; choose a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    gap = frame.in_situ_gap_days_verified
    subsets = {"common": frame, "same_day": frame[gap.eq(0)],
               "gap_le_2d": frame[gap.le(2)], "gap_le_4d": frame[gap.le(4)]}
    statistics = []
    for subset, part in subsets.items():
        variables = [*FEATURES, VWC] if subset == "common" else [VWC]
        if subset == "common" and MAP in part:
            variables.append(MAP)
        for x in variables:
            for y in ["common_residual_db", "differential_residual_db"]:
                modes = ["pooled", "within_field", "between_field"] if subset == "common" else ["pooled", "within_field"]
                for mode in modes:
                    item = association(part, x, y, mode, n_boot, seed + len(statistics))
                    statistics.append({"subset": subset, **item})
        print(f"Completed associations: {subset}, rows={len(part)}", flush=True)
    statistics = pd.DataFrame(statistics)
    statistics.to_csv(output / "association_statistics.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(output / "diagnostic_rows.csv", index=False, encoding="utf-8-sig")
    spectrum_check = None
    gauss = ["gaussian_spm_hh_raw_db", "gaussian_spm_vv_raw_db"]
    if all(c in frame for c in gauss):
        difference = (frame[gauss[1]] - frame[gauss[0]]) - frame.spm_polarization_difference_db
        spectrum_check = float(difference.abs().max())
    focus = statistics[(statistics.variable == VWC) & (statistics.subset == "common")]
    summary = {"cohort": counts(frame), "subsets": {k: counts(v) for k, v in subsets.items()},
               "input": fingerprint(cohort_path), "code": fingerprint(__file__),
               "bootstrap_draws": n_boot, "seed": seed, "python": platform.python_version(),
               "numpy": np.__version__, "pandas": pd.__version__,
               "spectrum_contrast_max_difference_db": spectrum_check,
               "mean_common_residual_db": float(frame.common_residual_db.mean()),
               "mean_differential_residual_db": float(frame.differential_residual_db.mean()),
               "vwc_associations": json.loads(focus.to_json(orient="records")),
               "interpretation": "Exploratory associations, not causal mechanisms or held-out predictive gains."}
    write_json(output / "analysis_summary.json", summary)
    draw_figures(frame, statistics, output)
    lines = ["# 地表散射极化失配诊断", "",
        f"统一样本：{len(frame)} 个地块—日期，{frame.field_id.nunique()} 个地块，{frame.acquisition_date.nunique()} 个日期。",
        "", "## 定义与数据来源", "",
        "采用未经观测偏置校正的指数谱 SPM。残差为实测减模拟，单位 dB。",
        "共同残差 rc=(rHH+rVV)/2；差异残差 rd=rVV-rHH。它们不是土壤/植被真实能量分解。",
        "实地 VWC 是同地块最近日期匹配；卫星图 VWC 保留原审计的产品来源标签，二者不混合填补。",
        "", "## 实地 VWC 关联结果", "",
        "| 残差 | 分析方式 | rho | 95% 区间 | 地块数 |", "|---|---|---:|---|---:|"]
    for _, row in focus.iterrows():
        interval = f"[{row.ci_low:.3f}, {row.ci_high:.3f}]" if np.isfinite(row.ci_low) else "不可估计"
        lines.append(f"| {row.residual} | {row['mode']} | {row.rho:.3f} | {interval} | {int(row.n_fields_used)} |")
    lines.extend(["", "## 四张图如何阅读", "",
        "1. 01：实测与 SPM 的极化差和共同强度；虚线为一致线，不是回归线。",
        "2. 02：共同残差与环境变量的总体关联；不同地块混合可能产生混杂。",
        "3. 03：极化差残差的总体与地块内部关联；后者在配对有限值上各自减地块均值。",
        "4. 04：实地匹配时间差敏感性；各子集样本与地块构成不同，区间变化不是配对改善检验。",
        "", "## 统计约定与论文边界", "",
        "- pooled：每个地块—日期一行。between_field：每个地块的配对均值一行。",
        "- within_field：仅保留至少两条配对记录且自变量有变化的地块；静态粗糙度不计算地块内部相关。",
        "- Bootstrap 整块重采样地块，保留块内记录；区间为点态区间，未做多重比较校正。",
        "- 少于四个地块或八个分析点不估计区间；少于十个地块标记为少地块探索性结果。",
        "- lofo_rho_min/max 为逐一去掉地块的描述性敏感性，不是留出预测验证。",
        "- 地块内部中心化不等于控制日期、土壤水分或作物变化，也不能证明植被因果效应。",
        "- 同一植被实测值可能匹配多个雷达日期；地块重采样保留这种依赖，但不解决共同日期效应。",
        "- 使用原模型有效域子集，不能将结论推广到全部粗糙度或所有植被条件。",
        "- 该步骤不训练新模型，不报告预测提升，不证明新的散射机制。",
        "", "## 下一步", "",
        "先检查时间匹配敏感性与地块影响，再在相同地块划分下比较有无植被输入的普通基线。",
        "只有诊断和控制变量实验共同支持时，才开发极化结构化修正，预训练与约束作为方法组成。"])
    (output / "README_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved analysis to: {output}", flush=True)
    return summary
