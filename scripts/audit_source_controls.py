"""Independently audit ancillary-control predictions, folds, tuning and metrics.

This entry point performs no training and does not import the training script.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["field_id", "acquisition_date"]
PRED = ["predicted_hh", "predicted_vv"]
OBS = ["observed_hh", "observed_vv"]


def read(path):
    return pd.read_csv(path, dtype={"field_id": str})


def audit_run(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "complete", f"Incomplete run: {folder}"
    data = read(folder / "analysis_input.csv")
    original_path = (folder.parent / "matching/matched_source.csv" if manifest["classical_only"]
                     else ROOT / "reproducibility/data/source/smapvex12_portable_source.csv")
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == manifest["inputs"][0]["sha256"]
    original = read(original_path).sort_values(KEY).reset_index(drop=True)
    original["soil_real_dielectric"] = np.polynomial.polynomial.polyval(
        original.soil_moisture_m3_m3, [3.03, 9.3, 146.0, -76.7])
    pd.testing.assert_frame_equal(original, data, check_exact=False, atol=1e-12, rtol=0)
    outer = read(folder / "outer_fold_assignments.csv")
    inner = read(folder / "inner_fold_assignments.csv")
    pred = read(folder / "oof_predictions.csv")
    ipred = read(folder / "inner_oof_predictions.csv")
    tuning = read(folder / "inner_tuning.csv")
    assert not data.duplicated(KEY).any()
    assert sorted(outer.row_id) == list(data.index)
    assert outer.groupby("field_id").fold.nunique().eq(1).all()
    assert set(outer.fold) == set(range(1, 6))
    for fold, block in inner.groupby("fold"):
        test_fields = set(outer.loc[outer.fold.eq(fold), "field_id"])
        assert not test_fields.intersection(block.field_id)
        assert set(block.row_id) == set(outer.loc[outer.fold.ne(fold), "row_id"])
        assert not block.row_id.duplicated().any()
        assert block.groupby("field_id").inner_fold.nunique().eq(1).all()
        assert set(block.inner_fold) == set(range(1, 5))
    for table in [outer, inner, pred, ipred]:
        expected = data.loc[table.row_id, KEY].reset_index(drop=True)
        pd.testing.assert_frame_equal(table[KEY].reset_index(drop=True), expected)
    for table in [pred, ipred]:
        assert np.isfinite(table[OBS + PRED].to_numpy(float)).all()
        expected = data.loc[table.row_id, ["sigma0_hh_db", "sigma0_vv_db"]].to_numpy(float)
        np.testing.assert_allclose(table[OBS], expected, atol=1e-12, rtol=0)
    check = pred.merge(outer[["row_id", "fold"]], on="row_id", validate="many_to_one", suffixes=("", "_expected"))
    assert check.fold.eq(check.fold_expected).all()
    check = ipred.merge(inner[["row_id", "fold", "inner_fold"]], on=["row_id", "fold"], validate="many_to_one", suffixes=("", "_expected"))
    assert check.inner_fold.eq(check.inner_fold_expected).all()
    methods = {"training_mean", "ridge"}
    if manifest["with_vwc"]:
        methods.add("ridge_vwc")
    if not manifest["classical_only"]:
        methods.update(f"{coord}_{init}" for coord in ["direct", "components", "fixed_common"]
                       for init in ["scratch", "spm", "multifidelity"])
    assert set(pred.method) == methods
    assert set(pred.repeat) == set(range(1, manifest["repeats"] + 1))
    metrics = []
    for (repeat, method), block in pred.groupby(["repeat", "method"]):
        assert sorted(block.row_id) == list(data.index)
        y, p = block[OBS].to_numpy(float), block[PRED].to_numpy(float)
        metrics.append({"repeat": repeat, "method": method,
                        "mean_channel_rmse_db": np.sqrt(np.square(p-y).mean(axis=0)).mean()})
    assert len(metrics) == manifest["repeats"] * len(methods)
    metrics = pd.DataFrame(metrics).sort_values(["repeat", "method"]).reset_index(drop=True)
    saved = read(folder / "metrics_by_repeat.csv").sort_values(["repeat", "method"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(metrics, saved, check_exact=False, atol=1e-12, rtol=0)
    summary = metrics.groupby("method").mean_channel_rmse_db.agg(["mean", "std"])
    saved_summary = pd.read_csv(folder / "summary.csv", index_col="method").sort_index()
    np.testing.assert_allclose(summary, saved_summary, atol=1e-12, rtol=0)

    centered = []
    for (repeat, fold, method), block in pred.groupby(["repeat", "fold", "method"]):
        y, p = block[OBS].to_numpy(float), block[PRED].to_numpy(float)
        cy = np.column_stack((y, y.mean(axis=1), y[:, 1]-y[:, 0]))
        cp = np.column_stack((p, p.mean(axis=1), p[:, 1]-p[:, 0]))
        for j, name in enumerate(["HH", "VV", "common", "differential"]):
            error = (cp[:, j]-cp[:, j].mean())-(cy[:, j]-cy[:, j].mean())
            centered.append({"repeat": repeat, "fold": fold, "method": method, "response": name,
                "centered_skill": 1-np.square(error).mean()/np.var(cy[:, j]),
                "centered_rmse_db": np.sqrt(np.square(error).mean()),
                "variance_ratio": np.var(cp[:, j])/np.var(cy[:, j])})
        train = data.loc[outer.loc[outer.fold.ne(fold), "row_id"], ["sigma0_hh_db", "sigma0_vv_db"]].to_numpy(float)
        if method == "training_mean":
            np.testing.assert_allclose(p, np.tile(train.mean(axis=0), (len(p), 1)), atol=1e-12, rtol=0)
        if method.startswith("fixed_common_"):
            other = pred.loc[pred.repeat.eq(repeat) & pred.fold.eq(fold) & pred.method.eq(method.replace("fixed_common", "components"))].set_index("row_id").loc[block.row_id, PRED].to_numpy(float)
            np.testing.assert_allclose(p[:, 1]-p[:, 0], other[:, 1]-other[:, 0], atol=1e-12, rtol=0)
            np.testing.assert_allclose(p.mean(axis=1), train.mean(), atol=1e-12, rtol=0)
    centered = pd.DataFrame(centered).groupby(["method", "response"])[["centered_skill", "centered_rmse_db", "variance_ratio"]].mean()
    saved_centered = pd.read_csv(folder / "centered_summary.csv", index_col=["method", "response"]).sort_index()
    np.testing.assert_allclose(centered, saved_centered, atol=1e-12, rtol=0)

    scores = []
    for (repeat, fold, method), block in ipred.groupby(["repeat", "fold", "method"]):
        candidate_col = "candidate_alpha" if method.startswith("ridge") else "candidate_epochs"
        expected_candidates = manifest["ridge_alphas"] if method.startswith("ridge") else manifest["epochs"]
        assert set(block[candidate_col]) == set(expected_candidates)
        candidates = []
        for candidate, part in block.groupby(candidate_col):
            assert sorted(part.row_id) == sorted(inner.loc[inner.fold.eq(fold), "row_id"])
            error = part[PRED].to_numpy(float)-part[OBS].to_numpy(float)
            score = np.sqrt(np.square(error).mean(axis=0)).mean()
            candidates.append((candidate, score))
            scores.append({"repeat": repeat, "fold": fold, "method": method,
                           "candidate": candidate, "rmse_db": score})
        selected = min(candidates, key=lambda v: v[1])[0]
        row = tuning.loc[tuning.repeat.eq(repeat) & tuning.fold.eq(fold) & tuning.method.eq(method)]
        selection_col = "selected_alpha" if method.startswith("ridge") else "selected_epochs"
        assert row[selection_col].dropna().tolist() == [selected]
    assert set(ipred.method) == methods - {"training_mean"} - {m for m in methods if m.startswith("fixed_common_")}
    assert ipred.groupby(["repeat", "fold", "method"]).ngroups == manifest["repeats"] * 5 * ipred.method.nunique()
    return {"rows": len(data), "fields": int(data.field_id.nunique()), "repeats": manifest["repeats"],
            "methods": sorted(methods), "outer_predictions": len(pred), "inner_predictions": len(ipred),
            "summary": summary.reset_index().replace({np.nan: None}).to_dict("records")}, pd.DataFrame(scores)


def audit_matching(folder, gap):
    matched = read(folder / "matched_source.csv")
    vegetation = read(folder / "vegetation_field_day.csv")
    source = read(ROOT / "reproducibility/data/source/smapvex12_portable_source.csv")
    vwc = "vegetation_water_content_in_situ_kg_m2"
    valid = vegetation.loc[vegetation[vwc].ge(0) & np.isfinite(vegetation[vwc])]
    candidates = source[KEY].merge(valid[["field_id", "sample_date", vwc]], on="field_id")
    candidates["gap"] = (pd.to_datetime(candidates.sample_date)-pd.to_datetime(candidates.acquisition_date)).dt.days.abs()
    expected = candidates.sort_values(KEY+["gap", "sample_date"]).drop_duplicates(KEY)
    expected = expected.loc[expected.gap.le(gap)].sort_values(KEY).reset_index(drop=True)
    matched = matched.sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected[KEY], matched[KEY])
    assert expected.sample_date.tolist() == matched.vegetation_sample_date.tolist()
    np.testing.assert_allclose(expected[vwc], matched[vwc], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(expected.gap, matched.vegetation_gap_days)
    assert len(matched) == {2: 132, 8: 240}[gap]
    observed = source.set_index(KEY).loc[pd.MultiIndex.from_frame(matched[KEY])].reset_index()
    for column in ["sigma0_hh_db", "sigma0_vv_db", "soil_moisture_m3_m3", "soil_real_dielectric",
                   "pals_rms_height_cm", "pals_correlation_length_cm"]:
        np.testing.assert_allclose(matched[column], observed[column], atol=1e-12, rtol=0)
    return {"rows": len(matched), "max_gap_days": gap, "nearest_date_and_tie_rule": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "reproducibility/results/source_controls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory.")
    report, scores = {}, []
    for name in ["source_structure", "vegetation_gap2/fit", "vegetation_gap8/fit"]:
        report[name], table = audit_run(args.results / name)
        table.insert(0, "control", name)
        scores.append(table)
    for gap in [2, 8]:
        report[f"matching_gap{gap}"] = audit_matching(args.results / f"vegetation_gap{gap}/matching", gap)
    args.output.mkdir(parents=True)
    pd.concat(scores, ignore_index=True).to_csv(args.output / "inner_scores_independent.csv", index=False)
    (args.output / "audit_summary.json").write_text(json.dumps({"status": "passed", "checks": report}, indent=2)+"\n")
    print(json.dumps({"status": "passed", "checks": report}, indent=2))


if __name__ == "__main__":
    main()
