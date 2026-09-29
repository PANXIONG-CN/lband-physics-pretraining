"""Data contract and validity checks for rough-surface physics teachers."""

from __future__ import annotations

import numpy as np
import pandas as pd


SPEED_OF_LIGHT_M_S = 299_792_458.0
REQUEST_COLUMNS = [
    "request_id",
    "scenario",
    "correlation_model",
    "frequency_ghz",
    "incidence_angle_deg",
    "rms_height_m",
    "correlation_length_m",
    "dielectric_real",
    "dielectric_loss_positive",
]
RESULT_COLUMNS = ["request_id", "i2em_hh_db", "i2em_vv_db"]


def add_i2em_validity(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the published I2EM implementation-domain diagnostics.

    The IEEE-GRSS reference implementation states ks < 1 and s/L <= 0.25.
    These conditions are implementation guardrails, not a proof of accuracy.
    """
    result = frame.copy()
    required = set(REQUEST_COLUMNS)
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"Missing request columns: {missing}")
    numeric = [
        "frequency_ghz",
        "incidence_angle_deg",
        "rms_height_m",
        "correlation_length_m",
        "dielectric_real",
        "dielectric_loss_positive",
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    finite = np.isfinite(result[numeric].to_numpy(dtype=float)).all(axis=1)
    frequency_hz = result["frequency_ghz"].to_numpy(dtype=float) * 1e9
    wavenumber = 2.0 * np.pi * frequency_hz / SPEED_OF_LIGHT_M_S
    height = result["rms_height_m"].to_numpy(dtype=float)
    length = result["correlation_length_m"].to_numpy(dtype=float)
    ks = wavenumber * height
    ratio = np.divide(
        height,
        length,
        out=np.full_like(height, np.nan),
        where=length > 0,
    )
    physical = (
        (frequency_hz > 0)
        & (height >= 0)
        & (length > 0)
        & (result["dielectric_real"].to_numpy(dtype=float) > 1)
        & (result["dielectric_loss_positive"].to_numpy(dtype=float) >= 0)
        & result["incidence_angle_deg"].between(0, 90, inclusive="left").to_numpy()
    )
    correlation_ok = result["correlation_model"].isin(
        ["exponential", "gaussian"]
    ).to_numpy()
    result["i2em_k_rms_height"] = ks
    result["i2em_rms_to_correlation_ratio"] = ratio
    result["i2em_ks_within_reference_limit"] = ks < 1.0
    result["i2em_slope_within_reference_limit"] = ratio <= 0.25
    result["i2em_request_valid"] = (
        finite
        & physical
        & correlation_ok
        & result["i2em_ks_within_reference_limit"].to_numpy()
        & result["i2em_slope_within_reference_limit"].to_numpy()
    )
    reasons = []
    for index in range(len(result)):
        row_reasons = []
        if not finite[index] or not physical[index]:
            row_reasons.append("invalid_physical_input")
        if not correlation_ok[index]:
            row_reasons.append("unsupported_correlation_model")
        if not bool(result.iloc[index]["i2em_ks_within_reference_limit"]):
            row_reasons.append("ks_not_below_1")
        if not bool(result.iloc[index]["i2em_slope_within_reference_limit"]):
            row_reasons.append("rms_to_correlation_above_0.25")
        reasons.append("valid" if not row_reasons else ";".join(row_reasons))
    result["i2em_validity_reason"] = reasons
    return result


def validate_teacher_requests(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_i2em_validity(frame)
    if result["request_id"].astype(str).duplicated().any():
        raise ValueError("request_id must be unique")
    if result["request_id"].isna().any():
        raise ValueError("request_id cannot be missing")
    if not result["i2em_request_valid"].all():
        invalid = result.loc[
            ~result["i2em_request_valid"],
            ["request_id", "i2em_validity_reason"],
        ]
        raise ValueError(f"Invalid I2EM requests: {invalid.to_dict(orient='records')}")
    return result


def validate_teacher_results(
    requests: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    checked_requests = validate_teacher_requests(requests)
    missing = sorted(set(RESULT_COLUMNS).difference(results.columns))
    if missing:
        raise ValueError(f"Missing result columns: {missing}")
    if results["request_id"].astype(str).duplicated().any():
        raise ValueError("Result request_id must be unique")
    expected = set(checked_requests["request_id"].astype(str))
    received = set(results["request_id"].astype(str))
    if expected != received:
        raise ValueError(
            f"Request/result ID mismatch; missing={sorted(expected-received)}, "
            f"unexpected={sorted(received-expected)}"
        )
    result = results.copy()
    for column in ["i2em_hh_db", "i2em_vv_db"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    values = result[["i2em_hh_db", "i2em_vv_db"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("I2EM HH/VV results must be finite")
    if np.any((values < -150.0) | (values > 50.0)):
        raise ValueError("I2EM HH/VV results fall outside the configured sanity range")
    return checked_requests.merge(result[RESULT_COLUMNS], on="request_id", validate="one_to_one")
