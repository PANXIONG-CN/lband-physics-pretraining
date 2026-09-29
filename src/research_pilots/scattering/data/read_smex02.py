"""Readers for the prospective SMEX02 external-domain campaign.

SMEX02 remains isolated from the SMAPVEX12/08 schemas until all products are
converted to the shared campaign-field-date contract.  This module performs
format parsing and input-domain checks only; it never fits or selects a model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


PALS_FILENAME_RE = re.compile(
    r"^(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})\.red$",
    re.IGNORECASE,
)

PALS_REQUIRED_COLUMNS = {
    "time",
    "GPS_time",
    "lat",
    "long",
    "beam_angle",
    "L_HH",
    "L_VV",
    "L_VH",
    "L_HV",
    "S_HH",
    "S_VV",
    "S_VH",
    "S_HV",
}

PALS_RENAME = {
    "time": "radar_time_seconds",
    "GPS_time": "gps_time_seconds",
    "lat": "latitude",
    "long": "longitude",
    "ant_azimuth": "look_azimuth_deg",
    "polar_angle": "antenna_roll_deg",
    "range": "range_m",
    "beam_angle": "incidence_angle_deg",
    "L_HH": "sigma0_hh_db",
    "L_VV": "sigma0_vv_db",
    "L_VH": "sigma0_vh_db",
    "L_HV": "sigma0_hv_db",
    "S_HH": "s_band_sigma0_hh_db",
    "S_VV": "s_band_sigma0_vv_db",
    "S_VH": "s_band_sigma0_vh_db",
    "S_HV": "s_band_sigma0_hv_db",
}

L_BAND_SIGMA0_COLUMNS = [
    "sigma0_hh_db",
    "sigma0_vv_db",
    "sigma0_vh_db",
    "sigma0_hv_db",
]


def require_file(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"File not found: {result}")
    return result


def parse_pals_filename(path: str | Path) -> dict[str, object]:
    path = Path(path)
    match = PALS_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected SMEX02 PALS filename: {path.name}")
    start = pd.Timestamp(
        year=2002,
        month=int(match.group("month")),
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
    )
    return {
        "source_file": path.name,
        "flight_start": start,
        "acquisition_date": start.normalize(),
        "l_band_frequency_ghz": 1.26,
        "s_band_frequency_ghz": 3.15,
    }


def list_pals_files(root: str | Path) -> list[Path]:
    files = sorted(Path(root).rglob("*.red"))
    if not files:
        raise FileNotFoundError(f"No SMEX02 PALS .red files below: {root}")
    for path in files:
        parse_pals_filename(path)
    return files


def iter_pals_chunks(
    path: str | Path,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Read a headered SMEX02 PALS radar file in bounded memory."""
    path = require_file(path)
    metadata = parse_pals_filename(path)
    reader = pd.read_csv(path, sep="\t", chunksize=chunksize)
    for raw in reader:
        raw.columns = [str(column).strip() for column in raw.columns]
        missing = sorted(PALS_REQUIRED_COLUMNS.difference(raw.columns))
        if missing:
            raise ValueError(f"{path.name} lacks PALS columns: {missing}")
        frame = raw.rename(columns=PALS_RENAME).copy()
        frame["source_file"] = metadata["source_file"]
        frame["flight_start"] = metadata["flight_start"]
        frame["acquisition_date"] = metadata["acquisition_date"]
        frame["frequency_ghz"] = metadata["l_band_frequency_ghz"]
        yield frame


def read_soil_moisture_summary(path: str | Path) -> pd.DataFrame:
    """Read the field/date ThetaProbe summary.

    ``VSM_ssc`` is retained as the primary site-specific calibrated moisture.
    ``VSM_gc`` remains an auxiliary general-calibration measurement.  Pandas
    safely disambiguates the three original ``Stdev`` columns.
    """
    frame = pd.read_csv(require_file(path), sep="\t")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")].copy()
    required = {
        "Date",
        "SiteID",
        "Latitude",
        "Longitude",
        "#ofsamples",
        "VSM_gc",
        "VSM_ssc",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SMEX02 ThetaProbe summary lacks columns: {missing}")

    frame = frame.rename(
        columns={
            "Date": "acquisition_date",
            "SiteID": "field_id",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "#ofsamples": "soil_sample_count",
            "VSM_gc": "soil_moisture_general_cal_m3_m3",
            "VSM_ssc": "soil_moisture_m3_m3",
        }
    )
    frame["acquisition_date"] = pd.to_datetime(
        frame["acquisition_date"], format="%m/%d/%y", errors="raise"
    ).dt.normalize()
    frame["field_id"] = frame["field_id"].astype("string").str.strip().str.upper()
    numeric = [
        "latitude",
        "longitude",
        "soil_sample_count",
        "soil_moisture_general_cal_m3_m3",
        "soil_moisture_m3_m3",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["soil_moisture_in_physical_range"] = frame[
        "soil_moisture_m3_m3"
    ].between(0.0, 1.0)
    frame["coordinate_available"] = np.isfinite(
        frame[["latitude", "longitude"]].to_numpy(dtype=float)
    ).all(axis=1)
    return frame


ROUGHNESS_COLUMNS = [
    "profile_id",
    "point_count",
    "rms_height_cm",
    "correlation_length_cm",
    "adjusted_rms_height_cm",
    "surface_exponent",
    "scorelat_exponent",
]


def read_surface_roughness_file(
    path: str | Path,
    scan_mode: str | None = None,
) -> pd.DataFrame:
    """Read one NSIDC-0204 roughness table (official sigma and L units: cm)."""
    path = require_file(path)
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=3,
        header=None,
        names=ROUGHNESS_COLUMNS,
        engine="python",
    )
    frame["profile_id"] = frame["profile_id"].astype("string").str.strip()
    number = frame["profile_id"].str.extract(
        r"(?i)^wc(?P<number>\d{2})", expand=True
    )["number"]
    frame["field_id"] = "WC" + number.astype("string")
    frame["scan_mode"] = scan_mode or path.stem.replace("_scanning", "")
    for column in ROUGHNESS_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["roughness_positive"] = (
        frame["rms_height_cm"].gt(0)
        & frame["correlation_length_cm"].gt(0)
    )
    return frame


def read_surface_roughness(
    grid_path: str | Path,
    slope_path: str | Path,
) -> pd.DataFrame:
    grid = read_surface_roughness_file(grid_path, "grid")
    slope = read_surface_roughness_file(slope_path, "slope")
    return pd.concat([grid, slope], ignore_index=True)


def discover_files(root: str | Path) -> dict[str, object]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    def exactly_one(name: str) -> Path:
        matches = sorted(root.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {name!r} below {root}, found {len(matches)}"
            )
        return matches[0]

    return {
        "root": root,
        "pals_files": list_pals_files(root),
        "soil_moisture_summary": exactly_one("WC_TP_SUM.txt"),
        "roughness_grid": exactly_one("grid_scanning.txt"),
        "roughness_slope": exactly_one("slope_scanning.txt"),
    }


def linear_power_mean_db(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.mean(np.power(10.0, values / 10.0))))
