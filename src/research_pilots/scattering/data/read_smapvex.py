"""Readers for the three SMAPVEX12 products used by the ground-scattering pilot."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd


PALS_COLUMNS = [
    "utc_seconds",
    "latitude",
    "longitude",
    "utm_x",
    "utm_y",
    "sigma0_vv_db",
    "sigma0_hh_db",
    "sigma0_hv_db",
    "sigma0_vh_db",
    "heading_uncertainty_flag",
]

SIGMA0_COLUMNS = [
    "sigma0_vv_db",
    "sigma0_hh_db",
    "sigma0_hv_db",
    "sigma0_vh_db",
]

PALS_FILENAME_RE = re.compile(
    r"^SV12PLBK_PALS_S0_(?P<date>\d{8})_"
    r"(?P<altitude>HiAlt|LoAlt)_v(?P<version>\d+)\.txt$"
)

NA_VALUES = ["", " ", "-9999", -9999, "NULL", "null", "NA", "N/A"]


def _require_file(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"File not found: {result}")
    return result


def _require_directory(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_dir():
        raise NotADirectoryError(f"Directory not found: {result}")
    return result


def parse_pals_filename(path: str | Path) -> dict[str, object]:
    """Parse acquisition metadata encoded in a PALS filename."""
    path = Path(path)
    match = PALS_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected PALS filename: {path.name}")

    altitude_mode = match.group("altitude")
    return {
        "source_file": path.name,
        "acquisition_date": pd.to_datetime(
            match.group("date"), format="%Y%m%d"
        ),
        "altitude_mode": altitude_mode,
        "footprint_m": 1500 if altitude_mode == "HiAlt" else 500,
        "product_version": match.group("version"),
        "frequency_ghz": 1.26,
        "nominal_incidence_angle_deg": 40.0,
    }


def list_pals_files(
    directory: str | Path,
    altitude_mode: str | None = None,
) -> list[Path]:
    """Return validated PALS files, optionally restricted to HiAlt or LoAlt."""
    directory = _require_directory(directory)
    if altitude_mode not in {None, "HiAlt", "LoAlt"}:
        raise ValueError("altitude_mode must be None, 'HiAlt', or 'LoAlt'")

    files = sorted(directory.glob("SV12PLBK_PALS_S0_*.txt"))
    selected: list[Path] = []
    for path in files:
        metadata = parse_pals_filename(path)
        if altitude_mode is None or metadata["altitude_mode"] == altitude_mode:
            selected.append(path)

    if not selected:
        raise FileNotFoundError(f"No matching PALS files found in: {directory}")
    return selected


def _prepare_pals_frame(
    frame: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    if frame.shape[1] != len(PALS_COLUMNS):
        raise ValueError(
            f"Expected {len(PALS_COLUMNS)} PALS columns, got {frame.shape[1]}"
        )

    frame = frame.copy()
    frame.columns = PALS_COLUMNS

    flag = frame["heading_uncertainty_flag"]
    non_integer = flag.notna() & ((flag - flag.round()).abs() > 1e-9)
    if non_integer.any():
        raise ValueError("heading_uncertainty_flag contains non-integer values")
    frame["heading_uncertainty_flag"] = flag.round().astype("Int8")

    acquisition_date = pd.Timestamp(metadata["acquisition_date"])
    frame["utc_datetime"] = acquisition_date + pd.to_timedelta(
        frame["utc_seconds"], unit="s"
    )
    frame["acquisition_date"] = acquisition_date
    frame["altitude_mode"] = metadata["altitude_mode"]
    frame["footprint_m"] = metadata["footprint_m"]
    frame["frequency_ghz"] = metadata["frequency_ghz"]
    frame["nominal_incidence_angle_deg"] = metadata[
        "nominal_incidence_angle_deg"
    ]
    frame["source_file"] = metadata["source_file"]
    return frame


def read_pals_file(path: str | Path) -> pd.DataFrame:
    """Read one complete headerless PALS backscatter text file."""
    path = _require_file(path)
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype="float64")
    return _prepare_pals_frame(raw, parse_pals_filename(path))


def iter_pals_chunks(
    path: str | Path,
    chunksize: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """Read a PALS file in bounded-memory chunks."""
    path = _require_file(path)
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")

    metadata = parse_pals_filename(path)
    reader = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        dtype="float64",
        chunksize=chunksize,
    )
    for raw in reader:
        yield _prepare_pals_frame(raw, metadata)


def _read_coordinates(path: str | Path) -> pd.DataFrame:
    path = _require_file(path)
    coordinates = pd.read_csv(
        path,
        na_values=NA_VALUES,
        keep_default_na=True,
        skipinitialspace=True,
    ).rename(
        columns={
            "OBJECTID": "coordinate_object_id",
            "Site_ID": "site_id",
            "X": "utm_x",
            "Y": "utm_y",
        }
    )
    expected = {"site_id", "utm_x", "utm_y"}
    missing = expected.difference(coordinates.columns)
    if missing:
        raise ValueError(f"Coordinate table lacks columns: {sorted(missing)}")

    coordinates["site_id"] = coordinates["site_id"].astype("string").str.strip()
    if coordinates["site_id"].duplicated().any():
        duplicates = coordinates.loc[
            coordinates["site_id"].duplicated(), "site_id"
        ].tolist()
        raise ValueError(f"Duplicate coordinate site_id values: {duplicates[:5]}")
    return coordinates


def read_soil_moisture(
    data_path: str | Path,
    coordinate_path: str | Path,
) -> pd.DataFrame:
    """Read soil moisture and attach UTM coordinates by Site_ID."""
    data_path = _require_file(data_path)
    frame = pd.read_csv(
        data_path,
        na_values=NA_VALUES,
        keep_default_na=True,
        skipinitialspace=True,
    ).rename(
        columns={
            "OBJECTID": "object_id",
            "Sample_Date": "sample_date",
            "Sample_Time": "sample_time",
            "Site_ID": "site_id",
            "Soil_Moisture_Cal": "soil_moisture_m3_m3",
            "Soil_Real_Dielectric": "soil_real_dielectric",
            "Source": "source",
            "Calibration": "calibration",
            "Comments": "comments",
        }
    )
    expected = {
        "sample_date",
        "sample_time",
        "site_id",
        "soil_moisture_m3_m3",
        "soil_real_dielectric",
    }
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Soil-moisture table lacks columns: {sorted(missing)}")

    frame["site_id"] = frame["site_id"].astype("string").str.strip()
    frame["sample_date"] = pd.to_datetime(
        frame["sample_date"], format="%m/%d/%Y", errors="coerce"
    )
    frame["sample_time"] = frame["sample_time"].astype("string").str.strip()
    time_delta = pd.to_timedelta(frame["sample_time"], errors="coerce")
    # Forty-eight records encode Excel time as a fraction of one day rather
    # than as HH:MM:SS. Preserve both encodings instead of losing their time.
    excel_fraction = pd.to_numeric(frame["sample_time"], errors="coerce")
    fraction_mask = time_delta.isna() & excel_fraction.notna()
    fraction_seconds = (excel_fraction.loc[fraction_mask] * 86_400.0).round()
    time_delta.loc[fraction_mask] = pd.to_timedelta(
        fraction_seconds,
        unit="s",
    )
    frame["sample_datetime"] = frame["sample_date"] + time_delta

    coordinates = _read_coordinates(coordinate_path)[["site_id", "utm_x", "utm_y"]]
    return frame.merge(coordinates, on="site_id", how="left", validate="many_to_one")


def read_surface_roughness(
    data_path: str | Path,
    coordinate_path: str | Path,
) -> pd.DataFrame:
    """Read surface roughness and attach UTM coordinates by Site_ID."""
    data_path = _require_file(data_path)
    frame = pd.read_csv(
        data_path,
        na_values=NA_VALUES,
        keep_default_na=True,
        skipinitialspace=True,
    ).rename(
        columns={
            "OBJECTID": "object_id",
            "Site_ID": "site_id",
            "UAV_Height": "uav_rms_height_cm",
            "UAV_Cor_L": "uav_correlation_length_cm",
            "PALS_Height": "pals_rms_height_cm",
            "PALS_Cor_L": "pals_correlation_length_cm",
            "R2_Height": "r2_rms_height_cm",
            "R2_Cor_L": "r2_correlation_length_cm",
        }
    )
    expected = {
        "site_id",
        "pals_rms_height_cm",
        "pals_correlation_length_cm",
    }
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Roughness table lacks columns: {sorted(missing)}")

    frame["site_id"] = frame["site_id"].astype("string").str.strip()
    coordinates = _read_coordinates(coordinate_path)[["site_id", "utm_x", "utm_y"]]
    return frame.merge(coordinates, on="site_id", how="left", validate="many_to_one")


def discover_smapvex_files(root: str | Path) -> dict[str, object]:
    """Resolve the expected files under the repository's SMAPVEX12 directory."""
    root = _require_directory(root)
    paths: dict[str, object] = {
        "root": root,
        "pals_files": list_pals_files(root / "pals_backscatter"),
        "soil_moisture_data": root
        / "soil_moisture"
        / "SV12PSMA_Soil_Moisture_Handheld_ver4.txt",
        "soil_moisture_coordinates": root
        / "soil_moisture"
        / "SV12PSMA_Field_Sites_ver4_coords",
        "surface_roughness_data": root
        / "surface_roughness"
        / "SV12SRA_Soil_Roughness_ver4.txt",
        "surface_roughness_coordinates": root
        / "surface_roughness"
        / "SV12SRA_Field_Sites_ver4_coords",
    }
    for key, value in paths.items():
        if key in {"root", "pals_files"}:
            continue
        _require_file(value)
    return paths
