"""Readers for the SMAPVEX08 external-domain ground-scattering products.

The module deliberately keeps the 2008 schema separate from the SMAPVEX12
reader.  The two campaigns use different file formats and must only be joined
after both have been converted to the same portable field-day contract.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


PALS_COLUMNS = [
    "radar_time_seconds",
    "dads_time_milliseconds",
    "latitude",
    "longitude",
    "look_azimuth_deg",
    "antenna_roll_deg",
    "range_m",
    "incidence_angle_deg",
    "sigma0_vv_db",
    "sigma0_hh_db",
    "sigma0_hv_db",
    "sigma0_vh_db",
    "corr_vvhh_real",
    "corr_vvhh_imag",
    "corr_vvhv_real",
    "corr_vvhv_imag",
    "corr_vvvh_real",
    "corr_vvvh_imag",
    "corr_hhhv_real",
    "corr_hhhv_imag",
    "corr_hhvh_real",
    "corr_hhvh_imag",
    "corr_hvvh_real",
    "corr_hvvh_imag",
]

SIGMA0_COLUMNS = [
    "sigma0_vv_db",
    "sigma0_hh_db",
    "sigma0_hv_db",
    "sigma0_vh_db",
]

PALS_FILENAME_RE = re.compile(
    r"^SV08PLBK_(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})\.red$"
)


def require_file(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"File not found: {result}")
    return result


def parse_pals_filename(path: str | Path) -> dict[str, object]:
    path = Path(path)
    match = PALS_FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected SMAPVEX08 PALS filename: {path.name}")
    start = pd.Timestamp(
        year=2008,
        month=int(match.group("month")),
        day=int(match.group("day")),
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
    )
    return {
        "source_file": path.name,
        "flight_start": start,
        "acquisition_date": start.normalize(),
        "frequency_ghz": 1.26,
    }


def list_pals_files(root: str | Path) -> list[Path]:
    files = sorted(Path(root).rglob("SV08PLBK_*.red"))
    if not files:
        raise FileNotFoundError(f"No SMAPVEX08 .red files below: {root}")
    for path in files:
        parse_pals_filename(path)
    return files


def iter_pals_chunks(
    path: str | Path,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """Read one headerless 24-column PALS ASCII file in bounded memory."""
    path = require_file(path)
    metadata = parse_pals_filename(path)
    reader = pd.read_csv(path, sep=r"\s+", header=None, chunksize=chunksize)
    for raw in reader:
        if raw.shape[1] != len(PALS_COLUMNS):
            raise ValueError(
                f"Expected 24 PALS columns in {path.name}, got {raw.shape[1]}"
            )
        raw.columns = PALS_COLUMNS
        raw["source_file"] = metadata["source_file"]
        raw["flight_start"] = metadata["flight_start"]
        raw["acquisition_date"] = metadata["acquisition_date"]
        raw["frequency_ghz"] = metadata["frequency_ghz"]
        yield raw


def discover_files(root: str | Path) -> dict[str, object]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    def one(pattern: str) -> Path:
        matches = sorted(root.rglob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {pattern!r} below {root}, found {len(matches)}"
            )
        return matches[0]

    return {
        "root": root,
        "pals_files": list_pals_files(root),
        "soil_moisture": one("SV08SM_SMAPVEX08_GVSM_Final.xls"),
        "soil_coordinates": one("SV08SM_AllTeamsGPS.txt"),
        "surface_roughness": one("SV08SR_SMAPVEX08_SR.txt"),
        "vegetation_in_situ": one("SV08V_Sum_VEG_SMAPVEX.xls"),
        "vwc_raster": one("SV08VWC_vwc.fst"),
        "vwc_header": one("SV08VWC_vwc.hdr"),
        "matchup": one("NSIDC0666_*_SMAPVEX08.txt"),
    }


def read_soil_coordinates(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path), sep="\t")
    frame = frame.rename(
        columns={"Point": "site_id", "Lat": "latitude", "Lon": "longitude"}
    )
    frame["site_id"] = frame["site_id"].astype("string").str.strip()
    frame["field_id"] = frame["site_id"].str.extract(r"^([A-Z]\d{2})", expand=False)
    if frame["site_id"].duplicated().any():
        raise ValueError("Soil-coordinate table contains duplicate site_id values")
    return frame


def read_soil_moisture(
    data_path: str | Path,
    coordinate_path: str | Path,
) -> pd.DataFrame:
    """Read calibrated probe moisture and attach exact sampling coordinates.

    The primary value is the mean of A/B/C site-specific calibration columns.
    GVSM is retained as an independent gravimetric diagnostic and is not used
    to fill missing probe values silently.
    """
    frame = pd.read_excel(require_file(data_path), sheet_name="Sheet1")
    required = {
        "Date",
        "Site_ID",
        "Sample",
        "A_VSM_SCC",
        "B_VSM_SCC",
        "C_VSM_SCC",
        "GVSM",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SMAPVEX08 soil table lacks columns: {missing}")

    frame["acquisition_date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    frame["field_id"] = frame["Site_ID"].astype("string").str.strip()
    sample = pd.to_numeric(frame["Sample"], errors="coerce").astype("Int64")
    frame["site_id"] = frame["field_id"] + "-" + sample.astype("string")
    scc = frame[["A_VSM_SCC", "B_VSM_SCC", "C_VSM_SCC"]].apply(
        pd.to_numeric, errors="coerce"
    )
    frame["soil_moisture_m3_m3"] = scc.mean(axis=1, skipna=True)
    frame["probe_position_count"] = scc.notna().sum(axis=1)
    frame["gravimetric_vsm_m3_m3"] = pd.to_numeric(frame["GVSM"], errors="coerce")

    coordinates = read_soil_coordinates(coordinate_path)[
        ["site_id", "latitude", "longitude"]
    ]
    result = frame.merge(coordinates, on="site_id", how="left", validate="many_to_one")
    # The distributed GPS table omits F02.  Preserve those observations with an
    # explicit flag; never invent coordinates.  They cannot enter spatial
    # collocation and are naturally excluded by the portable model contract.
    result["coordinate_available"] = result[["latitude", "longitude"]].notna().all(axis=1)
    return result


def read_surface_roughness(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(require_file(path), sep="\t")
    required = {"Name", "np", "sigma", "length", "asigma", "exponent"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SMAPVEX08 roughness table lacks columns: {missing}")
    frame["field_id"] = frame["Name"].astype("string").str.extract(
        r"^([A-Z]\d{2})", expand=False
    )
    # Official SV08SR units are millimetres; the portable contract uses cm.
    frame["pals_rms_height_cm"] = pd.to_numeric(frame["sigma"], errors="coerce") / 10.0
    frame["pals_correlation_length_cm"] = (
        pd.to_numeric(frame["length"], errors="coerce") / 10.0
    )
    return frame


def read_vegetation(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(require_file(path), sheet_name="Sheet1", header=5)
    frame = frame.rename(
        columns={
            "Field": "field_id",
            "Crop": "crop",
            "Part": "plant_part",
            "Date": "acquisition_date",
            "(kg/sq m)": "vwc_kg_m2",
            "LAI": "lai",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Notes": "notes",
        }
    )
    frame["field_id"] = frame["field_id"].astype("string").str.strip()
    frame["acquisition_date"] = pd.to_datetime(frame["acquisition_date"]).dt.normalize()
    frame["vwc_kg_m2"] = pd.to_numeric(frame["vwc_kg_m2"], errors="coerce")
    frame["lai"] = pd.to_numeric(frame["lai"], errors="coerce")
    # For multi-part corn records, use the supplied Total rather than summing it
    # together with its components.
    total_fields = set(
        frame.loc[frame["plant_part"].astype("string").str.lower().eq("total"), "field_id"]
    )
    keep = ~frame["field_id"].isin(total_fields) | frame["plant_part"].astype(
        "string"
    ).str.lower().eq("total")
    return frame.loc[keep].copy()


def read_matchup(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path), sep=r"\s+", na_values=["NaN"])


def linear_power_mean_db(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.mean(np.power(10.0, values / 10.0))))
