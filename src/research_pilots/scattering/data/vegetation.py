"""Read and collocate SMAPVEX12 vegetation observations.

This module adds two complementary vegetation sources to an existing
field-day SMAPVEX12 table:

* SV12VA agricultural in-situ vegetation measurements; and
* SV12VWC daily vegetation-water-content GeoTIFF maps.

The functions deliberately keep measurement provenance and time offsets.
Those columns are required later when the physical regularisation weight is
made sample-dependent rather than treated as a universal constant.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

try:  # Rasterio reads only the requested field window from large GeoTIFFs.
    import rasterio
    from rasterio.windows import Window
except ImportError:  # Pillow remains a functional fallback for small files.
    rasterio = None
    Window = None


BIOMASS_FILENAME = "SV12VA_Crop_Biomass_ver4.txt"
HEIGHT_FILENAME = "SV12VA_Crop_Height_Diam_ver4.txt"
COORDINATE_FILENAMES = (
    "SV12VA_Field_Sites_ver4_coords",
    "SV12VA_Field_Sites_ver4_coords.txt",
)

DIRECT_VWC_DATES = {
    pd.Timestamp("2012-05-14"),
    pd.Timestamp("2012-05-20"),
    pd.Timestamp("2012-06-04"),
    pd.Timestamp("2012-06-23"),
    pd.Timestamp("2012-06-28"),
    pd.Timestamp("2012-07-05"),
    pd.Timestamp("2012-07-14"),
}
LAST_DIRECT_VWC_DATE = pd.Timestamp("2012-07-14")


def _normalise_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _resolve_columns(frame: pd.DataFrame, aliases: dict[str, Iterable[str]]) -> pd.DataFrame:
    normalised = {_normalise_name(column): column for column in frame.columns}
    rename: dict[object, str] = {}
    missing: list[str] = []
    for canonical, candidates in aliases.items():
        source = next(
            (normalised[_normalise_name(name)] for name in candidates if _normalise_name(name) in normalised),
            None,
        )
        if source is None:
            missing.append(canonical)
        else:
            rename[source] = canonical
    if missing:
        raise ValueError(
            "Missing expected columns: "
            + ", ".join(missing)
            + f". Available columns: {list(frame.columns)}"
        )
    return frame.rename(columns=rename)


def _read_delimited(path: Path) -> pd.DataFrame:
    """Read a small NSIDC ASCII table with delimiter and encoding fallbacks."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "latin-1"):
        for separator in (None, "\t", ","):
            try:
                frame = pd.read_csv(
                    path,
                    sep=separator,
                    engine="python",
                    encoding=encoding,
                    skipinitialspace=True,
                )
                frame.columns = [str(column).strip() for column in frame.columns]
                if frame.shape[1] > 1:
                    return frame
            except Exception as exc:  # pragma: no cover - retained for field files
                errors.append(f"{encoding}/{separator!r}: {exc}")
    raise ValueError(f"Could not parse {path}. Attempts: {' | '.join(errors)}")


def _find_required(root: Path, names: Iterable[str]) -> Path:
    wanted = {name.lower() for name in names}
    matches = [path for path in root.rglob("*") if path.is_file() and path.name.lower() in wanted]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one of {sorted(wanted)} below {root}, found {matches}"
        )
    return matches[0]


def field_id_from_site_id(values: pd.Series) -> pd.Series:
    """Convert a sampling site such as ``11-3`` to agricultural field ``11``."""
    result = values.astype("string").str.strip().str.split("-", n=1).str[0]
    return result.replace({"": pd.NA})


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.normalize()


def _safe_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def read_in_situ_vegetation(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return field-day vegetation features and the site-coordinate audit table.

    Plant-part measurements are first averaged within a part and site, then
    summed across parts.  This prevents technical replicates from being counted
    as additional biomass while retaining the total canopy water mass.
    """
    root_path = Path(root)
    biomass_path = _find_required(root_path, [BIOMASS_FILENAME])
    height_path = _find_required(root_path, [HEIGHT_FILENAME])
    coordinate_path = _find_required(root_path, COORDINATE_FILENAMES)

    biomass = _resolve_columns(
        _read_delimited(biomass_path),
        {
            "sample_date": ["Sample_Date", "sample date", "date"],
            "site_id": ["Site_ID", "site id"],
            "crop_type": ["Crop_Type", "crop"],
            "crop_part": ["Crop_Part", "plant part", "part"],
            "plant_water_content_pct": ["Plant_Water_Cont_PCT"],
            "area_plant_water_content_g_m2": ["Area_Plant_Water_Cont_g_m2"],
        },
    )
    biomass["sample_date"] = _parse_dates(biomass["sample_date"])
    biomass["site_id"] = biomass["site_id"].astype("string").str.strip()
    biomass["field_id"] = field_id_from_site_id(biomass["site_id"])
    biomass["crop_type"] = biomass["crop_type"].astype("string").str.strip()
    biomass["crop_part"] = biomass["crop_part"].astype("string").str.strip()
    _safe_numeric(
        biomass,
        ["plant_water_content_pct", "area_plant_water_content_g_m2"],
    )
    biomass = biomass.dropna(
        subset=["sample_date", "site_id", "field_id", "area_plant_water_content_g_m2"]
    )
    biomass = biomass[biomass["area_plant_water_content_g_m2"].between(0.0, 40000.0)]

    part_day = (
        biomass.groupby(
            ["sample_date", "field_id", "site_id", "crop_type", "crop_part"],
            observed=True,
            dropna=False,
            as_index=False,
        )
        .agg(
            vegetation_water_content_part_g_m2=(
                "area_plant_water_content_g_m2",
                "mean",
            ),
            plant_water_content_pct=("plant_water_content_pct", "mean"),
        )
    )
    site_day = (
        part_day.groupby(
            ["sample_date", "field_id", "site_id", "crop_type"],
            observed=True,
            dropna=False,
            as_index=False,
        )
        .agg(
            vegetation_water_content_in_situ_kg_m2=(
                "vegetation_water_content_part_g_m2",
                lambda series: float(series.sum()) / 1000.0,
            ),
            plant_water_content_pct=("plant_water_content_pct", "mean"),
            vegetation_part_count=("crop_part", "nunique"),
        )
    )

    height = _resolve_columns(
        _read_delimited(height_path),
        {
            "sample_date": ["Date", "Sample_Date"],
            "site_id": ["Site_ID", "site id"],
            "crop_type": ["Crop", "Crop_Type"],
            "canopy_height_cm": ["Mean_Heigh", "Mean_Height", "mean height"],
            "stem_diameter_mm": ["Mean_Diame", "Mean_Diameter", "mean diameter"],
        },
    )
    height["sample_date"] = _parse_dates(height["sample_date"])
    height["site_id"] = height["site_id"].astype("string").str.strip()
    height["field_id"] = field_id_from_site_id(height["site_id"])
    height["crop_type"] = height["crop_type"].astype("string").str.strip()
    _safe_numeric(height, ["canopy_height_cm", "stem_diameter_mm"])
    height_site_day = (
        height.dropna(subset=["sample_date", "site_id", "field_id"])
        .groupby(
            ["sample_date", "field_id", "site_id", "crop_type"],
            observed=True,
            dropna=False,
            as_index=False,
        )
        .agg(
            canopy_height_cm=("canopy_height_cm", "mean"),
            stem_diameter_mm=("stem_diameter_mm", "mean"),
        )
    )

    combined_site_day = site_day.merge(
        height_site_day,
        on=["sample_date", "field_id", "site_id", "crop_type"],
        how="outer",
        validate="one_to_one",
    )
    field_day = (
        combined_site_day.groupby(
            ["sample_date", "field_id"],
            observed=True,
            as_index=False,
        )
        .agg(
            crop_type=("crop_type", lambda series: series.dropna().mode().iloc[0] if not series.dropna().empty else pd.NA),
            vegetation_water_content_in_situ_kg_m2=(
                "vegetation_water_content_in_situ_kg_m2",
                "mean",
            ),
            vegetation_water_content_in_situ_std_kg_m2=(
                "vegetation_water_content_in_situ_kg_m2",
                "std",
            ),
            plant_water_content_pct=("plant_water_content_pct", "mean"),
            canopy_height_cm=("canopy_height_cm", "mean"),
            stem_diameter_mm=("stem_diameter_mm", "mean"),
            vegetation_site_count=("site_id", "nunique"),
        )
        .sort_values(["field_id", "sample_date"])
        .reset_index(drop=True)
    )

    coordinates = _resolve_columns(
        _read_delimited(coordinate_path),
        {
            "site_id": ["Site_ID", "site id"],
            "utm_x": ["X", "UTM_X", "easting"],
            "utm_y": ["Y", "UTM_Y", "northing"],
        },
    )
    coordinates["site_id"] = coordinates["site_id"].astype("string").str.strip()
    coordinates["field_id"] = field_id_from_site_id(coordinates["site_id"])
    _safe_numeric(coordinates, ["utm_x", "utm_y"])
    coordinates = coordinates[["field_id", "site_id", "utm_x", "utm_y"]].drop_duplicates()
    return field_day, coordinates


def collocate_nearest_in_situ(
    observations: pd.DataFrame,
    vegetation_field_day: pd.DataFrame,
    tolerance_days: int = 8,
) -> pd.DataFrame:
    """Attach the nearest same-field in-situ vegetation visit to each radar date."""
    result = observations.copy()
    result["acquisition_date"] = _parse_dates(result["acquisition_date"])
    result["field_id"] = result["field_id"].astype("string").str.strip()
    vegetation = vegetation_field_day.copy()
    vegetation["sample_date"] = _parse_dates(vegetation["sample_date"])
    vegetation["field_id"] = vegetation["field_id"].astype("string").str.strip()

    vegetation_columns = [
        column
        for column in vegetation.columns
        if column not in {"field_id", "sample_date"}
    ]
    for column in vegetation_columns:
        result[column] = np.nan if pd.api.types.is_numeric_dtype(vegetation[column]) else pd.NA
    result["vegetation_sample_date"] = pd.NaT
    result["vegetation_time_offset_days"] = np.nan

    for field_id, row_indices in result.groupby("field_id", observed=True).groups.items():
        candidates = vegetation[vegetation["field_id"] == field_id]
        if candidates.empty:
            continue
        candidate_dates = candidates["sample_date"].to_numpy(dtype="datetime64[ns]")
        for index in row_indices:
            date = result.at[index, "acquisition_date"]
            if pd.isna(date):
                continue
            offsets = np.abs((candidate_dates - np.datetime64(date)) / np.timedelta64(1, "D"))
            best_position = int(np.argmin(offsets))
            best_offset = float(offsets[best_position])
            if best_offset > tolerance_days:
                continue
            source = candidates.iloc[best_position]
            result.at[index, "vegetation_sample_date"] = source["sample_date"]
            result.at[index, "vegetation_time_offset_days"] = best_offset
            for column in vegetation_columns:
                result.at[index, column] = source[column]

    result["vegetation_in_situ_available"] = result[
        "vegetation_water_content_in_situ_kg_m2"
    ].notna()
    return result


def _parse_vwc_date(path: Path) -> pd.Timestamp | None:
    match = re.search(r"SV12VWC_(\d{8})_", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def discover_vwc_maps(root: str | Path) -> dict[pd.Timestamp, Path]:
    maps: dict[pd.Timestamp, Path] = {}
    duplicates: dict[pd.Timestamp, list[Path]] = {}
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        date = _parse_vwc_date(path)
        if date is None or pd.isna(date):
            continue
        date = date.normalize()
        if date in maps:
            duplicates.setdefault(date, [maps[date]]).append(path)
        else:
            maps[date] = path
    if duplicates:
        raise ValueError(f"Multiple SV12VWC maps found for dates: {duplicates}")
    return maps


def _find_world_file(tif_path: Path) -> Path:
    candidates = {tif_path.with_suffix(".tfw").name.lower(), tif_path.with_suffix(".TFW").name.lower()}
    matches = [path for path in tif_path.parent.iterdir() if path.is_file() and path.name.lower() in candidates]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one matching TFW for {tif_path}, found {matches}")
    return matches[0]


def _read_world_transform(path: Path) -> tuple[float, float, float, float, float, float]:
    values = [float(line.strip()) for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if len(values) != 6:
        raise ValueError(f"World file {path} must contain six numeric lines")
    return tuple(values)  # type: ignore[return-value]


def _world_to_pixel(
    x: float,
    y: float,
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    a, d, b, e, c, f = transform
    matrix = np.array([[a, b], [d, e]], dtype=float)
    if abs(np.linalg.det(matrix)) < 1.0e-12:
        raise ValueError("The TFW affine transform is singular")
    col, row = np.linalg.solve(matrix, np.array([x - c, y - f], dtype=float))
    return float(col), float(row)


def sample_vwc_field(
    tif_path: str | Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> dict[str, float]:
    """Summarise valid 0--40 kg/m2 pixels inside a field bounding box."""
    path = Path(tif_path)
    transform = _read_world_transform(_find_world_file(path))
    corners = [
        _world_to_pixel(x, y, transform)
        for x in (float(x_min), float(x_max))
        for y in (float(y_min), float(y_max))
    ]
    cols = [corner[0] for corner in corners]
    rows = [corner[1] for corner in corners]

    if rasterio is not None:
        with rasterio.open(path) as dataset:
            height, width = dataset.height, dataset.width
    else:
        with Image.open(path) as image:
            width, height = image.size
    col_start = max(0, int(np.floor(min(cols))))
    col_stop = min(width, int(np.ceil(max(cols))) + 1)
    row_start = max(0, int(np.floor(min(rows))))
    row_stop = min(height, int(np.ceil(max(rows))) + 1)
    if col_start >= col_stop or row_start >= row_stop:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "count": 0.0}
    if rasterio is not None:
        assert Window is not None
        with rasterio.open(path) as dataset:
            window = Window(
                col_off=col_start,
                row_off=row_start,
                width=col_stop - col_start,
                height=row_stop - row_start,
            )
            values = dataset.read(1, window=window).astype(float, copy=False).reshape(-1)
    else:
        try:
            with Image.open(path) as image:
                crop = image.crop((col_start, row_start, col_stop, row_stop))
                values = np.asarray(crop, dtype=float).reshape(-1)
        except Image.DecompressionBombError as exc:  # pragma: no cover - large field file
            raise RuntimeError(
                "This SV12VWC GeoTIFF is too large for the Pillow fallback. "
                "Install rasterio in the research-pilots environment."
            ) from exc
    values = values[np.isfinite(values) & (values >= 0.0) & (values <= 40.0)]
    if values.size == 0:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "count": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "count": float(values.size),
    }


def vwc_source_quality(date: pd.Timestamp, path: Path) -> tuple[str, float]:
    """Encode map provenance without pretending interpolation equals observation."""
    upper_name = path.stem.upper()
    if date.normalize() in DIRECT_VWC_DATES or upper_name.endswith(("_RE", "_SP", "_RE_SP")):
        return "direct_satellite", 1.0
    if date.normalize() > LAST_DIRECT_VWC_DATE:
        return "extrapolated", 0.4
    return "interpolated", 0.7


def collocate_vwc_maps(observations: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Attach daily field-average SV12VWC and its provenance to observations."""
    result = observations.copy()
    result["acquisition_date"] = _parse_dates(result["acquisition_date"])
    maps = discover_vwc_maps(root)
    required_geometry = [
        "field_utm_x_min",
        "field_utm_x_max",
        "field_utm_y_min",
        "field_utm_y_max",
    ]
    missing_geometry = [column for column in required_geometry if column not in result]
    if missing_geometry:
        raise ValueError(f"Input table is missing field geometry: {missing_geometry}")

    records: list[dict[str, object]] = []
    cache: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}
    for _, row in result.iterrows():
        date = row["acquisition_date"]
        field_id = str(row["field_id"])
        path = maps.get(date)
        record: dict[str, object] = {
            "vwc_map_filename": pd.NA,
            "vwc_map_source": pd.NA,
            "vwc_map_quality": np.nan,
            "vegetation_water_content_map_kg_m2": np.nan,
            "vegetation_water_content_map_median_kg_m2": np.nan,
            "vegetation_water_content_map_std_kg_m2": np.nan,
            "vwc_map_pixel_count": 0,
        }
        if path is not None and all(pd.notna(row[column]) for column in required_geometry):
            cache_key = (date, field_id)
            if cache_key not in cache:
                cache[cache_key] = sample_vwc_field(
                    path,
                    row["field_utm_x_min"],
                    row["field_utm_x_max"],
                    row["field_utm_y_min"],
                    row["field_utm_y_max"],
                )
            statistics = cache[cache_key]
            source, quality = vwc_source_quality(date, path)
            record.update(
                {
                    "vwc_map_filename": path.name,
                    "vwc_map_source": source,
                    "vwc_map_quality": quality,
                    "vegetation_water_content_map_kg_m2": statistics["mean"],
                    "vegetation_water_content_map_median_kg_m2": statistics["median"],
                    "vegetation_water_content_map_std_kg_m2": statistics["std"],
                    "vwc_map_pixel_count": int(statistics["count"]),
                }
            )
        records.append(record)

    map_frame = pd.DataFrame.from_records(records, index=result.index)
    result = pd.concat([result, map_frame], axis=1)
    result["vwc_map_available"] = result["vegetation_water_content_map_kg_m2"].notna()
    return result


def calculate_physics_confidence(frame: pd.DataFrame, alpha: float = 0.8) -> pd.DataFrame:
    """Build a transparent initial gate for the bare-soil SPM constraint.

    The gate is not a learned result.  ``alpha`` and any use of the gate as a
    loss weight must be selected inside inner grouped cross-validation.
    """
    result = frame.copy()
    ks = pd.to_numeric(result.get("spm_k_rms_height"), errors="coerce")
    slope = pd.to_numeric(result.get("spm_rms_slope_proxy"), errors="coerce")
    q_roughness = np.minimum(
        np.clip((0.30 - ks) / 0.30, 0.0, 1.0),
        np.clip((0.21 - slope) / 0.21, 0.0, 1.0),
    )
    vwc = pd.to_numeric(result.get("vegetation_water_content_map_kg_m2"), errors="coerce")
    vwc = vwc.fillna(pd.to_numeric(result.get("vegetation_water_content_in_situ_kg_m2"), errors="coerce"))
    q_vegetation = np.exp(-float(alpha) * np.clip(vwc, 0.0, None))
    q_source = pd.to_numeric(result.get("vwc_map_quality"), errors="coerce").fillna(0.5)
    result["physics_validity_roughness"] = q_roughness
    result["physics_validity_vegetation"] = q_vegetation
    result["physics_confidence_initial"] = np.clip(q_roughness * q_vegetation * q_source, 0.0, 1.0)
    return result
