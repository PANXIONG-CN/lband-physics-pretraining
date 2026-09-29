"""Fast schema and unit tests for the SMEX02 reader."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_pilots.scattering.data.read_smex02 import (  # noqa: E402
    iter_pals_chunks,
    linear_power_mean_db,
    parse_pals_filename,
    read_soil_moisture_summary,
    read_surface_roughness,
)


def main() -> None:
    metadata = parse_pals_filename("06250752.red")
    assert str(metadata["acquisition_date"].date()) == "2002-06-25"
    assert metadata["l_band_frequency_ghz"] == 1.26

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        pals = root / "06250752.red"
        pals_columns = [
            "time", "GPS_time", "lat", "long", "ant_azimuth", "polar_angle",
            "range", "beam_angle", "L_HH", "L_VV", "L_VH", "L_HV",
            "S_HH", "S_VV", "S_VH", "S_HV",
        ]
        pd.DataFrame(
            [[1, 2, 41.9, -93.7, 270, 0, 1100, 43.5, -15, -14, -23, -24,
              -12, -11, -20, -21]],
            columns=pals_columns,
        ).to_csv(pals, sep="\t", index=False)
        parsed = pd.concat(iter_pals_chunks(pals), ignore_index=True)
        assert parsed.loc[0, "sigma0_hh_db"] == -15
        assert parsed.loc[0, "sigma0_vv_db"] == -14

        soil = root / "WC_TP_SUM.txt"
        soil.write_text(
            "Date\tSiteID\tLatitude\tLongitude\t#ofsamples\tV\tStdev\t"
            "VSM_gc\tStdev\tVSM_ssc\tStdev\t\n"
            "6/25/02\tWC01\t41.96\t-93.76\t14\t0.4\t0.1\t0.18\t0.03\t0.16\t0.02\t\n",
            encoding="utf-8",
        )
        soil_frame = read_soil_moisture_summary(soil)
        assert soil_frame.loc[0, "field_id"] == "WC01"
        assert abs(soil_frame.loc[0, "soil_moisture_m3_m3"] - 0.16) < 1e-12

        rough_header = "title\nsubtitle\ncolumns\n"
        grid = root / "grid_scanning.txt"
        slope = root / "slope_scanning.txt"
        grid.write_text(rough_header + "wc01a1 101 0.778 4.079 0.772 1\n")
        slope.write_text(rough_header + "WC01A1 122 0.681 7.464 0.671 1 1\n")
        rough = read_surface_roughness(grid, slope)
        assert set(rough["scan_mode"]) == {"grid", "slope"}
        assert set(rough["field_id"]) == {"WC01"}
        assert rough["roughness_positive"].all()

    assert abs(linear_power_mean_db(pd.Series([-10.0, -10.0])) + 10.0) < 1e-12
    print("SMEX02 reader tests passed")


if __name__ == "__main__":
    main()
