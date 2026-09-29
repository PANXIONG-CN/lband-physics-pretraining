"""Fast contract tests for the cross-campaign data pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1] / "scripts")]

import numpy as np
import pandas as pd

import audit_external_domain as audit


def make_frame(campaign: str, year: int, offset: float = 0.0) -> pd.DataFrame:
    moisture = np.linspace(0.1, 0.5, 12) + offset
    return pd.DataFrame(
        {
            "campaign_id": campaign,
            "acquisition_date": pd.date_range(f"{year}-06-01", periods=12),
            "field_id": [f"F{index % 6}" for index in range(12)],
            "soil_moisture_m3_m3": moisture,
            "pals_rms_height_cm": np.linspace(0.3, 0.9, 12),
            "pals_correlation_length_cm": np.linspace(5.0, 15.0, 12),
            "sigma0_hh_db": -18 + 5 * moisture,
            "sigma0_vv_db": -15 + 6 * moisture,
        }
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_path = root / "source.csv"
        external_path = root / "external.csv"
        make_frame("SMAPVEX12", 2012).to_csv(source_path, index=False)
        make_frame("SMEX02", 2002, 0.02).to_csv(external_path, index=False)
        source = audit.prepare_table(source_path, "source", "topp_both")
        external = audit.prepare_table(external_path, "external", "topp_both")
        assert np.isfinite(source[audit.FEATURES].to_numpy()).all()
        assert set(source.campaign_id).isdisjoint(set(external.campaign_id))
        shift = audit.feature_shift(source, external)
        assert list(shift.feature) == audit.FEATURES
        assert shift.external_fraction_inside_source_q01_q99.between(0, 1).all()
        validity = audit.validity_summary(external, 1.26)
        assert 0 <= validity["fraction_spm_valid_ks_lt_0_30_and_sigma_over_L_lt_0_21"] <= 1
    print("external-domain contract tests passed")


if __name__ == "__main__":
    main()
