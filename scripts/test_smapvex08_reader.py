"""Fast schema tests for the SMAPVEX08 reader using synthetic files."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from research_pilots.scattering.data.read_smapvex08 import (  # noqa: E402
    PALS_COLUMNS,
    iter_pals_chunks,
    linear_power_mean_db,
    parse_pals_filename,
)


def main() -> None:
    metadata = parse_pals_filename("SV08PLBK_09291306.red")
    assert str(metadata["acquisition_date"].date()) == "2008-09-29"
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "SV08PLBK_09291306.red"
        pd.DataFrame([np.arange(24), np.arange(24) + 1]).to_csv(
            path, sep=" ", header=False, index=False
        )
        frame = pd.concat(iter_pals_chunks(path, chunksize=1), ignore_index=True)
        assert list(frame.columns[:24]) == PALS_COLUMNS
        assert len(frame) == 2
    assert abs(linear_power_mean_db(pd.Series([-10.0, -10.0])) + 10.0) < 1e-12
    print("SMAPVEX08 reader tests passed")


if __name__ == "__main__":
    main()
