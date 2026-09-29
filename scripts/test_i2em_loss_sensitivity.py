"""Tests for the I2EM dielectric-loss sensitivity request design."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "prepare_i2em_loss_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("prepare_i2em_loss_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LossSensitivityDesignTests(unittest.TestCase):
    def test_complete_unique_valid_grid(self) -> None:
        frame = pd.DataFrame(
            {
                "soil_real_dielectric": [6.0, 12.0, 28.0],
                "rms_height_m": [0.0046, 0.0067, 0.0104],
                "correlation_length_m": [0.055, 0.1025, 0.1825],
            }
        )
        losses = [0.0, 0.02, 0.05, 0.10]
        grid, _ = MODULE.make_sensitivity_grid(frame, 1.26, 40.0, losses)
        self.assertEqual(len(grid), 72)
        self.assertEqual(grid["request_id"].nunique(), 72)
        self.assertEqual(sorted(grid["loss_tangent"].unique().tolist()), losses)
        self.assertTrue(bool(grid["i2em_request_valid"].all()))
        for loss in losses:
            self.assertEqual(int((grid["loss_tangent"] == loss).sum()), 18)


if __name__ == "__main__":
    unittest.main()
