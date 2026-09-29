"""Analyze common/differential SPM residuals with field-block bootstrap intervals."""
import argparse
import sys
from datetime import datetime
sys.dont_write_bytecode = True
from paper_diagnostics_core import DATA, analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DATA / "paper_diagnostics/cohort_common.csv"))
    parser.add_argument("--output", default=str(DATA / "paper_diagnostics" / ("analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    analyze(args.input, args.output, args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
