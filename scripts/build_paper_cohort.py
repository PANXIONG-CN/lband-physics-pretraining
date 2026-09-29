"""Build an auditable, common field-day cohort without modifying existing results."""
import argparse
import sys
sys.dont_write_bytecode = True
from paper_diagnostics_core import DATA, build_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(DATA / "spectrum_comparison/spectrum_predictions.csv"))
    parser.add_argument("--vegetation", default=str(DATA / "vegetation_audit/vegetation_enriched_predictions.csv"))
    parser.add_argument("--output", default=str(DATA / "paper_diagnostics"))
    parser.add_argument("--quality-days", type=int, default=2)
    args = parser.parse_args()
    build_cohort(args.base, args.vegetation, args.output, args.quality_days)


if __name__ == "__main__":
    main()
