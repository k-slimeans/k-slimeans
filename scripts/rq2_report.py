"""Build the RQ2 report from completed `rq2_*` runs: table, paired comparisons, Pareto plot.

Writes rq2_clinc150_{table.md,table.csv,per_seed.csv,matched.md,pareto.png} to --out-dir. The runs used are the
planned runs of phases A-D, built from --base-config (+ --set, which must match what the driver was given) and
identity-checked like the driver's done-detection; other rq2_* dirs are warned about and ignored. The H2
verdict is judged on the held-out seeds 3-9; the Pareto plot shows seeds 0-2 for every variant.

    python scripts/rq2_report.py
    python scripts/rq2_report.py --results-dir results --out-dir docs/results
"""

from __future__ import annotations

import argparse

from kslimeans.rq2 import load_base, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="docs/results")
    parser.add_argument("--base-config", default="configs/rq2_clinc150.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="the driver's --set overrides, if it was run with any (smoke tests)")
    parser.add_argument("--no-plot", action="store_true", help="skip the Pareto PNG (matplotlib not needed)")
    args = parser.parse_args()
    try:
        paths = write_report(args.results_dir, args.out_dir, load_base(args.base_config), args.overrides, plot=not args.no_plot)
    except ImportError as e:
        raise SystemExit(f"{e}; install the plot extra (pip install -e '.[plot]') or pass --no-plot") from e
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
