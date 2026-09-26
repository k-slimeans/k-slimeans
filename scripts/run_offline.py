"""Run an offline experiment from a YAML config.

Loads the config, applies any `--set key.path=value` overrides on top of
it (repeatable; values are parsed as YAML, so numbers/bools/lists work),
and hands the result to `kslimeans.experiment.run_offline`.

Usage:

    # Run a config exactly as written.
    python scripts/run_offline.py configs/bank77_knlpmeans.yaml

    # Swap the SLM a summarizer config uses, without touching the YAML.
    python scripts/run_offline.py configs/bank77_slm.yaml \\
        --set summarizer.model_id=Qwen/Qwen2.5-1.5B-Instruct

    # Swap dataset and seeds for a quick multi-seed check.
    python scripts/run_offline.py configs/offline_knlpmeans.yaml \\
        --set dataset=clinic --set seeds=[0,1,2]
"""

from __future__ import annotations

import argparse

from kslimeans.config import load_config
from kslimeans.experiment import run_offline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key.path=value",
        help="override a config key, e.g. --set schedule.max_iter=50 (repeatable)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_offline(cfg)


if __name__ == "__main__":
    main()
