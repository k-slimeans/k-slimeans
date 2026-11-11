"""Verify the logged cost counters of run directories and write a markdown report (#35, part 1).

For each run directory it rebuilds what the counters count from the logged partitions and summaries (calls,
input and output tokens, truncation, empty summaries) and checks the seconds for consistency; see
`kslimeans.costcheck`. Tokenizers are loaded from the local Hugging Face cache only (no download, no model
weights, no GPU), and embeddings come from the cache (nothing is encoded).

Usage:

    python scripts/verify_cost.py results/clinc150_lsa/<stamp> results/clinc150_qwen3-4b/<stamp> \\
        --cache-dir /path/to/k-slimeans/cache --out report.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from kslimeans.costcheck import OUTPUT_TOL, format_report, verify_run
from kslimeans.runlog import load_run


def load_local_tokenizer(model_id: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, padding_side="left", local_files_only=True, trust_remote_code=trust_remote_code
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    parser.add_argument("--cache-dir", help="the embeddings cache (default: each run's cache_dir)")
    parser.add_argument("--out", help="write the report here (default: stdout)")
    parser.add_argument(
        "--output-tol",
        type=float,
        default=OUTPUT_TOL,
        help="relative tolerance of the output-token comparison per seed (default %(default)s, i.e. 0.3%%). "
        "Any error above 0.3%% of a seed's output tokens is flagged, e.g. one token missing on every ~20th call; "
        "a missed EOS on every call (4-7%%) fails by a wide margin. The observed residual is at most 0.1%%.",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    reports = [
        verify_run(load_run(d), args.cache_dir, load_local_tokenizer, args.output_tol) for d in args.run_dirs
    ]
    report = format_report(reports)
    if args.out:
        Path(args.out).write_text(report)
    else:
        print(report)
    print(f"verified {len(reports)} run(s) in {time.perf_counter() - start:.1f} s", file=sys.stderr)
    failed = any(not c.passed for r in reports for m in r.methods for c in m.checks)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
