"""Resumable driver for the RQ2 sweep (adaptive vs fixed vs random summary schedules on CLINC150).

Phases (see `kslimeans.rq2` and `configs/rq2_clinc150.yaml`): A fixed family (seeds 0-4), B adaptive grid
(seeds 0-2), C random controls matched to B's per-step calls, D held-out evaluation: the headline config
(chosen on seeds 0-2) and its controls on seeds 3-9, plus fixed-n5 on seeds 5-9. Runs are executed in-process,
one after another. A run is skipped when `<results-dir>/<name>/<stamp>/` already holds a `summary.csv` whose
config defines the same experiment ([done]); a matching dir without `summary.csv` that was written to within
--stale-minutes is assumed to be running elsewhere ([running], skipped); older ones count as crashed and are redone.

`--set` applies to EVERY run, Phase D's extension runs included (a `--set seeds=[0]` would also shrink the
held-out runs), so use it for smoke tests only.

    python scripts/run_rq2.py --dry-run
    python scripts/run_rq2.py --phases A B --limit 3
    python scripts/run_rq2.py --phases A B C --set dataset=bank77 --set 'summarizer={type: lsa, top_k: 5}' \\
        --set 'seeds=[0]' --results-dir /tmp/rq2_smoke
"""

from __future__ import annotations

import argparse
import time

from kslimeans import rq2
from kslimeans.experiment import run_offline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phases", nargs="+", choices=rq2.PHASES, default=list(rq2.PHASES))
    parser.add_argument("--base-config", default="configs/rq2_clinc150.yaml")
    parser.add_argument("--results-dir", default=None, help="override the config's results_dir")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and which runs are done; no compute")
    parser.add_argument("--stale-minutes", type=float, default=rq2.STALE_MINUTES,
                        help="an unfinished matching run dir touched within this many minutes counts as running")
    parser.add_argument("--limit", type=int, default=None, help="run at most N runs, then stop")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="config override applied to every run, Phase D included (dotted keys, YAML values); "
                             "for smoke tests only: e.g. seeds=[0] also shrinks the held-out runs")
    args = parser.parse_args()

    base = rq2.load_base(args.base_config, args.results_dir)
    results_dir = base["results_dir"]
    ran = 0
    for phase in sorted(args.phases):
        # Phase D's random controls need the extension run's trace, so plan it a second time once that run is done.
        for _ in range(2 if phase == "D" and not args.dry_run else 1):
            ran_before = ran
            specs, rule = rq2.plan_phase(phase, base, results_dir, args.overrides)
            counts = {"done": 0, "running": 0, "todo": 0, "blocked": 0}
            print(f"== Phase {phase}: {len(specs)} run(s)" + (f" (headline rule: {rule})" if rule else ""))
            for spec in specs:
                if spec.cfg is None:
                    counts["blocked"] += 1
                    print(f"  [blocked] {spec.name}: {spec.blocked}")
                    continue
                status, _ = rq2.run_status(results_dir, spec.cfg, args.stale_minutes)
                if status != "todo":
                    counts[status] += 1
                    print(f"  [{status}]".ljust(12) + spec.name)
                    continue
                counts["todo"] += 1
                if args.dry_run:
                    print(f"  [todo]    {spec.name}")
                    continue
                if args.limit is not None and ran >= args.limit:
                    print(f"  [todo]    {spec.name} (limit reached)")
                    continue
                print(f"  [start]   {spec.name}", flush=True)
                start = time.perf_counter()
                run_offline(spec.cfg)
                ran += 1
                counts["todo"] -= 1
                counts["done"] += 1
                print(f"  [finish]  {spec.name} in {time.perf_counter() - start:.0f}s", flush=True)
            print(f"   Phase {phase}: {counts['done']} done, {counts['running']} running, {counts['todo']} to run, {counts['blocked']} blocked")
            if ran == ran_before:
                break
        if args.limit is not None and ran >= args.limit and not args.dry_run:
            break


if __name__ == "__main__":
    main()
