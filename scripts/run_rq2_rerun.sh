#!/usr/bin/env bash
# RQ2 headline re-run on the #31 protocol (#37 step 1, seeds 0-9): fixed-n1, fixed-n2, fixed-n5 (with the k-means
# baseline), the headline adaptive config adaptive-d0.6-r0.7 and its call-matched random control, all Qwen3-0.6B on
# CLINC150. The #15 runs trimmed overflowing prompts to a dataset-order prefix; this code keeps a seeded random subset
# (#28), as the #31 runs do. See configs/rq2_rerun.yaml and docs/results/rq2_truncation_check.md for why.
#
#   PYTHON=/path/to/venv/bin/python scripts/run_rq2_rerun.sh           # plan, the runs, then the table
#   PYTHON=... scripts/run_rq2_rerun.sh plan                           # only print which runs are todo / done
#   PYTHON=... scripts/run_rq2_rerun.sh table                          # only rebuild the table
#
# Resumable: scripts/run_rq2.py skips a run whose directory already holds a summary.csv for the same config, so a
# crashed run is simply redone by running this script again. The five run groups go one after another in one
# process, in this order: fixed-n5, fixed-n1, fixed-n2, adaptive, then one random control per seed (each needs that
# seed's adaptive trace, so the driver plans them once the adaptive run is done). The log goes to
# results/rq2_rerun_logs/rq2_rerun.log; the table (results/rq2_rerun_table.md + .csv, and the per-seed CSV) uses the
# newest completed run of every name.
#
# Time: about 16,000 summarizer calls (150 + 293 + 703 + 237 + 236 per seed, from #15's means, times 10 seeds). At the
# #31 Qwen3-0.6B cost of 1.28 s per call, wall clock included (903.9 s per seed for ~707 calls; the background GPU load
# of that run inflated it, #31 section 3) that is about 5.8 h; at the 0.75 s per call of #15's fixed-n5 runs, about
# 3.4 h. Budget 6 h. Throughput is not steady (#15 saw 0.7 to 3 s per call), so judge progress by the call count, not
# the clock.
#
# Before a run (WSL2, one 12 GB card): check that nothing else uses the GPU (`nvidia-smi` on both the WSL and the
# Windows side). Do not edit this script, the config or the code while it runs: bash reads a script as it goes, and a
# changed file changes later steps. Batch size 8 as in #31 and #15; the 0.6B peaked at 6.7 GB reserved in #31.
set -euo pipefail

PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(run)

mkdir -p results/rq2_rerun_logs

for step in "${STEPS[@]}"; do
    case $step in
        plan) "$PYTHON" scripts/run_rq2.py --phases R --base-config configs/rq2_rerun.yaml --dry-run ;;
        run)
            echo "=== rq2 rerun ==="
            "$PYTHON" scripts/run_rq2.py --phases R --base-config configs/rq2_rerun.yaml 2>&1 | tee results/rq2_rerun_logs/rq2_rerun.log
            ;;
        table) ;;
        *) echo "Unknown step '$step', expected plan | run | table" >&2; exit 1 ;;
    esac
done
[ "${STEPS[*]}" = "plan" ] && exit 0

# Table over the newest *completed* run of each name, fixed-n5 first (it carries the k-means row), then fixed-n1,
# fixed-n2, the adaptive run and the ten random controls. summary.csv is written only when a run finishes.
RUN_DIRS=()
for name in rq2r_fixed_n5 rq2r_fixed_n1 rq2r_fixed_n2 rq2r_adaptive_d0.6_r0.7 rq2r_random_d0.6_r0.7_seed{0..9}; do
    latest=$(ls -d "results/$name"/*/summary.csv 2>/dev/null | sort | tail -1 || true)
    [ -n "$latest" ] && RUN_DIRS+=("$(dirname "$latest")")
done
[ ${#RUN_DIRS[@]} -gt 0 ] || { echo "No completed rq2r runs found under results/" >&2; exit 1; }
printf 'Aggregating %s\n' "${RUN_DIRS[@]}"
"$PYTHON" scripts/aggregate_results.py "${RUN_DIRS[@]}" \
    --out-md results/rq2_rerun_table.md --out-csv results/rq2_rerun_table.csv \
    --per-seed-csv results/rq2_rerun_per_seed.csv
echo
cat results/rq2_rerun_table.md
