#!/usr/bin/env bash
# CLINC150 summary-guard diagnostic (#47, stage 2, seeds 0-4): k-SLMmeans with Qwen3-4B where flagged summaries (an
# exact duplicate of another cluster's, or talking about the cluster) fall back to the cluster's k-means center at every
# summary step; see configs/clinc150_guard.yaml. A diagnostic variant, not a method. Then one table next to the #31 4B
# and k-means rows. Same protocol as scripts/run_clinc150.sh (read its notes); only the guard differs.
#
#   PYTHON=/path/to/venv/bin/python scripts/run_clinc150_guard.sh          # the guarded 4B run, then the table
#   PYTHON=... scripts/run_clinc150_guard.sh table                          # only rebuild the table
#
# Needs the #31 runs clinc150_lsa (carries the k-means row) and clinc150_qwen3-4b under results/. The run's log goes to
# results/clinc150_logs/clinc150_guard_qwen3-4b.log. The table (results/clinc150_guard_table.md + .csv, and the per-seed
# CSV) uses the newest completed run of every name, so a failed run can be rerun on its own.
#
# Time: no extra model calls, so about the #31 4B cost of 1,862 s per seed (section 3 of
# docs/results/clinc150_summarizers.md), which a shared GPU inflated ~1.7x: about 1,100 s per seed and 1.5 h for five
# seeds on an idle card, up to 2.6 h under the old inflation. Guarded runs follow other partitions, so the number of
# summarizer calls (714 per seed in #31) can differ a little.
#
# Before a run (WSL2, one 12 GB card): check that nothing else uses the GPU (`nvidia-smi` on both the WSL and the
# Windows side). Do not edit this script, the config or the code while it runs: bash reads a script as it goes, and a
# changed file changes later steps. The 4B runs at batch size 1 as in #31: at batch 2 it overflows the 12 GB card on
# capped prompts, and under WSL2 an overflow hangs the job instead of raising an out-of-memory error.
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set below as in #31 (it changes how memory is reserved, not
# what is computed).
set -euo pipefail

PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(4b)

mkdir -p results/clinc150_logs

for step in "${STEPS[@]}"; do
    case $step in
        4b)
            name=clinc150_guard_qwen3-4b
            echo "=== $name ==="
            "$PYTHON" scripts/run_offline.py configs/clinc150_guard.yaml --set "name=$name" 2>&1 | tee "results/clinc150_logs/$name.log"
            echo "Finished $name: $(grep '^Run directory:' "results/clinc150_logs/$name.log" | tail -1)"
            ;;
        table) ;;
        *) echo "Unknown step '$step', expected 4b | table" >&2; exit 1 ;;
    esac
done

# Table over the newest *completed* run of each name, in a fixed order: the #31 LSA run (it carries the k-means row),
# the #31 4B, then the guarded 4B. summary.csv is written only when a run finishes.
RUN_DIRS=()
for name in clinc150_lsa clinc150_qwen3-4b clinc150_guard_qwen3-4b; do
    latest=$(ls -d "results/$name"/*/summary.csv 2>/dev/null | sort | tail -1 || true)
    [ -n "$latest" ] && RUN_DIRS+=("$(dirname "$latest")")
done
[ ${#RUN_DIRS[@]} -gt 0 ] || { echo "No completed clinc150 runs found under results/" >&2; exit 1; }
printf 'Aggregating %s\n' "${RUN_DIRS[@]}"
"$PYTHON" scripts/aggregate_results.py "${RUN_DIRS[@]}" \
    --out-md results/clinc150_guard_table.md --out-csv results/clinc150_guard_table.csv \
    --per-seed-csv results/clinc150_guard_per_seed.csv
echo
cat results/clinc150_guard_table.md
