#!/usr/bin/env bash
# CLINC150 prompt test (#46, seeds 0-4): k-SLMmeans with Qwen3-4B / 0.6B / 1.7B under the clearer prompt of
# configs/clinc150_prompt_v2.yaml, then one table next to the #31 runs. Same protocol as scripts/run_clinc150.sh
# (read its notes); only the prompt changes. k-means and LSA are not rerun: they do not depend on the prompt and
# the #31 runs have them.
#
#   PYTHON=/path/to/venv/bin/python scripts/run_clinc150_prompt_v2.sh          # 4b, 0.6b, 1.7b, then the table
#   PYTHON=... scripts/run_clinc150_prompt_v2.sh 4b                              # only the 4B, then the table
#   PYTHON=... scripts/run_clinc150_prompt_v2.sh table                           # only rebuild the table
#
# Steps: 4b | 0.6b | 1.7b | table (no run, just the table). The 4B goes first because it carries the question. Each
# run's log goes to results/clinc150_logs/<name>.log. The table (results/clinc150_prompt_v2_table.md + .csv, and the
# per-seed CSV) uses the newest completed run of every step, v2 and #31, so a failed step can be rerun on its own.
#
# Before a run (WSL2, one 12 GB card): check that nothing else uses the GPU (`nvidia-smi` on both the WSL and
# the Windows side). The #31 timings were inflated ~1.7x by another process on the card. Do not edit this script,
# the config or the code while it runs: bash reads a script as it goes, and a changed file changes later steps.
# batch_size per model is the #31 choice (0.6B 8, 1.7B 4, 4B 1): the 4B at batch 2 overflows the 12 GB card
# on capped prompts, and under WSL2 an overflow hangs the job instead of raising an out-of-memory error.
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set below as in #31 (it changes how memory is reserved,
# not what is computed).
set -euo pipefail

PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(4b 0.6b 1.7b)

mkdir -p results/clinc150_logs

# run_slm SIZE MODEL BATCH: run one experiment, tee its log, print its run directory.
run_slm() {
    local name="clinc150_v2_qwen3-$1"
    echo "=== $name ==="
    "$PYTHON" scripts/run_offline.py configs/clinc150_prompt_v2.yaml --set "name=$name" \
        --set "summarizer.model_id=Qwen/Qwen3-$2" --set "summarizer.batch_size=$3" 2>&1 | tee "results/clinc150_logs/$name.log"
    echo "Finished $name: $(grep '^Run directory:' "results/clinc150_logs/$name.log" | tail -1)"
}

for step in "${STEPS[@]}"; do
    case $step in
        4b) run_slm 4b 4B 1 ;;
        0.6b) run_slm 0.6b 0.6B 8 ;;
        1.7b) run_slm 1.7b 1.7B 4 ;;
        table) ;;
        *) echo "Unknown step '$step', expected 4b | 0.6b | 1.7b | table" >&2; exit 1 ;;
    esac
done

# Table over the newest *completed* run of each step, in a fixed order: the #31 LSA run first (it carries the
# k-means and LSA rows), the #31 sizes, then the v2 sizes. summary.csv is written only when a run finishes.
RUN_DIRS=()
for name in clinc150_lsa clinc150_qwen3-0.6b clinc150_qwen3-1.7b clinc150_qwen3-4b \
            clinc150_v2_qwen3-0.6b clinc150_v2_qwen3-1.7b clinc150_v2_qwen3-4b; do
    latest=$(ls -d "results/$name"/*/summary.csv 2>/dev/null | sort | tail -1 || true)
    [ -n "$latest" ] && RUN_DIRS+=("$(dirname "$latest")")
done
[ ${#RUN_DIRS[@]} -gt 0 ] || { echo "No completed clinc150 runs found under results/" >&2; exit 1; }
printf 'Aggregating %s\n' "${RUN_DIRS[@]}"
"$PYTHON" scripts/aggregate_results.py "${RUN_DIRS[@]}" \
    --out-md results/clinc150_prompt_v2_table.md --out-csv results/clinc150_prompt_v2_table.csv \
    --per-seed-csv results/clinc150_prompt_v2_per_seed.csv
echo
cat results/clinc150_prompt_v2_table.md
