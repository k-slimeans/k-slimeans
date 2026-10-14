#!/usr/bin/env bash
# Preliminary Bank77 experiment (seeds 0-4): k-NLPmeans (LSA) plus k-SLMmeans with Qwen3-0.6B / 1.7B / 4B,
# then one comparison table. See configs/prelim_bank77.yaml for the protocol.
#
#   PYTHON=/path/to/venv/bin/python scripts/run_prelim.sh            # everything
#   PYTHON=... scripts/run_prelim.sh lsa 0.6b                         # only these steps, then the table
#   PYTHON=... scripts/run_prelim.sh table                            # only rebuild the table
#
# Steps: lsa | 0.6b | 1.7b | 4b | table (no run, just the table). Each run's log goes to
# results/prelim_logs/<name>.log. The table (results/prelim_bank77_table.md + .csv) is built from the
# newest completed run of every step that has one, so a failed step can be rerun on its own.
#
# batch_size per model: largest of {16, 8, 4, 2} whose peak VRAM stayed <= 10 GB when summarizing the
# largest k-means clusters (77 clusters, seeds 0-4; largest cluster 107 docs, longest prompt 2911 tokens,
# prompts are capped at max_input_tokens=4096), then one notch smaller as headroom for the bigger clusters
# the summary steps can produce. Measured peaks (GB) at bs 16 / 8 / 4 / 2:
#   Qwen3-0.6B  8.37 / 4.92 / 3.19 / 2.29  -> 8
#   Qwen3-1.7B  11.70 / 7.71 / 5.71 / 4.67 -> 4
#   Qwen3-4B    >10 (not run; bs 16 spilled out of VRAM) / >10 / >10 / 9.65 -> 2 (bs 1 peaked at 8.71; fall back to
#               it with `--set summarizer.batch_size=1` if bs 2 runs out of memory on a long prompt)
set -euo pipefail

PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(lsa 0.6b 1.7b 4b)

mkdir -p results/prelim_logs

# run_step NAME CONFIG [--set ...]: run one experiment, tee its log, print its run directory.
run_step() {
    local name=$1 config=$2
    shift 2
    echo "=== $name ==="
    "$PYTHON" scripts/run_offline.py "$config" --set "name=$name" "$@" 2>&1 | tee "results/prelim_logs/$name.log"
    echo "Finished $name: $(grep '^Run directory:' "results/prelim_logs/$name.log" | tail -1)"
}

run_slm() {
    run_step "prelim_bank77_qwen3-$1" configs/prelim_bank77.yaml \
        --set "summarizer.model_id=Qwen/Qwen3-$2" --set "summarizer.batch_size=$3"
}

for step in "${STEPS[@]}"; do
    case $step in
        lsa) run_step prelim_bank77_lsa configs/offline_knlpmeans.yaml --set 'seeds=[0,1,2,3,4]' ;;
        0.6b) run_slm 0.6b 0.6B 8 ;;
        1.7b) run_slm 1.7b 1.7B 4 ;;
        4b) run_slm 4b 4B 2 ;;
        table) ;;
        *) echo "Unknown step '$step', expected lsa | 0.6b | 1.7b | 4b | table" >&2; exit 1 ;;
    esac
done

# Table over the newest *completed* run directory of each step (in a fixed order), whichever steps ran
# now or before. summary.csv is written only when a run finishes, so crashed runs are skipped.
RUN_DIRS=()
for name in prelim_bank77_lsa prelim_bank77_qwen3-0.6b prelim_bank77_qwen3-1.7b prelim_bank77_qwen3-4b; do
    latest=$(ls -d "results/$name"/*/summary.csv 2>/dev/null | sort | tail -1 || true)
    [ -n "$latest" ] && RUN_DIRS+=("$(dirname "$latest")")
done
[ ${#RUN_DIRS[@]} -gt 0 ] || { echo "No completed prelim runs found under results/" >&2; exit 1; }
printf 'Aggregating %s\n' "${RUN_DIRS[@]}"
"$PYTHON" scripts/aggregate_results.py "${RUN_DIRS[@]}" \
    --out-md results/prelim_bank77_table.md --out-csv results/prelim_bank77_table.csv
echo
cat results/prelim_bank77_table.md
