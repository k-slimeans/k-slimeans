#!/usr/bin/env bash
# CLINC150 summarizer comparison (#31, seeds 0-4): k-NLPmeans (LSA) plus k-SLMmeans with Qwen3-0.6B / 1.7B / 4B,
# then one comparison table. See configs/clinc150_summarizers.yaml for the protocol.
#
#   PYTHON=/path/to/venv/bin/python scripts/run_clinc150.sh            # everything
#   PYTHON=... scripts/run_clinc150.sh lsa 0.6b                         # only these steps, then the table
#   PYTHON=... scripts/run_clinc150.sh table                            # only rebuild the table
#
# Steps: lsa | 0.6b | 1.7b | 4b | table (no run, just the table). Each run's log goes to
# results/clinc150_logs/<name>.log. The table (results/clinc150_summarizers_table.md + .csv, and the per-seed CSV)
# is built from the newest completed run of every step that has one, so a failed step can be rerun on its own.
#
# batch_size per model: 0.6B 8 and 1.7B 4, as on Bank77 (the 0.6B also ran all of #15 on CLINC150 at 8). The 4B
# runs at 1, not Bank77's 2: CLINC150 prompts reach the max_input_tokens=4096 cap (Bank77's longest was 2911
# tokens). On two capped CLINC150 prompts (~3,700 tokens) the 4B peaked at 9.18 GB reserved at batch 1 (4.5 s/call),
# while batch 2 spilled out of the 12 GB card, which under WSL2 hangs instead of raising an OOM. The 1.7B at batch 4
# peaked at 6.78 GB reserved on the same prompts. Because batch sizes differ between models, per-call times mix
# model size with batching; #35 measures that separately.
set -euo pipefail

PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(lsa 0.6b 1.7b 4b)

mkdir -p results/clinc150_logs

# run_step NAME CONFIG [--set ...]: run one experiment, tee its log, print its run directory.
run_step() {
    local name=$1 config=$2
    shift 2
    echo "=== $name ==="
    "$PYTHON" scripts/run_offline.py "$config" --set "name=$name" "$@" 2>&1 | tee "results/clinc150_logs/$name.log"
    echo "Finished $name: $(grep '^Run directory:' "results/clinc150_logs/$name.log" | tail -1)"
}

run_slm() {
    run_step "clinc150_qwen3-$1" configs/clinc150_summarizers.yaml \
        --set "summarizer.model_id=Qwen/Qwen3-$2" --set "summarizer.batch_size=$3"
}

for step in "${STEPS[@]}"; do
    case $step in
        lsa) run_step clinc150_lsa configs/offline_knlpmeans.yaml --set dataset=clinc150_intent --set 'seeds=[0,1,2,3,4]' ;;
        0.6b) run_slm 0.6b 0.6B 8 ;;
        1.7b) run_slm 1.7b 1.7B 4 ;;
        4b) run_slm 4b 4B 1 ;;
        table) ;;
        *) echo "Unknown step '$step', expected lsa | 0.6b | 1.7b | 4b | table" >&2; exit 1 ;;
    esac
done

# Table over the newest *completed* run directory of each step (in a fixed order), whichever steps ran
# now or before. summary.csv is written only when a run finishes, so crashed runs are skipped.
RUN_DIRS=()
for name in clinc150_lsa clinc150_qwen3-0.6b clinc150_qwen3-1.7b clinc150_qwen3-4b; do
    latest=$(ls -d "results/$name"/*/summary.csv 2>/dev/null | sort | tail -1 || true)
    [ -n "$latest" ] && RUN_DIRS+=("$(dirname "$latest")")
done
[ ${#RUN_DIRS[@]} -gt 0 ] || { echo "No completed clinc150 runs found under results/" >&2; exit 1; }
printf 'Aggregating %s\n' "${RUN_DIRS[@]}"
"$PYTHON" scripts/aggregate_results.py "${RUN_DIRS[@]}" \
    --out-md results/clinc150_summarizers_table.md --out-csv results/clinc150_summarizers_table.csv \
    --per-seed-csv results/clinc150_summarizers_per_seed.csv
echo
cat results/clinc150_summarizers_table.md
