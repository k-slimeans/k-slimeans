#!/usr/bin/env bash
# BBC News comparison (#18, seeds 0-4): k-means, k-NLPmeans (LSA) and k-SLMmeans with Qwen3-0.6B / 1.7B / 4B on
# 500 articles, in a TF-IDF space or with a dense MiniLM control, then one comparison table.
# See configs/bbc_tfidf.yaml and configs/bbc_dense.yaml for the protocol.
#
#   PYTHON=/path/to/venv/bin/python scripts/run_bbc.sh tfidf               # everything, TF-IDF space
#   PYTHON=... scripts/run_bbc.sh dense lsa 0.6b                            # only these steps, then the table
#   PYTHON=... scripts/run_bbc.sh tfidf table                               # only rebuild the table
#
# Variant: tfidf | dense. Steps: lsa | 0.6b | 1.7b | 4b | 0.6b-random | table (no run, just the table).
# 0.6b-random is the member-selection ablation: whole articles with the #28 random subset instead of each
# article's first 30 words; it is not part of the default steps. Logs go to results/bbc_logs/<name>.log.
# The table (results/bbc_<variant>_table.md + .csv, and the per-seed CSV) uses the newest completed run of every
# step that has one, so a failed step can be rerun on its own.
#
# batch_size per model as in scripts/run_clinc150.sh (8 / 4 / 1): the prompts can reach max_input_tokens=4096.
set -euo pipefail

PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

VARIANT=${1:-}
case $VARIANT in
    tfidf|dense) shift ;;
    *) echo "Usage: $0 tfidf|dense [lsa 0.6b 1.7b 4b 0.6b-random table]" >&2; exit 1 ;;
esac
CONFIG=configs/bbc_$VARIANT.yaml

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(lsa 0.6b 1.7b 4b)

mkdir -p results/bbc_logs

# run_step NAME [--set ...]: run one experiment, tee its log, print its run directory.
run_step() {
    local name=$1
    shift
    echo "=== $name ==="
    "$PYTHON" scripts/run_offline.py "$CONFIG" --set "name=$name" "$@" 2>&1 | tee "results/bbc_logs/$name.log"
    echo "Finished $name: $(grep '^Run directory:' "results/bbc_logs/$name.log" | tail -1)"
}

run_slm() {
    run_step "bbc_${VARIANT}_qwen3-$1" --set "summarizer.model_id=Qwen/Qwen3-$2" --set "summarizer.batch_size=$3"
}

for step in "${STEPS[@]}"; do
    case $step in
        lsa) run_step "bbc_${VARIANT}_lsa" --set method_name=k-NLPmeans --set 'summarizer={type: lsa, top_k: 5}' ;;
        0.6b) run_slm 0.6b 0.6B 8 ;;
        1.7b) run_slm 1.7b 1.7B 4 ;;
        4b) run_slm 4b 4B 1 ;;
        0.6b-random)
            run_step "bbc_${VARIANT}_qwen3-0.6b-random" --set summarizer.max_words_per_text=null --set method_name=k-SLMmeans-random \
                --set summarizer.model_id=Qwen/Qwen3-0.6B --set summarizer.batch_size=8
            ;;
        table) ;;
        *) echo "Unknown step '$step', expected lsa | 0.6b | 1.7b | 4b | 0.6b-random | table" >&2; exit 1 ;;
    esac
done

# Table over the newest *completed* run directory of each step (in a fixed order). summary.csv is written
# only when a run finishes, so crashed runs are skipped.
RUN_DIRS=()
for name in lsa qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-0.6b-random; do
    latest=$(ls -d "results/bbc_${VARIANT}_$name"/*/summary.csv 2>/dev/null | sort | tail -1 || true)
    [ -n "$latest" ] && RUN_DIRS+=("$(dirname "$latest")")
done
[ ${#RUN_DIRS[@]} -gt 0 ] || { echo "No completed bbc_$VARIANT runs found under results/" >&2; exit 1; }
printf 'Aggregating %s\n' "${RUN_DIRS[@]}"
"$PYTHON" scripts/aggregate_results.py "${RUN_DIRS[@]}" \
    --out-md "results/bbc_${VARIANT}_table.md" --out-csv "results/bbc_${VARIANT}_table.csv" \
    --per-seed-csv "results/bbc_${VARIANT}_per_seed.csv"
echo
cat "results/bbc_${VARIANT}_table.md"
