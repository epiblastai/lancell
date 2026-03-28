#!/usr/bin/env bash
# Run all unimodal gene-expression ingestion scripts against a shared atlas.
# Usage: ./run_gene_expression.sh /path/to/atlas [--limit N]
set -euo pipefail

ATLAS_PATH="${1:?Usage: $0 <atlas-path> [--limit N]}"
shift
EXTRA_ARGS=("$@")

SCRIPTS=(
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE90546
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE92872
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE120861
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE133344
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE135497
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE149383
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE161824
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE273677
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSM4150377
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSM4150378
    lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSM4150379
    # lancell_examples.multimodal_perturbation_atlas.scripts.ingest_figshare_20029387
)

MAX_PARALLEL=4
PIDS=()
FAILED=()

run_script() {
    local script="$1"
    local logfile="/tmp/ingest_${script##*.}.log"
    echo "[START] $script -> $logfile"
    python -m "$script" --atlas-path "$ATLAS_PATH" "${EXTRA_ARGS[@]}" \
        > "$logfile" 2>&1 \
        && echo "[DONE]  $script" \
        || { echo "[FAIL]  $script (see $logfile)"; FAILED+=("$script"); }
}

for script in "${SCRIPTS[@]}"; do
    # If we've hit the parallelism cap, wait for one to finish
    while (( ${#PIDS[@]} >= MAX_PARALLEL )); do
        wait -n || true
        # Prune finished PIDs
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
    done

    run_script "$script" &
    PIDS+=($!)
done

# Wait for all remaining
wait

echo ""
echo "============================="
if (( ${#FAILED[@]} > 0 )); then
    echo "FAILED (${#FAILED[@]}/${#SCRIPTS[@]}):"
    for s in "${FAILED[@]}"; do
        echo "  - $s"
    done
    exit 1
else
    echo "ALL ${#SCRIPTS[@]} scripts completed successfully."
fi
