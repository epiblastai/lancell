#!/usr/bin/env bash
set -euo pipefail

DATASETS_DIR="${HOME}/datasets/cellxgene_census"
ATLAS_DIR="s3://epiblast/ragged_atlases/cellxgene_mini_bp"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_DIR"

h5ad_files=("$DATASETS_DIR"/*.h5ad)
total=${#h5ad_files[@]}
echo "Found $total h5ad files in $DATASETS_DIR"

for i in "${!h5ad_files[@]}"; do
    h5ad="${h5ad_files[$i]}"
    echo ""
    echo "[$(( i + 1 ))/$total] Ingesting: $(basename "$h5ad")"
    python -m lancell_examples.cellxgene_census.ingest \
        --h5ad "$h5ad" \
        --atlas-dir "$ATLAS_DIR"
done

echo ""
echo "All $total files ingested."
