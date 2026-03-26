"""Reconcile barcodes across modalities for multimodal GEO entries.

For multimodal entries (e.g., CITE-seq, NEAT-seq, Multiome), different modalities
may represent the same cells but with different barcode string formats:
  - GEX barcodes from CellRanger: ACGTACGT-1 (with well suffix)
  - ADT barcodes from CSV export: ACGTACGT (no suffix)
  - ATAC fragment barcodes: lane1#ACGTACGT-1 (with lane prefix)

This script detects the normalization needed, writes a `multimodal_barcode`
column to each feature space's preparer fragment CSV, and reports overlap statistics.

Usage:
    python scripts/reconcile_barcodes.py <experiment_dir>

Arguments:
    experiment_dir  Path to the experiment directory (e.g., /tmp/geo_agent/GSE264667/HepG2)

The script reads metadata.json from the experiment directory to find feature spaces
and source files, extracts barcodes from each modality, finds the best normalization,
and writes `multimodal_barcode` to {fs}_fragment_preparer_obs.csv for each
feature space.
"""

import json
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA barcode sequence (ignoring non-ACGT suffixes)."""
    # Split off any suffix like -1
    if "-" in seq:
        base, suffix = seq.rsplit("-", 1)
        return base.translate(COMPLEMENT)[::-1]
    return seq.translate(COMPLEMENT)[::-1]


# Each normalization returns a canonical form from a raw barcode.
# Order matters — we try the most common/cheapest first.
NORMALIZATIONS = [
    ("exact", lambda bc: bc),
    ("strip_suffix", lambda bc: bc.rsplit("-", 1)[0] if "-" in bc else bc),
    ("strip_prefix", lambda bc: bc.split("#", 1)[-1] if "#" in bc else bc),
    ("strip_both", lambda bc: (bc.split("#", 1)[-1]).rsplit("-", 1)[0]),
    ("reverse_complement", reverse_complement),
]


def extract_barcodes(data_dir: Path, meta: dict, feature_space: str, idx: int) -> set[str]:
    """Extract barcodes from a single modality's source file."""
    anndata_file = meta["anndata"][idx]
    matrix_file = meta["matrix_files"][idx]

    if feature_space == "chromatin_accessibility":
        frag_path = data_dir / matrix_file
        # Sample first 200K lines to get a representative barcode set
        chunks = pd.read_csv(
            frag_path,
            sep="\t",
            header=None,
            comment="#",
            usecols=[3],
            names=["barcode"],
            chunksize=200_000,
        )
        barcodes = set()
        for chunk in chunks:
            barcodes.update(chunk["barcode"].unique())
            if len(barcodes) > 50_000:
                break
        return barcodes

    if feature_space == "protein_abundance" and matrix_file:
        # ADT CSVs can be cells×proteins (barcodes as row index) or proteins×cells
        # (barcodes as column headers). Read a small sample to detect orientation.
        sep = "\t" if matrix_file.endswith(".tsv.gz") else ","
        adt_sample = pd.read_csv(data_dir / matrix_file, sep=sep, index_col=0, nrows=5)
        # Heuristic: if many more columns than rows in sample, it's proteins×cells
        # (few protein rows, many cell columns). Otherwise cells×proteins.
        if adt_sample.shape[1] > 100:
            # proteins×cells: barcodes are column headers
            return set(adt_sample.columns)
        else:
            # cells×proteins: barcodes are in the index; read just the index column
            adt_full_idx = pd.read_csv(data_dir / matrix_file, sep=sep, usecols=[0])
            return set(adt_full_idx.iloc[:, 0].tolist())

    if anndata_file:
        adata = ad.read_h5ad(data_dir / anndata_file, backed="r")
        barcodes = set(adata.obs.index)
        adata.file.close()
        return barcodes

    if matrix_file:
        matrix_path = data_dir / matrix_file
        if matrix_path.suffix == ".h5":
            adata = sc.read_10x_h5(matrix_path)
            return set(adata.obs.index)
        elif matrix_path.name.endswith((".tsv.gz", ".tsv", ".csv.gz", ".csv")):
            # Dense count matrix — detect orientation (genes×cells vs cells×genes)
            sep = "\t" if ".tsv" in matrix_path.name else ","
            sample = pd.read_csv(data_dir / matrix_file, sep=sep, index_col=0, nrows=5)
            if sample.shape[1] > 100:
                # genes×cells: barcodes are column headers
                return set(sample.columns)
            else:
                # cells×genes: barcodes are in the index; read just the index column
                full_idx = pd.read_csv(data_dir / matrix_file, sep=sep, usecols=[0])
                return set(full_idx.iloc[:, 0].tolist())
        elif "mtx" in matrix_path.name:
            # Non-standard mtx filenames — read barcodes from cell_metadata companion
            for cm_file in meta.get("cell_metadata", []):
                if "barcode" in cm_file.lower():
                    barcodes_df = pd.read_csv(data_dir / cm_file, sep="\t", header=None)
                    return set(barcodes_df[0].tolist())
            # Fallback: try standard naming in same directory
            adata = sc.read_10x_mtx(matrix_path.parent)
            return set(adata.obs.index)

    raise ValueError(
        f"Cannot extract barcodes for {feature_space} from {anndata_file or matrix_file}"
    )


def reconcile(barcode_sets: dict[str, set[str]]) -> tuple[dict[str, str], str]:
    """Find the normalization that maximizes barcode overlap across modalities.

    Args:
        barcode_sets: {feature_space: set_of_barcodes}

    Returns:
        (barcode_to_normalized, normalization_name) where barcode_to_normalized maps
        every raw barcode (from any modality) to its normalized form.
    """
    if len(barcode_sets) < 2:
        only_key = next(iter(barcode_sets))
        return {bc: bc for bc in barcode_sets[only_key]}, "single_modality"

    reference_key = next(iter(barcode_sets))
    reference_bcs = barcode_sets[reference_key]

    best_norm_fn = NORMALIZATIONS[0][1]  # default to identity
    best_overlap = 0
    best_name = "exact"

    for name, norm_fn in NORMALIZATIONS:
        ref_normalized = {norm_fn(bc) for bc in reference_bcs}
        min_overlap = float("inf")
        for key, bcs in barcode_sets.items():
            if key == reference_key:
                continue
            other_normalized = {norm_fn(bc) for bc in bcs}
            overlap = len(ref_normalized & other_normalized)
            min_overlap = min(min_overlap, overlap)
        if min_overlap > best_overlap:
            best_overlap = min_overlap
            best_norm_fn = norm_fn
            best_name = name

    barcode_to_normalized = {}
    for bcs in barcode_sets.values():
        for bc in bcs:
            barcode_to_normalized[bc] = best_norm_fn(bc)

    return barcode_to_normalized, best_name


def main(experiment_dir: str) -> None:
    experiment_dir = Path(experiment_dir)

    meta_path = experiment_dir / "metadata.json"
    assert meta_path.exists(), f"metadata.json not found in {experiment_dir}"
    with open(meta_path) as f:
        meta = json.load(f)

    feature_spaces = meta["feature_spaces"]

    if len(feature_spaces) < 2:
        print(
            f"{experiment_dir.name}: single modality ({feature_spaces[0]}), skipping reconciliation"
        )
        return

    # Extract barcodes from each modality
    barcode_sets = {}
    for i, fs in enumerate(feature_spaces):
        barcodes = extract_barcodes(experiment_dir, meta, fs, i)
        barcode_sets[fs] = barcodes
        print(f"  {fs}: {len(barcodes)} barcodes")

    # Find best normalization
    barcode_map, norm_name = reconcile(barcode_sets)

    # Report overlap
    normalized_sets = {}
    for fs, bcs in barcode_sets.items():
        normalized_sets[fs] = {barcode_map[bc] for bc in bcs}
    common = set.intersection(*normalized_sets.values())
    print(f"  normalization: {norm_name}")
    print(f"  common barcodes: {len(common)}")
    for fs, norm_bcs in normalized_sets.items():
        print(f"    {fs}: {len(norm_bcs)} unique, {len(norm_bcs - common)} unmatched")

    if len(common) < 0.5 * min(len(s) for s in normalized_sets.values()):
        print(f"  WARNING: <50% overlap — check file pairing for {experiment_dir.name}")

    # Write multimodal_barcode to each feature space's preparer fragment
    for fs in feature_spaces:
        fragment_path = experiment_dir / f"{fs}_fragment_preparer_obs.csv"
        assert fragment_path.exists(), f"{fragment_path} not found — write preparer fragment first"
        preparer_fragment = pd.read_csv(fragment_path, index_col=0)
        preparer_fragment["multimodal_barcode"] = [
            barcode_map.get(str(bc), str(bc)) for bc in preparer_fragment.index
        ]
        preparer_fragment.to_csv(fragment_path)
        print(f"  wrote multimodal_barcode to {fragment_path.name}")


if __name__ == "__main__":
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} <experiment_dir>"
    main(sys.argv[1])
