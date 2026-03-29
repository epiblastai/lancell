"""Ingest prepared GSE149383 (drug resistance scRNA-seq SuperSeries) into a RaggedAtlas.

This SuperSeries contains seven sub-series profiling drug-tolerant cancer cell
lines (PC9, M14, A549) treated with various small-molecule inhibitors:

  - GSE134836  PC9 + erlotinib, 10x Genomics (2 GSMs, ~24.5k cells)
  - GSE134838  M14 + vemurafenib, Drop-seq  (2 GSMs, ~12.3k cells)
  - GSE134839  PC9 + erlotinib time-course, Drop-seq (6 GSMs, ~6.5k cells)
  - GSE134841  PC9 + erlotinib long time-course, Drop-seq (5 GSMs, ~28.7k cells)
  - GSE149214  PC9 + erlotinib, Drop-seq  (3 GSMs, ~26.6k cells)
  - GSE149215  PC9 + erlotinib/crizotinib/etoposide, Drop-seq (5 GSMs, ~37.4k cells)
  - GSE160244  PC9 + osimertinib/crizotinib, Drop-seq (4 GSMs, ~119k cells)

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/GSE149383/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE149383 \
        --atlas-path /tmp/atlas/GSE149383 [--limit 1000]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.io as sio
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas, create_or_open_atlas
from lancell.ingestion import add_anndata_batch, add_csc, deduplicate_var
from lancell.schema import make_uid

from lancell_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    REGISTRY_SCHEMAS,
    GenomicFeatureSchema,
    PublicationSchema,
    PublicationSectionSchema,
    SmallMoleculeSchema,
)

VALIDATE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "geo-data-curator" / "scripts" / "validate_obs.py"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION = "GSE149383"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/GSE149383")
FEATURE_SPACE = "gene_expression"

# Sub-series and their data format
EXPERIMENTS = {
    "GSE134836": "10x",
    "GSE134838": "dge",
    "GSE134839": "dge",
    "GSE134841": "dge",
    "GSE149214": "dge",
    "GSE149215": "dge",
    "GSE160244": "dge",
}

# GSE134836 has 10x bundles extracted into named subdirectories.
# Map GSM IDs to their 10x bundle directories.
_10X_BUNDLES = {
    "GSM3972651": "PC9D0_10x",
    "GSM3972652": "PC9D3Erl_10x",
}


# ---------------------------------------------------------------------------
# Step 1 & 2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs(experiment: str) -> Path:
    """Merge fragment CSVs into a standardized obs CSV.

    Fragments are positionally aligned with raw_obs. The preparer fragment
    is indexed by batch_id (non-unique), and barcodes may be shared across
    GSMs, so we use positional concatenation (reset_index) rather than
    index-based joining.

    The molecule fragment uses ``|SmallMolecule`` suffixed columns.
    Since this dataset only has small molecule perturbations, we strip
    the suffix to produce the final schema column names.
    """
    exp_dir = ACCESSION_DIR / experiment
    output_path = exp_dir / f"{FEATURE_SPACE}_standardized_obs.csv"
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping assembly")
        return output_path

    print(f"  Assembling obs for {experiment}...")

    # Load all fragments positionally (drop their index columns)
    ont = pd.read_csv(exp_dir / f"{FEATURE_SPACE}_fragment_ontology_obs.csv")
    mol = pd.read_csv(exp_dir / f"{FEATURE_SPACE}_fragment_molecule_obs.csv")
    prep = pd.read_csv(exp_dir / f"{FEATURE_SPACE}_fragment_preparer_obs.csv")

    # Drop the barcode/batch_id index columns (they're not schema fields)
    ont = ont.drop(columns=["barcode"], errors="ignore")
    mol = mol.drop(columns=["barcode"], errors="ignore")
    prep = prep.drop(columns=["batch_id"], errors="ignore")

    # Strip |SmallMolecule suffix from molecule fragment columns
    mol.columns = [c.split("|")[0] for c in mol.columns]

    # Drop ontology_resolved flag (not a schema field)
    ont = ont.drop(columns=["ontology_resolved"], errors="ignore")

    # Combine positionally
    assembled = pd.concat([ont, mol, prep], axis=1)

    # Handle overlapping columns: ontology provides cell_line, molecule
    # provides is_negative_control/negative_control_type. If both have
    # the same column, keep the resolver (molecule/ontology) version.
    # pandas concat will create duplicates; drop_duplicates on columns
    assembled = assembled.loc[:, ~assembled.columns.duplicated(keep="last")]

    assembled.index.name = "cell_index"
    assembled.to_csv(output_path)
    print(f"    wrote {output_path.name}: {len(assembled.columns)} cols, {len(assembled)} rows")
    return output_path


def validate_obs(experiment: str) -> Path:
    """Run validate_obs.py to coerce types and strip non-schema columns."""
    exp_dir = ACCESSION_DIR / experiment
    standardized_obs = exp_dir / f"{FEATURE_SPACE}_standardized_obs.csv"
    validated_obs = exp_dir / f"{FEATURE_SPACE}_validated_obs.parquet"

    if validated_obs.exists():
        print(f"  {validated_obs.name} already exists, skipping validation")
        return validated_obs

    print(f"  Validating obs for {experiment}...")
    subprocess.run(
        [
            sys.executable, str(VALIDATE_SCRIPT),
            str(standardized_obs),
            str(validated_obs),
            "lancell_examples.multimodal_perturbation_atlas.schema",
            "CellIndex",
            "--column", "cell_type=None",
            "--column", "tissue=None",
            "--column", "development_stage=None",
            "--column", "disease=None",
            "--column", "donor_uid=None",
            "--column", "replicate=None",
            "--column", "well_position=None",
        ],
        check=True,
    )
    return validated_obs


def assemble_and_validate(experiment: str) -> None:
    """Assemble fragment CSVs and validate obs for one experiment."""
    assemble_obs(experiment)
    validate_obs(experiment)


# ---------------------------------------------------------------------------
# Step 3: Load expression matrices
# ---------------------------------------------------------------------------


def _load_10x_experiment(
    experiment: str,
    obs_parquet: Path,
    var_csv: Path,
    limit: int | None = None,
) -> ad.AnnData:
    """Load a 10x-based experiment (GSE134836) from extracted MTX bundles.

    Each GSM has its own bundle directory. Reads each, aligns features
    to the standardized var, and vertically stacks.
    """
    exp_dir = ACCESSION_DIR / experiment
    obs_df = pd.read_parquet(obs_parquet)
    var_df = pd.read_csv(var_csv, index_col=0)
    n_genes = len(var_df)
    gene_to_idx = {g: i for i, g in enumerate(var_df.index)}

    # Read raw_obs to get barcode → gsm_id mapping
    raw_obs = pd.read_csv(exp_dir / f"{FEATURE_SPACE}_raw_obs.csv")
    gsm_order = raw_obs["gsm_id"].unique()

    matrices = []
    for gsm_id in gsm_order:
        bundle_dir = exp_dir / _10X_BUNDLES[gsm_id]
        print(f"    Loading 10x bundle: {bundle_dir.name}")

        mat = sio.mmread(str(bundle_dir / "matrix.mtx.gz"))
        if sp.issparse(mat):
            mat = mat.T.tocsr()
        else:
            mat = sp.csr_matrix(mat.T)

        # Read features to build gene name mapping
        features = pd.read_csv(
            bundle_dir / "features.tsv.gz",
            sep="\t", header=None, names=["id", "name", "type"],
        )

        # Remap columns to standardized var order
        col_remap = []
        keep_mask = np.zeros(features.shape[0], dtype=bool)
        for i, gene_name in enumerate(features["name"]):
            if gene_name in gene_to_idx:
                col_remap.append(gene_to_idx[gene_name])
                keep_mask[i] = True
            else:
                keep_mask[i] = False

        # Build reindexed sparse matrix
        mat_filtered = mat[:, keep_mask]
        col_indices = np.array(col_remap, dtype=np.int32)

        # Create a matrix with the full gene count
        n_cells = mat_filtered.shape[0]
        coo = mat_filtered.tocoo()
        remapped_cols = col_indices[coo.col]
        aligned = sp.csr_matrix(
            (coo.data, (coo.row, remapped_cols)),
            shape=(n_cells, n_genes),
        )
        matrices.append(aligned)
        print(f"      {n_cells} cells, {keep_mask.sum()}/{len(features)} genes mapped")

    combined = sp.vstack(matrices, format="csr")
    print(f"    Combined matrix: {combined.shape}")

    assert combined.shape[0] == len(obs_df), (
        f"Matrix rows ({combined.shape[0]}) != obs rows ({len(obs_df)})"
    )

    if limit is not None and limit < combined.shape[0]:
        print(f"    Limiting to {limit} cells (of {combined.shape[0]})")
        combined = combined[:limit]
        obs_df = obs_df.iloc[:limit]

    # Deduplicate var (multiple raw features may map to the same global_feature_uid)
    combined, var_df = deduplicate_var(combined, var_df)

    adata = ad.AnnData(X=combined, obs=obs_df, var=var_df)
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    return adata


def _load_dge_experiment(
    experiment: str,
    obs_parquet: Path,
    var_csv: Path,
    limit: int | None = None,
) -> ad.AnnData:
    """Load a DGE-based experiment from gzipped gene×cell text files.

    Each GSM has a DGE file with genes as rows and cell barcodes as columns.
    Reads each, transposes, aligns to standardized var, and vstacks.
    """
    exp_dir = ACCESSION_DIR / experiment
    obs_df = pd.read_parquet(obs_parquet)
    var_df = pd.read_csv(var_csv, index_col=0)
    n_genes = len(var_df)
    gene_to_idx = {g: i for i, g in enumerate(var_df.index)}

    # Read raw_obs to get barcode → gsm_id mapping and barcode order
    raw_obs = pd.read_csv(exp_dir / f"{FEATURE_SPACE}_raw_obs.csv")
    gsm_order = raw_obs["gsm_id"].unique()

    # Build a map from GSM ID to its DGE file
    dge_files = sorted(exp_dir.glob("GSM*.dge.txt.gz")) + sorted(
        exp_dir.glob("GSM*_expression_matrix.txt.gz")
    )
    gsm_to_file = {}
    for f in dge_files:
        # Extract GSM ID from filename (e.g., GSM3972655_M14Day0.dge.txt.gz)
        gsm = f.name.split("_")[0]
        gsm_to_file[gsm] = f

    matrices = []
    for gsm_id in gsm_order:
        dge_path = gsm_to_file[gsm_id]
        print(f"    Loading DGE: {dge_path.name}")

        # Read DGE: first column is GENE, rest are cell barcodes
        dge = pd.read_csv(dge_path, sep="\t", index_col=0)
        # dge is gene × cell; we need cell × gene
        gene_names = dge.index.values
        n_cells_gsm = dge.shape[1]

        # Map genes to standardized var positions
        keep_mask = np.array([g in gene_to_idx for g in gene_names])
        col_remap = np.array([gene_to_idx[g] for g in gene_names if g in gene_to_idx])

        # Filter and transpose
        dge_filtered = dge.values[keep_mask, :]  # (n_genes_kept, n_cells)
        # Create sparse cell × gene matrix
        dge_sparse = sp.csc_matrix(dge_filtered)  # gene × cell as CSC
        dge_t = dge_sparse.T.tocoo()  # cell × gene_filtered

        # Remap to full gene space
        remapped_cols = col_remap[dge_t.col]
        aligned = sp.csr_matrix(
            (dge_t.data, (dge_t.row, remapped_cols)),
            shape=(n_cells_gsm, n_genes),
        )
        matrices.append(aligned)
        print(f"      {n_cells_gsm} cells, {keep_mask.sum()}/{len(gene_names)} genes mapped")

    combined = sp.vstack(matrices, format="csr")
    print(f"    Combined matrix: {combined.shape}")

    assert combined.shape[0] == len(obs_df), (
        f"Matrix rows ({combined.shape[0]}) != obs rows ({len(obs_df)})"
    )

    # Deduplicate var (multiple raw features may map to the same global_feature_uid)
    combined, var_df = deduplicate_var(combined, var_df)

    if limit is not None and limit < combined.shape[0]:
        print(f"    Limiting to {limit} cells (of {combined.shape[0]})")
        combined = combined[:limit]
        obs_df = obs_df.iloc[:limit]

    adata = ad.AnnData(X=combined, obs=obs_df, var=var_df)
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    return adata


def load_experiment(
    experiment: str,
    obs_parquet: Path,
    var_csv: Path,
    limit: int | None = None,
) -> ad.AnnData:
    """Load an experiment's expression matrix as AnnData."""
    fmt = EXPERIMENTS[experiment]
    if fmt == "10x":
        return _load_10x_experiment(experiment, obs_parquet, var_csv, limit)
    else:
        return _load_dge_experiment(experiment, obs_parquet, var_csv, limit)


# ---------------------------------------------------------------------------
# Step 5: Populate foreign key tables
# ---------------------------------------------------------------------------


def populate_fk_tables(db_uri: str) -> str:
    """Create publication, publication_sections, and small_molecule tables.

    Returns the publication_uid for use in DatasetSchema.
    """
    db = lancedb.connect(db_uri)
    existing = db.list_tables().tables

    # --- Publications ---
    pub_parquet = ACCESSION_DIR / "PublicationSchema.parquet"
    pub_df = pd.read_parquet(pub_parquet)
    publication_uid = pub_df["uid"].iloc[0]
    print(f"  Publication UID: {publication_uid}")

    if "publications" not in existing:
        pub_table = db.create_table(
            "publications", schema=PublicationSchema.to_arrow_schema()
        )
    else:
        pub_table = db.open_table("publications")
    pub_table.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(pub_df, schema=PublicationSchema.to_arrow_schema())
    )
    print(f"  Added {len(pub_df)} publication record(s)")

    # --- Publication sections ---
    section_parquet = ACCESSION_DIR / "PublicationSectionSchema.parquet"
    if section_parquet.exists():
        section_df = pd.read_parquet(section_parquet)
        if "publication_sections" not in existing:
            sec_table = db.create_table(
                "publication_sections",
                schema=PublicationSectionSchema.to_arrow_schema(),
            )
            sec_table.add(
                pa.Table.from_pandas(
                    section_df, schema=PublicationSectionSchema.to_arrow_schema()
                )
            )
            print(f"  Added {len(section_df)} publication section(s)")
        else:
            sec_table = db.open_table("publication_sections")
            existing_pubs = set(
                sec_table.search()
                .select(["publication_uid"])
                .to_pandas()["publication_uid"]
            )
            new_sections = section_df[~section_df["publication_uid"].isin(existing_pubs)]
            if not new_sections.empty:
                sec_table.add(
                    pa.Table.from_pandas(
                        new_sections, schema=PublicationSectionSchema.to_arrow_schema()
                    )
                )
            print(f"  Added {len(new_sections)} publication section(s) (skipped {len(section_df) - len(new_sections)} existing)")

    # --- Small molecules ---
    sm_parquet = ACCESSION_DIR / "SmallMoleculeSchema.parquet"
    sm_df = pd.read_parquet(sm_parquet)
    if "small_molecules" not in existing:
        sm_table = db.create_table(
            "small_molecules",
            schema=SmallMoleculeSchema.to_arrow_schema(),
        )
    else:
        sm_table = db.open_table("small_molecules")
    sm_table.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(
            sm_df, schema=SmallMoleculeSchema.to_arrow_schema()
        )
    )
    print(f"  Added {len(sm_df)} small molecule record(s)")

    return publication_uid


# ---------------------------------------------------------------------------
# Step 6: Register features
# ---------------------------------------------------------------------------


def register_features(atlas: RaggedAtlas) -> None:
    """Register genomic features from the finalized parquet."""
    feature_parquet = ACCESSION_DIR / "GenomicFeatureSchema.parquet"
    feature_df = pd.read_parquet(feature_parquet)

    # Fill null feature_id with gene_name (genes without Ensembl IDs)
    mask = feature_df["feature_id"].isna()
    if mask.any():
        feature_df.loc[mask, "feature_id"] = feature_df.loc[mask, "gene_name"]
        print(f"  Filled {mask.sum()} null feature_id values with gene_name")

    records = [
        GenomicFeatureSchema(**row.to_dict()) for _, row in feature_df.iterrows()
    ]
    n_new = atlas.register_features("gene_expression", records)
    print(f"  Registered {n_new} new features ({len(records)} total)")


# ---------------------------------------------------------------------------
# Step 7: Ingest per-experiment data
# ---------------------------------------------------------------------------


def ingest_experiment(
    atlas: RaggedAtlas,
    experiment: str,
    publication_uid: str,
    limit: int | None = None,
) -> int:
    """Ingest one experiment into the atlas. Returns number of cells ingested."""
    exp_dir = ACCESSION_DIR / experiment
    validated_obs = exp_dir / f"{FEATURE_SPACE}_validated_obs.parquet"
    standardized_var = exp_dir / f"{FEATURE_SPACE}_standardized_var.csv"

    # Load experiment metadata
    meta_path = exp_dir / f"{experiment}_metadata.json"
    with open(meta_path) as f:
        exp_meta = json.load(f)

    print(f"\n  Loading data for {experiment}...")
    adata = load_experiment(experiment, validated_obs, standardized_var, limit)

    dataset_uid = make_uid()

    def _unique_non_null(col: str) -> list[str] | None:
        if col not in adata.obs.columns:
            return None
        vals = adata.obs[col].dropna().unique().tolist()
        return sorted(vals) if vals else None

    dataset_record = DatasetSchema(
        uid=dataset_uid,
        zarr_group=dataset_uid,
        feature_space=FEATURE_SPACE,
        n_cells=adata.n_obs,
        publication_uid=publication_uid,
        accession_database="GEO",
        accession_id=experiment,
        dataset_description=exp_meta.get("summary"),
        organism=_unique_non_null("organism"),
        tissue=_unique_non_null("tissue"),
        cell_line=_unique_non_null("cell_line"),
        disease=_unique_non_null("disease"),
    )

    print(f"  Ingesting {adata.n_obs:,} cells × {adata.n_vars:,} genes...")
    n_ingested = add_anndata_batch(
        atlas,
        adata,
        feature_space=FEATURE_SPACE,
        zarr_layer="counts",
        dataset_record=dataset_record,
    )
    print(f"  Ingested {n_ingested:,} cells for {experiment} (dataset_uid={dataset_uid})")
    return n_ingested, dataset_uid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest GSE149383 into a RaggedAtlas"
    )
    parser.add_argument(
        "--atlas-path",
        type=str,
        required=True,
        help="Directory for the atlas (created if it doesn't exist)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cells to ingest per experiment (for testing)",
    )
    args = parser.parse_args()

    atlas_path = Path(args.atlas_path)

    # Load metadata
    metadata_path = ACCESSION_DIR / f"{ACCESSION}_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"Dataset: {metadata['title']}")
    print(f"Accession: {ACCESSION}")
    print(f"Atlas path: {atlas_path}")
    if args.limit:
        print(f"Cell limit per experiment: {args.limit:,}")

    # Step 1-2: Assemble fragments and validate obs for all experiments
    print(f"\n{'='*60}")
    print("Step 1-2: Assemble fragments & validate obs")
    print(f"{'='*60}")
    for exp in EXPERIMENTS:
        assemble_and_validate(exp)

    # Step 3: Create or open atlas
    print(f"\n{'='*60}")
    print("Step 3: Create or open atlas")
    print(f"{'='*60}")
    atlas = create_or_open_atlas(
        str(atlas_path),
        cell_table_name="cells",
        cell_schema=CellIndex,
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas=REGISTRY_SCHEMAS,
    )

    # Step 4: Populate FK tables
    print(f"\n{'='*60}")
    print("Step 4: Populate foreign key tables")
    print(f"{'='*60}")
    db_uri = str(atlas_path / "lance_db")
    publication_uid = populate_fk_tables(db_uri)

    # Step 5: Register features
    print(f"\n{'='*60}")
    print("Step 5: Register features")
    print(f"{'='*60}")
    register_features(atlas)

    # Step 6: Ingest experiments
    print(f"\n{'='*60}")
    print("Step 6: Ingest experiments")
    print(f"{'='*60}")
    total_cells = 0
    dataset_uids = []
    for exp in EXPERIMENTS:
        n, dataset_uid = ingest_experiment(atlas, exp, publication_uid, args.limit)
        total_cells += n
        dataset_uids.append(dataset_uid)

    # Build CSC arrays for feature-filtered queries
    print(f"\n{'='*60}")
    print("Building CSC arrays")
    print(f"{'='*60}")
    for dataset_uid in dataset_uids:
        print(f"  Building CSC for {dataset_uid}...")
        add_csc(atlas, zarr_group=dataset_uid, feature_space=FEATURE_SPACE)
    print("  Done.")

    # Step 7: Summary
    print(f"\n{'='*60}")
    print("Ingestion complete")
    print(f"{'='*60}")
    print(f"  Accession: {ACCESSION}")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"  Total cells ingested: {total_cells:,}")
    print(f"  Feature space: {FEATURE_SPACE}")
    print(f"  Atlas path: {atlas_path}")


if __name__ == "__main__":
    main()
