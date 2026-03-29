"""Ingest prepared GSE273677 (Nan et al. 2026) into a RaggedAtlas.

This dataset contains two Perturb-seq CRISPR screens in EndoC-betaH1
(pancreatic beta) cells:
  - GWAS   (41,642 cells × 36,601 genes, 7 replicates)
  - RQC    (20,385 cells × 36,601 genes, 2 replicates)

Each replicate is a 10x MTX bundle (gene×cell) that includes both
Gene Expression and CRISPR Guide Capture features. Only Gene Expression
features (first 36,601 rows) are ingested.

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/GSE273677/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE273677 \
        --atlas-path /tmp/atlas/GSE273677 [--limit 1000]
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
    GeneticPerturbationSchema,
    GenomicFeatureSchema,
    PublicationSchema,
    PublicationSectionSchema,
)

VALIDATE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "geo-data-curator" / "scripts" / "validate_obs.py"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION = "GSE273677"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/GSE273677")
FEATURE_SPACE = "gene_expression"

EXPERIMENTS = ["GWAS", "RQC"]

# Number of Gene Expression features in the MTX bundles (CRISPR guides follow)
N_GENE_FEATURES = 36601


# ---------------------------------------------------------------------------
# Step 1 & 2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs(experiment: str) -> Path:
    """Merge fragment CSVs into a standardized obs CSV.

    Barcodes are NOT unique across replicates, so fragments are merged
    positionally (by row order) rather than by index.
    """
    exp_dir = ACCESSION_DIR / experiment
    output_path = exp_dir / f"{FEATURE_SPACE}_standardized_obs.csv"
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping assembly")
        return output_path

    print(f"  Assembling obs for {experiment}...")

    # Load fragments positionally — all have same row count as raw_obs
    fragments = []
    for frag_type in ["ontology", "perturbation", "preparer"]:
        frag_path = exp_dir / f"{FEATURE_SPACE}_fragment_{frag_type}_obs.csv"
        if frag_path.exists():
            df = pd.read_csv(frag_path)
            # Drop barcode column if present (it's not a schema field; used only as index)
            if "barcode" in df.columns:
                df = df.drop(columns=["barcode"])
            fragments.append(df)
            print(f"    loaded {frag_path.name}: {list(df.columns)}")

    assembled = pd.concat(fragments, axis=1)
    # Handle duplicate columns — keep first occurrence (resolver overrides preparer)
    assembled = assembled.loc[:, ~assembled.columns.duplicated(keep="first")]
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
            "--column", "days_in_vitro=None",
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
# Step 3: Load expression matrices from MTX bundles
# ---------------------------------------------------------------------------


def discover_replicates(experiment: str) -> list[dict]:
    """Discover MTX bundles for an experiment, ordered by raw_obs GSM order.

    Returns list of dicts with keys: gsm_id, mtx_path, n_cells.
    """
    exp_dir = ACCESSION_DIR / experiment
    raw_obs = pd.read_csv(exp_dir / f"{FEATURE_SPACE}_raw_obs.csv", usecols=["gsm_id"])
    # Get ordered unique GSM IDs (preserving raw_obs order)
    gsm_ids = list(dict.fromkeys(raw_obs["gsm_id"]))

    replicates = []
    for gsm_id in gsm_ids:
        mtx_files = list(exp_dir.glob(f"{gsm_id}_*_matrix.mtx.gz"))
        if not mtx_files:
            raise FileNotFoundError(f"No MTX file found for {gsm_id} in {exp_dir}")
        n_cells = (raw_obs["gsm_id"] == gsm_id).sum()
        replicates.append({
            "gsm_id": gsm_id,
            "mtx_path": mtx_files[0],
            "n_cells": n_cells,
        })
    return replicates


def load_mtx_bundles(
    experiment: str,
    obs_parquet: Path,
    var_csv: Path,
    limit: int | None = None,
) -> ad.AnnData:
    """Load all replicate MTX bundles, concatenate, and attach obs/var.

    MTX files are gene×cell (36,826 features × n_cells). We take only
    the first N_GENE_FEATURES rows (Gene Expression) and transpose to
    cell×gene.
    """
    replicates = discover_replicates(experiment)

    print(f"  Loading {len(replicates)} replicate MTX bundles...")
    matrices = []
    for rep in replicates:
        mat = sio.mmread(str(rep["mtx_path"]))
        if sp.issparse(mat):
            mat = mat.tocsc()
        else:
            mat = sp.csc_matrix(mat)
        # Keep only Gene Expression features (first N_GENE_FEATURES rows)
        mat = mat[:N_GENE_FEATURES, :]
        # Transpose to cell × gene
        mat = mat.T.tocsr()
        assert mat.shape[0] == rep["n_cells"], (
            f"{rep['gsm_id']}: expected {rep['n_cells']} cells, got {mat.shape[0]}"
        )
        matrices.append(mat)
        print(f"    {rep['gsm_id']}: {mat.shape[0]:,} cells × {mat.shape[1]:,} genes")

    # Vertical stack (all have same gene set)
    combined = sp.vstack(matrices, format="csr")
    print(f"  Combined matrix: {combined.shape[0]:,} cells × {combined.shape[1]:,} genes")

    # Load obs and var
    obs_df = pd.read_parquet(obs_parquet)
    var_df = pd.read_csv(var_csv, index_col=0)

    assert combined.shape[0] == len(obs_df), (
        f"Matrix rows ({combined.shape[0]}) != obs rows ({len(obs_df)})"
    )
    assert combined.shape[1] == len(var_df), (
        f"Matrix cols ({combined.shape[1]}) != var rows ({len(var_df)})"
    )

    # Apply limit
    if limit is not None and limit < combined.shape[0]:
        print(f"  Limiting to {limit} cells (of {combined.shape[0]:,})")
        combined = combined[:limit]
        obs_df = obs_df.iloc[:limit]

    # Deduplicate features that share the same global_feature_uid (sum counts)
    combined, var_df = deduplicate_var(combined, var_df)
    print(f"  After dedup: {combined.shape[0]:,} cells × {combined.shape[1]:,} genes")

    adata = ad.AnnData(X=combined, obs=obs_df, var=var_df)
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    return adata



# ---------------------------------------------------------------------------
# Step 5: Populate foreign key tables
# ---------------------------------------------------------------------------


def populate_fk_tables(db_uri: str) -> str:
    """Create publication, publication_sections, and genetic_perturbation tables.

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

    # --- Genetic perturbations ---
    gp_parquet = ACCESSION_DIR / "GeneticPerturbationSchema.parquet"
    gp_df = pd.read_parquet(gp_parquet)
    if "genetic_perturbations" not in existing:
        gp_table = db.create_table(
            "genetic_perturbations",
            schema=GeneticPerturbationSchema.to_arrow_schema(),
        )
    else:
        gp_table = db.open_table("genetic_perturbations")
    gp_table.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(
            gp_df, schema=GeneticPerturbationSchema.to_arrow_schema()
        )
    )
    print(f"  Added {len(gp_df)} genetic perturbation record(s)")

    return publication_uid


# ---------------------------------------------------------------------------
# Step 6: Register features
# ---------------------------------------------------------------------------


def register_features(atlas: RaggedAtlas) -> None:
    """Register genomic features from the finalized parquet."""
    feature_parquet = ACCESSION_DIR / "GenomicFeatureSchema.parquet"
    feature_df = pd.read_parquet(feature_parquet)

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
    metadata: dict,
    limit: int | None = None,
) -> int:
    """Ingest one experiment into the atlas. Returns number of cells ingested."""
    exp_dir = ACCESSION_DIR / experiment
    validated_obs = exp_dir / f"{FEATURE_SPACE}_validated_obs.parquet"
    standardized_var = exp_dir / f"{FEATURE_SPACE}_standardized_var.csv"

    print(f"\n  Loading data for {experiment}...")
    adata = load_mtx_bundles(experiment, validated_obs, standardized_var, limit)

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
        accession_id=ACCESSION,
        dataset_description=metadata.get("summary"),
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
        description="Ingest GSE273677 into a RaggedAtlas"
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
        n, dataset_uid = ingest_experiment(atlas, exp, publication_uid, metadata, args.limit)
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
