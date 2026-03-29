"""Ingest prepared GSE135497 (Schraivogel et al. 2020) into a RaggedAtlas.

TAP-seq dataset with four CRISPR-based experiments profiling gene expression
in K562 cells:
  - TAP_DIFFEX    (21,977 cells × 74 genes)
  - WTX_DIFFEX    (38,024 cells × 17,192 genes)
  - SCREEN_chr8   (112,260 cells × 4,191 genes)
  - SCREEN_chr11  (120,310 cells × 3,185 genes)

Data format: gene×cell CSV files (one per sample), transposed and concatenated
at load time.

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/GSE135497/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE135497 \
        --atlas-path /tmp/atlas/GSE135497 [--limit 1000]
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

ACCESSION = "GSE135497"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/GSE135497")
FEATURE_SPACE = "gene_expression"

EXPERIMENTS = ["TAP_DIFFEX", "WTX_DIFFEX", "SCREEN_chr8", "SCREEN_chr11"]


# ---------------------------------------------------------------------------
# Step 1 & 2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs(experiment: str) -> Path:
    """Merge fragment CSVs into a standardized obs CSV."""
    exp_dir = ACCESSION_DIR / experiment
    output_path = exp_dir / f"{FEATURE_SPACE}_standardized_obs.csv"
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping assembly")
        return output_path

    print(f"  Assembling obs for {experiment}...")

    # Load cell_barcode-indexed fragments
    fragments = []
    for frag_type in ["ontology", "perturbation", "preparer"]:
        frag_path = exp_dir / f"{FEATURE_SPACE}_fragment_{frag_type}_obs.csv"
        if frag_path.exists():
            frag_df = pd.read_csv(frag_path, index_col=0)
            if not frag_df.empty:
                fragments.append(frag_df)
                print(f"    loaded {frag_path.name}: {len(frag_df.columns)} columns")

    assembled = pd.concat(fragments, axis=1)
    # Drop duplicate columns (keep first occurrence)
    assembled = assembled.loc[:, ~assembled.columns.duplicated()]
    assembled.index.name = "cell_barcode"
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
            "--column", "replicate=None",
            "--column", "well_position=None",
            "--column", "additional_metadata=None",
        ],
        check=True,
    )
    return validated_obs


def assemble_and_validate(experiment: str) -> None:
    """Assemble fragment CSVs and validate obs for one experiment."""
    assemble_obs(experiment)
    validate_obs(experiment)


# ---------------------------------------------------------------------------
# Step 3: Load expression matrices from per-sample gene×cell CSVs
# ---------------------------------------------------------------------------


def load_csv_matrices_as_anndata(
    experiment: str,
    obs_parquet: Path,
    var_csv: Path,
    limit: int | None = None,
) -> ad.AnnData:
    """Load gene×cell CSV files for all samples, transpose and concatenate.

    The CSV files have genes as rows and cells as columns. We read them,
    transpose to cell×gene, concatenate across samples, and reorder genes
    to match the standardized var order.
    """
    exp_dir = ACCESSION_DIR / experiment
    count_files = sorted(exp_dir.glob("*.counts.csv.gz"))
    if not count_files:
        raise FileNotFoundError(f"No .counts.csv.gz files in {exp_dir}")

    print(f"  Loading {len(count_files)} sample CSV files...")

    # Load standardized var for gene ordering and global_feature_uid
    std_var = pd.read_csv(var_csv)
    target_genes = std_var["gene_name"].tolist()

    # Read and concatenate all sample matrices
    sample_dfs = []
    for f in count_files:
        df = pd.read_csv(f, index_col=0)  # genes × cells
        sample_dfs.append(df)
        print(f"    {f.name}: {df.shape[1]} cells × {df.shape[0]} genes")

    # Concatenate along columns (cells)
    combined = pd.concat(sample_dfs, axis=1)
    print(f"  Combined: {combined.shape[1]} cells × {combined.shape[0]} genes")

    # Reorder genes to match standardized_var
    combined = combined.loc[target_genes]

    # Transpose to cell × gene
    mat = sp.csr_matrix(combined.values.T)

    # Build var DataFrame
    var_df = std_var.set_index("gene_name")

    # Deduplicate features with the same global_feature_uid (sum counts)
    mat, var_df = deduplicate_var(mat, var_df)

    # Load validated obs
    obs_df = pd.read_parquet(obs_parquet)

    # Verify dimensions
    assert mat.shape[0] == len(obs_df), (
        f"Matrix rows ({mat.shape[0]}) != obs rows ({len(obs_df)})"
    )
    assert mat.shape[1] == len(var_df), (
        f"Matrix cols ({mat.shape[1]}) != var rows ({len(var_df)})"
    )

    # Apply limit
    if limit is not None and limit < mat.shape[0]:
        print(f"  Limiting to {limit} cells (of {mat.shape[0]})")
        mat = mat[:limit]
        obs_df = obs_df.iloc[:limit]

    # Build AnnData
    adata = ad.AnnData(X=mat, obs=obs_df, var=var_df)
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
    adata = load_csv_matrices_as_anndata(experiment, validated_obs, standardized_var, limit)

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
        description="Ingest GSE135497 into a RaggedAtlas"
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

    # Step 1-2: Assemble fragments and validate obs
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

    # Summary
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
