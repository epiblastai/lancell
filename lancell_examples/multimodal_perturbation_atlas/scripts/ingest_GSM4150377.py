"""Ingest prepared GSM4150377 (Srivatsan et al. 2019) into a RaggedAtlas.

sciPlex2 dataset: single-nucleus RNA-seq of A549 lung adenocarcinoma cells
treated with small molecule transcription modulators (BMS-345541,
dexamethasone, nutlin-3A, SAHA) or DMSO vehicle control.

Data format: cell-sorted COO triplet matrix (gene_idx, cell_idx, count),
gzipped, 1-indexed, tab-separated. 24,262 cells × 58,347 genes (56 genes
share global_feature_uid with another gene, so deduplication sums counts
for those pairs).

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/GSM4150377/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSM4150377 \
        --atlas-path /tmp/atlas/GSM4150377 [--limit 1000]
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
    GenomicFeatureSchema,
    PublicationSchema,
    PublicationSectionSchema,
    SmallMoleculeSchema,
)

ASSEMBLE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "geo-data-curator" / "scripts" / "assemble_fragments.py"
)
VALIDATE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "geo-data-curator" / "scripts" / "validate_obs.py"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION = "GSM4150377"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/GSM4150377")
FEATURE_SPACE = "gene_expression"
SCHEMA_FILE = (
    Path(__file__).resolve().parents[1] / "schema.py"
)

EXPERIMENTS = ["A549_Transcription_Modulators"]

COO_FILENAME = (
    "GSM4150377_sciPlex2_A549_Transcription_Modulators_UMI.count.matrix.gz"
)


# ---------------------------------------------------------------------------
# Step 1-2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs(experiment: str) -> Path:
    """Merge fragment CSVs into a standardized obs CSV using assemble_fragments.py."""
    exp_dir = ACCESSION_DIR / experiment
    output_path = exp_dir / f"{FEATURE_SPACE}_standardized_obs.csv"
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping assembly")
        return output_path

    print(f"  Assembling obs for {experiment}...")
    subprocess.run(
        [
            sys.executable, str(ASSEMBLE_SCRIPT),
            str(exp_dir),
            "--schema", str(SCHEMA_FILE),
        ],
        check=True,
    )
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
        ],
        check=True,
    )
    return validated_obs


def assemble_and_validate(experiment: str) -> None:
    """Assemble fragment CSVs and validate obs for one experiment."""
    assemble_obs(experiment)
    validate_obs(experiment)



# ---------------------------------------------------------------------------
# Step 4: Populate foreign key tables
# ---------------------------------------------------------------------------


def populate_fk_tables(db_uri: str) -> str:
    """Create publication, publication_sections, and small_molecules tables.

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
# Step 5: Register features
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
# Step 6: Load COO matrix and ingest via AnnData path
# ---------------------------------------------------------------------------


def load_coo_as_anndata(
    experiment: str,
    obs_parquet: Path,
    var_csv: Path,
    limit: int | None = None,
) -> ad.AnnData:
    """Load a gzipped COO triplet matrix and build an AnnData object.

    The COO file has columns: gene_idx, cell_idx, count (1-indexed, tab-sep).
    Features with duplicate global_feature_uid are summed via deduplicate_var.
    """
    exp_dir = ACCESSION_DIR / experiment
    coo_path = exp_dir / COO_FILENAME

    # Read standardized var for global_feature_uid mapping
    std_var = pd.read_csv(str(var_csv))
    n_features = len(std_var)

    # Load validated obs
    obs_df = pd.read_parquet(obs_parquet)
    n_cells = len(obs_df)

    print(f"  Loading COO matrix from {coo_path.name}...")
    import subprocess
    proc = subprocess.Popen(
        ["gzip", "-dc", str(coo_path)], stdout=subprocess.PIPE
    )
    import polars as pl
    triplets = pl.read_csv(
        proc.stdout,
        has_header=False,
        separator="\t",
        schema_overrides={
            "column_1": pl.Int32,
            "column_2": pl.Int32,
            "column_3": pl.Int32,
        },
    )
    proc.wait()

    gene_idx = triplets["column_1"].to_numpy() - 1  # 1-indexed → 0-indexed
    cell_idx = triplets["column_2"].to_numpy() - 1
    values = triplets["column_3"].to_numpy()
    del triplets

    mat = sp.csr_matrix(
        (values, (cell_idx, gene_idx)),
        shape=(n_cells, n_features),
    )
    del gene_idx, cell_idx, values
    print(f"  Matrix: {mat.shape[0]:,} × {mat.shape[1]:,}, {mat.nnz:,} nonzeros")

    # Build var DataFrame with global_feature_uid
    var_df = std_var.set_index(std_var.columns[0])  # index by ensembl_id

    # Deduplicate features sharing the same global_feature_uid (sum counts)
    mat, var_df = deduplicate_var(mat, var_df)
    print(f"  After dedup: {mat.shape[0]:,} × {mat.shape[1]:,}")

    # Apply limit
    if limit is not None and limit < mat.shape[0]:
        print(f"  Limiting to {limit} cells (of {mat.shape[0]})")
        mat = mat[:limit]
        obs_df = obs_df.iloc[:limit]

    adata = ad.AnnData(X=mat, obs=obs_df, var=var_df)
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    return adata


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
    adata = load_coo_as_anndata(experiment, validated_obs, standardized_var, limit)

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
        dataset_description=metadata.get("description"),
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
    print(f"  Ingested {n_ingested:,} cells (dataset_uid={dataset_uid})")
    return n_ingested, dataset_uid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest GSM4150377 into a RaggedAtlas"
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
        help="Maximum number of cells to ingest (for testing)",
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
        print(f"Cell limit: {args.limit:,}")

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
