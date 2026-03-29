"""Ingest prepared GSE133344 (Norman et al. 2019) into a RaggedAtlas.

This dataset contains a CRISPRa Perturb-seq screen profiling genetic
interactions via single-cell RNA-seq in K562 cells:
  - Norman2019 (~111,445 cells × 33,694 genes)

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/GSE133344/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE133344 \
        --atlas-path /tmp/atlas/GSE133344 [--limit 1000]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import lancedb
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

ASSEMBLE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "geo-data-curator" / "scripts" / "assemble_fragments.py"
)
VALIDATE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "geo-data-curator" / "scripts" / "validate_obs.py"
)
SCHEMA_FILE = Path(__file__).resolve().parent.parent / "schema.py"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION = "GSE133344"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/GSE133344")
FEATURE_SPACE = "gene_expression"

EXPERIMENTS = ["Norman2019"]


# ---------------------------------------------------------------------------
# Step 1 & 2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs(experiment: str) -> Path:
    """Run assemble_fragments.py to merge fragment CSVs into standardized obs."""
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
# Step 3: Load expression matrix from MTX bundle
# ---------------------------------------------------------------------------


def load_mtx_as_anndata(
    experiment: str,
    obs_parquet: Path,
    var_csv: Path,
    limit: int | None = None,
) -> ad.AnnData:
    """Load an MTX bundle into an AnnData, attaching validated obs and var.

    The MTX barcodes file may contain more cells than the obs (which was
    derived from the cell_identities file). We load the barcodes, map obs
    barcodes to MTX row indices, and subset the matrix accordingly.
    """
    import numpy as np

    exp_dir = ACCESSION_DIR / experiment
    mtx_path = exp_dir / "GSE133344_filtered_matrix.mtx.gz"
    barcodes_path = exp_dir / "GSE133344_filtered_barcodes.tsv.gz"

    print(f"  Loading expression matrix: {mtx_path.name} ...")
    mat = sio.mmread(str(mtx_path))
    # MTX files from GEO are often gene×cell — transpose to cell×gene
    if sp.issparse(mat):
        mat = mat.T.tocsr()
    else:
        mat = sp.csr_matrix(mat.T)

    print(f"  Matrix shape (cell × gene): {mat.shape}")

    # Load barcodes from the MTX bundle
    mtx_barcodes = pd.read_csv(barcodes_path, header=None, sep="\t")[0].values

    # Load obs and var
    obs_df = pd.read_parquet(obs_parquet)
    var_df = pd.read_csv(var_csv, index_col=0)

    # Map obs barcodes to MTX row indices and subset the matrix
    barcode_to_idx = {bc: i for i, bc in enumerate(mtx_barcodes)}
    obs_barcodes = obs_df.index if obs_df.index.name == "cell_barcode" else obs_df.iloc[:, 0]
    # Try the obs index first; if it's just integers, look for a cell_barcode column
    if obs_df.index.dtype == "int64" or obs_df.index.dtype == "int32":
        if "cell_barcode" in obs_df.columns:
            obs_barcodes = obs_df["cell_barcode"].values
        else:
            obs_barcodes = obs_df.index.astype(str).values
    else:
        obs_barcodes = obs_df.index.values

    row_indices = np.array([barcode_to_idx[bc] for bc in obs_barcodes])
    mat = mat[row_indices]
    print(f"  Subset matrix to {mat.shape[0]} cells matching obs barcodes")

    assert mat.shape[0] == len(obs_df), (
        f"Matrix rows ({mat.shape[0]}) != obs rows ({len(obs_df)})"
    )
    assert mat.shape[1] == len(var_df), (
        f"Matrix cols ({mat.shape[1]}) != var rows ({len(var_df)})"
    )

    # Deduplicate var features that share the same global_feature_uid
    # (e.g., multiple Ensembl IDs mapping to the same gene). Sum columns.
    n_before = len(var_df)
    mat, var_df = deduplicate_var(mat, var_df)
    if len(var_df) < n_before:
        print(f"  Deduplicated var: {n_before} → {len(var_df)} features ({n_before - len(var_df)} merged)")

    # Apply limit
    if limit is not None and limit < mat.shape[0]:
        print(f"  Limiting to {limit} cells (of {mat.shape[0]})")
        mat = mat[:limit]
        obs_df = obs_df.iloc[:limit]

    adata = ad.AnnData(X=mat, obs=obs_df, var=var_df)
    # obs index must be string for anndata
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
    adata = load_mtx_as_anndata(experiment, validated_obs, standardized_var, limit)

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
        description="Ingest GSE133344 into a RaggedAtlas"
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
