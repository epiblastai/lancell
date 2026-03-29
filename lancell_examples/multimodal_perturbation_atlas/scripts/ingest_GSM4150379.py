"""Ingest prepared GSM4150379 (Srivatsan et al. 2019) into a RaggedAtlas.

sciPlex4 dataset: single-nucleus RNA-seq of A549 and MCF7 cancer cell lines
exposed to HDAC inhibitors and metabolite-related small molecules.

Data format: cell-sorted COO triplet matrix (gene_idx, cell_idx, count),
gzipped, 1-indexed, tab-separated. The cell annotations file lists 98,437
barcodes but the matrix only covers the first 31,403 cells. The obs is
filtered down to match the matrix before ingestion.

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/GSM4150379/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSM4150379 \
        --atlas-path /tmp/atlas/GSM4150379 [--limit 1000]
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
import polars as pl
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

ACCESSION = "GSM4150379"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/GSM4150379")
FEATURE_SPACE = "gene_expression"
SCHEMA_FILE = Path(__file__).resolve().parents[1] / "schema.py"

EXPERIMENTS = ["sciPlex4_A549_MCF7_HDACi"]

COO_FILENAME = (
    "GSM4150379_sciPlex4_A549_MCF7_HDACi_UMI.count.matrix.gz"
)

CELL_ANNOTATIONS_FILENAME = (
    "GSM4150379_sciPlex4_A549_MCF7_HDACi_cell.annotations.txt.gz"
)


# ---------------------------------------------------------------------------
# COO loading helper
# ---------------------------------------------------------------------------


def load_coo_as_anndata(
    coo_path: Path,
    cell_annotations_path: Path,
    var_df: pd.DataFrame,
    n_features: int,
    limit: int | None = None,
) -> tuple[ad.AnnData, list[str]]:
    """Load a gzipped COO triplet file into an AnnData with deduplication.

    The matrix covers only a subset of the cells listed in cell_annotations.
    We determine the number of cells from the max cell index in the matrix,
    then return the matching barcodes from the annotations file.

    Uses subprocess zcat to handle truncated/corrupt gzip streams gracefully.

    Returns (adata, matrix_barcodes) where matrix_barcodes are the barcodes
    corresponding to matrix rows (in order).
    """
    print(f"  Loading COO file: {coo_path.name} ...")

    # Read cell annotations to get barcode ordering
    annotations = pl.read_csv(
        cell_annotations_path, has_header=False, separator="\t"
    )
    all_barcodes = annotations["column_1"].to_list()

    # Read COO triplets — use zcat subprocess to tolerate corrupt gzip tail,
    # then feed the decompressed stream to polars
    import subprocess as _sp
    proc = _sp.Popen(
        ["zcat", str(coo_path)], stdout=_sp.PIPE, stderr=_sp.DEVNULL,
    )
    df = pl.read_csv(
        proc.stdout, has_header=False, separator="\t",
    )
    proc.wait()  # zcat may exit non-zero on truncated stream — that's OK

    if limit is not None:
        df = df.filter(pl.col("column_2") <= limit)

    gene_idx = df["column_1"].to_numpy() - 1  # 1-indexed → 0-indexed
    cell_idx = df["column_2"].to_numpy() - 1

    # Determine actual number of cells from the data
    n_cells = int(cell_idx.max()) + 1 if len(cell_idx) > 0 else 0
    if limit is not None and limit < n_cells:
        n_cells = limit

    values = df["column_3"].to_numpy()
    del df

    # Get the barcodes for cells in the matrix
    matrix_barcodes = all_barcodes[:n_cells]
    print(f"  Loaded {len(values):,} nonzeros for {n_cells:,} cells "
          f"(of {len(all_barcodes):,} annotated)")

    mat = sp.coo_matrix(
        (values.astype(np.int32), (cell_idx, gene_idx)),
        shape=(n_cells, n_features),
    ).tocsr()
    del gene_idx, cell_idx, values

    # Deduplicate features sharing the same global_feature_uid
    n_dupes = var_df["global_feature_uid"].duplicated().sum()
    if n_dupes > 0:
        print(f"  Deduplicating {n_dupes} features with shared UIDs...")
        mat, var_df = deduplicate_var(mat, var_df)
        print(f"  After dedup: {var_df.shape[0]:,} features")

    adata = ad.AnnData(X=mat, var=var_df)
    return adata, matrix_barcodes


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
        ],
        check=True,
    )
    return validated_obs


def assemble_and_validate(experiment: str) -> None:
    """Assemble fragment CSVs and validate obs for one experiment."""
    assemble_obs(experiment)
    validate_obs(experiment)


# ---------------------------------------------------------------------------
# Step 3: Populate foreign key tables
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
# Step 6: Ingest per-experiment data
# ---------------------------------------------------------------------------


def ingest_experiment(
    atlas: RaggedAtlas,
    experiment: str,
    publication_uid: str,
    metadata: dict,
    limit: int | None = None,
) -> int:
    """Ingest one experiment into the atlas. Returns cells ingested."""
    exp_dir = ACCESSION_DIR / experiment
    validated_obs = exp_dir / f"{FEATURE_SPACE}_validated_obs.parquet"
    standardized_var = exp_dir / f"{FEATURE_SPACE}_standardized_var.csv"
    coo_path = exp_dir / COO_FILENAME
    cell_ann_path = exp_dir / CELL_ANNOTATIONS_FILENAME

    # Load var
    var_df = pd.read_csv(str(standardized_var), index_col=0)
    n_features = len(var_df)

    # Load COO into AnnData — also returns the barcodes covered by the matrix
    adata, matrix_barcodes = load_coo_as_anndata(
        coo_path, cell_ann_path, var_df, n_features, limit,
    )

    # Load validated obs and filter to cells present in the matrix
    obs_df = pd.read_parquet(validated_obs)
    n_obs_total = len(obs_df)

    # Build a barcode→row mapping from the full obs
    obs_df["_barcode"] = obs_df.index if obs_df.index.name == "cell_barcode" else (
        obs_df["cell_barcode"] if "cell_barcode" in obs_df.columns else obs_df.index
    )
    # Reindex obs to match matrix barcode order
    barcode_set = set(matrix_barcodes)
    mask = obs_df["_barcode"].isin(barcode_set)
    obs_df = obs_df.loc[mask].copy()

    # Ensure obs is in the same order as matrix rows
    barcode_order = {b: i for i, b in enumerate(matrix_barcodes)}
    obs_df["_sort_key"] = obs_df["_barcode"].map(barcode_order)
    obs_df = obs_df.sort_values("_sort_key").drop(columns=["_barcode", "_sort_key"])
    obs_df = obs_df.reset_index(drop=True)

    n_cells = len(obs_df)
    print(f"  Filtered obs: {n_cells:,} cells (of {n_obs_total:,} annotated)")

    assert n_cells == adata.n_obs, (
        f"obs rows ({n_cells}) != matrix rows ({adata.n_obs})"
    )

    adata.obs = obs_df
    adata.obs.index = adata.obs.index.astype(str)

    dataset_uid = make_uid()

    def _unique_non_null(col: str) -> list[str] | None:
        if col not in obs_df.columns:
            return None
        vals = obs_df[col].dropna().unique().tolist()
        return sorted(vals) if vals else None

    dataset_record = DatasetSchema(
        uid=dataset_uid,
        zarr_group=dataset_uid,
        feature_space=FEATURE_SPACE,
        n_cells=n_cells,
        publication_uid=publication_uid,
        accession_database="GEO",
        accession_id=ACCESSION,
        dataset_description=metadata.get("summary"),
        organism=_unique_non_null("organism"),
        tissue=_unique_non_null("tissue"),
        cell_line=_unique_non_null("cell_line"),
        disease=_unique_non_null("disease"),
    )

    print(f"  Ingesting {n_cells:,} cells...")
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
        description="Ingest GSM4150379 into a RaggedAtlas"
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
    metadata_path = ACCESSION_DIR / "GSE139944_metadata.json"
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
