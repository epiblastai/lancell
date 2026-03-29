"""Ingest prepared GSE153056 (Papalexi et al. 2021) into a RaggedAtlas.

This dataset contains multimodal ECCITE-seq screens profiling PD-L1
regulation via CRISPR perturbation with single-cell RNA + surface
protein (ADT) measurements in THP-1 cells:
  - ECCITE          (20,729 cells × 18,649 genes + 4 proteins)
  - ECCITE_Arrayed  (16,695 cells × 16,826 genes + 4 proteins)

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/GSE153056/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE153056 \
        --atlas-path /tmp/atlas/GSE153056 [--limit 1000]
"""

import argparse
import gzip
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
from lancell.ingestion import add_csc, deduplicate_var
from lancell.schema import make_uid

from lancell_examples.multimodal_perturbation_atlas.ingestion import (
    add_multimodal_batch,
)
from lancell_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    REGISTRY_SCHEMAS,
    GeneticPerturbationSchema,
    GenomicFeatureSchema,
    ProteinSchema,
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

ACCESSION = "GSE153056"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/GSE153056")

# Each experiment maps to its cDNA and ADT count matrix files
EXPERIMENTS = {
    "ECCITE": {
        "cDNA": "GSM4633614_ECCITE_cDNA_counts.tsv.gz",
        "ADT": "GSM4633615_ECCITE_ADT_counts.tsv.gz",
    },
    "ECCITE_Arrayed": {
        "cDNA": "GSM4633608_ECCITE_Arrayed_cDNA_counts.tsv.gz",
        "ADT": "GSM4633609_ECCITE_Arrayed_ADT_counts.tsv.gz",
    },
}


# ---------------------------------------------------------------------------
# Step 1 & 2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs(experiment: str) -> Path:
    """Merge fragment CSVs into a standardized obs CSV for gene_expression."""
    exp_dir = ACCESSION_DIR / experiment
    output_path = exp_dir / "gene_expression_standardized_obs.csv"
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping assembly")
        return output_path

    print(f"  Assembling obs for {experiment}...")
    subprocess.run(
        [
            sys.executable, str(ASSEMBLE_SCRIPT),
            str(exp_dir),
            "--feature-spaces", "gene_expression",
            "--schema", str(SCHEMA_FILE),
        ],
        check=True,
    )
    return output_path


def validate_obs(experiment: str) -> Path:
    """Run validate_obs.py to coerce types and strip non-schema columns."""
    exp_dir = ACCESSION_DIR / experiment
    standardized_obs = exp_dir / "gene_expression_standardized_obs.csv"
    validated_obs = exp_dir / "gene_expression_validated_obs.parquet"

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
# Step 3: Load TSV count matrices
# ---------------------------------------------------------------------------


def load_tsv_matrix(path: Path) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Load a gene×cell TSV.gz count matrix.

    Returns (cell×gene CSR matrix, gene_names, cell_barcodes).
    """
    print(f"  Loading {path.name} ...")
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        # First column is empty (row label header); rest are cell barcodes
        barcodes = [h.strip('"') for h in header[1:]]

        gene_names = []
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            gene_names.append(parts[0].strip('"'))
            rows.append(np.array([int(x) for x in parts[1:]], dtype=np.int32))

    # gene×cell → cell×gene
    mat = sp.csr_matrix(np.vstack(rows)).T.tocsr()
    print(f"    Shape (cell × gene): {mat.shape}")
    return mat, gene_names, barcodes


def load_dense_matrix(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """Load a protein×cell TSV.gz count matrix as dense.

    Returns (cell×protein dense array, protein_names, cell_barcodes).
    """
    print(f"  Loading {path.name} ...")
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        barcodes = [h.strip('"') for h in header[1:]]

        protein_names = []
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            protein_names.append(parts[0].strip('"'))
            rows.append(np.array([int(x) for x in parts[1:]], dtype=np.float32))

    # protein×cell → cell×protein
    mat = np.vstack(rows).T
    print(f"    Shape (cell × protein): {mat.shape}")
    return mat, protein_names, barcodes


def build_multimodal_anndata(
    experiment: str,
    validated_obs_path: Path,
    limit: int | None = None,
) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
    """Build gene_expression and protein_abundance AnnData objects.

    Returns (gene_adata, protein_adata, obs_df).
    """
    exp_dir = ACCESSION_DIR / experiment
    files = EXPERIMENTS[experiment]

    # Load gene expression (sparse)
    ge_mat, _, ge_barcodes = load_tsv_matrix(exp_dir / files["cDNA"])
    ge_var = pd.read_csv(exp_dir / "gene_expression_standardized_var.csv", index_col=0)
    assert ge_mat.shape[1] == len(ge_var), (
        f"Gene matrix cols ({ge_mat.shape[1]}) != var rows ({len(ge_var)})"
    )

    # Load protein abundance (dense)
    pa_mat, _, pa_barcodes = load_dense_matrix(exp_dir / files["ADT"])
    pa_var = pd.read_csv(exp_dir / "protein_abundance_standardized_var.csv", index_col=0)
    assert pa_mat.shape[1] == len(pa_var), (
        f"Protein matrix cols ({pa_mat.shape[1]}) != var rows ({len(pa_var)})"
    )

    # Verify barcodes match
    assert ge_barcodes == pa_barcodes, "Cell barcodes differ between cDNA and ADT matrices"

    # Load validated obs
    obs_df = pd.read_parquet(validated_obs_path)
    assert ge_mat.shape[0] == len(obs_df), (
        f"Matrix rows ({ge_mat.shape[0]}) != obs rows ({len(obs_df)})"
    )

    # Apply limit
    if limit is not None and limit < ge_mat.shape[0]:
        print(f"  Limiting to {limit} cells (of {ge_mat.shape[0]})")
        ge_mat = ge_mat[:limit]
        pa_mat = pa_mat[:limit]
        obs_df = obs_df.iloc[:limit].copy()

    # Deduplicate var (multiple gene names can share the same global_feature_uid)
    ge_mat, ge_var = deduplicate_var(ge_mat, ge_var)

    # Build AnnData objects
    gene_adata = ad.AnnData(X=ge_mat, var=ge_var)
    gene_adata.obs.index = gene_adata.obs.index.astype(str)
    gene_adata.var.index = gene_adata.var.index.astype(str)

    protein_adata = ad.AnnData(X=pa_mat, var=pa_var)
    protein_adata.obs.index = protein_adata.obs.index.astype(str)
    protein_adata.var.index = protein_adata.var.index.astype(str)

    return gene_adata, protein_adata, obs_df


# ---------------------------------------------------------------------------
# Step 4: Create or open atlas
# ---------------------------------------------------------------------------


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
    """Register genomic features and protein features."""
    # Gene expression features
    ge_parquet = ACCESSION_DIR / "GenomicFeatureSchema.parquet"
    ge_df = pd.read_parquet(ge_parquet)
    ge_records = [GenomicFeatureSchema(**row.to_dict()) for _, row in ge_df.iterrows()]
    n_new = atlas.register_features("gene_expression", ge_records)
    print(f"  Gene expression: {n_new} new features ({len(ge_records)} total)")

    # Protein abundance features
    pa_parquet = ACCESSION_DIR / "ProteinSchema.parquet"
    pa_df = pd.read_parquet(pa_parquet)
    pa_records = [ProteinSchema(**row.to_dict()) for _, row in pa_df.iterrows()]
    n_new = atlas.register_features("protein_abundance", pa_records)
    print(f"  Protein abundance: {n_new} new features ({len(pa_records)} total)")


# ---------------------------------------------------------------------------
# Step 7: Ingest per-experiment data (multimodal)
# ---------------------------------------------------------------------------


def ingest_experiment(
    atlas: RaggedAtlas,
    experiment: str,
    publication_uid: str,
    metadata: dict,
    limit: int | None = None,
) -> int:
    """Ingest one experiment (gene_expression + protein_abundance) into the atlas."""
    exp_dir = ACCESSION_DIR / experiment
    validated_obs = exp_dir / "gene_expression_validated_obs.parquet"

    print(f"\n  Loading data for {experiment}...")
    gene_adata, protein_adata, obs_df = build_multimodal_anndata(
        experiment, validated_obs, limit
    )

    def _unique_non_null(col: str) -> list[str] | None:
        if col not in obs_df.columns:
            return None
        vals = obs_df[col].dropna().unique().tolist()
        return sorted(vals) if vals else None

    # Create dataset records — one per modality, each with its own zarr_group
    ge_uid = make_uid()
    pa_uid = make_uid()

    shared_kwargs = dict(
        publication_uid=publication_uid,
        accession_database="GEO",
        accession_id=ACCESSION,
        dataset_description=metadata.get("summary"),
        organism=_unique_non_null("organism"),
        tissue=_unique_non_null("tissue"),
        cell_line=_unique_non_null("cell_line"),
        disease=_unique_non_null("disease"),
    )

    ge_record = DatasetSchema(
        uid=ge_uid,
        zarr_group=ge_uid,
        feature_space="gene_expression",
        n_cells=gene_adata.n_obs,
        **shared_kwargs,
    )
    pa_record = DatasetSchema(
        uid=pa_uid,
        zarr_group=pa_uid,
        feature_space="protein_abundance",
        n_cells=protein_adata.n_obs,
        **shared_kwargs,
    )

    print(f"  Ingesting {gene_adata.n_obs:,} cells × {gene_adata.n_vars:,} genes + {protein_adata.n_vars} proteins...")
    n_ingested = add_multimodal_batch(
        atlas,
        modalities={
            "gene_expression": gene_adata,
            "protein_abundance": protein_adata,
        },
        obs_df=obs_df,
        zarr_layer="counts",
        dataset_records={
            "gene_expression": ge_record,
            "protein_abundance": pa_record,
        },
    )
    print(f"  Ingested {n_ingested:,} cells for {experiment}")
    return n_ingested, ge_uid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest GSE153056 into a RaggedAtlas"
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
    ge_uids = []
    for exp in EXPERIMENTS:
        n, ge_uid = ingest_experiment(atlas, exp, publication_uid, metadata, args.limit)
        total_cells += n
        ge_uids.append(ge_uid)

    # Build CSC arrays for feature-filtered queries
    print(f"\n{'='*60}")
    print("Building CSC arrays")
    print(f"{'='*60}")
    for ge_uid in ge_uids:
        print(f"  Building CSC for {ge_uid}...")
        add_csc(atlas, zarr_group=ge_uid, feature_space="gene_expression")
    print("  Done.")

    # Summary
    print(f"\n{'='*60}")
    print("Ingestion complete")
    print(f"{'='*60}")
    print(f"  Accession: {ACCESSION}")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"  Total cells ingested: {total_cells:,}")
    print(f"  Feature spaces: gene_expression, protein_abundance")
    print(f"  Atlas path: {atlas_path}")


if __name__ == "__main__":
    main()
