"""Ingest prepared SCP1064 (Frangieh et al. 2021) into a RaggedAtlas.

This dataset contains multimodal Perturb-CITE-seq data profiling
cancer immune evasion in A375 melanoma cells:
  - perturb_cite_seq (218K cells × 23,712 genes + 24 proteins)

Three conditions: Control, IFN-gamma stimulation, TIL co-culture.
Genetic perturbations via CRISPRko sgRNAs; biologic perturbations
(IFN-gamma cytokine, TIL co-culture).

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/SCP1064/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_SCP1064 \
        --atlas-path /tmp/atlas/SCP1064 [--limit 1000]
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
    BiologicPerturbationSchema,
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

ACCESSION = "SCP1064"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/SCP1064")
EXPERIMENT = "perturb_cite_seq"


# ---------------------------------------------------------------------------
# Step 1-2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs() -> Path:
    """Merge fragment CSVs into a standardized obs CSV for gene_expression."""
    exp_dir = ACCESSION_DIR / EXPERIMENT
    output_path = exp_dir / "gene_expression_standardized_obs.csv"
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping assembly")
        return output_path

    print("  Assembling obs for gene_expression...")
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


def enrich_perturbation_columns() -> None:
    """Add perturbation_uids/types columns to the standardized obs CSV.

    Maps sgRNA reagent IDs to GeneticPerturbationSchema UIDs and combines
    them with biologic perturbation UIDs from the biologic fragment.
    """
    exp_dir = ACCESSION_DIR / EXPERIMENT
    obs_path = exp_dir / "gene_expression_standardized_obs.csv"
    obs_df = pd.read_csv(obs_path, low_memory=False)

    # Already enriched?
    if "perturbation_uids" in obs_df.columns:
        print("  perturbation_uids already present, skipping enrichment")
        return

    # Load raw obs for sgRNA column
    raw_obs = pd.read_csv(exp_dir / "gene_expression_raw_obs.csv", usecols=["cell_id", "sgRNA"])

    # Build sgRNA → genetic perturbation UID mapping
    gp_df = pd.read_parquet(ACCESSION_DIR / "GeneticPerturbationSchema.parquet")
    sgrna_to_uid = dict(zip(gp_df["reagent_id"], gp_df["uid"]))

    # Map each cell's sgRNA to genetic perturbation UID
    raw_obs["genetic_perturbation_uid"] = raw_obs["sgRNA"].map(sgrna_to_uid)

    # Merge into obs on cell_id
    obs_df = obs_df.merge(
        raw_obs[["cell_id", "genetic_perturbation_uid"]],
        on="cell_id",
        how="left",
    )

    # Build perturbation_uids and perturbation_types lists
    # Combine genetic perturbation and biologic perturbation UIDs
    bio_col = "biologic_perturbation_uid" if "biologic_perturbation_uid" in obs_df.columns else None

    perturbation_uids = []
    perturbation_types = []
    perturbation_concentrations = []
    perturbation_durations = []
    perturbation_metadata = []

    for _, row in obs_df.iterrows():
        uids = []
        types = []
        concs = []
        durs = []
        meta = []

        gp_uid = row.get("genetic_perturbation_uid")
        if pd.notna(gp_uid):
            uids.append(gp_uid)
            types.append("genetic_perturbation")
            concs.append(-1.0)
            durs.append(-1.0)
            meta.append("{}")

        if bio_col:
            bio_uid = row.get(bio_col)
            if pd.notna(bio_uid):
                uids.append(bio_uid)
                types.append("biologic_perturbation")
                concs.append(-1.0)
                durs.append(-1.0)
                meta.append("{}")

        if uids:
            perturbation_uids.append(json.dumps(uids))
            perturbation_types.append(json.dumps(types))
            perturbation_concentrations.append(json.dumps(concs))
            perturbation_durations.append(json.dumps(durs))
            perturbation_metadata.append(json.dumps(meta))
        else:
            perturbation_uids.append(None)
            perturbation_types.append(None)
            perturbation_concentrations.append(None)
            perturbation_durations.append(None)
            perturbation_metadata.append(None)

    obs_df["perturbation_uids"] = perturbation_uids
    obs_df["perturbation_types"] = perturbation_types
    obs_df["perturbation_concentrations_um"] = perturbation_concentrations
    obs_df["perturbation_durations_hr"] = perturbation_durations
    obs_df["perturbation_additional_metadata"] = perturbation_metadata

    # Drop intermediate columns
    obs_df.drop(columns=["genetic_perturbation_uid"], inplace=True, errors="ignore")

    obs_df.to_csv(obs_path, index=False)
    print(f"  Enriched perturbation columns for {len(obs_df):,} cells")


def validate_obs() -> Path:
    """Run validate_obs.py to coerce types and strip non-schema columns."""
    exp_dir = ACCESSION_DIR / EXPERIMENT
    standardized_obs = exp_dir / "gene_expression_standardized_obs.csv"
    validated_obs = exp_dir / "gene_expression_validated_obs.parquet"

    if validated_obs.exists():
        print(f"  {validated_obs.name} already exists, skipping validation")
        return validated_obs

    print("  Validating obs...")
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
            "--column", "donor_uid=None",
            "--column", "days_in_vitro=None",
            "--column", "well_position=None",
            "--column", "replicate=None",
            "--column", "batch_id=None",
        ],
        check=True,
    )
    return validated_obs


def assemble_and_validate() -> None:
    """Assemble fragment CSVs, enrich perturbation columns, and validate obs."""
    assemble_obs()
    enrich_perturbation_columns()
    validate_obs()


# ---------------------------------------------------------------------------
# Step 3: Load CSV count matrices
# ---------------------------------------------------------------------------


def load_csv_sparse_matrix(
    path: Path,
    cell_subset: list[str] | None = None,
) -> tuple[sp.csr_matrix, list[str], list[str]]:
    """Load a gene×cell CSV.gz matrix as sparse (float32).

    Parameters
    ----------
    path
        Path to CSV.gz file with GENE header row, genes as rows, cells as columns.
    cell_subset
        If provided, only load these cell barcodes (must be a subset of header).

    Returns (cell×gene CSR matrix, gene_names, cell_barcodes).
    """
    print(f"  Loading {path.name} ...")

    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
        all_barcodes = header[1:]  # First column is gene name header

        if cell_subset is not None:
            keep_set = set(cell_subset)
            col_mask = [b in keep_set for b in all_barcodes]
            barcodes = [b for b, keep in zip(all_barcodes, col_mask) if keep]
            keep_indices = [i for i, keep in enumerate(col_mask) if keep]
        else:
            barcodes = all_barcodes
            keep_indices = None

        gene_names = []
        row_indices = []
        col_indices = []
        values = []

        for gene_idx, line in enumerate(f):
            parts = line.rstrip("\n").split(",")
            gene_names.append(parts[0])

            if keep_indices is not None:
                for new_col, orig_col in enumerate(keep_indices):
                    val = float(parts[orig_col + 1])
                    if val != 0.0:
                        row_indices.append(new_col)  # cell index (transposed)
                        col_indices.append(gene_idx)  # gene index
                        values.append(val)
            else:
                for cell_idx, raw_val in enumerate(parts[1:]):
                    val = float(raw_val)
                    if val != 0.0:
                        row_indices.append(cell_idx)
                        col_indices.append(gene_idx)
                        values.append(val)

            if (gene_idx + 1) % 5000 == 0:
                print(f"    Processed {gene_idx + 1} genes, {len(values):,} nonzeros...")

    n_cells = len(barcodes)
    n_genes = len(gene_names)
    mat = sp.coo_matrix(
        (np.array(values, dtype=np.float32),
         (np.array(row_indices, dtype=np.int32),
          np.array(col_indices, dtype=np.int32))),
        shape=(n_cells, n_genes),
    ).tocsr()

    print(f"    Shape (cell × gene): {mat.shape}, nnz: {mat.nnz:,}")
    return mat, gene_names, barcodes


def load_csv_dense_matrix(
    path: Path,
    cell_subset: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load a protein×cell CSV.gz matrix as dense (float32).

    Returns (cell×protein dense array, protein_names, cell_barcodes).
    """
    print(f"  Loading {path.name} ...")

    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
        # First element may be empty (row label header)
        all_barcodes = [h for h in header[1:] if h]

        if cell_subset is not None:
            keep_set = set(cell_subset)
            col_mask = [b in keep_set for b in all_barcodes]
            barcodes = [b for b, keep in zip(all_barcodes, col_mask) if keep]
            keep_indices = [i for i, keep in enumerate(col_mask) if keep]
        else:
            barcodes = all_barcodes
            keep_indices = None

        protein_names = []
        rows = []
        for line in f:
            parts = line.rstrip("\n").split(",")
            protein_names.append(parts[0])
            if keep_indices is not None:
                row = np.array([float(parts[i + 1]) for i in keep_indices], dtype=np.float32)
            else:
                row = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            rows.append(row)

    # protein×cell → cell×protein
    mat = np.vstack(rows).T
    print(f"    Shape (cell × protein): {mat.shape}")
    return mat, protein_names, barcodes


def build_multimodal_anndata(
    validated_obs_path: Path,
    limit: int | None = None,
) -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame]:
    """Build gene_expression and protein_abundance AnnData objects.

    Returns (gene_adata, protein_adata, obs_df) aligned to common cells.
    """
    exp_dir = ACCESSION_DIR / EXPERIMENT

    # Load validated obs (gene_expression has all metadata)
    # cell_id is the index in the validated parquet
    obs_df = pd.read_parquet(validated_obs_path)
    obs_df = obs_df.reset_index()  # cell_id becomes a column

    # Read headers to find common cells
    with gzip.open(exp_dir / "RNA_expression.csv.gz", "rt") as f:
        ge_header = f.readline().rstrip("\n").split(",")
    ge_barcodes = ge_header[1:]

    with gzip.open(exp_dir / "raw_CITE_expression.csv.gz", "rt") as f:
        pa_header = f.readline().rstrip("\n").split(",")
    pa_barcodes = [h for h in pa_header[1:] if h]

    # Find cells present in both modalities and in the validated obs
    obs_cell_set = set(obs_df["cell_id"])
    common_cells = sorted(
        set(ge_barcodes) & set(pa_barcodes) & obs_cell_set,
        key=lambda c: ge_barcodes.index(c),
    )
    print(f"  Common cells across modalities + obs: {len(common_cells):,}")

    # Apply limit
    if limit is not None and limit < len(common_cells):
        common_cells = common_cells[:limit]
        print(f"  Limiting to {limit} cells")

    # Load matrices (subset to common cells)
    ge_mat, _, ge_loaded_barcodes = load_csv_sparse_matrix(
        exp_dir / "RNA_expression.csv.gz",
        cell_subset=common_cells,
    )
    pa_mat, _, pa_loaded_barcodes = load_csv_dense_matrix(
        exp_dir / "raw_CITE_expression.csv.gz",
        cell_subset=common_cells,
    )

    # Align protein matrix to gene expression cell order
    pa_barcode_to_idx = {b: i for i, b in enumerate(pa_loaded_barcodes)}
    reorder = [pa_barcode_to_idx[b] for b in ge_loaded_barcodes]
    pa_mat = pa_mat[reorder]

    # Align obs to gene expression cell order
    obs_df = obs_df.set_index("cell_id").loc[ge_loaded_barcodes].reset_index()

    # Load standardized var DataFrames
    ge_var = pd.read_csv(exp_dir / "gene_expression_standardized_var.csv", index_col=0)
    pa_var = pd.read_csv(exp_dir / "protein_abundance_standardized_var.csv", index_col=0)

    assert ge_mat.shape[1] == len(ge_var), (
        f"Gene matrix cols ({ge_mat.shape[1]}) != var rows ({len(ge_var)})"
    )
    assert pa_mat.shape[1] == len(pa_var), (
        f"Protein matrix cols ({pa_mat.shape[1]}) != var rows ({len(pa_var)})"
    )

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
# Step 4: Populate foreign key tables
# ---------------------------------------------------------------------------


def populate_fk_tables(db_uri: str) -> str:
    """Create publication, publication sections, genetic perturbation,
    and biologic perturbation tables.

    Returns the publication_uid for use in DatasetSchema.
    """
    db = lancedb.connect(db_uri)
    existing = db.list_tables().tables

    # --- Publications ---
    pub_df = pd.read_parquet(ACCESSION_DIR / "PublicationSchema.parquet")
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
    gp_df = pd.read_parquet(ACCESSION_DIR / "GeneticPerturbationSchema.parquet")
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

    # --- Biologic perturbations ---
    bio_df = pd.read_parquet(ACCESSION_DIR / "BiologicPerturbationSchema.parquet")
    if "biologic_perturbations" not in existing:
        bio_table = db.create_table(
            "biologic_perturbations",
            schema=BiologicPerturbationSchema.to_arrow_schema(),
        )
    else:
        bio_table = db.open_table("biologic_perturbations")
    bio_table.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(
            bio_df, schema=BiologicPerturbationSchema.to_arrow_schema()
        )
    )
    print(f"  Added {len(bio_df)} biologic perturbation record(s)")

    return publication_uid


# ---------------------------------------------------------------------------
# Step 6: Register features
# ---------------------------------------------------------------------------


def register_features(atlas: RaggedAtlas) -> None:
    """Register genomic features and protein features."""
    ge_df = pd.read_parquet(ACCESSION_DIR / "GenomicFeatureSchema.parquet")
    ge_records = [GenomicFeatureSchema(**row.to_dict()) for _, row in ge_df.iterrows()]
    n_new = atlas.register_features("gene_expression", ge_records)
    print(f"  Gene expression: {n_new} new features ({len(ge_records)} total)")

    pa_df = pd.read_parquet(ACCESSION_DIR / "ProteinSchema.parquet")
    pa_records = [ProteinSchema(**row.to_dict()) for _, row in pa_df.iterrows()]
    n_new = atlas.register_features("protein_abundance", pa_records)
    print(f"  Protein abundance: {n_new} new features ({len(pa_records)} total)")


# ---------------------------------------------------------------------------
# Step 7: Ingest experiment data (multimodal)
# ---------------------------------------------------------------------------


def ingest_experiment(
    atlas: RaggedAtlas,
    publication_uid: str,
    metadata: dict,
    limit: int | None = None,
) -> int:
    """Ingest gene_expression + protein_abundance into the atlas."""
    exp_dir = ACCESSION_DIR / EXPERIMENT
    validated_obs = exp_dir / "gene_expression_validated_obs.parquet"

    print("\n  Loading multimodal data...")
    gene_adata, protein_adata, obs_df = build_multimodal_anndata(
        validated_obs, limit
    )

    def _unique_non_null(col: str) -> list[str] | None:
        if col not in obs_df.columns:
            return None
        vals = obs_df[col].dropna().unique().tolist()
        return sorted(vals) if vals else None

    ge_uid = make_uid()
    pa_uid = make_uid()

    shared_kwargs = dict(
        publication_uid=publication_uid,
        accession_database="Single Cell Portal",
        accession_id=ACCESSION,
        dataset_description=metadata.get("title"),
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

    print(
        f"  Ingesting {gene_adata.n_obs:,} cells × "
        f"{gene_adata.n_vars:,} genes + {protein_adata.n_vars} proteins..."
    )
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
    print(f"  Ingested {n_ingested:,} cells")
    return n_ingested, ge_uid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest SCP1064 into a RaggedAtlas"
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
    assemble_and_validate()

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

    # Step 6: Ingest experiment
    print(f"\n{'='*60}")
    print("Step 6: Ingest experiment")
    print(f"{'='*60}")
    n, ge_uid = ingest_experiment(atlas, publication_uid, metadata, args.limit)

    # Build CSC arrays for feature-filtered queries
    print(f"\n{'='*60}")
    print("Building CSC arrays")
    print(f"{'='*60}")
    print(f"  Building CSC for {ge_uid}...")
    add_csc(atlas, zarr_group=ge_uid, feature_space="gene_expression")
    print("  Done.")

    # Summary
    print(f"\n{'='*60}")
    print("Ingestion complete")
    print(f"{'='*60}")
    print(f"  Accession: {ACCESSION}")
    print(f"  Total cells ingested: {n:,}")
    print(f"  Feature spaces: gene_expression, protein_abundance")
    print(f"  Atlas path: {atlas_path}")


if __name__ == "__main__":
    main()
