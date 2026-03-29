"""Ingest prepared figshare_20029387 (Replogle et al. 2022) into a RaggedAtlas.

This dataset contains three CRISPRi Perturb-seq experiments profiling gene
expression via single-cell RNA-seq:
  - K562_essential  (310,385 cells × 8,563 genes)  — day 6
  - K562_gwps       (1,989,578 cells × 8,248 genes) — day 8
  - RPE1            (247,914 cells × 8,749 genes)   — day 7

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/figshare_20029387/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_figshare_20029387 \
        --atlas-path /tmp/atlas/figshare_20029387 [--limit 1000]
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

from lancell.atlas import RaggedAtlas, create_or_open_atlas
from lancell.ingestion import add_anndata_batch, add_csc
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

ACCESSION = "figshare_20029387"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/figshare_20029387")
FEATURE_SPACE = "gene_expression"

# Experiment name → singlecell h5ad filename
EXPERIMENTS = {
    "K562_essential": "K562_essential_raw_singlecell_01.h5ad",
    "K562_gwps": "K562_gwps_raw_singlecell_01.h5ad",
    "RPE1": "rpe1_raw_singlecell_01.h5ad",
}


# ---------------------------------------------------------------------------
# Step 1 & 2: Assemble fragments and validate obs
# ---------------------------------------------------------------------------


def assemble_obs(experiment: str) -> Path:
    """Merge fragment CSVs into a standardized obs CSV.

    The ontology fragment is indexed by gem_group (non-unique), while
    perturbation and preparer fragments are indexed by cell_barcode.
    We handle this by loading the ontology fragment separately,
    broadcasting via the gem_group column from raw_obs, then
    concatenating with the cell_barcode-indexed fragments.
    """
    exp_dir = ACCESSION_DIR / experiment
    output_path = exp_dir / f"{FEATURE_SPACE}_standardized_obs.csv"
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping assembly")
        return output_path

    print(f"  Assembling obs for {experiment}...")

    # Load raw_obs to get cell_barcode → gem_group mapping
    raw_obs = pd.read_csv(
        exp_dir / f"{FEATURE_SPACE}_raw_obs.csv",
        usecols=["cell_barcode", "gem_group"],
    )
    cell_barcodes = raw_obs["cell_barcode"].values

    # Load ontology fragment (indexed by gem_group, non-unique)
    ont_path = exp_dir / f"{FEATURE_SPACE}_fragment_ontology_obs.csv"
    ont_df = pd.read_csv(ont_path, index_col=0)
    # Keep one row per gem_group (values are identical within each group)
    ont_unique = ont_df.groupby(ont_df.index).first()
    # Join: raw_obs.gem_group → ont_unique (indexed by gem_group)
    ont_broadcast = raw_obs[["gem_group"]].join(ont_unique, on="gem_group").drop(columns=["gem_group"])
    ont_broadcast.index = cell_barcodes

    # Load cell_barcode-indexed fragments
    fragments = [ont_broadcast]
    for frag_type in ["perturbation", "preparer"]:
        frag_path = exp_dir / f"{FEATURE_SPACE}_fragment_{frag_type}_obs.csv"
        if frag_path.exists():
            frag_df = pd.read_csv(frag_path, index_col=0)
            if not frag_df.empty:
                fragments.append(frag_df)
                print(f"    loaded {frag_path.name}: {len(frag_df.columns)} columns")

    assembled = pd.concat(fragments, axis=1)
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
# Step 3: Load h5ad as backed AnnData
# ---------------------------------------------------------------------------


def load_h5ad(
    experiment: str,
    obs_parquet: Path,
    var_csv: Path,
    limit: int | None = None,
) -> ad.AnnData:
    """Load a backed h5ad, attach validated obs and standardized var.

    The h5ad var index already matches the standardized var exactly
    (same genes, same order), so no subsetting or deduplication is needed.
    """
    exp_dir = ACCESSION_DIR / experiment
    h5ad_name = EXPERIMENTS[experiment]
    h5ad_path = exp_dir / h5ad_name

    print(f"  Loading h5ad: {h5ad_name} ...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Load validated obs and standardized var
    obs_df = pd.read_parquet(obs_parquet)
    var_df = pd.read_csv(var_csv, index_col=0)

    assert adata.n_obs == len(obs_df), (
        f"h5ad cells ({adata.n_obs}) != obs rows ({len(obs_df)})"
    )
    assert adata.n_vars == len(var_df), (
        f"h5ad genes ({adata.n_vars}) != var rows ({len(var_df)})"
    )

    # Apply limit: read a slice into memory
    if limit is not None and limit < adata.n_obs:
        print(f"  Limiting to {limit} cells (of {adata.n_obs:,})")
        import scipy.sparse as sp
        X_slice = adata.X[:limit, :]
        if not sp.issparse(X_slice):
            X_slice = sp.csr_matrix(X_slice)
        obs_df = obs_df.iloc[:limit]
        adata_out = ad.AnnData(X=X_slice, obs=obs_df, var=var_df)
        adata_out.obs.index = adata_out.obs.index.astype(str)
        adata_out.var.index = adata_out.var.index.astype(str)
        adata.file.close()
        return adata_out

    # Full backed mode: replace obs and var in place
    adata.obs = obs_df
    adata.var = var_df
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    return adata


# ---------------------------------------------------------------------------
# Step 4: Populate foreign key tables
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
    adata = load_h5ad(experiment, validated_obs, standardized_var, limit)

    dataset_uid = make_uid()

    def _unique_non_null(col: str) -> list[str] | None:
        if col not in adata.obs.columns:
            return None
        vals = adata.obs[col].dropna().unique().tolist()
        return sorted(vals) if vals else None

    exp_meta = metadata.get("experiments", {}).get(experiment, {})
    dataset_record = DatasetSchema(
        uid=dataset_uid,
        zarr_group=dataset_uid,
        feature_space=FEATURE_SPACE,
        n_cells=adata.n_obs,
        publication_uid=publication_uid,
        accession_database="FigShare",
        accession_id=metadata.get("accession_id", ACCESSION),
        dataset_description=exp_meta.get("description", metadata.get("summary")),
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

    # Close the backing file if still open
    if hasattr(adata, "file") and adata.file is not None:
        try:
            adata.file.close()
        except Exception:
            pass

    return n_ingested, dataset_uid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest figshare_20029387 (Replogle 2022) into a RaggedAtlas"
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
