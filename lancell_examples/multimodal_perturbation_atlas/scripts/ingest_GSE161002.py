"""Ingest prepared GSE161002 (Liscovitch-Brauer et al. 2021) into a RaggedAtlas.

This dataset contains two CRISPR-sciATAC screens profiling chromatin
accessibility in K562 cells via single-cell ATAC-seq BED fragment files:
  - screen1  (28,755 cells, 66 guides targeting cancer-mutated chromatin modifiers)
  - screen2  (108,893 cells, 251 guides targeting chromatin remodeling complexes)

The raw data is in BED fragment format (chrom, start, end, barcode), not count
matrices. Ingestion uses ``add_fragment_batch`` which writes cell-sorted and
genome-sorted fragment arrays to zarr, with ReferenceSequenceSchema features
(chromosomes) as the feature registry.

Prerequisites:
  - Prepared data in /home/ubuntu/geo_agent_resolution/GSE161002/ (from geo-data-preparer)

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.ingest_GSE161002 \
        --atlas-path /tmp/atlas/GSE161002 [--limit 1000]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import lancedb
import pandas as pd
import polars as pl
import pyarrow as pa

from lancell.atlas import RaggedAtlas, create_or_open_atlas
from lancell.fragments.ingestion import build_chrom_order, parse_bed_fragments
from lancell.schema import make_stable_uid, make_uid

from lancell_examples.multimodal_perturbation_atlas.ingestion import add_fragment_batch
from lancell_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    REGISTRY_SCHEMAS,
    GeneticPerturbationSchema,
    PublicationSchema,
    PublicationSectionSchema,
    ReferenceSequenceSchema,
)

VALIDATE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "geo-data-curator" / "scripts" / "validate_obs.py"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION = "GSE161002"
ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/GSE161002")
FEATURE_SPACE = "chromatin_accessibility"

EXPERIMENTS = ["screen1", "screen2"]

BED_FILES = {
    "screen1": "GSM4887677_screen1_snATAC.bed.gz",
    "screen2": "GSM4887679_screen2_snATAC.bed.gz",
}

# hg19 assembly - the BED files are aligned to hg19
ASSEMBLY = "hg19"
ORGANISM = "human"


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

    # Load raw_obs for the authoritative index
    raw_obs = pd.read_csv(exp_dir / f"{FEATURE_SPACE}_raw_obs.csv", index_col=0)
    cell_barcodes = raw_obs.index

    # Load fragment CSVs
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

    # Verify index alignment
    if not assembled.index.equals(cell_barcodes):
        print("  WARNING: fragment indices differ from raw obs, reindexing")
        assembled = assembled.reindex(cell_barcodes)

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
        ],
        check=True,
    )
    return validated_obs


def assemble_and_validate(experiment: str) -> None:
    """Assemble fragment CSVs and validate obs for one experiment."""
    assemble_obs(experiment)
    validate_obs(experiment)


# ---------------------------------------------------------------------------
# Step 3: Build chromosome feature registry
# ---------------------------------------------------------------------------


def build_chromosome_registry(experiments: list[str]) -> tuple[list[ReferenceSequenceSchema], dict[str, str]]:
    """Scan all BED files to discover chromosomes and build ReferenceSequenceSchema records.

    Returns (records, chrom_uids) where chrom_uids maps chromosome names to UIDs.
    """
    print("  Scanning BED files for chromosome names...")
    all_chroms: set[str] = set()
    for exp in experiments:
        bed_path = ACCESSION_DIR / exp / BED_FILES[exp]
        # Use polars to just get unique chromosome names from the first column
        df = pl.read_csv(
            bed_path,
            separator="\t",
            has_header=False,
            columns=[0],
        )
        chroms = df["column_1"].unique().to_list()
        all_chroms.update(chroms)
        print(f"    {exp}: {len(chroms)} chromosomes")

    print(f"  Total unique chromosomes: {len(all_chroms)}")

    # Build deterministic UIDs based on (assembly, chrom_name)
    chrom_uids: dict[str, str] = {}
    records: list[ReferenceSequenceSchema] = []

    # Sort for deterministic ordering
    sorted_chroms = sorted(all_chroms)
    for chrom in sorted_chroms:
        uid = make_stable_uid(ASSEMBLY, chrom)
        chrom_uids[chrom] = uid

        # Determine sequence role
        if chrom.startswith("chr") and chrom[3:].isdigit():
            role = "chromosome"
        elif chrom in ("chrX", "chrY"):
            role = "chromosome"
        elif chrom == "chrM":
            role = "mitochondrial"
        else:
            role = "scaffold"

        records.append(ReferenceSequenceSchema(
            uid=uid,
            global_index=None,
            sequence_name=chrom,
            sequence_role=role,
            organism=ORGANISM,
            assembly=ASSEMBLY,
            is_primary_assembly=role == "chromosome",
        ))

    return records, chrom_uids


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


def register_features(
    atlas: RaggedAtlas,
    records: list[ReferenceSequenceSchema],
) -> None:
    """Register chromosome features in the atlas."""
    n_new = atlas.register_features(FEATURE_SPACE, records)
    print(f"  Registered {n_new} new features ({len(records)} total)")


# ---------------------------------------------------------------------------
# Step 7: Ingest per-experiment data
# ---------------------------------------------------------------------------


def ingest_experiment(
    atlas: RaggedAtlas,
    experiment: str,
    publication_uid: str,
    metadata: dict,
    chrom_uids: dict[str, str],
    limit: int | None = None,
) -> int:
    """Ingest one experiment into the atlas. Returns number of cells ingested."""
    exp_dir = ACCESSION_DIR / experiment
    validated_obs = exp_dir / f"{FEATURE_SPACE}_validated_obs.parquet"
    bed_path = exp_dir / BED_FILES[experiment]

    print(f"\n  Loading validated obs for {experiment}...")
    obs_df = pd.read_parquet(validated_obs)

    # Apply limit
    if limit is not None and limit < len(obs_df):
        print(f"  Limiting to {limit} cells (of {len(obs_df)})")
        obs_df = obs_df.iloc[:limit]

    # obs index must be cell barcodes (strings)
    obs_df.index = obs_df.index.astype(str)

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
        n_cells=len(obs_df),
        publication_uid=publication_uid,
        accession_database="GEO",
        accession_id=ACCESSION,
        dataset_description=metadata.get("summary"),
        organism=_unique_non_null("organism"),
        tissue=_unique_non_null("tissue"),
        cell_line=_unique_non_null("cell_line"),
        disease=_unique_non_null("disease"),
    )

    print(f"  Ingesting {len(obs_df):,} cells from {bed_path.name}...")
    n_ingested = add_fragment_batch(
        atlas,
        bed_path,
        obs_df=obs_df,
        chrom_uids=chrom_uids,
        feature_space=FEATURE_SPACE,
        dataset_record=dataset_record,
        barcode_col="barcode",
    )
    print(f"  Ingested {n_ingested:,} cells for {experiment} (dataset_uid={dataset_uid})")
    return n_ingested


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest GSE161002 into a RaggedAtlas"
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

    # Step 3: Build chromosome feature registry
    print(f"\n{'='*60}")
    print("Step 3: Build chromosome feature registry")
    print(f"{'='*60}")
    chrom_records, chrom_uids = build_chromosome_registry(EXPERIMENTS)

    # Step 4: Create or open atlas
    print(f"\n{'='*60}")
    print("Step 4: Create or open atlas")
    print(f"{'='*60}")
    atlas = create_or_open_atlas(
        str(atlas_path),
        cell_table_name="cells",
        cell_schema=CellIndex,
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas=REGISTRY_SCHEMAS,
    )

    # Step 5: Populate FK tables
    print(f"\n{'='*60}")
    print("Step 5: Populate foreign key tables")
    print(f"{'='*60}")
    db_uri = str(atlas_path / "lance_db")
    publication_uid = populate_fk_tables(db_uri)

    # Step 6: Register features
    print(f"\n{'='*60}")
    print("Step 6: Register features")
    print(f"{'='*60}")
    register_features(atlas, chrom_records)

    # Step 7: Ingest experiments
    print(f"\n{'='*60}")
    print("Step 7: Ingest experiments")
    print(f"{'='*60}")
    total_cells = 0
    for exp in EXPERIMENTS:
        n = ingest_experiment(atlas, exp, publication_uid, metadata, chrom_uids, args.limit)
        total_cells += n

    # Step 8: Summary
    print(f"\n{'='*60}")
    print("Ingestion complete")
    print(f"{'='*60}")
    print(f"  Accession: {ACCESSION}")
    print(f"  Experiments: {len(EXPERIMENTS)}")
    print(f"  Total cells ingested: {total_cells:,}")
    print(f"  Feature space: {FEATURE_SPACE}")
    print(f"  Chromosomes: {len(chrom_uids)}")
    print(f"  Atlas path: {atlas_path}")


if __name__ == "__main__":
    main()
