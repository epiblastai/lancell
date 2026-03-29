"""Deduplicate, optimize, index, and snapshot the perturbation atlas.

Intended to be run once after all ingestion scripts have finished.
Steps:
    1. Open the atlas as a PerturbationAtlas
    2. Deduplicate and optimize all tables (core + FK)
    3. Create scalar and FTS indexes on cells, datasets, and FK tables
    4. Snapshot

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.snapshot_atlas \
        --atlas-path /path/to/atlas
"""

import argparse

import lancedb

from lancell_examples.multimodal_perturbation_atlas.atlas import PerturbationAtlas
from lancell_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    REGISTRY_SCHEMAS,
)


# ---------------------------------------------------------------------------
# Index creation (moved from create_indexes.py)
# ---------------------------------------------------------------------------


def _scalar(table: lancedb.table.Table, column: str, index_type: str = "BTREE") -> None:
    print(f"  {table.name}.{column} ({index_type})")
    table.create_scalar_index(column, index_type=index_type, replace=True)


def _fts(
    table: lancedb.table.Table,
    column: str,
    *,
    tokenizer_name: str = "default",
    base_tokenizer: str = "simple",
    stem: bool = True,
    lower_case: bool = True,
) -> None:
    print(f"  {table.name}.{column} (FTS, base_tokenizer={base_tokenizer})")
    table.create_fts_index(
        column,
        tokenizer_name=tokenizer_name,
        base_tokenizer=base_tokenizer,
        stem=stem,
        lower_case=lower_case,
        replace=True,
    )


def create_indexes(db: lancedb.DBConnection) -> None:
    existing = set(db.list_tables().tables)

    # -- cells -----------------------------------------------------------------
    if "cells" in existing:
        t = db.open_table("cells")
        print("cells:")
        _scalar(t, "dataset_uid")
        _scalar(t, "assay")
        _scalar(t, "organism")
        _scalar(t, "cell_line")
        _scalar(t, "cell_type")
        _scalar(t, "disease")
        _scalar(t, "tissue")
        _scalar(t, "negative_control_type")
        _scalar(t, "is_negative_control", index_type="BITMAP")
        _fts(
            t,
            "perturbation_search_string",
            base_tokenizer="whitespace",
            stem=False,
            lower_case=False,
        )

    # -- datasets --------------------------------------------------------------
    if "datasets" in existing:
        t = db.open_table("datasets")
        print("datasets:")
        _scalar(t, "uid")
        _scalar(t, "feature_space")
        _scalar(t, "layout_uid")
        _scalar(t, "publication_uid")
        _scalar(t, "accession_id")

    # -- publications ----------------------------------------------------------
    if "publications" in existing:
        t = db.open_table("publications")
        print("publications:")
        _scalar(t, "uid")
        _scalar(t, "doi")
        _scalar(t, "pmid")

    # -- publication_sections --------------------------------------------------
    if "publication_sections" in existing:
        t = db.open_table("publication_sections")
        print("publication_sections:")
        _scalar(t, "publication_uid")

    # -- genetic_perturbations -------------------------------------------------
    if "genetic_perturbations" in existing:
        t = db.open_table("genetic_perturbations")
        print("genetic_perturbations:")
        _scalar(t, "uid")
        _scalar(t, "perturbation_type")
        _scalar(t, "intended_gene_name")
        _scalar(t, "intended_ensembl_gene_id")
        _scalar(t, "library_name")
        _scalar(t, "target_chromosome")

    # -- small_molecules -------------------------------------------------------
    if "small_molecules" in existing:
        t = db.open_table("small_molecules")
        print("small_molecules:")
        _scalar(t, "uid")
        _scalar(t, "pubchem_cid")
        _scalar(t, "inchi_key")
        _scalar(t, "chembl_id")
        _scalar(t, "name")

    # -- biologic_perturbations ------------------------------------------------
    if "biologic_perturbations" in existing:
        t = db.open_table("biologic_perturbations")
        print("biologic_perturbations:")
        _scalar(t, "uid")
        _scalar(t, "biologic_name")
        _scalar(t, "biologic_type")
        _scalar(t, "uniprot_id")

    # -- dataset_perturbation_index -------------------------------------------
    if "dataset_perturbation_index" in existing:
        t = db.open_table("dataset_perturbation_index")
        print("dataset_perturbation_index:")
        _scalar(t, "dataset_uid")
        _scalar(t, "perturbation_uid")
        _scalar(t, "perturbation_type")
        _scalar(t, "intended_gene_name")
        _scalar(t, "compound_name")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate, optimize, index, and snapshot the perturbation atlas"
    )
    parser.add_argument("--atlas-path", required=True, help="Root path for the atlas")
    args = parser.parse_args()

    atlas_path = str(args.atlas_path).rstrip("/")
    db_uri = atlas_path + "/lance_db"
    zarr_uri = atlas_path + "/zarr_store"

    # Determine store
    import obstore.store

    if atlas_path.startswith(("s3://", "gs://", "az://")):
        store = obstore.store.from_url(zarr_uri)
    else:
        store = obstore.store.LocalStore(zarr_uri)

    registry_tables = {fs: f"{fs}_registry" for fs in REGISTRY_SCHEMAS}

    atlas = PerturbationAtlas.open(
        db_uri=db_uri,
        cell_table_name="cells",
        cell_schema=CellIndex,
        dataset_table_name="datasets",
        store=store,
        registry_tables=registry_tables,
    )

    # Step 1: Deduplicate and optimize all tables
    print("=" * 60)
    print("Step 1: Deduplicate and optimize")
    print("=" * 60)
    atlas.optimize()

    # Step 2: Create indexes (on clean, compacted data)
    print()
    print("=" * 60)
    print("Step 2: Create indexes")
    print("=" * 60)
    create_indexes(atlas.db)

    # Step 3: Snapshot
    print()
    print("=" * 60)
    print("Step 3: Snapshot")
    print("=" * 60)
    version = atlas.snapshot()
    print(f"Snapshot created: version {version}")


if __name__ == "__main__":
    main()
