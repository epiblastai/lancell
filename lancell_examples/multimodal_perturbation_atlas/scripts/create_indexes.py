"""Create scalar and FTS indexes on all atlas tables.

Intended to be run after ingestion + optimize(). Indexes on feature
registries and _feature_layouts are already handled by optimize(), so
this script covers: cells, datasets, and all FK tables.

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.create_indexes \
        --atlas-path /path/to/atlas
"""

import argparse

import lancedb


def _scalar(table: lancedb.table.Table, column: str, index_type: str = "BTREE") -> None:
    """Create a scalar index, printing progress."""
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
    """Create a full-text search index, printing progress."""
    print(f"  {table.name}.{column} (FTS, base_tokenizer={base_tokenizer})")
    table.create_fts_index(
        column,
        tokenizer_name=tokenizer_name,
        base_tokenizer=base_tokenizer,
        stem=stem,
        lower_case=lower_case,
        replace=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create indexes on atlas tables")
    parser.add_argument("--atlas-path", required=True, help="Root path for the atlas")
    args = parser.parse_args()

    db = lancedb.connect(str(args.atlas_path).rstrip("/") + "/lance_db")
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

    print("\nDone.")


if __name__ == "__main__":
    main()
