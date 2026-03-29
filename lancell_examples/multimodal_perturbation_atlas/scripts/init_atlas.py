"""Initialize an empty multimodal perturbation atlas.

Creates the RaggedAtlas (cells, datasets, feature registries, zarr store)
and all foreign-key tables (publications, perturbations, etc.) so that
parallel ingestion scripts never race on table creation.

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.init_atlas \
        --atlas-path /path/to/atlas
"""

import argparse

import lancedb

from lancell.atlas import create_or_open_atlas
from lancell_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    FK_TABLE_SCHEMAS,
    REGISTRY_SCHEMAS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize an empty perturbation atlas")
    parser.add_argument("--atlas-path", required=True, help="Root path for the atlas")
    args = parser.parse_args()

    # 1. Create the core atlas (cells, datasets, registries, zarr store)
    atlas = create_or_open_atlas(
        str(args.atlas_path),
        cell_table_name="cells",
        cell_schema=CellIndex,
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas=REGISTRY_SCHEMAS,
    )

    # 2. Create all FK tables that ingestion scripts expect
    db = lancedb.connect(str(args.atlas_path).rstrip("/") + "/lance_db")
    existing = set(db.list_tables().tables)

    for table_name, schema_cls in FK_TABLE_SCHEMAS.items():
        if table_name not in existing:
            db.create_table(table_name, schema=schema_cls.to_arrow_schema())
            print(f"  Created table: {table_name}")
        else:
            print(f"  Already exists: {table_name}")

    print(f"\nAtlas initialized at {args.atlas_path}")
    print(f"  Tables: {sorted(db.list_tables().tables)}")


if __name__ == "__main__":
    main()
