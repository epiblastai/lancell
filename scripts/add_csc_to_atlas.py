"""Add CSC index to an existing mus_musculus lancell atlas.

Usage:
    python scripts/add_csc_to_atlas.py [--atlas-dir ~/datasets/mus_musculus_lancell]
"""

import argparse
import time
from pathlib import Path

import obstore.store

from examples.cellxgene_census_tiledb.schema import CellObs
from lancell.atlas import RaggedAtlas
from lancell.ingestion import add_csc

FEATURE_SPACE = "gene_expression"
LAYER_NAME = "counts"
CHUNK_SIZE = 40_960
SHARD_SIZE = 1024 * CHUNK_SIZE


def main():
    parser = argparse.ArgumentParser(description="Add CSC index to a lancell atlas")
    parser.add_argument(
        "--atlas-dir",
        default=str(Path.home() / "datasets" / "mus_musculus_lancell"),
        help="Path to atlas directory",
    )
    args = parser.parse_args()

    atlas_dir = args.atlas_dir

    zarr_path = Path(atlas_dir) / "zarr_store"
    db_uri = str(Path(atlas_dir) / "lance_db")
    store = obstore.store.LocalStore(str(zarr_path))

    print(f"Opening atlas at {atlas_dir}")
    atlas = RaggedAtlas.open(
        db_uri=db_uri,
        cell_table_name="cells",
        cell_schema=CellObs,
        dataset_table_name="datasets",
        store=store,
        registry_tables={FEATURE_SPACE: "gene_expression_registry"},
    )

    # Find the zarr group(s) from the dataset table
    datasets_df = (
        atlas._dataset_table.search().select(["zarr_group", "feature_space", "n_cells"]).to_polars()
    )
    print(f"Found {len(datasets_df)} dataset(s):")
    print(datasets_df)

    for row in datasets_df.iter_rows(named=True):
        zarr_group = row["zarr_group"]
        feature_space = row["feature_space"]
        n_cells = row["n_cells"]

        # Check if CSC already exists
        try:
            atlas._root[f"{zarr_group}/csc"]
            print(f"\n  {zarr_group}: CSC already exists, skipping")
            continue
        except KeyError:
            pass

        print(f"\n  Building CSC for {zarr_group} ({n_cells:,} cells)...")
        t0 = time.perf_counter()
        add_csc(
            atlas,
            zarr_group=zarr_group,
            feature_space=feature_space,
            layer_name=LAYER_NAME,
            chunk_size=CHUNK_SIZE,
            shard_size=SHARD_SIZE,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s")

    print("\nAll done!")


if __name__ == "__main__":
    main()
