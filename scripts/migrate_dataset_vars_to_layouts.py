#!/usr/bin/env python3
"""Migrate a lancell atlas from _dataset_vars to _feature_layouts.

This script converts an atlas that uses the old per-(dataset, feature)
``_dataset_vars`` table to the new per-(layout, feature)
``_feature_layouts`` table.

Steps performed:
1. Read all rows from ``_dataset_vars``, grouped by ``dataset_uid``.
2. Compute ``layout_uid`` for each unique feature ordering.
3. Create the ``_feature_layouts`` table with the new rows.
4. Add ``layout_uid`` column to the ``datasets`` table.
5. Recreate the ``atlas_versions`` table with the new schema (old snapshots
   are dropped — they referenced ``dataset_vars_table_version``).
6. If any rows had ``csc_start``/``csc_end``, write zarr ``csc/indptr`` arrays.
7. Optionally drop the old ``_dataset_vars`` table.

Usage:
    python scripts/migrate_dataset_vars_to_layouts.py /path/to/atlas/lance_db [--zarr-store /path/to/zarr_store] [--drop-old]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lancedb
import numpy as np
import polars as pl

# Add project root to path so we can import lancell
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lancell.feature_layouts import compute_layout_uid
from lancell.schema import AtlasVersionRecord, FeatureLayout


def migrate(
    db_uri: str,
    *,
    zarr_store_path: str | None = None,
    drop_old: bool = False,
) -> None:
    db = lancedb.connect(db_uri)
    table_names = db.table_names()

    # ── Preflight checks ──────────────────────────────────────────────
    if "_dataset_vars" not in table_names:
        print("ERROR: No _dataset_vars table found. Nothing to migrate.")
        return

    if "_feature_layouts" in table_names:
        print("ERROR: _feature_layouts table already exists. Atlas may already be migrated.")
        return

    # ── Step 1: Read old _dataset_vars ────────────────────────────────
    old_table = db.open_table("_dataset_vars")
    old_df = old_table.search().to_polars()
    print(f"Read {len(old_df)} rows from _dataset_vars")

    if old_df.is_empty():
        print("WARNING: _dataset_vars is empty. Creating empty _feature_layouts table.")
        db.create_table("_feature_layouts", schema=FeatureLayout)
        _migrate_datasets_table(db)
        _migrate_versions_table(db, table_names)
        print("Migration complete (empty atlas).")
        return

    # ── Step 2: Group by dataset_uid, compute layouts ─────────────────
    dataset_uids = old_df["dataset_uid"].unique().to_list()
    print(f"Found {len(dataset_uids)} unique dataset(s)")

    # Track layout_uid -> layout_df and dataset_uid -> layout_uid
    layout_dfs: dict[str, pl.DataFrame] = {}
    dataset_to_layout: dict[str, str] = {}

    for ds_uid in dataset_uids:
        ds_rows = old_df.filter(pl.col("dataset_uid") == ds_uid).sort("local_index")
        feature_uids = ds_rows["feature_uid"].to_list()
        layout_uid = compute_layout_uid(feature_uids)

        dataset_to_layout[ds_uid] = layout_uid

        if layout_uid not in layout_dfs:
            layout_dfs[layout_uid] = pl.DataFrame(
                {
                    "layout_uid": [layout_uid] * len(feature_uids),
                    "feature_uid": feature_uids,
                    "local_index": list(range(len(feature_uids))),
                    "global_index": ds_rows["global_index"].to_list(),
                }
            )

    print(f"Computed {len(layout_dfs)} unique layout(s) from {len(dataset_uids)} dataset(s)")
    for layout_uid, ldf in layout_dfs.items():
        print(f"  layout {layout_uid}: {len(ldf)} features")

    # ── Step 3: Create _feature_layouts table ─────────────────────────
    all_layouts = pl.concat(list(layout_dfs.values()))
    new_table = db.create_table("_feature_layouts", schema=FeatureLayout)
    new_table.add(all_layouts)
    new_table.create_fts_index("feature_uid")
    new_table.create_fts_index("layout_uid")
    print(f"Created _feature_layouts table with {len(all_layouts)} rows")

    # ── Step 4: Update datasets table with layout_uid ─────────────────
    _migrate_datasets_table(db, dataset_to_layout)

    # ── Step 5: Recreate atlas_versions table ─────────────────────────
    _migrate_versions_table(db, table_names)

    # ── Step 6: Migrate CSC data to zarr indptr (if applicable) ───────
    _migrate_csc_to_zarr(old_df, dataset_to_layout, zarr_store_path)

    # ── Step 7: Optionally drop old table ─────────────────────────────
    if drop_old:
        db.drop_table("_dataset_vars")
        print("Dropped old _dataset_vars table")
    else:
        print("Old _dataset_vars table retained. Re-run with --drop-old to remove it.")

    # ── Verify ────────────────────────────────────────────────────────
    verify_table = db.open_table("_feature_layouts")
    print(f"\nVerification: _feature_layouts has {verify_table.count_rows()} rows")
    print("Migration complete!")


def _migrate_datasets_table(
    db: lancedb.DBConnection,
    dataset_to_layout: dict[str, str] | None = None,
) -> None:
    """Add layout_uid column to datasets table."""
    ds_table = db.open_table("datasets")
    ds_df = ds_table.search().to_polars()

    if ds_df.is_empty():
        print("datasets table is empty, skipping layout_uid update")
        return

    if "layout_uid" in ds_df.columns:
        # Column exists, just update values
        if dataset_to_layout:
            ds_df = ds_df.with_columns(
                pl.col("uid")
                .map_elements(
                    lambda uid: dataset_to_layout.get(uid, ""),
                    return_dtype=pl.Utf8,
                )
                .alias("layout_uid")
            )
            ds_table.merge_insert(on="uid").when_matched_update_all().execute(ds_df)
            print(f"Updated layout_uid for {len(ds_df)} dataset(s)")
    else:
        # Need to add the column — drop and recreate with new data
        if dataset_to_layout:
            ds_df = ds_df.with_columns(
                pl.col("uid")
                .map_elements(
                    lambda uid: dataset_to_layout.get(uid, ""),
                    return_dtype=pl.Utf8,
                )
                .alias("layout_uid")
            )
        else:
            ds_df = ds_df.with_columns(pl.lit("").alias("layout_uid"))

        # LanceDB doesn't support ALTER TABLE ADD COLUMN, so we overwrite
        db.drop_table("datasets")
        db.create_table("datasets", data=ds_df)
        print(f"Recreated datasets table with layout_uid for {len(ds_df)} row(s)")


def _migrate_versions_table(
    db: lancedb.DBConnection,
    table_names: list[str],
) -> None:
    """Recreate atlas_versions with the new schema.

    Old snapshots referenced dataset_vars_table_version which no longer
    exists. Since snapshots are cheap to recreate, we drop and recreate
    the table with the new schema.
    """
    version_table_name = "atlas_versions"
    if version_table_name in table_names:
        old_vt = db.open_table(version_table_name)
        n_versions = old_vt.count_rows()
        if n_versions > 0:
            print(
                f"WARNING: Dropping {n_versions} old snapshot(s) from "
                f"atlas_versions (they referenced dataset_vars_table_version). "
                f"You can re-snapshot after migration."
            )
        db.drop_table(version_table_name)

    db.create_table(version_table_name, schema=AtlasVersionRecord)
    print("Recreated atlas_versions table with new schema")


def _migrate_csc_to_zarr(
    old_df: pl.DataFrame,
    dataset_to_layout: dict[str, str],
    zarr_store_path: str | None,
) -> None:
    """If old _dataset_vars had csc_start/csc_end, write zarr indptr arrays."""
    if "csc_start" not in old_df.columns or "csc_end" not in old_df.columns:
        return

    has_csc = old_df.filter(pl.col("csc_start").is_not_null() & pl.col("csc_end").is_not_null())
    if has_csc.is_empty():
        print("No CSC data found in _dataset_vars (csc_start/csc_end all null)")
        return

    if zarr_store_path is None:
        print(
            "WARNING: CSC data exists in _dataset_vars but no --zarr-store "
            "path provided. Skipping CSC indptr migration. Re-run with "
            "--zarr-store to migrate CSC data."
        )
        return

    import zarr as zarr_lib

    store = zarr_lib.storage.LocalStore(zarr_store_path)
    root = zarr_lib.open_group(store, mode="a")

    csc_datasets = has_csc["dataset_uid"].unique().to_list()
    print(f"Migrating CSC indptr for {len(csc_datasets)} dataset(s)...")

    for ds_uid in csc_datasets:
        ds_rows = has_csc.filter(pl.col("dataset_uid") == ds_uid).sort("local_index")
        csc_start = ds_rows["csc_start"].to_numpy()
        csc_end = ds_rows["csc_end"].to_numpy()
        n_features = len(ds_rows)

        # Build indptr: [csc_start[0], csc_start[1], ..., csc_end[-1]]
        indptr = np.zeros(n_features + 1, dtype=np.int64)
        indptr[:n_features] = csc_start
        indptr[n_features] = csc_end[-1] if len(csc_end) > 0 else 0

        # Write to zarr at {zarr_group}/csc/indptr
        # The zarr_group is typically the same as the dataset_uid
        zarr_group = ds_uid
        if zarr_group in root:
            group = root[zarr_group]
            if "csc" not in group:
                csc_group = group.create_group("csc")
            else:
                csc_group = group["csc"]

            if "indptr" not in csc_group:
                csc_group.create_array("indptr", data=indptr)
                print(f"  Wrote csc/indptr for {ds_uid} ({n_features + 1} entries)")
            else:
                print(f"  csc/indptr already exists for {ds_uid}, skipping")
        else:
            print(f"  WARNING: zarr group {zarr_group} not found, skipping CSC migration")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate a lancell atlas from _dataset_vars to _feature_layouts"
    )
    parser.add_argument(
        "db_uri",
        help="Path to the LanceDB database directory",
    )
    parser.add_argument(
        "--zarr-store",
        help="Path to the zarr store (needed only if CSC data exists in _dataset_vars)",
    )
    parser.add_argument(
        "--drop-old",
        action="store_true",
        help="Drop the old _dataset_vars table after migration",
    )
    args = parser.parse_args()

    migrate(
        args.db_uri,
        zarr_store_path=args.zarr_store,
        drop_old=args.drop_old,
    )


if __name__ == "__main__":
    main()
