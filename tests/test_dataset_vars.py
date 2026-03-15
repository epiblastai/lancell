"""Tests for lancell.dataset_vars — _dataset_vars Lance table helpers."""

from pathlib import Path

import lancedb
import numpy as np
import polars as pl
import pytest
import zarr

from lancell.dataset_vars import (
    build_dataset_vars_df,
    read_dataset_vars,
    reindex_registry,
    sync_dataset_vars_global_index,
    validate_dataset_vars,
)
from lancell.group_specs import get_spec
from lancell.schema import DatasetVar, FeatureBaseSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


def _make_registry(
    tmp_path: Path,
    uids: list[str],
    *,
    indexed: bool = True,
    db_name: str = "lancedb",
) -> lancedb.table.Table:
    """Create a LanceDB table with a GeneFeatureSchema registry."""
    db = lancedb.connect(str(tmp_path / db_name))
    records = [
        GeneFeatureSchema(
            uid=uid,
            global_index=i if indexed else None,
            gene_name=f"GENE{i}",
        )
        for i, uid in enumerate(uids)
    ]
    return db.create_table("genes", data=records, schema=GeneFeatureSchema)


def _make_dataset_vars_table(
    tmp_path: Path,
    db_name: str = "lancedb",
) -> lancedb.table.Table:
    """Create an empty _dataset_vars table."""
    db = lancedb.connect(str(tmp_path / db_name))
    return db.create_table("_dataset_vars", schema=DatasetVar)


def _make_store_and_group(tmp_path: Path):
    """Create a zarr group with a 10x5 dense data array."""
    group_prefix = "datasets/test/protein_abundance"
    zarr_root = zarr.open_group(str(tmp_path / group_prefix), mode="w")
    zarr_root.create_array("data", shape=(10, 5), dtype="float32", chunks=(10, 5))
    return group_prefix, zarr_root


# ---------------------------------------------------------------------------
# build_dataset_vars_df
# ---------------------------------------------------------------------------


class TestBuildDatasetVarsDf:
    def test_basic(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        var_df = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a"]})
        df = build_dataset_vars_df(var_df, "ds_001", registry)

        assert df.columns == [
            "feature_uid",
            "dataset_uid",
            "local_index",
            "global_index",
            "csc_start",
            "csc_end",
        ]
        assert df["feature_uid"].to_list() == ["uid_c", "uid_a"]
        assert df["dataset_uid"].to_list() == ["ds_001", "ds_001"]
        assert df["local_index"].to_list() == [0, 1]
        # uid_c has global_index=2, uid_a has global_index=0
        assert df["global_index"].to_list() == [2, 0]
        assert df["csc_start"].null_count() == 2
        assert df["csc_end"].null_count() == 2

    def test_missing_uid_raises(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])
        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_missing"]})
        with pytest.raises(ValueError, match="not found in registry"):
            build_dataset_vars_df(var_df, "ds_001", registry)

    def test_unindexed_registry_raises(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a", "uid_b"], indexed=False)
        var_df = pl.DataFrame({"global_feature_uid": ["uid_a"]})
        with pytest.raises(ValueError, match="no global_index"):
            build_dataset_vars_df(var_df, "ds_001", registry)

    def test_missing_column_raises(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])
        var_df = pl.DataFrame({"other_col": ["uid_a"]})
        with pytest.raises(ValueError, match="global_feature_uid"):
            build_dataset_vars_df(var_df, "ds_001", registry)

    def test_roundtrip_via_table(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        table = _make_dataset_vars_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b", "uid_c"]})
        df = build_dataset_vars_df(var_df, "ds_001", registry)
        table.add(df)

        result = read_dataset_vars(table, "ds_001")
        assert len(result) == 3
        assert result["local_index"].to_list() == [0, 1, 2]
        assert result["feature_uid"].to_list() == ["uid_a", "uid_b", "uid_c"]


# ---------------------------------------------------------------------------
# read_dataset_vars
# ---------------------------------------------------------------------------


class TestReadDatasetVars:
    def test_sorted_by_local_index(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        table = _make_dataset_vars_table(tmp_path)

        # Insert two datasets
        var_df1 = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a"]})
        var_df2 = pl.DataFrame({"global_feature_uid": ["uid_b"]})
        table.add(build_dataset_vars_df(var_df1, "ds_001", registry))
        table.add(build_dataset_vars_df(var_df2, "ds_002", registry))

        result = read_dataset_vars(table, "ds_001")
        assert result["feature_uid"].to_list() == ["uid_c", "uid_a"]
        assert result["local_index"].to_list() == [0, 1]

        result2 = read_dataset_vars(table, "ds_002")
        assert result2["feature_uid"].to_list() == ["uid_b"]

    def test_empty_for_unknown_dataset(self, tmp_path):
        table = _make_dataset_vars_table(tmp_path)
        result = read_dataset_vars(table, "nonexistent")
        assert result.is_empty()


# ---------------------------------------------------------------------------
# sync_dataset_vars_global_index
# ---------------------------------------------------------------------------


class TestSyncDatasetVarsGlobalIndex:
    def test_basic_sync(self, tmp_path):
        uids = ["uid_c", "uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids, indexed=False)
        table = _make_dataset_vars_table(tmp_path)

        # Reindex assigns uid_a=0, uid_b=1, uid_c=2
        reindex_registry(registry)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a"]})
        df = build_dataset_vars_df(var_df, "ds_001", registry)
        table.add(df)

        # Simulate registry change: add a new feature and reindex
        # (global_index values in _dataset_vars are now stale)
        # For this test, manually mess up global_index then sync
        rows = read_dataset_vars(table, "ds_001")
        stale = rows.with_columns(pl.Series("global_index", [-1, -1]))
        (
            table.merge_insert(on=["feature_uid", "dataset_uid"])
            .when_matched_update_all()
            .execute(stale)
        )

        # Sync restores correct values
        n = sync_dataset_vars_global_index(table, registry)
        assert n == 2

        result = read_dataset_vars(table, "ds_001")
        # uid_c=2, uid_a=0 after reindex
        assert result["global_index"].to_list() == [2, 0]

    def test_empty_registry(self, tmp_path):
        db = lancedb.connect(str(tmp_path / "lancedb"))
        registry = db.create_table("genes_empty", schema=GeneFeatureSchema)
        table = _make_dataset_vars_table(tmp_path)
        assert sync_dataset_vars_global_index(table, registry) == 0

    def test_empty_dataset_vars(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])
        table = _make_dataset_vars_table(tmp_path)
        assert sync_dataset_vars_global_index(table, registry) == 0


# ---------------------------------------------------------------------------
# validate_dataset_vars
# ---------------------------------------------------------------------------


class TestValidateDatasetVars:
    def test_valid_dense(self, tmp_path):
        group_prefix, zarr_group = _make_store_and_group(tmp_path)
        spec = get_spec("protein_abundance")
        uids = [f"u{i}" for i in range(5)]
        registry = _make_registry(tmp_path, uids)
        table = _make_dataset_vars_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": uids})
        table.add(build_dataset_vars_df(var_df, "ds_001", registry))

        errors = validate_dataset_vars(
            table, "ds_001", spec=spec, group=zarr_group, registry_table=registry
        )
        assert errors == []

    def test_wrong_row_count(self, tmp_path):
        group_prefix, zarr_group = _make_store_and_group(tmp_path)
        spec = get_spec("protein_abundance")
        # zarr has 5 features but we only insert 2
        uids = ["u0", "u1"]
        registry = _make_registry(tmp_path, [f"u{i}" for i in range(5)])
        table = _make_dataset_vars_table(tmp_path)

        # Only insert 2 rows (zarr has 5)
        var_df = pl.DataFrame({"global_feature_uid": uids})
        table.add(build_dataset_vars_df(var_df, "ds_001", registry))

        errors = validate_dataset_vars(table, "ds_001", spec=spec, group=zarr_group)
        assert any("2 rows but expected 5" in e for e in errors)

    def test_registry_unresolved(self, tmp_path):
        spec = get_spec("gene_expression")
        registry = _make_registry(tmp_path, ["uid_a"])
        table = _make_dataset_vars_table(tmp_path)

        # Manually insert a row with a uid not in registry (bypassing build_dataset_vars_df)
        table.add(
            pl.DataFrame(
                {
                    "feature_uid": ["uid_MISSING"],
                    "dataset_uid": ["ds_001"],
                    "local_index": [0],
                    "global_index": [0],
                    "csc_start": pl.Series([None], dtype=pl.Int64),
                    "csc_end": pl.Series([None], dtype=pl.Int64),
                }
            )
        )

        errors = validate_dataset_vars(table, "ds_001", spec=spec, registry_table=registry)
        assert any("not found in registry" in e for e in errors)

    def test_sparse_no_count_needed(self, tmp_path):
        spec = get_spec("gene_expression")
        registry = _make_registry(tmp_path, ["uid_a", "uid_b"])
        table = _make_dataset_vars_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b"]})
        table.add(build_dataset_vars_df(var_df, "ds_001", registry))

        # No group — sparse can't derive count, so no row count error
        errors = validate_dataset_vars(table, "ds_001", spec=spec)
        assert errors == []


# ---------------------------------------------------------------------------
# reindex_registry (kept from old test_var_df.py)
# ---------------------------------------------------------------------------


class TestReindexRegistry:
    def test_assigns_contiguous_indices(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_c", "uid_a", "uid_b"], indexed=False)
        n = reindex_registry(registry)
        assert n == 3

        df = registry.search().select(["uid", "global_index"]).to_polars()
        df = df.sort("uid")
        assert df["uid"].to_list() == ["uid_a", "uid_b", "uid_c"]
        assert df["global_index"].to_list() == [0, 1, 2]

    def test_reindex_is_deterministic(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_z", "uid_a", "uid_m"], indexed=False)
        reindex_registry(registry)
        df1 = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")

        reindex_registry(registry)
        df2 = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")

        assert df1["global_index"].to_list() == df2["global_index"].to_list()

    def test_reindex_overwrites_existing_indices(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_b", "uid_a"], indexed=True)
        reindex_registry(registry)
        df = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")
        assert df["uid"].to_list() == ["uid_a", "uid_b"]
        assert df["global_index"].to_list() == [0, 1]

    def test_empty_table(self, tmp_path):
        db = lancedb.connect(str(tmp_path / "lancedb"))
        table = db.create_table("empty", schema=GeneFeatureSchema)
        assert reindex_registry(table) == 0

    def test_build_after_reindex(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_c", "uid_a", "uid_b"], indexed=False)
        reindex_registry(registry)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_b", "uid_c"]})
        df = build_dataset_vars_df(var_df, "ds_001", registry)
        # uid_a=0, uid_b=1, uid_c=2 after reindex
        assert df["global_index"].to_list() == [1, 2]


# ---------------------------------------------------------------------------
# GroupReader remap caching via Lance version
# ---------------------------------------------------------------------------


class TestGroupReaderRemap:
    def test_cold_cache(self, tmp_path):
        from lancell.group_reader import GroupReader

        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        table = _make_dataset_vars_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a", "uid_b"]})
        table.add(build_dataset_vars_df(var_df, "ds_001", registry))

        import obstore

        store = obstore.store.MemoryStore()
        zarr_root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
        zarr_root.create_group("my_group")

        gr = GroupReader.from_atlas_root(
            zarr_group="my_group",
            feature_space="gene_expression",
            store=store,
            dataset_vars_table=table,
            dataset_uid="ds_001",
        )
        remap = gr.get_remap()
        # uid_c=2, uid_a=0, uid_b=1
        np.testing.assert_array_equal(remap, [2, 0, 1])
        assert remap.dtype == np.int32

    def test_warm_cache(self, tmp_path):
        from lancell.group_reader import GroupReader

        uids = ["uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids)
        table = _make_dataset_vars_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b"]})
        table.add(build_dataset_vars_df(var_df, "ds_001", registry))

        import obstore

        store = obstore.store.MemoryStore()
        zarr_root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
        zarr_root.create_group("my_group")

        gr = GroupReader.from_atlas_root(
            zarr_group="my_group",
            feature_space="gene_expression",
            store=store,
            dataset_vars_table=table,
            dataset_uid="ds_001",
        )
        remap1 = gr.get_remap()
        remap2 = gr.get_remap()
        # Same object (warm cache)
        assert remap1 is remap2

    def test_cache_invalidated_by_table_version(self, tmp_path):
        from lancell.group_reader import GroupReader

        uids = ["uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids)
        table = _make_dataset_vars_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b"]})
        table.add(build_dataset_vars_df(var_df, "ds_001", registry))

        import obstore

        store = obstore.store.MemoryStore()
        zarr_root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
        zarr_root.create_group("my_group")

        gr = GroupReader.from_atlas_root(
            zarr_group="my_group",
            feature_space="gene_expression",
            store=store,
            dataset_vars_table=table,
            dataset_uid="ds_001",
        )
        remap1 = gr.get_remap()
        np.testing.assert_array_equal(remap1, [0, 1])

        # Simulate reindex: swap global indices
        rows = read_dataset_vars(table, "ds_001")
        rows = rows.with_columns(pl.Series("global_index", [1, 0]))
        (
            table.merge_insert(on=["feature_uid", "dataset_uid"])
            .when_matched_update_all()
            .execute(rows)
        )

        # Cache is stale — next call rebuilds
        remap2 = gr.get_remap()
        np.testing.assert_array_equal(remap2, [1, 0])
        assert remap1 is not remap2

    def test_worker_path_returns_frozen_remap(self, tmp_path):
        import obstore

        from lancell.group_reader import GroupReader

        store = obstore.store.MemoryStore()
        remap = np.array([3, 1, 2], dtype=np.int32)
        gr = GroupReader.for_worker("my_group", "gene_expression", store, remap)

        result = gr.get_remap()
        np.testing.assert_array_equal(result, remap)
        assert result is remap  # frozen — same object


# ---------------------------------------------------------------------------
# FeatureBaseSchema (kept for completeness)
# ---------------------------------------------------------------------------


class TestFeatureBaseSchema:
    def test_global_index_optional(self):
        f = FeatureBaseSchema(uid="test")
        assert f.global_index is None

    def test_accepts_global_index(self):
        f = FeatureBaseSchema(uid="test", global_index=42)
        assert f.global_index == 42

    def test_subclass(self):
        g = GeneFeatureSchema(uid="g1", global_index=0, gene_name="TP53")
        assert g.uid == "g1"
        assert g.global_index == 0
        assert g.gene_name == "TP53"
