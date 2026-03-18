"""Tests for lancell.feature_layouts — _feature_layouts Lance table helpers."""

from pathlib import Path

import lancedb
import numpy as np
import polars as pl
import pytest
import zarr

from lancell.feature_layouts import (
    build_feature_layout_df,
    compute_layout_uid,
    layout_exists,
    read_feature_layout,
    reindex_registry,
    sync_layouts_global_index,
    validate_feature_layout,
)
from lancell.group_specs import get_spec
from lancell.schema import FeatureBaseSchema, FeatureLayout

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


def _make_feature_layouts_table(
    tmp_path: Path,
    db_name: str = "lancedb",
) -> lancedb.table.Table:
    """Create an empty _feature_layouts table."""
    db = lancedb.connect(str(tmp_path / db_name))
    return db.create_table("_feature_layouts", schema=FeatureLayout)


# ---------------------------------------------------------------------------
# compute_layout_uid
# ---------------------------------------------------------------------------


class TestComputeLayoutUid:
    def test_deterministic(self):
        uids = ["uid_a", "uid_b", "uid_c"]
        assert compute_layout_uid(uids) == compute_layout_uid(uids)

    def test_order_matters(self):
        assert compute_layout_uid(["uid_a", "uid_b"]) != compute_layout_uid(["uid_b", "uid_a"])

    def test_length_16_hex(self):
        result = compute_layout_uid(["uid_a", "uid_b"])
        assert len(result) == 16
        int(result, 16)  # must be valid hex

    def test_different_inputs_different_uids(self):
        uid1 = compute_layout_uid(["uid_a"])
        uid2 = compute_layout_uid(["uid_b"])
        assert uid1 != uid2


# ---------------------------------------------------------------------------
# build_feature_layout_df
# ---------------------------------------------------------------------------


class TestBuildFeatureLayoutDf:
    def test_basic(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        var_df = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a"]})
        layout_uid, df = build_feature_layout_df(var_df, registry)

        assert df.columns == [
            "layout_uid",
            "feature_uid",
            "local_index",
            "global_index",
        ]
        assert df["feature_uid"].to_list() == ["uid_c", "uid_a"]
        assert df["layout_uid"].to_list() == [layout_uid, layout_uid]
        assert df["local_index"].to_list() == [0, 1]
        # uid_c has global_index=2, uid_a has global_index=0
        assert df["global_index"].to_list() == [2, 0]

    def test_missing_uid_raises(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])
        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_missing"]})
        with pytest.raises(ValueError, match="not found in registry"):
            build_feature_layout_df(var_df, registry)

    def test_unindexed_registry_returns_null_global_index(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a", "uid_b"], indexed=False)
        var_df = pl.DataFrame({"global_feature_uid": ["uid_a"]})
        layout_uid, df = build_feature_layout_df(var_df, registry)
        assert df["feature_uid"].to_list() == ["uid_a"]
        assert df["global_index"].null_count() == 1

    def test_missing_column_raises(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])
        var_df = pl.DataFrame({"other_col": ["uid_a"]})
        with pytest.raises(ValueError, match="global_feature_uid"):
            build_feature_layout_df(var_df, registry)

    def test_roundtrip_via_table(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        table = _make_feature_layouts_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b", "uid_c"]})
        layout_uid, df = build_feature_layout_df(var_df, registry)
        table.add(df)

        result = read_feature_layout(table, layout_uid)
        assert len(result) == 3
        assert result["local_index"].to_list() == [0, 1, 2]
        assert result["feature_uid"].to_list() == ["uid_a", "uid_b", "uid_c"]


# ---------------------------------------------------------------------------
# layout_exists / read_feature_layout
# ---------------------------------------------------------------------------


class TestLayoutExistsAndRead:
    def test_exists_false_on_empty(self, tmp_path):
        table = _make_feature_layouts_table(tmp_path)
        assert not layout_exists(table, "nonexistent")

    def test_exists_true_after_insert(self, tmp_path):
        uids = ["uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids)
        table = _make_feature_layouts_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": uids})
        layout_uid, df = build_feature_layout_df(var_df, registry)
        table.add(df)

        assert layout_exists(table, layout_uid)
        assert not layout_exists(table, "nonexistent")

    def test_read_sorted_by_local_index(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        table = _make_feature_layouts_table(tmp_path)

        # Insert two layouts with different orderings
        var_df1 = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a"]})
        var_df2 = pl.DataFrame({"global_feature_uid": ["uid_b"]})
        lid1, df1 = build_feature_layout_df(var_df1, registry)
        lid2, df2 = build_feature_layout_df(var_df2, registry)
        table.add(df1)
        table.add(df2)

        result1 = read_feature_layout(table, lid1)
        assert result1["feature_uid"].to_list() == ["uid_c", "uid_a"]
        assert result1["local_index"].to_list() == [0, 1]

        result2 = read_feature_layout(table, lid2)
        assert result2["feature_uid"].to_list() == ["uid_b"]

    def test_empty_for_unknown_layout(self, tmp_path):
        table = _make_feature_layouts_table(tmp_path)
        result = read_feature_layout(table, "nonexistent")
        assert result.is_empty()


# ---------------------------------------------------------------------------
# Layout reuse
# ---------------------------------------------------------------------------


class TestLayoutReuse:
    def test_same_features_same_layout_uid(self, tmp_path):
        """Two datasets with identical feature orderings get the same layout_uid."""
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)

        var_df = pl.DataFrame({"global_feature_uid": uids})
        lid1, _ = build_feature_layout_df(var_df, registry)
        lid2, _ = build_feature_layout_df(var_df, registry)
        assert lid1 == lid2

    def test_different_order_different_layout_uid(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)

        var_df1 = pl.DataFrame({"global_feature_uid": uids})
        var_df2 = pl.DataFrame({"global_feature_uid": list(reversed(uids))})
        lid1, _ = build_feature_layout_df(var_df1, registry)
        lid2, _ = build_feature_layout_df(var_df2, registry)
        assert lid1 != lid2


# ---------------------------------------------------------------------------
# sync_layouts_global_index
# ---------------------------------------------------------------------------


class TestSyncLayoutsGlobalIndex:
    def test_basic_sync(self, tmp_path):
        """Sync fills NULL global_index from registry (the post-reindex_registry path)."""
        uids = ["uid_c", "uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids, indexed=False)
        table = _make_feature_layouts_table(tmp_path)

        # Build layout BEFORE reindex — global_index will be NULL
        var_df = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a"]})
        layout_uid, df = build_feature_layout_df(var_df, registry)
        assert df["global_index"].null_count() == 2
        table.add(df)

        # Now index the registry
        reindex_registry(registry)

        # Look up actual assigned indices
        reg_df = registry.search().select(["uid", "global_index"]).to_polars()
        gi_c = int(reg_df.filter(pl.col("uid") == "uid_c")["global_index"][0])
        gi_a = int(reg_df.filter(pl.col("uid") == "uid_a")["global_index"][0])

        # Sync propagates the new indices to NULL layout rows
        n = sync_layouts_global_index(table, registry)
        assert n == 2

        result = read_feature_layout(table, layout_uid)
        assert result["global_index"].to_list() == [gi_c, gi_a]

    def test_empty_registry(self, tmp_path):
        db = lancedb.connect(str(tmp_path / "lancedb"))
        registry = db.create_table("genes_empty", schema=GeneFeatureSchema)
        table = _make_feature_layouts_table(tmp_path)
        assert sync_layouts_global_index(table, registry) == 0

    def test_empty_layouts(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])
        table = _make_feature_layouts_table(tmp_path)
        assert sync_layouts_global_index(table, registry) == 0


# ---------------------------------------------------------------------------
# validate_feature_layout
# ---------------------------------------------------------------------------


class TestValidateFeatureLayout:
    def test_registry_unresolved(self, tmp_path):
        spec = get_spec("gene_expression")
        registry = _make_registry(tmp_path, ["uid_a"])
        table = _make_feature_layouts_table(tmp_path)

        # Manually insert a row with a uid not in registry
        table.add(
            pl.DataFrame(
                {
                    "layout_uid": ["lay_001"],
                    "feature_uid": ["uid_MISSING"],
                    "local_index": [0],
                    "global_index": [0],
                }
            )
        )

        errors = validate_feature_layout(table, "lay_001", spec=spec, registry_table=registry)
        assert any("not found in registry" in e for e in errors)

    def test_sparse_no_count_needed(self, tmp_path):
        spec = get_spec("gene_expression")
        registry = _make_registry(tmp_path, ["uid_a", "uid_b"])
        table = _make_feature_layouts_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b"]})
        layout_uid, df = build_feature_layout_df(var_df, registry)
        table.add(df)

        # No group — sparse can't derive count, so no row count error
        errors = validate_feature_layout(table, layout_uid, spec=spec)
        assert errors == []


# ---------------------------------------------------------------------------
# reindex_registry
# ---------------------------------------------------------------------------


class TestReindexRegistry:
    def test_assigns_indices_to_unindexed(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_c", "uid_a", "uid_b"], indexed=False)
        n = reindex_registry(registry)
        assert n == 3

        df = registry.search().select(["uid", "global_index"]).to_polars()
        assert df["global_index"].null_count() == 0
        assert df["global_index"].n_unique() == 3

    def test_reindex_is_deterministic(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_z", "uid_a", "uid_m"], indexed=False)
        reindex_registry(registry)
        df1 = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")

        reindex_registry(registry)
        df2 = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")

        assert df1["global_index"].to_list() == df2["global_index"].to_list()

    def test_reindex_is_noop_when_already_indexed(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_b", "uid_a"], indexed=True)
        df_before = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")
        n = reindex_registry(registry)
        assert n == 0
        df_after = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")
        assert df_before["global_index"].to_list() == df_after["global_index"].to_list()

    def test_empty_table(self, tmp_path):
        db = lancedb.connect(str(tmp_path / "lancedb"))
        table = db.create_table("empty", schema=GeneFeatureSchema)
        assert reindex_registry(table) == 0

    def test_build_after_reindex(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_c", "uid_a", "uid_b"], indexed=False)
        reindex_registry(registry)

        reg_df = registry.search().select(["uid", "global_index"]).to_polars()
        gi_b = int(reg_df.filter(pl.col("uid") == "uid_b")["global_index"][0])
        gi_c = int(reg_df.filter(pl.col("uid") == "uid_c")["global_index"][0])

        var_df = pl.DataFrame({"global_feature_uid": ["uid_b", "uid_c"]})
        _, df = build_feature_layout_df(var_df, registry)
        assert df["global_index"].to_list() == [gi_b, gi_c]


# ---------------------------------------------------------------------------
# GroupReader remap caching via feature layouts
# ---------------------------------------------------------------------------


class TestGroupReaderRemap:
    def test_cold_cache(self, tmp_path):
        from lancell.group_reader import GroupReader

        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        table = _make_feature_layouts_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a", "uid_b"]})
        layout_uid, df = build_feature_layout_df(var_df, registry)
        table.add(df)

        import obstore

        store = obstore.store.MemoryStore()
        zarr_root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
        zarr_root.create_group("my_group")

        gr = GroupReader.from_atlas_root(
            zarr_group="my_group",
            feature_space="gene_expression",
            store=store,
            feature_layouts_table=table,
            layout_uid=layout_uid,
        )
        remap = gr.get_remap()
        # uid_c=2, uid_a=0, uid_b=1
        np.testing.assert_array_equal(remap, [2, 0, 1])
        assert remap.dtype == np.int32

    def test_warm_cache(self, tmp_path):
        from lancell.group_reader import GroupReader

        uids = ["uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids)
        table = _make_feature_layouts_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b"]})
        layout_uid, df = build_feature_layout_df(var_df, registry)
        table.add(df)

        import obstore

        store = obstore.store.MemoryStore()
        zarr_root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
        zarr_root.create_group("my_group")

        gr = GroupReader.from_atlas_root(
            zarr_group="my_group",
            feature_space="gene_expression",
            store=store,
            feature_layouts_table=table,
            layout_uid=layout_uid,
        )
        remap1 = gr.get_remap()
        remap2 = gr.get_remap()
        # Same object (warm cache)
        assert remap1 is remap2

    def test_remap_load_once_ignores_table_mutations(self, tmp_path):
        """get_remap() is load-once: mutations to the table after first load are not seen."""
        from lancell.group_reader import GroupReader

        uids = ["uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids)
        table = _make_feature_layouts_table(tmp_path)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b"]})
        layout_uid, df = build_feature_layout_df(var_df, registry)
        table.add(df)

        import obstore

        store = obstore.store.MemoryStore()
        zarr_root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
        zarr_root.create_group("my_group")

        gr = GroupReader.from_atlas_root(
            zarr_group="my_group",
            feature_space="gene_expression",
            store=store,
            feature_layouts_table=table,
            layout_uid=layout_uid,
        )
        remap1 = gr.get_remap()
        np.testing.assert_array_equal(remap1, [0, 1])

        # Mutate the underlying table
        rows = read_feature_layout(table, layout_uid)
        rows = rows.with_columns(pl.Series("global_index", [1, 0]))
        (
            table.merge_insert(on=["layout_uid", "feature_uid"])
            .when_matched_update_all()
            .execute(rows)
        )

        # Load-once: same cached object returned, mutation is not visible
        remap2 = gr.get_remap()
        np.testing.assert_array_equal(remap2, [0, 1])
        assert remap1 is remap2

    def test_worker_path_returns_frozen_remap(self, tmp_path):
        import obstore

        from lancell.group_reader import GroupReader

        store = obstore.store.MemoryStore()
        remap = np.array([3, 1, 2], dtype=np.int32)
        gr = GroupReader.for_worker("my_group", "gene_expression", store, remap)

        result = gr.get_remap()
        np.testing.assert_array_equal(result, remap)
        assert result is remap  # frozen — same object

    def test_has_csc_via_zarr(self, tmp_path):
        """has_csc checks for zarr indptr existence."""
        import obstore

        from lancell.group_reader import GroupReader

        store = obstore.store.MemoryStore()
        zarr_root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
        grp = zarr_root.create_group("my_group")

        gr = GroupReader.for_worker(
            "my_group", "gene_expression", store, np.array([0], dtype=np.int32)
        )
        assert not gr.has_csc

        # Create csc/indptr
        csc = grp.create_group("csc")
        csc.create_array("indptr", data=np.array([0, 5, 10], dtype=np.int64))

        # Need fresh GroupReader since zarr handle is cached
        gr2 = GroupReader.for_worker(
            "my_group", "gene_expression", store, np.array([0], dtype=np.int32)
        )
        assert gr2.has_csc

    def test_get_csc_indptr(self, tmp_path):
        """get_csc_indptr() loads from zarr and caches."""
        import obstore

        from lancell.group_reader import GroupReader

        store = obstore.store.MemoryStore()
        zarr_root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")
        grp = zarr_root.create_group("my_group")
        csc = grp.create_group("csc")
        expected_indptr = np.array([0, 3, 7, 10], dtype=np.int64)
        csc.create_array("indptr", data=expected_indptr)

        gr = GroupReader.for_worker(
            "my_group", "gene_expression", store, np.array([0, 1, 2], dtype=np.int32)
        )
        indptr = gr.get_csc_indptr()
        np.testing.assert_array_equal(indptr, expected_indptr)
        # Cached
        assert gr.get_csc_indptr() is indptr


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
