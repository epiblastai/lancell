"""Tests for lancell.var_df — sidecar read/write/validate/remap."""

from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np
import polars as pl
import pytest
import zarr

from lancell.group_specs import ZARR_SPECS, FeatureSpace
from lancell.schema import FeatureBaseSchema
from lancell.var_df import (
    REMAP_FILENAME,
    VAR_DF_FILENAME,
    VarDfColumnSchema,
    build_remap,
    read_remap,
    read_var_df,
    reindex_registry,
    remap_path,
    validate_remap,
    validate_var_df,
    var_df_path,
    write_remap,
    write_var_df,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


def _make_store_and_group(tmp_path: Path):
    """Create a local obstore + zarr group with a 10x5 dense data array."""
    import obstore

    store = obstore.store.LocalStore(prefix=str(tmp_path))
    group_prefix = "datasets/test/protein_abundance"

    zarr_root = zarr.open_group(str(tmp_path / group_prefix), mode="w")
    zarr_root.create_array("data", shape=(10, 5), dtype="float32", chunks=(10, 5))
    return store, group_prefix, zarr_root


def _make_registry(
    tmp_path: Path,
    uids: list[str],
    *,
    indexed: bool = True,
) -> lancedb.table.Table:
    """Create a LanceDB table with a GeneFeatureSchema registry."""
    db = lancedb.connect(str(tmp_path / "lancedb"))
    records = [
        GeneFeatureSchema(
            uid=uid,
            global_index=i if indexed else None,
            gene_name=f"GENE{i}",
        )
        for i, uid in enumerate(uids)
    ]
    return db.create_table("genes", data=records, schema=GeneFeatureSchema)


# ---------------------------------------------------------------------------
# VarDfColumnSchema
# ---------------------------------------------------------------------------


class TestVarDfColumnSchema:
    def test_required_columns_base(self):
        assert VarDfColumnSchema.required_columns() == {"global_feature_uid"}

    def test_subclass_adds_required(self):
        class Extended(VarDfColumnSchema):
            gene_name: str

        assert Extended.required_columns() == {"global_feature_uid", "gene_name"}

    def test_subclass_optional_not_required(self):
        class Extended(VarDfColumnSchema):
            gene_name: str | None = None

        assert Extended.required_columns() == {"global_feature_uid"}

    def test_validate_df_passes(self):
        df = pl.DataFrame({"global_feature_uid": ["a"]})
        assert VarDfColumnSchema.validate_df(df) == []

    def test_validate_df_fails(self):
        df = pl.DataFrame({"other": ["a"]})
        errors = VarDfColumnSchema.validate_df(df)
        assert len(errors) == 1
        assert "Missing required columns" in errors[0]

    def test_validate_df_subclass(self):
        class Extended(VarDfColumnSchema):
            gene_name: str

        df = pl.DataFrame({"global_feature_uid": ["a"]})
        errors = Extended.validate_df(df)
        assert any("gene_name" in str(e) for e in errors)

        df2 = pl.DataFrame({"global_feature_uid": ["a"], "gene_name": ["TP53"]})
        assert Extended.validate_df(df2) == []


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


class TestPaths:
    def test_var_df_path(self):
        assert var_df_path("s3://b/group") == f"s3://b/group/{VAR_DF_FILENAME}"

    def test_var_df_path_strips_trailing_slash(self):
        assert var_df_path("s3://b/group/") == f"s3://b/group/{VAR_DF_FILENAME}"

    def test_remap_path(self):
        assert remap_path("local/group") == f"local/group/{REMAP_FILENAME}"


# ---------------------------------------------------------------------------
# Write / read round-trip
# ---------------------------------------------------------------------------


class TestWriteRead:
    def test_var_df_roundtrip(self, tmp_path):
        store, group_prefix, _ = _make_store_and_group(tmp_path)
        df = pl.DataFrame({
            "global_feature_uid": ["a", "b", "c", "d", "e"],
            "feature_name": ["F0", "F1", "F2", "F3", "F4"],
        })
        write_var_df(store, group_prefix, df)
        result = read_var_df(store, group_prefix)
        assert result.equals(df)

    def test_write_var_df_rejects_missing_column(self, tmp_path):
        store, group_prefix, _ = _make_store_and_group(tmp_path)
        df = pl.DataFrame({"feature_name": ["F0"]})
        with pytest.raises(ValueError, match="missing required columns"):
            write_var_df(store, group_prefix, df)

    def test_write_var_df_with_custom_schema(self, tmp_path):
        class MySchema(VarDfColumnSchema):
            gene_name: str

        store, group_prefix, _ = _make_store_and_group(tmp_path)
        df = pl.DataFrame({"global_feature_uid": ["a"], "gene_name": ["TP53"]})
        write_var_df(store, group_prefix, df, schema=MySchema)
        result = read_var_df(store, group_prefix)
        assert result.equals(df)

    def test_write_var_df_custom_schema_rejects(self, tmp_path):
        class MySchema(VarDfColumnSchema):
            gene_name: str

        store, group_prefix, _ = _make_store_and_group(tmp_path)
        df = pl.DataFrame({"global_feature_uid": ["a"]})
        with pytest.raises(ValueError, match="missing required columns"):
            write_var_df(store, group_prefix, df, schema=MySchema)

    def test_remap_roundtrip(self, tmp_path):
        store, group_prefix, _ = _make_store_and_group(tmp_path)
        remap = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        write_remap(store, group_prefix, remap)
        result = read_remap(store, group_prefix)
        np.testing.assert_array_equal(result, remap)

    def test_write_remap_rejects_2d(self, tmp_path):
        store, group_prefix, _ = _make_store_and_group(tmp_path)
        with pytest.raises(ValueError, match="1-D"):
            write_remap(store, group_prefix, np.zeros((2, 3), dtype=np.int32))


# ---------------------------------------------------------------------------
# build_remap
# ---------------------------------------------------------------------------


class TestBuildRemap:
    def test_basic(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        var_df = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a"]})
        remap = build_remap(var_df, registry)
        np.testing.assert_array_equal(remap, [2, 0])

    def test_missing_uid_raises(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])
        var_df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_missing"]})
        with pytest.raises(ValueError, match="not found in registry"):
            build_remap(var_df, registry)

    def test_unindexed_registry_raises(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a", "uid_b"], indexed=False)
        var_df = pl.DataFrame({"global_feature_uid": ["uid_a"]})
        with pytest.raises(ValueError, match="no global_index"):
            build_remap(var_df, registry)


# ---------------------------------------------------------------------------
# validate_var_df
# ---------------------------------------------------------------------------


class TestValidateVarDf:
    def test_valid_dense(self, tmp_path):
        _, _, group = _make_store_and_group(tmp_path)
        spec = ZARR_SPECS[FeatureSpace.PROTEIN_ABUNDANCE]
        df = pl.DataFrame({
            "global_feature_uid": [f"u{i}" for i in range(5)],
        })
        errors = validate_var_df(df, spec=spec, group=group)
        assert errors == []

    def test_wrong_row_count_dense(self, tmp_path):
        _, _, group = _make_store_and_group(tmp_path)
        spec = ZARR_SPECS[FeatureSpace.PROTEIN_ABUNDANCE]
        df = pl.DataFrame({"global_feature_uid": ["u0", "u1"]})
        errors = validate_var_df(df, spec=spec, group=group)
        assert any("2 rows but expected 5" in e for e in errors)

    def test_explicit_feature_count_overrides_group(self, tmp_path):
        _, _, group = _make_store_and_group(tmp_path)
        spec = ZARR_SPECS[FeatureSpace.PROTEIN_ABUNDANCE]
        df = pl.DataFrame({"global_feature_uid": ["u0", "u1", "u2"]})
        # group says 5, but explicit says 3 — explicit wins
        errors = validate_var_df(
            df, spec=spec, group=group, expected_feature_count=3
        )
        assert errors == []

    def test_sparse_needs_explicit_count(self, tmp_path):
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        df = pl.DataFrame({"global_feature_uid": ["u0", "u1"]})
        # no group, no explicit count → no row count error (can't check)
        errors = validate_var_df(df, spec=spec)
        assert errors == []

    def test_sparse_with_explicit_count(self, tmp_path):
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        df = pl.DataFrame({"global_feature_uid": ["u0", "u1"]})
        errors = validate_var_df(df, spec=spec, expected_feature_count=10)
        assert any("2 rows but expected 10" in e for e in errors)

    def test_missing_required_column(self):
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        df = pl.DataFrame({"feature_name": ["x"]})
        errors = validate_var_df(df, spec=spec)
        assert any("Missing required columns" in e for e in errors)

    def test_custom_schema_validation(self):
        class MySchema(VarDfColumnSchema):
            gene_name: str

        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        df = pl.DataFrame({"global_feature_uid": ["u0"]})
        errors = validate_var_df(df, spec=spec, schema=MySchema)
        assert any("gene_name" in str(e) for e in errors)

    def test_null_uids(self):
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        df = pl.DataFrame({"global_feature_uid": ["u0", None, "u2"]})
        errors = validate_var_df(df, spec=spec)
        assert any("null" in e for e in errors)

    def test_duplicate_uids(self):
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        df = pl.DataFrame({"global_feature_uid": ["u0", "u0", "u2"]})
        errors = validate_var_df(df, spec=spec)
        assert any("duplicate" in e.lower() for e in errors)

    def test_registry_resolution(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_b"]})
        errors = validate_var_df(df, spec=spec, registry_table=registry)
        assert errors == []

    def test_registry_unresolved(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        df = pl.DataFrame({"global_feature_uid": ["uid_a", "uid_MISSING"]})
        errors = validate_var_df(df, spec=spec, registry_table=registry)
        assert any("not found in registry" in e for e in errors)

    def test_global_index_agreement(self, tmp_path):
        uids = ["uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids)
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        # correct indices
        df = pl.DataFrame({
            "global_feature_uid": ["uid_a", "uid_b"],
            "global_index": [0, 1],
        })
        errors = validate_var_df(df, spec=spec, registry_table=registry)
        assert errors == []

    def test_global_index_mismatch(self, tmp_path):
        uids = ["uid_a", "uid_b"]
        registry = _make_registry(tmp_path, uids)
        spec = ZARR_SPECS[FeatureSpace.GENE_EXPRESSION]
        # swapped indices
        df = pl.DataFrame({
            "global_feature_uid": ["uid_a", "uid_b"],
            "global_index": [1, 0],
        })
        errors = validate_var_df(df, spec=spec, registry_table=registry)
        assert any("mismatch" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# validate_remap
# ---------------------------------------------------------------------------


class TestValidateRemap:
    def test_valid(self, tmp_path):
        uids = ["uid_a", "uid_b", "uid_c"]
        registry = _make_registry(tmp_path, uids)
        var_df = pl.DataFrame({"global_feature_uid": ["uid_c", "uid_a"]})
        remap = np.array([2, 0], dtype=np.int32)
        errors = validate_remap(remap, var_df=var_df, registry_table=registry)
        assert errors == []

    def test_length_mismatch(self):
        var_df = pl.DataFrame({"global_feature_uid": ["a", "b", "c"]})
        remap = np.array([0, 1], dtype=np.int32)
        errors = validate_remap(remap, var_df=var_df)
        assert any("length" in e for e in errors)

    def test_invalid_index(self, tmp_path):
        registry = _make_registry(tmp_path, ["uid_a"])  # only global_index=0
        remap = np.array([0, 999], dtype=np.int32)
        errors = validate_remap(remap, registry_table=registry)
        assert any("not in registry" in e for e in errors)

    def test_rejects_2d(self):
        errors = validate_remap(np.zeros((2, 3), dtype=np.int32))
        assert any("1-D" in e for e in errors)


# ---------------------------------------------------------------------------
# reindex_registry
# ---------------------------------------------------------------------------


class TestReindexRegistry:
    def test_assigns_contiguous_indices(self, tmp_path):
        # Insert with no global_index
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

        # Reindex again — should produce same result
        reindex_registry(registry)
        df2 = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")

        assert df1["global_index"].to_list() == df2["global_index"].to_list()

    def test_reindex_overwrites_existing_indices(self, tmp_path):
        # Start with arbitrary indices
        registry = _make_registry(tmp_path, ["uid_b", "uid_a"], indexed=True)
        # Before reindex: uid_b=0, uid_a=1 (insertion order)
        reindex_registry(registry)
        # After reindex: sorted by uid → uid_a=0, uid_b=1
        df = registry.search().select(["uid", "global_index"]).to_polars().sort("uid")
        assert df["uid"].to_list() == ["uid_a", "uid_b"]
        assert df["global_index"].to_list() == [0, 1]

    def test_empty_table(self, tmp_path):
        db = lancedb.connect(str(tmp_path / "lancedb"))
        table = db.create_table("empty", schema=GeneFeatureSchema)
        assert reindex_registry(table) == 0

    def test_build_remap_after_reindex(self, tmp_path):
        """End-to-end: ingest without indices, reindex, then build remap."""
        registry = _make_registry(
            tmp_path, ["uid_c", "uid_a", "uid_b"], indexed=False
        )
        reindex_registry(registry)

        var_df = pl.DataFrame({"global_feature_uid": ["uid_b", "uid_c"]})
        remap = build_remap(var_df, registry)
        # sorted: uid_a=0, uid_b=1, uid_c=2
        np.testing.assert_array_equal(remap, [1, 2])


# ---------------------------------------------------------------------------
# FeatureBaseSchema
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

    def test_subclass_without_index(self):
        g = GeneFeatureSchema(uid="g1", gene_name="TP53")
        assert g.global_index is None
