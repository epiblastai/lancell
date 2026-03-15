"""Tests for atlas versioning: snapshot(), list_versions(), checkout()."""

import numpy as np
import obstore
import pytest
import scipy.sparse as sp
import polars as pl

import anndata as ad

from lancell.atlas import RaggedAtlas, align_obs_to_schema
from lancell.ingestion import add_from_anndata
from lancell.schema import (
    DatasetRecord,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)
from lancell.var_df import reindex_registry


# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class TestCellSchema(LancellBaseSchema):
    gene_expression: SparseZarrPointer | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ds(adata: ad.AnnData, zarr_group: str) -> DatasetRecord:
    return DatasetRecord(zarr_group=zarr_group, feature_space="gene_expression", n_cells=adata.n_obs)


def _make_sparse_adata(n_obs: int, n_vars: int, feature_uids: list[str]) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    var = pl.DataFrame({"global_feature_uid": feature_uids}).to_pandas()
    return ad.AnnData(X=X, var=var)


def _make_atlas(tmp_path, store) -> RaggedAtlas:
    atlas = RaggedAtlas.create(
        db_uri=str(tmp_path / "atlas.lancedb"),
        cell_table_name="cells",
        cell_schema=TestCellSchema,
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="_datasets",
        dataset_schema=DatasetRecord,
    )
    gene_uids = [f"gene_{i}" for i in range(10)]
    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])
    return atlas, gene_uids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_first_snapshot_returns_zero(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata = align_obs_to_schema(
            _make_sparse_adata(20, 10, gene_uids), TestCellSchema
        )
        add_from_anndata(atlas, adata, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata, "ds1/gene_expression"))

        v = atlas.snapshot()
        assert v == 0

    def test_second_snapshot_returns_one(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata1 = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata1, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata1, "ds1/gene_expression"))
        atlas.snapshot()

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata2, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata2, "ds2/gene_expression"))
        v1 = atlas.snapshot()
        assert v1 == 1

    def test_snapshot_records_total_cells(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata, "ds1/gene_expression"))
        atlas.snapshot()

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata2, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata2, "ds2/gene_expression"))
        atlas.snapshot()

        versions = RaggedAtlas.list_versions(str(tmp_path / "atlas.lancedb"))
        assert versions["total_cells"].to_list() == [20, 35]

    def test_snapshot_raises_without_version_table(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)
        atlas._version_table = None

        with pytest.raises(ValueError, match="no version table"):
            atlas.snapshot()


class TestListVersions:
    def test_returns_two_rows_sorted(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata1 = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata1, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata1, "ds1/gene_expression"))
        atlas.snapshot()

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata2, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata2, "ds2/gene_expression"))
        atlas.snapshot()

        df = RaggedAtlas.list_versions(str(tmp_path / "atlas.lancedb"))
        assert len(df) == 2
        assert df["version"].to_list() == [0, 1]

    def test_returns_polars_dataframe(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata = align_obs_to_schema(_make_sparse_adata(5, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata, "ds1/gene_expression"))
        atlas.snapshot()

        df = RaggedAtlas.list_versions(str(tmp_path / "atlas.lancedb"))
        assert isinstance(df, pl.DataFrame)


class TestCheckout:
    def test_checkout_v0_sees_only_first_batch(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata1 = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata1, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata1, "ds1/gene_expression"))
        atlas.snapshot()  # v0: 20 cells

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata2, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata2, "ds2/gene_expression"))
        atlas.snapshot()  # v1: 35 cells

        old = RaggedAtlas.checkout(
            db_uri=str(tmp_path / "atlas.lancedb"),
            version=0,
            cell_schema=TestCellSchema,
            store=store,
        )
        assert old.cell_table.count_rows() == 20

    def test_checkout_v1_sees_both_batches(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata1 = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata1, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata1, "ds1/gene_expression"))
        atlas.snapshot()

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata2, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata2, "ds2/gene_expression"))
        atlas.snapshot()

        new = RaggedAtlas.checkout(
            db_uri=str(tmp_path / "atlas.lancedb"),
            version=1,
            cell_schema=TestCellSchema,
            store=store,
        )
        assert new.cell_table.count_rows() == 35

    def test_checkout_invalid_version_raises(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata = align_obs_to_schema(_make_sparse_adata(5, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata, "ds1/gene_expression"))
        atlas.snapshot()

        with pytest.raises(ValueError, match="version 99 not found"):
            RaggedAtlas.checkout(
                db_uri=str(tmp_path / "atlas.lancedb"),
                version=99,
                cell_schema=TestCellSchema,
                store=store,
            )


class TestBackwardCompat:
    def test_open_without_version_table_succeeds(self, tmp_path):
        """Atlas opened without a version table sets _version_table=None."""
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        # Write data so the tables are persisted on disk
        adata = align_obs_to_schema(_make_sparse_adata(5, 10, gene_uids), TestCellSchema)
        add_from_anndata(atlas, adata, feature_space="gene_expression",
                         zarr_layer="counts", dataset_record=_ds(adata, "ds1/gene_expression"))

        # Pass a non-existent version table name to simulate an older atlas
        opened = RaggedAtlas.open(
            db_uri=str(tmp_path / "atlas.lancedb"),
            cell_table_name="cells",
            cell_schema=TestCellSchema,
            dataset_table_name="_datasets",
            store=store,
            registry_tables={"gene_expression": "gene_expression_registry"},
            version_table_name="nonexistent_versions",
        )
        assert opened._version_table is None

    def test_snapshot_on_no_version_table_raises(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, _ = _make_atlas(tmp_path, store)
        atlas._version_table = None

        with pytest.raises(ValueError, match="no version table"):
            atlas.snapshot()
