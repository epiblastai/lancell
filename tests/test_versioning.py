"""Tests for atlas versioning: snapshot(), list_versions(), checkout()."""

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pytest
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas
from lancell.feature_layouts import read_feature_layout, reindex_registry
from lancell.ingestion import add_from_anndata
from lancell.obs_alignment import align_obs_to_schema
from lancell.schema import (
    DatasetRecord,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)

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
    return DatasetRecord(
        zarr_group=zarr_group, feature_space="gene_expression", n_cells=adata.n_obs
    )


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
        dataset_table_name="datasets",
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

        adata = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )

        v = atlas.snapshot()
        assert v == 0

    def test_second_snapshot_returns_one(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata1 = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata1,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata1, "ds1/gene_expression"),
        )
        atlas.snapshot()

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata2,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata2, "ds2/gene_expression"),
        )
        v1 = atlas.snapshot()
        assert v1 == 1

    def test_snapshot_records_total_cells(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata2,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata2, "ds2/gene_expression"),
        )
        atlas.snapshot()

        versions = RaggedAtlas.list_versions(str(tmp_path / "atlas.lancedb"))
        assert versions["total_cells"].to_list() == [20, 35]

    def test_snapshot_raises_without_version_table(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        # Opening with a non-existent version table name raises immediately.
        with pytest.raises(ValueError, match="not found"):
            RaggedAtlas.open(
                db_uri=str(tmp_path / "atlas.lancedb"),
                cell_table_name="cells",
                cell_schema=TestCellSchema,
                dataset_table_name="datasets",
                store=store,
                registry_tables={"gene_expression": "gene_expression_registry"},
                version_table_name="nonexistent_versions",
            )

    def test_snapshot_raises_if_registry_invalid(self, tmp_path):
        """snapshot() must fail if registries are not fully indexed."""
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas = RaggedAtlas.create(
            db_uri=str(tmp_path / "atlas.lancedb"),
            cell_table_name="cells",
            cell_schema=TestCellSchema,
            store=store,
            registry_schemas={"gene_expression": GeneFeatureSchema},
            dataset_table_name="datasets",
            dataset_schema=DatasetRecord,
        )
        # Register features but deliberately skip reindex_registry
        atlas.register_features(
            "gene_expression",
            [GeneFeatureSchema(uid=f"gene_{i}", gene_name=f"GENE{i}") for i in range(5)],
        )
        with pytest.raises(ValueError, match="validation failed"):
            atlas.snapshot()


class TestListVersions:
    def test_returns_two_rows_sorted(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata1 = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata1,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata1, "ds1/gene_expression"),
        )
        atlas.snapshot()

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata2,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata2, "ds2/gene_expression"),
        )
        atlas.snapshot()

        df = RaggedAtlas.list_versions(str(tmp_path / "atlas.lancedb"))
        assert len(df) == 2
        assert df["version"].to_list() == [0, 1]

    def test_returns_polars_dataframe(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata = align_obs_to_schema(_make_sparse_adata(5, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()

        df = RaggedAtlas.list_versions(str(tmp_path / "atlas.lancedb"))
        assert isinstance(df, pl.DataFrame)


class TestCheckout:
    def test_checkout_v0_sees_only_first_batch(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata1 = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata1,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata1, "ds1/gene_expression"),
        )
        atlas.snapshot()  # v0: 20 cells

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata2,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata2, "ds2/gene_expression"),
        )
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
        add_from_anndata(
            atlas,
            adata1,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata1, "ds1/gene_expression"),
        )
        atlas.snapshot()

        adata2 = align_obs_to_schema(_make_sparse_adata(15, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata2,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata2, "ds2/gene_expression"),
        )
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
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()

        with pytest.raises(ValueError, match="version 99 not found"):
            RaggedAtlas.checkout(
                db_uri=str(tmp_path / "atlas.lancedb"),
                version=99,
                cell_schema=TestCellSchema,
                store=store,
            )

    def test_checkout_feature_layouts_pinned(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata = align_obs_to_schema(_make_sparse_adata(10, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()  # v0 — pins feature_layouts at current version

        # Record the remap as it was at snapshot time
        ds_rows = (
            atlas._dataset_table.search()
            .where("zarr_group = 'ds1/gene_expression'", prefilter=True)
            .to_polars()
        )
        layout_uid = ds_rows["layout_uid"][0]
        rows_at_v0 = read_feature_layout(atlas._feature_layouts_table, layout_uid)
        remap_at_v0 = rows_at_v0["global_index"].to_numpy().astype(np.int32, copy=False).copy()

        # Mutate _feature_layouts on the live atlas: reverse global_index
        reversed_rows = rows_at_v0.with_columns(pl.col("global_index").reverse())
        (
            atlas._feature_layouts_table.merge_insert(on=["layout_uid", "feature_uid"])
            .when_matched_update_all()
            .execute(reversed_rows)
        )

        # Checkout v0 — _feature_layouts must be pinned to the snapshot version
        pinned = RaggedAtlas.checkout(
            db_uri=str(tmp_path / "atlas.lancedb"),
            version=0,
            cell_schema=TestCellSchema,
            store=store,
        )
        gr = pinned._get_group_reader("ds1/gene_expression", "gene_expression")
        np.testing.assert_array_equal(gr.get_remap(), remap_at_v0)


class TestSchemalessCheckout:
    """Checkout without providing cell_schema — pointer fields inferred from Arrow."""

    def _make_snapshotted_atlas(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)
        adata = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()
        return atlas, store

    def test_checkout_without_schema(self, tmp_path):
        self._make_snapshotted_atlas(tmp_path)
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        checked = RaggedAtlas.checkout(
            db_uri=str(tmp_path / "atlas.lancedb"),
            version=0,
            store=store,
        )
        assert checked.cell_table.count_rows() == 20
        assert "gene_expression" in checked._pointer_fields
        assert checked._cell_schema is None

    def test_checkout_latest_without_schema(self, tmp_path):
        self._make_snapshotted_atlas(tmp_path)
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        checked = RaggedAtlas.checkout_latest(
            db_uri=str(tmp_path / "atlas.lancedb"),
            store=store,
        )
        assert checked.cell_table.count_rows() == 20
        assert "gene_expression" in checked._pointer_fields

    def test_query_works_without_schema(self, tmp_path):
        self._make_snapshotted_atlas(tmp_path)
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        checked = RaggedAtlas.checkout(
            db_uri=str(tmp_path / "atlas.lancedb"),
            version=0,
            store=store,
        )
        q = checked.query()
        assert q is not None


class TestStorelessCheckout:
    """Checkout without providing store — reconstructed from version record."""

    def test_checkout_without_store(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)
        adata = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()

        checked = RaggedAtlas.checkout(
            db_uri=str(tmp_path / "atlas.lancedb"),
            version=0,
        )
        assert checked.cell_table.count_rows() == 20

    def test_checkout_latest_without_store(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)
        adata = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()

        checked = RaggedAtlas.checkout_latest(
            db_uri=str(tmp_path / "atlas.lancedb"),
        )
        assert checked.cell_table.count_rows() == 20

    def test_version_record_has_uris(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)
        adata = align_obs_to_schema(_make_sparse_adata(5, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()

        versions = RaggedAtlas.list_versions(str(tmp_path / "atlas.lancedb"))
        row = versions.row(0, named=True)
        assert row["zarr_store_uri"] != ""

    def test_backward_compat_convention_fallback(self, tmp_path):
        """Old version records without zarr_store_uri fall back to convention."""
        from lancell.atlas import _zarr_uri_from_db_uri

        assert _zarr_uri_from_db_uri("/data/my_atlas/lance_db") == "/data/my_atlas/zarr_store"
        assert (
            _zarr_uri_from_db_uri("s3://bucket/prefix/lance_db") == "s3://bucket/prefix/zarr_store"
        )


class TestIngestionGuard:
    """Ingestion must fail with a clear message when schema is None."""

    def test_ingest_without_schema_raises(self, tmp_path):
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)
        adata = align_obs_to_schema(_make_sparse_adata(20, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )
        atlas.snapshot()

        checked = RaggedAtlas.checkout(
            db_uri=str(tmp_path / "atlas.lancedb"),
            version=0,
            store=store,
        )
        adata2 = align_obs_to_schema(_make_sparse_adata(5, 10, gene_uids), TestCellSchema)
        with pytest.raises(ValueError, match="without a cell schema"):
            add_from_anndata(
                checked,
                adata2,
                feature_space="gene_expression",
                zarr_layer="counts",
                dataset_record=_ds(adata2, "ds2/gene_expression"),
            )


class TestOpenDefaults:
    """open() with optional parameters."""

    def test_open_with_defaults(self, tmp_path):
        (tmp_path / "zarr_store").mkdir()
        store = obstore.store.LocalStore(prefix=str(tmp_path / "zarr_store"))
        atlas, gene_uids = _make_atlas(tmp_path, store)
        adata = align_obs_to_schema(_make_sparse_adata(5, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )

        reopened = RaggedAtlas.open(
            db_uri=str(tmp_path / "atlas.lancedb"),
            cell_table_name="cells",
            store=store,
        )
        assert reopened.cell_table.count_rows() == 5
        assert "gene_expression" in reopened._pointer_fields
        assert "gene_expression" in reopened._registry_tables


class TestBackwardCompat:
    def test_open_without_version_table_raises(self, tmp_path):
        """Opening with a non-existent version table name raises immediately."""
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas, gene_uids = _make_atlas(tmp_path, store)

        adata = align_obs_to_schema(_make_sparse_adata(5, 10, gene_uids), TestCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=_ds(adata, "ds1/gene_expression"),
        )

        with pytest.raises(ValueError, match="not found"):
            RaggedAtlas.open(
                db_uri=str(tmp_path / "atlas.lancedb"),
                cell_table_name="cells",
                cell_schema=TestCellSchema,
                dataset_table_name="datasets",
                store=store,
                registry_tables={"gene_expression": "gene_expression_registry"},
                version_table_name="nonexistent_versions",
            )
