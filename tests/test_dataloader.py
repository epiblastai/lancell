"""Tests for lancell.dataloader — fast batch dataloading for ML training."""

from pathlib import Path

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pytest
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas, align_obs_to_schema
from lancell.dataloader import CellDataset, SparseBatch, sparse_to_dense_collate
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
    tissue: str | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sparse_adata(
    n_obs: int,
    n_vars: int,
    feature_uids: list[str],
    rng: np.random.Generator,
    tissues: list[str] | None = None,
) -> ad.AnnData:
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.float32)
    obs = {"tissue": tissues or [f"tissue_{i % 3}" for i in range(n_obs)]}
    var = pl.DataFrame({"global_feature_uid": feature_uids}).to_pandas()
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def two_group_atlas(tmp_path):
    """Atlas with 2 zarr groups, 10 genes, 35 total cells (20 + 15)."""
    store = obstore.store.LocalStore(prefix=str(tmp_path))
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
    gene_records = [
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}")
        for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(42)

    # Dataset 1: 20 cells, all 10 genes
    adata1 = _make_sparse_adata(20, 10, gene_uids, rng)
    adata1 = align_obs_to_schema(adata1, TestCellSchema)
    add_from_anndata(
        atlas, adata1,
        feature_space="gene_expression",
        zarr_group="ds1/gene_expression",
        layer_name="counts",
    )

    # Dataset 2: 15 cells, first 7 genes
    adata2 = _make_sparse_adata(15, 7, gene_uids[:7], rng)
    adata2 = align_obs_to_schema(adata2, TestCellSchema)
    add_from_anndata(
        atlas, adata2,
        feature_space="gene_expression",
        zarr_group="ds2/gene_expression",
        layer_name="counts",
    )

    return atlas


@pytest.fixture
def single_group_atlas(tmp_path):
    """Atlas with 1 zarr group for exact round-trip comparison."""
    store = obstore.store.LocalStore(prefix=str(tmp_path))
    atlas = RaggedAtlas.create(
        db_uri=str(tmp_path / "atlas.lancedb"),
        cell_table_name="cells",
        cell_schema=TestCellSchema,
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="_datasets",
        dataset_schema=DatasetRecord,
    )

    gene_uids = [f"gene_{i}" for i in range(5)]
    gene_records = [
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}")
        for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(123)
    adata = _make_sparse_adata(10, 5, gene_uids, rng)
    adata = align_obs_to_schema(adata, TestCellSchema)
    add_from_anndata(
        atlas, adata,
        feature_space="gene_expression",
        zarr_group="ds/gene_expression",
        layer_name="counts",
    )

    return atlas


# ---------------------------------------------------------------------------
# Tests: CellDataset basics
# ---------------------------------------------------------------------------


def test_cell_dataset_shapes(two_group_atlas):
    """CellDataset yields SparseBatch with correct shapes."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset(
        batch_size=10, shuffle=False
    )

    assert ds.n_cells == 35
    assert ds.n_features == 10
    assert len(ds) == 4  # ceil(35/10)

    total_cells = 0
    for batch in ds:
        assert isinstance(batch, SparseBatch)
        n = len(batch.offsets) - 1
        assert batch.n_features == 10
        assert batch.offsets[0] == 0
        assert len(batch.indices) == batch.offsets[-1]
        assert len(batch.values) == batch.offsets[-1]
        # All indices must be valid global feature indices
        if len(batch.indices) > 0:
            assert np.all(batch.indices >= 0)
            assert np.all(batch.indices < batch.n_features)
        total_cells += n

    assert total_cells == 35


def test_cell_dataset_drop_last(two_group_atlas):
    """drop_last=True skips the last incomplete batch."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset(
        batch_size=10, shuffle=False, drop_last=True
    )

    assert len(ds) == 3  # 35 // 10
    batches = list(ds)
    assert len(batches) == 3
    assert all(len(b.offsets) - 1 == 10 for b in batches)


def test_cell_dataset_empty(two_group_atlas):
    """CellDataset handles empty query results."""
    ds = two_group_atlas.query().where("tissue = 'nonexistent'").to_cell_dataset(
        batch_size=10, shuffle=False
    )

    assert ds.n_cells == 0
    assert len(ds) == 0
    assert list(ds) == []


# ---------------------------------------------------------------------------
# Tests: round-trip data integrity
# ---------------------------------------------------------------------------


def test_round_trip_values(single_group_atlas):
    """Data from CellDataset matches to_anndata() for a single-group atlas."""
    atlas = single_group_atlas
    q = atlas.query().feature_spaces("gene_expression")

    # Reference via AnnData path
    adata = q.to_anndata()
    ref_dense = adata.X.toarray()
    ref_uids = list(adata.obs.index)

    # CellDataset path (single batch, no shuffle, with uid metadata)
    ds = q.to_cell_dataset(
        batch_size=10, shuffle=False, metadata_columns=["uid"]
    )
    batch = next(iter(ds))

    # Reconstruct dense from SparseBatch
    n_cells = len(batch.offsets) - 1
    # CellDataset uses full registry size (5), AnnData might use union (also 5 here)
    cd_dense = np.zeros((n_cells, ds.n_features), dtype=np.float32)
    for i in range(n_cells):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        cd_dense[i, batch.indices[s:e]] = batch.values[s:e]

    # Match rows by uid (order may differ between AnnData and CellDataset)
    cd_uids = batch.metadata["uid"].tolist()

    for cd_idx, uid in enumerate(cd_uids):
        ref_idx = ref_uids.index(uid)
        np.testing.assert_allclose(
            cd_dense[cd_idx, :ref_dense.shape[1]],
            ref_dense[ref_idx],
            err_msg=f"Mismatch for cell uid={uid}",
        )


def test_round_trip_two_groups(two_group_atlas):
    """Data from CellDataset matches across two zarr groups."""
    atlas = two_group_atlas
    q = atlas.query().feature_spaces("gene_expression")

    adata = q.to_anndata()
    ref_dense = adata.X.toarray()
    ref_uids = list(adata.obs.index)

    ds = q.to_cell_dataset(
        batch_size=100, shuffle=False, metadata_columns=["uid"]
    )
    batch = next(iter(ds))
    n_cells = len(batch.offsets) - 1

    cd_dense = np.zeros((n_cells, ds.n_features), dtype=np.float32)
    for i in range(n_cells):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        cd_dense[i, batch.indices[s:e]] = batch.values[s:e]

    cd_uids = batch.metadata["uid"].tolist()

    for cd_idx, uid in enumerate(cd_uids):
        ref_idx = ref_uids.index(uid)
        np.testing.assert_allclose(
            cd_dense[cd_idx, :ref_dense.shape[1]],
            ref_dense[ref_idx],
            err_msg=f"Mismatch for cell uid={uid}",
        )


# ---------------------------------------------------------------------------
# Tests: shuffling
# ---------------------------------------------------------------------------


def test_shuffle_different_epochs(two_group_atlas):
    """Two epochs with shuffle produce different batch compositions."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset(
        batch_size=10, shuffle=True, seed=42, metadata_columns=["uid"]
    )

    epoch1_uids = []
    for batch in ds:
        epoch1_uids.extend(batch.metadata["uid"].tolist())

    epoch2_uids = []
    for batch in ds:
        epoch2_uids.extend(batch.metadata["uid"].tolist())

    # Same cells overall
    assert sorted(epoch1_uids) == sorted(epoch2_uids)
    # But different order
    assert epoch1_uids != epoch2_uids


def test_shuffle_reproducible(two_group_atlas):
    """Same seed + epoch produces same order when base cell order is identical."""
    # Sort cells to ensure deterministic base ordering (LanceDB order is not guaranteed)
    cells_pl = (
        two_group_atlas.query().feature_spaces("gene_expression")
        ._build_scanner().to_polars().sort("uid")
    )

    ds1 = CellDataset(
        atlas=two_group_atlas, cells_pl=cells_pl,
        batch_size=10, shuffle=True, seed=42, metadata_columns=["uid"],
    )
    ds2 = CellDataset(
        atlas=two_group_atlas, cells_pl=cells_pl,
        batch_size=10, shuffle=True, seed=42, metadata_columns=["uid"],
    )

    uids1 = []
    for batch in ds1:
        uids1.extend(batch.metadata["uid"].tolist())

    uids2 = []
    for batch in ds2:
        uids2.extend(batch.metadata["uid"].tolist())

    assert uids1 == uids2


# ---------------------------------------------------------------------------
# Tests: metadata
# ---------------------------------------------------------------------------


def test_metadata_columns(two_group_atlas):
    """Metadata columns are included and aligned with cells."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset(
        batch_size=100, shuffle=False, metadata_columns=["tissue", "uid"]
    )

    batch = next(iter(ds))
    assert batch.metadata is not None
    assert "tissue" in batch.metadata
    assert "uid" in batch.metadata
    assert len(batch.metadata["tissue"]) == len(batch.offsets) - 1
    assert len(batch.metadata["uid"]) == len(batch.offsets) - 1


def test_no_metadata(two_group_atlas):
    """Without metadata_columns, metadata is None."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset(
        batch_size=10, shuffle=False
    )

    batch = next(iter(ds))
    assert batch.metadata is None


# ---------------------------------------------------------------------------
# Tests: collate functions
# ---------------------------------------------------------------------------


def test_sparse_to_dense_collate(single_group_atlas):
    """sparse_to_dense_collate produces correct dense tensor."""
    ds = single_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset(
        batch_size=10, shuffle=False
    )
    batch = next(iter(ds))

    result = sparse_to_dense_collate(batch)
    X = result["X"]

    assert X.shape == (10, 5)
    assert X.dtype.is_floating_point

    # Verify round-trip: dense -> CSR -> compare with original batch
    for i in range(10):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        for j in range(s, e):
            assert X[i, batch.indices[j]].item() == pytest.approx(batch.values[j])


def test_collate_with_metadata(two_group_atlas):
    """Collate functions pass through metadata as tensors."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset(
        batch_size=10, shuffle=False, metadata_columns=["tissue"]
    )
    batch = next(iter(ds))

    result = sparse_to_dense_collate(batch)
    assert "X" in result
    # tissue is string dtype, so it stays as numpy array
    assert "tissue" in result


# ---------------------------------------------------------------------------
# Tests: DataLoader integration
# ---------------------------------------------------------------------------


def test_to_dataloader(single_group_atlas):
    """to_dataloader returns a working DataLoader."""
    torch = pytest.importorskip("torch")

    dl = single_group_atlas.query().feature_spaces("gene_expression").to_dataloader(
        batch_size=5, shuffle=False
    )

    batches = list(dl)
    assert len(batches) == 2  # 10 cells / 5
    for batch in batches:
        assert isinstance(batch, SparseBatch)


def test_to_dataloader_with_collate(single_group_atlas):
    """to_dataloader with collate_fn returns transformed batches."""
    torch = pytest.importorskip("torch")

    dl = single_group_atlas.query().feature_spaces("gene_expression").to_dataloader(
        collate_fn=sparse_to_dense_collate,
        batch_size=10,
        shuffle=False,
    )

    batches = list(dl)
    assert len(batches) == 1
    assert "X" in batches[0]
    assert batches[0]["X"].shape == (10, 5)
