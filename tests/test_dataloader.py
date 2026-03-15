"""Tests for lancell.dataloader and lancell.sampler."""

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pytest
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas, align_obs_to_schema
from lancell.dataloader import (
    CellDataset,
    DenseBatch,
    MultimodalBatch,
    MultimodalCellDataset,
    SparseBatch,
    multimodal_to_dense_collate,
    sparse_to_dense_collate,
)
from lancell.ingestion import add_from_anndata
from lancell.sampler import BalancedCellSampler, CellSampler
from lancell.schema import (
    DatasetRecord,
    DenseZarrPointer,
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


class ProteinFeatureSchema(FeatureBaseSchema):
    protein_name: str


class TestCellSchema(LancellBaseSchema):
    gene_expression: SparseZarrPointer | None = None
    tissue: str | None = None


class MultimodalCellSchema(LancellBaseSchema):
    gene_expression: SparseZarrPointer | None = None
    protein_abundance: DenseZarrPointer | None = None
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
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(42)

    # Dataset 1: 20 cells, all 10 genes
    adata1 = _make_sparse_adata(20, 10, gene_uids, rng)
    adata1 = align_obs_to_schema(adata1, TestCellSchema)
    add_from_anndata(
        atlas,
        adata1,
        feature_space="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetRecord(
            zarr_group="ds1/gene_expression",
            feature_space="gene_expression",
            n_cells=20,
        ),
    )

    # Dataset 2: 15 cells, first 7 genes
    adata2 = _make_sparse_adata(15, 7, gene_uids[:7], rng)
    adata2 = align_obs_to_schema(adata2, TestCellSchema)
    add_from_anndata(
        atlas,
        adata2,
        feature_space="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetRecord(
            zarr_group="ds2/gene_expression",
            feature_space="gene_expression",
            n_cells=15,
        ),
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
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(123)
    adata = _make_sparse_adata(10, 5, gene_uids, rng)
    adata = align_obs_to_schema(adata, TestCellSchema)
    add_from_anndata(
        atlas,
        adata,
        feature_space="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetRecord(
            zarr_group="ds/gene_expression",
            feature_space="gene_expression",
            n_cells=10,
        ),
    )

    return atlas


# ---------------------------------------------------------------------------
# Tests: CellDataset basics
# ---------------------------------------------------------------------------


def test_cell_dataset_shapes(two_group_atlas):
    """CellDataset + CellSampler yield SparseBatch with correct shapes."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset()
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, num_workers=1)

    assert ds.n_cells == 35
    assert ds.n_features == 10
    assert len(sampler) == 4  # ceil(35/10)

    total_cells = 0
    for indices in sampler:
        batch = ds.__getitems__(indices)
        assert isinstance(batch, SparseBatch)
        n = len(batch.offsets) - 1
        assert batch.n_features == 10
        assert batch.offsets[0] == 0
        assert len(batch.indices) == batch.offsets[-1]
        assert len(batch.values) == batch.offsets[-1]
        if len(batch.indices) > 0:
            assert np.all(batch.indices >= 0)
            assert np.all(batch.indices < batch.n_features)
        total_cells += n

    assert total_cells == 35


def test_cell_dataset_drop_last(two_group_atlas):
    """drop_last=True on sampler skips the last incomplete batch."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset()
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, drop_last=True, num_workers=1)

    assert len(sampler) == 3  # 35 // 10
    batches = [ds.__getitems__(indices) for indices in sampler]
    assert len(batches) == 3
    assert all(len(b.offsets) - 1 == 10 for b in batches)


def test_cell_dataset_empty(two_group_atlas):
    """CellDataset handles empty query results."""
    ds = two_group_atlas.query().where("tissue = 'nonexistent'").to_cell_dataset()
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, num_workers=1)

    assert ds.n_cells == 0
    assert len(sampler) == 0
    assert list(sampler) == []


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
    ds = q.to_cell_dataset(metadata_columns=["uid"])
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, num_workers=1)
    batch = ds.__getitems__(next(iter(sampler)))

    # Reconstruct dense from SparseBatch
    n_cells = len(batch.offsets) - 1
    cd_dense = np.zeros((n_cells, ds.n_features), dtype=np.float32)
    for i in range(n_cells):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        cd_dense[i, batch.indices[s:e]] = batch.values[s:e]

    # Match rows by uid (order may differ between AnnData and CellDataset)
    cd_uids = batch.metadata["uid"].tolist()

    for cd_idx, uid in enumerate(cd_uids):
        ref_idx = ref_uids.index(uid)
        np.testing.assert_allclose(
            cd_dense[cd_idx, : ref_dense.shape[1]],
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

    ds = q.to_cell_dataset(metadata_columns=["uid"])
    sampler = CellSampler(ds.groups_np, batch_size=100, shuffle=False, num_workers=1)
    batch = ds.__getitems__(next(iter(sampler)))
    n_cells = len(batch.offsets) - 1

    cd_dense = np.zeros((n_cells, ds.n_features), dtype=np.float32)
    for i in range(n_cells):
        s, e = batch.offsets[i], batch.offsets[i + 1]
        cd_dense[i, batch.indices[s:e]] = batch.values[s:e]

    cd_uids = batch.metadata["uid"].tolist()

    for cd_idx, uid in enumerate(cd_uids):
        ref_idx = ref_uids.index(uid)
        np.testing.assert_allclose(
            cd_dense[cd_idx, : ref_dense.shape[1]],
            ref_dense[ref_idx],
            err_msg=f"Mismatch for cell uid={uid}",
        )


# ---------------------------------------------------------------------------
# Tests: shuffling
# ---------------------------------------------------------------------------


def test_shuffle_different_epochs(two_group_atlas):
    """Two epochs with shuffle produce different batch compositions."""
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_cell_dataset(metadata_columns=["uid"])
    )
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=True, seed=42, num_workers=1)

    sampler.set_epoch(0)
    epoch1_uids = []
    for indices in sampler:
        batch = ds.__getitems__(indices)
        epoch1_uids.extend(batch.metadata["uid"].tolist())

    sampler.set_epoch(1)
    epoch2_uids = []
    for indices in sampler:
        batch = ds.__getitems__(indices)
        epoch2_uids.extend(batch.metadata["uid"].tolist())

    # Same cells overall
    assert sorted(epoch1_uids) == sorted(epoch2_uids)
    # But different order
    assert epoch1_uids != epoch2_uids


def test_shuffle_reproducible(two_group_atlas):
    """Same seed + epoch produces same order when base cell order is identical."""
    cells_pl = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        ._build_scanner()
        .to_polars()
        .sort("uid")
    )

    ds1 = CellDataset(
        atlas=two_group_atlas,
        cells_pl=cells_pl,
        metadata_columns=["uid"],
    )
    ds2 = CellDataset(
        atlas=two_group_atlas,
        cells_pl=cells_pl,
        metadata_columns=["uid"],
    )

    sampler1 = CellSampler(ds1.groups_np, batch_size=10, shuffle=True, seed=42, num_workers=1)
    sampler2 = CellSampler(ds2.groups_np, batch_size=10, shuffle=True, seed=42, num_workers=1)

    uids1 = []
    for indices in sampler1:
        batch = ds1.__getitems__(indices)
        uids1.extend(batch.metadata["uid"].tolist())

    uids2 = []
    for indices in sampler2:
        batch = ds2.__getitems__(indices)
        uids2.extend(batch.metadata["uid"].tolist())

    assert uids1 == uids2


# ---------------------------------------------------------------------------
# Tests: CellSampler
# ---------------------------------------------------------------------------


def test_cell_sampler_set_epoch_reproducible(two_group_atlas):
    """CellSampler with same seed+epoch produces identical batches."""
    cells_pl = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        ._build_scanner()
        .to_polars()
        .sort("uid")
    )
    ds = CellDataset(atlas=two_group_atlas, cells_pl=cells_pl)

    sampler1 = CellSampler(ds.groups_np, batch_size=10, shuffle=True, seed=42, num_workers=1)
    sampler2 = CellSampler(ds.groups_np, batch_size=10, shuffle=True, seed=42, num_workers=1)

    # Same epoch → same batches
    sampler1.set_epoch(3)
    sampler2.set_epoch(3)
    assert list(sampler1) == list(sampler2)

    # Different epoch → different batches
    sampler1.set_epoch(0)
    plan_e0 = list(sampler1)
    sampler1.set_epoch(1)
    plan_e1 = list(sampler1)
    assert plan_e0 != plan_e1


# ---------------------------------------------------------------------------
# Tests: BalancedCellSampler
# ---------------------------------------------------------------------------


def test_balanced_cell_sampler_from_column(two_group_atlas):
    """BalancedCellSampler.from_column produces equal cells per category per batch."""
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_cell_dataset(metadata_columns=["tissue"])
    )
    # 3 tissue types, batch_size=9, drop_last=True → cells_per_cat=3, all batches full
    sampler = BalancedCellSampler.from_column(
        ds.cells_pl, "tissue", batch_size=9, shuffle=False, drop_last=True, num_workers=1
    )

    assert len(sampler) > 0
    for indices in sampler:
        batch = ds.__getitems__(indices)
        _, counts = np.unique(batch.metadata["tissue"], return_counts=True)
        # All categories should appear equally (cells_per_cat=3 each)
        assert counts.min() == counts.max()


# ---------------------------------------------------------------------------
# Tests: metadata
# ---------------------------------------------------------------------------


def test_metadata_columns(two_group_atlas):
    """Metadata columns are included and aligned with cells."""
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_cell_dataset(metadata_columns=["tissue", "uid"])
    )
    sampler = CellSampler(ds.groups_np, batch_size=100, shuffle=False, num_workers=1)
    batch = ds.__getitems__(next(iter(sampler)))

    assert batch.metadata is not None
    assert "tissue" in batch.metadata
    assert "uid" in batch.metadata
    assert len(batch.metadata["tissue"]) == len(batch.offsets) - 1
    assert len(batch.metadata["uid"]) == len(batch.offsets) - 1


def test_no_metadata(two_group_atlas):
    """Without metadata_columns, metadata is None."""
    ds = two_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset()
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, num_workers=1)
    batch = ds.__getitems__(next(iter(sampler)))
    assert batch.metadata is None


# ---------------------------------------------------------------------------
# Tests: collate functions
# ---------------------------------------------------------------------------


def test_sparse_to_dense_collate(single_group_atlas):
    """sparse_to_dense_collate produces correct dense tensor."""
    ds = single_group_atlas.query().feature_spaces("gene_expression").to_cell_dataset()
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, num_workers=1)
    batch = ds.__getitems__(next(iter(sampler)))

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
    ds = (
        two_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_cell_dataset(metadata_columns=["tissue"])
    )
    sampler = CellSampler(ds.groups_np, batch_size=10, shuffle=False, num_workers=1)
    batch = ds.__getitems__(next(iter(sampler)))

    result = sparse_to_dense_collate(batch)
    assert "X" in result
    # tissue is string dtype, so it stays as numpy array
    assert "tissue" in result


# ---------------------------------------------------------------------------
# Tests: DataLoader integration
# ---------------------------------------------------------------------------


def test_to_dataloader(single_group_atlas):
    """to_dataloader returns a working DataLoader."""
    pytest.importorskip("torch")

    dl = (
        single_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_dataloader(batch_size=5, shuffle=False)
    )

    batches = list(dl)
    assert len(batches) == 2  # 10 cells / 5
    for batch in batches:
        assert isinstance(batch, SparseBatch)


def test_to_dataloader_with_collate(single_group_atlas):
    """to_dataloader with collate_fn returns transformed batches."""
    pytest.importorskip("torch")

    dl = (
        single_group_atlas.query()
        .feature_spaces("gene_expression")
        .to_dataloader(
            collate_fn=sparse_to_dense_collate,
            batch_size=10,
            shuffle=False,
        )
    )

    batches = list(dl)
    assert len(batches) == 1
    assert "X" in batches[0]
    assert batches[0]["X"].shape == (10, 5)


# ---------------------------------------------------------------------------
# Fixtures: multimodal atlas (sparse RNA + dense protein)
# ---------------------------------------------------------------------------


def _make_dense_adata(
    n_obs: int,
    n_vars: int,
    protein_uids: list[str],
    rng: np.random.Generator,
) -> ad.AnnData:
    X = rng.random((n_obs, n_vars)).astype(np.float32)
    obs = {"tissue": [f"tissue_{i % 3}" for i in range(n_obs)]}
    var = pl.DataFrame({"global_feature_uid": protein_uids}).to_pandas()
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def multimodal_atlas(tmp_path):
    """Atlas with sparse RNA (gene_expression) and dense protein (protein_abundance).

    Each modality is ingested separately, so cell rows have exactly one pointer
    populated (the other is null).  This naturally exercises the presence-mask
    logic in MultimodalCellDataset:

    - 30 cells with RNA only  (ds1/gene_expression)
    - 70 cells with RNA only  (ds2/gene_expression)
    - 80 cells with protein only  (ds1/protein_abundance)

    Total: 180 rows.  130 have RNA, 80 have protein, 0 have both (by design for
    simplicity in ingestion; the code path is the same as paired cells — the
    present_mask correctly reflects which rows have each pointer non-null).
    """
    store = obstore.store.LocalStore(prefix=str(tmp_path))
    atlas = RaggedAtlas.create(
        db_uri=str(tmp_path / "atlas.lancedb"),
        cell_table_name="cells",
        cell_schema=MultimodalCellSchema,
        store=store,
        registry_schemas={
            "gene_expression": GeneFeatureSchema,
            "protein_abundance": ProteinFeatureSchema,
        },
        dataset_table_name="_datasets",
        dataset_schema=DatasetRecord,
    )

    gene_uids = [f"gene_{i}" for i in range(8)]
    protein_uids = [f"protein_{i}" for i in range(5)]

    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=u, gene_name=f"GENE{i}") for i, u in enumerate(gene_uids)],
    )
    atlas.register_features(
        "protein_abundance",
        [ProteinFeatureSchema(uid=u, protein_name=f"PROT{i}") for i, u in enumerate(protein_uids)],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])
    reindex_registry(atlas._registry_tables["protein_abundance"])

    rng = np.random.default_rng(7)

    # RNA group 1: 30 cells
    adata_rna1 = _make_sparse_adata(30, 8, gene_uids, rng)
    adata_rna1 = align_obs_to_schema(adata_rna1, MultimodalCellSchema)
    add_from_anndata(
        atlas,
        adata_rna1,
        feature_space="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetRecord(
            zarr_group="ds1/gene_expression",
            feature_space="gene_expression",
            n_cells=30,
        ),
    )

    # RNA group 2: 70 cells (first 6 genes)
    adata_rna2 = _make_sparse_adata(70, 6, gene_uids[:6], rng)
    adata_rna2 = align_obs_to_schema(adata_rna2, MultimodalCellSchema)
    add_from_anndata(
        atlas,
        adata_rna2,
        feature_space="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetRecord(
            zarr_group="ds2/gene_expression",
            feature_space="gene_expression",
            n_cells=70,
        ),
    )

    # Protein group: 80 cells
    adata_prot = _make_dense_adata(80, 5, protein_uids, rng)
    adata_prot = align_obs_to_schema(adata_prot, MultimodalCellSchema)
    add_from_anndata(
        atlas,
        adata_prot,
        feature_space="protein_abundance",
        zarr_layer="counts",
        dataset_record=DatasetRecord(
            zarr_group="ds1/protein_abundance",
            feature_space="protein_abundance",
            n_cells=80,
        ),
    )

    return atlas


# ---------------------------------------------------------------------------
# Tests: MultimodalCellDataset
# ---------------------------------------------------------------------------


def test_multimodal_dataset_presence(multimodal_atlas):
    """present masks reflect which cells have each modality."""
    atlas = multimodal_atlas

    ds = atlas.query().to_multimodal_dataset(
        ["gene_expression", "protein_abundance"],
        metadata_columns=["uid"],
    )

    # 30 + 70 RNA cells = 100; 80 protein cells; total = 180
    assert ds.n_cells == 180

    batch = ds.__getitems__(list(range(180)))

    assert isinstance(batch, MultimodalBatch)
    assert batch.n_cells == 180

    # 100 cells have RNA, 80 have protein
    assert batch.present["gene_expression"].sum() == 100
    assert batch.present["protein_abundance"].sum() == 80

    # Row counts of sub-batches must match presence sums
    rna_batch = batch.modalities["gene_expression"]
    prot_batch = batch.modalities["protein_abundance"]
    assert isinstance(rna_batch, SparseBatch)
    assert isinstance(prot_batch, DenseBatch)
    assert len(rna_batch.offsets) - 1 == 100
    assert prot_batch.data.shape == (80, 5)


def test_multimodal_query_order_alignment(multimodal_atlas):
    """Modality row count equals present mask sum for any batch slice."""
    atlas = multimodal_atlas

    ds = atlas.query().to_multimodal_dataset(
        ["gene_expression", "protein_abundance"],
        metadata_columns=["uid"],
    )

    # Fetch a batch of 40 cells from various positions
    indices = list(range(0, 40))
    batch = ds.__getitems__(indices)

    for fs in ["gene_expression", "protein_abundance"]:
        present_mask = batch.present[fs]
        n_present = int(present_mask.sum())
        mod_batch = batch.modalities[fs]

        if isinstance(mod_batch, SparseBatch):
            assert len(mod_batch.offsets) - 1 == n_present
        else:
            assert mod_batch.data.shape[0] == n_present


def test_multimodal_collate_shapes(multimodal_atlas):
    """multimodal_to_dense_collate produces correct tensor shapes."""
    pytest.importorskip("torch")
    atlas = multimodal_atlas

    ds = atlas.query().to_multimodal_dataset(
        ["gene_expression", "protein_abundance"],
    )

    batch = ds.__getitems__(list(range(60)))
    result = multimodal_to_dense_collate(batch)

    assert "present" in result
    assert "gene_expression" in result
    assert "protein_abundance" in result

    n_rna_present = int(batch.present["gene_expression"].sum())
    n_prot_present = int(batch.present["protein_abundance"].sum())

    assert result["gene_expression"]["X"].shape == (n_rna_present, 8)
    assert result["protein_abundance"]["X"].shape == (n_prot_present, 5)
    assert result["present"]["gene_expression"].shape == (60,)
    assert result["present"]["protein_abundance"].shape == (60,)


def test_multimodal_sampler_compatibility(multimodal_atlas):
    """CellSampler over groups_np covers all cells exactly once."""
    atlas = multimodal_atlas

    ds = atlas.query().to_multimodal_dataset(["gene_expression", "protein_abundance"])

    sampler = CellSampler(ds.groups_np, batch_size=32, shuffle=False, num_workers=1)
    seen = []
    for indices in sampler:
        seen.extend(indices)

    assert sorted(seen) == list(range(ds.n_cells))


def test_multimodal_pickle_roundtrip(multimodal_atlas):
    """MultimodalCellDataset survives pickle (simulates DataLoader worker spawn)."""
    import pickle

    atlas = multimodal_atlas
    ds = atlas.query().to_multimodal_dataset(
        ["gene_expression", "protein_abundance"],
        metadata_columns=["uid"],
    )

    data = pickle.dumps(ds)
    ds2 = pickle.loads(data)

    batch = ds2.__getitems__([0, 1, 2])
    assert isinstance(batch, MultimodalBatch)
    assert batch.n_cells == 3
