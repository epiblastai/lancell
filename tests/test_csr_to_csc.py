"""Tests for memory-efficient CSR-to-CSC conversion via Rust."""

import anndata as ad
import numpy as np
import obstore
import polars as pl
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas
from lancell.batch_array import BatchArray
from lancell.feature_layouts import reindex_registry
from lancell.ingestion import add_csc, add_from_anndata
from lancell.obs_alignment import align_obs_to_schema
from lancell.schema import (
    DatasetRecord,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class TestCellSchema(LancellBaseSchema):
    gene_expression: SparseZarrPointer | None = None


def _make_sparse_adata(
    n_obs: int,
    n_vars: int,
    feature_uids: list[str],
    rng: np.random.Generator,
) -> ad.AnnData:
    X = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.float32, random_state=rng)
    X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.float32)
    var = pl.DataFrame({"global_feature_uid": feature_uids}).to_pandas()
    return ad.AnnData(X=X, var=var)


def _create_atlas_with_data(tmp_path, n_obs=100, n_vars=50, seed=42):
    """Create an atlas, ingest data as CSR, return (atlas, zarr_group, expected_csc)."""
    db_uri = str(tmp_path / "atlas.lancedb")
    store = obstore.store.LocalStore(prefix=str(tmp_path))
    atlas = RaggedAtlas.create(
        db_uri=db_uri,
        cell_table_name="cells",
        cell_schema=TestCellSchema,
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="_datasets",
        dataset_schema=DatasetRecord,
    )

    gene_uids = [f"gene_{i}" for i in range(n_vars)]
    gene_records = [
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(seed)
    adata = _make_sparse_adata(n_obs, n_vars, gene_uids, rng)
    adata = align_obs_to_schema(adata, TestCellSchema)

    zarr_group = "ds1/gene_expression"
    add_from_anndata(
        atlas,
        adata,
        feature_space="gene_expression",
        zarr_layer="counts",
        dataset_record=DatasetRecord(
            zarr_group=zarr_group,
            feature_space="gene_expression",
            n_cells=n_obs,
        ),
    )

    # Compute expected CSC from scipy for verification
    csr = adata.X if isinstance(adata.X, sp.csr_matrix) else sp.csr_matrix(adata.X)
    expected_csc = csr.tocsc()

    return atlas, zarr_group, expected_csc


class TestAddCsc:
    def test_round_trip(self, tmp_path):
        """CSC data matches scipy's CSR->CSC conversion."""
        atlas, zarr_group, expected_csc = _create_atlas_with_data(tmp_path, n_obs=100, n_vars=50)

        add_csc(atlas, zarr_group, feature_space="gene_expression", layer_name="counts")

        # Read back CSC indices and values
        csc_indices_arr = BatchArray.from_array(atlas._root[f"{zarr_group}/csc/indices"])
        csc_values_arr = BatchArray.from_array(atlas._root[f"{zarr_group}/csc/layers/counts"])

        # Get indptr from zarr
        indptr = np.asarray(atlas._root[f"{zarr_group}/csc/indptr"][:])
        csc_start = indptr[:-1]
        csc_end = indptr[1:]

        # Read all CSC data
        all_indices, _ = csc_indices_arr.read_ranges(
            csc_start.astype(np.int64), csc_end.astype(np.int64)
        )
        all_values, _ = csc_values_arr.read_ranges(
            csc_start.astype(np.int64), csc_end.astype(np.int64)
        )

        # Verify against expected CSC
        expected_indices = expected_csc.indices.astype(np.uint32)
        expected_indptr = expected_csc.indptr

        n_features = expected_csc.shape[1]
        for j in range(n_features):
            exp_start = expected_indptr[j]
            exp_end = expected_indptr[j + 1]

            # Our csc_start/csc_end should match the expected indptr structure
            our_start = int(csc_start[j])
            our_end = int(csc_end[j])
            assert our_end - our_start == exp_end - exp_start, (
                f"Feature {j}: count mismatch "
                f"(ours={our_end - our_start}, expected={exp_end - exp_start})"
            )

        # Verify total nnz
        total_ours = int(csc_end[-1]) if len(csc_end) > 0 else 0
        assert total_ours == expected_csc.nnz

        # Verify cell IDs match (per-feature)
        for j in range(n_features):
            exp_start = expected_indptr[j]
            exp_end = expected_indptr[j + 1]
            our_start = int(csc_start[j])
            our_end = int(csc_end[j])

            if exp_end == exp_start:
                continue

            our_cell_ids = all_indices[our_start:our_end]
            exp_cell_ids = expected_indices[exp_start:exp_end]
            np.testing.assert_array_equal(
                our_cell_ids,
                exp_cell_ids,
                err_msg=f"Feature {j} cell IDs mismatch",
            )

    def test_single_cell(self, tmp_path):
        """Edge case: single cell."""
        atlas, zarr_group, expected_csc = _create_atlas_with_data(tmp_path, n_obs=1, n_vars=10)
        add_csc(atlas, zarr_group, feature_space="gene_expression", layer_name="counts")

        # Verify nnz matches via zarr indptr
        indptr = np.asarray(atlas._root[f"{zarr_group}/csc/indptr"][:])
        total_ours = int(indptr[-1])
        assert total_ours == expected_csc.nnz

    def test_single_feature(self, tmp_path):
        """Edge case: single feature."""
        atlas, zarr_group, expected_csc = _create_atlas_with_data(tmp_path, n_obs=20, n_vars=1)
        add_csc(atlas, zarr_group, feature_space="gene_expression", layer_name="counts")

        indptr = np.asarray(atlas._root[f"{zarr_group}/csc/indptr"][:])
        csc_start = indptr[:-1]
        csc_end = indptr[1:]

        assert len(csc_start) == 1
        assert int(csc_start[0]) == 0
        assert int(csc_end[0]) == expected_csc.nnz

    def test_indptr_written_to_zarr(self, tmp_path):
        """indptr zarr array is created after add_csc."""
        atlas, zarr_group, _ = _create_atlas_with_data(tmp_path, n_obs=50, n_vars=20)

        # Before: no csc group
        assert "csc" not in atlas._root[zarr_group]

        add_csc(atlas, zarr_group, feature_space="gene_expression", layer_name="counts")

        # After: indptr exists
        assert "csc" in atlas._root[zarr_group]
        assert "indptr" in atlas._root[f"{zarr_group}/csc"]
        indptr = np.asarray(atlas._root[f"{zarr_group}/csc/indptr"][:])
        assert len(indptr) == 21  # n_features + 1
        assert indptr[0] == 0
