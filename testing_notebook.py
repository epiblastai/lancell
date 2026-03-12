"""End-to-end test for the RaggedAtlas refactored API.

Run with: python testing_notebook.py
"""

import shutil
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import obstore
import polars as pl
import scipy.sparse as sp

from lancell.atlas import (
    RaggedAtlas,
    align_obs_to_schema,
    validate_obs_columns,
)
from lancell.group_specs import FeatureSpace, LayerName
from lancell.schema import (
    DatasetRecord,
    DenseZarrPointer,
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
    protein_abundance: DenseZarrPointer | None = None
    tissue: str | None = None
    organism: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sparse_adata(n_obs: int, n_vars: int, feature_uids: list[str]) -> ad.AnnData:
    """Create a sparse AnnData with random data."""
    rng = np.random.default_rng(42)
    density = 0.3
    X = sp.random(n_obs, n_vars, density=density, format="csr", dtype=np.float32, random_state=rng)
    X.data[:] = rng.integers(1, 100, size=X.nnz).astype(np.float32)

    obs = {
        "tissue": [f"tissue_{i % 3}" for i in range(n_obs)],
        "organism": ["human"] * n_obs,
    }
    var = {"global_feature_uid": feature_uids}

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pl.DataFrame(var).to_pandas(),
    )
    return adata


def make_dense_adata(n_obs: int, n_vars: int, feature_uids: list[str]) -> ad.AnnData:
    """Create a dense AnnData with random data."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((n_obs, n_vars)).astype(np.float32)

    obs = {
        "tissue": [f"tissue_{i % 2}" for i in range(n_obs)],
        "organism": ["mouse"] * n_obs,
    }
    var = {"global_feature_uid": feature_uids}

    return ad.AnnData(
        X=X,
        obs=obs,
        var=pl.DataFrame(var).to_pandas(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validate_obs_columns():
    """Test validate_obs_columns and align_obs_to_schema."""
    print("--- test_validate_obs_columns ---")

    # Valid obs with required fields present
    adata = make_sparse_adata(5, 3, ["g1", "g2", "g3"])
    errors = validate_obs_columns(adata.obs, TestCellSchema)
    assert errors == [], f"Expected no errors, got: {errors}"

    # Missing required field (none for TestCellSchema, all optional)
    # So this should pass even with empty obs
    adata_empty_obs = ad.AnnData(
        X=sp.csr_matrix((5, 3), dtype=np.float32),
    )
    errors = validate_obs_columns(adata_empty_obs.obs, TestCellSchema)
    assert errors == [], f"Expected no errors for schema with all optional fields, got: {errors}"

    # Test align_obs_to_schema
    adata_extra = make_sparse_adata(5, 3, ["g1", "g2", "g3"])
    adata_extra.obs["extra_col"] = "foo"
    aligned = align_obs_to_schema(adata_extra, TestCellSchema)
    assert "extra_col" not in aligned.obs.columns, "Extra columns should be dropped"
    assert "tissue" in aligned.obs.columns
    assert "organism" in aligned.obs.columns

    print("PASSED")


def test_full_workflow():
    """Test register_features -> align_obs_to_schema -> add_from_anndata -> query."""
    print("--- test_full_workflow ---")

    tmp_dir = tempfile.mkdtemp()
    try:
        store = obstore.store.LocalStore(prefix=tmp_dir)

        # Create atlas
        atlas = RaggedAtlas.create(
            db_uri=str(Path(tmp_dir) / "atlas.lancedb"),
            cell_table_name="cells",
            cell_schema=TestCellSchema,
            store=store,
            registry_schemas={
                FeatureSpace.GENE_EXPRESSION: GeneFeatureSchema,
                FeatureSpace.PROTEIN_ABUNDANCE: FeatureBaseSchema,
            },
            dataset_table_name="_datasets",
            dataset_schema=DatasetRecord,
        )

        # 1. Register features
        gene_uids = [f"gene_{i}" for i in range(10)]
        gene_records = [
            GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}")
            for i, uid in enumerate(gene_uids)
        ]
        n_registered = atlas.register_features(FeatureSpace.GENE_EXPRESSION, gene_records)
        assert n_registered == 10, f"Expected 10 registered, got {n_registered}"
        print(f"  Registered {n_registered} gene features")

        # Register again — should be 0 new
        n_dup = atlas.register_features(FeatureSpace.GENE_EXPRESSION, gene_records)
        assert n_dup == 0, f"Expected 0 duplicates, got {n_dup}"

        # Register via DataFrame
        protein_uids = [f"protein_{i}" for i in range(5)]
        protein_df = pl.DataFrame({"uid": protein_uids})
        n_prot = atlas.register_features(FeatureSpace.PROTEIN_ABUNDANCE, protein_df)
        assert n_prot == 5, f"Expected 5 proteins, got {n_prot}"
        print(f"  Registered {n_prot} protein features")

        # 2. Create and align AnnDatas
        adata1 = make_sparse_adata(20, 10, gene_uids)
        adata1 = align_obs_to_schema(adata1, TestCellSchema)

        adata2 = make_sparse_adata(15, 7, gene_uids[:7])
        adata2 = align_obs_to_schema(adata2, TestCellSchema)

        # 3. Add from anndata
        n1 = atlas.add_from_anndata(
            adata1,
            feature_space=FeatureSpace.GENE_EXPRESSION,
            zarr_group="dataset_1/gene_expression",
            layer_name=LayerName.COUNTS,
        )
        assert n1 == 20
        print(f"  Ingested dataset_1: {n1} cells")

        n2 = atlas.add_from_anndata(
            adata2,
            feature_space=FeatureSpace.GENE_EXPRESSION,
            zarr_group="dataset_2/gene_expression",
            layer_name=LayerName.COUNTS,
        )
        assert n2 == 15
        print(f"  Ingested dataset_2: {n2} cells")

        # Dense dataset
        adata_dense = make_dense_adata(10, 5, protein_uids)
        adata_dense = align_obs_to_schema(adata_dense, TestCellSchema)
        n3 = atlas.add_from_anndata(
            adata_dense,
            feature_space=FeatureSpace.PROTEIN_ABUNDANCE,
            zarr_group="dataset_3/protein_abundance",
            layer_name=LayerName.COUNTS,
        )
        assert n3 == 10
        print(f"  Ingested dataset_3 (dense): {n3} cells")

        # 4. Check list_datasets
        datasets = atlas.list_datasets()
        assert datasets.height == 3, f"Expected 3 datasets, got {datasets.height}"
        print(f"  Datasets:\n{datasets}")

        # 5. Validate
        errors = atlas.validate()
        assert errors == [], f"Validation errors: {errors}"
        print("  Validation passed")

        # 6. Query — to_polars
        cells = atlas.query().to_polars()
        assert cells.height == 45, f"Expected 45 cells, got {cells.height}"
        print(f"  Query returned {cells.height} cells")

        # 7. Query — to_anndata (first feature space = gene_expression)
        adata_out = atlas.query().feature_spaces(FeatureSpace.GENE_EXPRESSION).to_anndata()
        assert adata_out.n_obs == 35, f"Expected 35 obs, got {adata_out.n_obs}"
        assert adata_out.n_vars == 10, f"Expected 10 vars, got {adata_out.n_vars}"
        print(f"  Gene expression AnnData: {adata_out.n_obs} x {adata_out.n_vars}")

        # 8. Query — dense to_anndata
        adata_prot = atlas.query().feature_spaces(FeatureSpace.PROTEIN_ABUNDANCE).to_anndata()
        assert adata_prot.n_obs == 10, f"Expected 10 obs, got {adata_prot.n_obs}"
        assert adata_prot.n_vars == 5, f"Expected 5 vars, got {adata_prot.n_vars}"
        print(f"  Protein abundance AnnData: {adata_prot.n_obs} x {adata_prot.n_vars}")

        # 9. Batched query
        batches = list(atlas.query().feature_spaces(FeatureSpace.GENE_EXPRESSION).to_batches(batch_size=10))
        total_cells = sum(b.n_obs for b in batches)
        assert total_cells == 35, f"Batched total: {total_cells}"
        print(f"  Batched query: {len(batches)} batches, {total_cells} total cells")

        print("PASSED")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_layer_name_required_for_sparse():
    """Verify that layer_name is required for sparse feature spaces."""
    print("--- test_layer_name_required_for_sparse ---")

    tmp_dir = tempfile.mkdtemp()
    try:
        store = obstore.store.LocalStore(prefix=tmp_dir)
        atlas = RaggedAtlas.create(
            db_uri=str(Path(tmp_dir) / "atlas.lancedb"),
            cell_table_name="cells",
            cell_schema=TestCellSchema,
            store=store,
            registry_schemas={FeatureSpace.GENE_EXPRESSION: GeneFeatureSchema},
            dataset_table_name="_datasets",
            dataset_schema=DatasetRecord,
        )

        gene_uids = ["g1", "g2", "g3"]
        atlas.register_features(
            FeatureSpace.GENE_EXPRESSION,
            [GeneFeatureSchema(uid=u, gene_name=f"G{i}") for i, u in enumerate(gene_uids)],
        )

        adata = make_sparse_adata(5, 3, gene_uids)
        adata = align_obs_to_schema(adata, TestCellSchema)

        try:
            atlas.add_from_anndata(
                adata,
                feature_space=FeatureSpace.GENE_EXPRESSION,
                zarr_group="test_group",
                layer_name=None,
            )
            assert False, "Should have raised ValueError for missing layer_name"
        except ValueError as e:
            assert "layer_name is required" in str(e)

        print("PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_obs_validation_before_write():
    """Verify that obs validation fails before any zarr writes."""
    print("--- test_obs_validation_before_write ---")

    # Create a schema with a required field
    class StrictCellSchema(LancellBaseSchema):
        gene_expression: SparseZarrPointer | None = None
        required_field: str  # Required, not optional

    tmp_dir = tempfile.mkdtemp()
    try:
        store = obstore.store.LocalStore(prefix=tmp_dir)
        atlas = RaggedAtlas.create(
            db_uri=str(Path(tmp_dir) / "atlas.lancedb"),
            cell_table_name="cells",
            cell_schema=StrictCellSchema,
            store=store,
            registry_schemas={FeatureSpace.GENE_EXPRESSION: GeneFeatureSchema},
            dataset_table_name="_datasets",
            dataset_schema=DatasetRecord,
        )

        gene_uids = ["g1", "g2"]
        atlas.register_features(
            FeatureSpace.GENE_EXPRESSION,
            [GeneFeatureSchema(uid=u, gene_name=f"G{i}") for i, u in enumerate(gene_uids)],
        )

        # AnnData without required_field in obs
        adata = make_sparse_adata(5, 2, gene_uids)

        try:
            atlas.add_from_anndata(
                adata,
                feature_space=FeatureSpace.GENE_EXPRESSION,
                zarr_group="test_group",
                layer_name=LayerName.COUNTS,
            )
            assert False, "Should have raised ValueError for missing required_field"
        except ValueError as e:
            assert "required_field" in str(e)

        print("PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_validate_obs_columns()
    test_full_workflow()
    test_layer_name_required_for_sparse()
    test_obs_validation_before_write()
    print("\nAll tests passed!")
