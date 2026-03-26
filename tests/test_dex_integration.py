"""End-to-end tests for lancell.dex.dex() with real atlas fixtures."""

import anndata as ad
import numpy as np
import obstore
import polars as pl
import pytest
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas
from lancell.dex import dex
from lancell.feature_layouts import reindex_registry
from lancell.ingestion import add_from_anndata
from lancell.obs_alignment import align_obs_to_schema
from lancell.schema import (
    DatasetRecord,
    DenseZarrPointer,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class DexCellSchema(LancellBaseSchema):
    gene_expression: SparseZarrPointer | None = None


class ImageFeatureSchema(FeatureBaseSchema):
    feature_name: str


class DenseCellSchema(LancellBaseSchema):
    image_features: DenseZarrPointer | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sparse_adata(
    n_obs: int,
    n_vars: int,
    feature_uids: list[str],
    rng: np.random.Generator,
    upregulated_cols: list[int] | None = None,
    up_factor: float = 1.0,
) -> ad.AnnData:
    """Create sparse AnnData with optional proportional upregulation.

    All genes start with baseline counts in [1, 20). Genes in
    ``upregulated_cols`` are multiplied by ``up_factor`` to create
    proportional differences that survive library-size normalization.
    """
    X = sp.random(n_obs, n_vars, density=0.8, format="csr", dtype=np.float32, random_state=rng)
    X.data[:] = rng.integers(1, 20, size=X.nnz).astype(np.float32)
    if upregulated_cols:
        X_dense = X.toarray()
        for col in upregulated_cols:
            X_dense[:, col] *= up_factor
        X = sp.csr_matrix(X_dense)
    var = pl.DataFrame({"global_feature_uid": feature_uids}).to_pandas()
    return ad.AnnData(X=X, var=var)


@pytest.fixture
def dex_atlas(tmp_path):
    """Atlas with 3 datasets (brain/liver/blood), 10 genes, for dex tests.

    - brain: 30 cells, counts in [1, 50) (control)
    - liver: 25 cells, counts in [50, 150) (target, clearly different)
    - blood: 20 cells, counts in [1, 50) (target, similar to brain)
    """
    db_uri = str(tmp_path / "atlas.lancedb")
    store = obstore.store.LocalStore(prefix=str(tmp_path))
    atlas = RaggedAtlas.create(
        db_uri=db_uri,
        cell_table_name="cells",
        cell_schema=DexCellSchema,
        store=store,
        registry_schemas={"gene_expression": GeneFeatureSchema},
        dataset_table_name="datasets",
        dataset_schema=DatasetRecord,
    )

    gene_uids = [f"gene_{i}" for i in range(10)]
    gene_records = [
        GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)
    ]
    atlas.register_features("gene_expression", gene_records)
    reindex_registry(atlas._registry_tables["gene_expression"])

    rng = np.random.default_rng(42)

    # brain: baseline expression (control)
    # liver: genes 0-4 strongly upregulated (proportionally different after normalization)
    # blood: similar to brain (no upregulation)
    datasets = [
        ("brain", 30, None, 1.0),
        ("liver", 25, [0, 1, 2, 3, 4], 20.0),
        ("blood", 20, None, 1.0),
    ]

    for label, n_cells, up_cols, up_factor in datasets:
        adata = _make_sparse_adata(
            n_cells, 10, gene_uids, rng, upregulated_cols=up_cols, up_factor=up_factor
        )
        adata = align_obs_to_schema(adata, DexCellSchema)
        add_from_anndata(
            atlas,
            adata,
            feature_space="gene_expression",
            zarr_layer="counts",
            dataset_record=DatasetRecord(
                uid=label,
                zarr_group=f"{label}/gene_expression",
                feature_space="gene_expression",
                n_cells=n_cells,
            ),
        )

    atlas.snapshot()
    return RaggedAtlas.checkout_latest(db_uri, DexCellSchema, store=store)


# ---------------------------------------------------------------------------
# TestDexEndToEnd
# ---------------------------------------------------------------------------


class TestDexEndToEnd:
    def test_mwu_basic(self, dex_atlas):
        result = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["liver"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
        )
        assert result.shape[0] == 10
        assert "target" in result.columns
        assert all(t == "liver" for t in result["target"].to_list())
        assert all(0 <= p <= 1 for p in result["p_value"].to_list())
        assert all(
            f >= p
            for f, p in zip(result["fdr"].to_list(), result["p_value"].to_list(), strict=False)
        )
        assert result["target_n"].to_list() == [25] * 10
        assert result["ref_n"].to_list() == [30] * 10

    def test_ttest_basic(self, dex_atlas):
        result = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["liver"],
            control="brain",
            feature_space="gene_expression",
            test="ttest",
        )
        assert result.shape[0] == 10
        assert all(0 <= p <= 1 for p in result["p_value"].to_list())
        assert result["target_n"].to_list() == [25] * 10

    def test_multiple_targets(self, dex_atlas):
        result = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["liver", "blood"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
        )
        assert result.shape[0] == 20
        targets = result["target"].to_list()
        assert targets.count("liver") == 10
        assert targets.count("blood") == 10

    def test_significant_separation(self, dex_atlas):
        result = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["liver"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
        )
        # liver [50,150) vs brain [1,50) — distributions clearly differ
        # at least some genes should be significantly different
        assert any(p < 0.05 for p in result["p_value"].to_list())
        # fold changes should be non-trivial (not all near zero)
        fcs = result["fold_change"].to_numpy()
        assert np.max(np.abs(fcs)) > 0.5

    def test_output_schema(self, dex_atlas):
        result = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["liver"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
        )
        expected_columns = [
            "feature",
            "target_mean",
            "ref_mean",
            "target_n",
            "ref_n",
            "fold_change",
            "percent_change",
            "p_value",
            "statistic",
            "fdr",
            "target",
        ]
        assert result.columns == expected_columns
        assert result.schema["feature"] == pl.String
        assert result.schema["target_mean"] in (pl.Float32, pl.Float64)
        assert result.schema["ref_mean"] in (pl.Float32, pl.Float64)
        assert result.schema["target_n"] == pl.Int64
        assert result.schema["ref_n"] == pl.Int64
        for col in ("fold_change", "percent_change", "p_value", "statistic", "fdr"):
            assert result.schema[col] in (pl.Float32, pl.Float64), f"{col} not float"
        assert result.schema["target"] == pl.String

    def test_max_records(self, dex_atlas):
        result = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["liver"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
            max_records=5,
        )
        assert result.shape[0] == 10
        assert all(n <= 5 for n in result["target_n"].to_list())
        assert all(n <= 5 for n in result["ref_n"].to_list())


# ---------------------------------------------------------------------------
# TestDexEdgeCases
# ---------------------------------------------------------------------------


class TestDexEdgeCases:
    def test_nonexistent_control(self, dex_atlas):
        with pytest.raises(AssertionError, match="No control cells"):
            dex(
                dex_atlas,
                groupby="dataset_uid",
                target=["liver"],
                control="nonexistent",
                feature_space="gene_expression",
                test="mwu",
            )

    def test_nonexistent_target_skipped(self, dex_atlas):
        result = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["nonexistent", "liver"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
        )
        # "nonexistent" is skipped, only liver results
        assert result.shape[0] == 10
        assert all(t == "liver" for t in result["target"].to_list())

    def test_all_targets_empty(self, dex_atlas):
        result = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["nonexistent1", "nonexistent2"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
        )
        assert result.shape[0] == 0
        assert "target" in result.columns

    def test_geometric_mean_false(self, dex_atlas):
        result_geo = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["liver"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
            geometric_mean=True,
        )
        result_arith = dex(
            dex_atlas,
            groupby="dataset_uid",
            target=["liver"],
            control="brain",
            feature_space="gene_expression",
            test="mwu",
            geometric_mean=False,
        )
        assert result_arith.shape[0] == 10
        # Arithmetic and geometric means should produce different target_mean values
        geo_means = result_geo["target_mean"].to_numpy()
        arith_means = result_arith["target_mean"].to_numpy()
        assert not np.allclose(geo_means, arith_means)


# ---------------------------------------------------------------------------
# TestDexMWUOnDensePath
# ---------------------------------------------------------------------------


class TestDexMWUOnDensePath:
    def test_dense_mwu(self, tmp_path):
        """MWU test works end-to-end on dense (image_features) data."""
        db_uri = str(tmp_path / "dense_atlas.lancedb")
        store = obstore.store.LocalStore(prefix=str(tmp_path))
        atlas = RaggedAtlas.create(
            db_uri=db_uri,
            cell_table_name="cells",
            cell_schema=DenseCellSchema,
            store=store,
            registry_schemas={"image_features": ImageFeatureSchema},
            dataset_table_name="datasets",
            dataset_schema=DatasetRecord,
        )

        feat_uids = [f"feat_{i}" for i in range(5)]
        feat_records = [
            ImageFeatureSchema(uid=uid, feature_name=f"FEAT{i}") for i, uid in enumerate(feat_uids)
        ]
        atlas.register_features("image_features", feat_records)
        reindex_registry(atlas._registry_tables["image_features"])

        rng = np.random.default_rng(99)

        for label, n_cells, loc in [("ctrl", 15, 0.0), ("test", 12, 5.0)]:
            X = rng.normal(loc, 1.0, size=(n_cells, 5)).astype(np.float32)
            var = pl.DataFrame({"global_feature_uid": feat_uids}).to_pandas()
            adata = ad.AnnData(X=X, var=var)
            adata = align_obs_to_schema(adata, DenseCellSchema)
            add_from_anndata(
                atlas,
                adata,
                feature_space="image_features",
                zarr_layer="raw",
                dataset_record=DatasetRecord(
                    uid=label,
                    zarr_group=f"{label}/image_features",
                    feature_space="image_features",
                    n_cells=n_cells,
                ),
            )

        atlas.snapshot()
        atlas = RaggedAtlas.checkout_latest(db_uri, DenseCellSchema, store=store)

        result = dex(
            atlas,
            groupby="dataset_uid",
            target=["test"],
            control="ctrl",
            feature_space="image_features",
            test="mwu",
        )
        assert result.shape[0] == 5
        assert all(0 <= p <= 1 for p in result["p_value"].to_list())
        # Clear separation — most p-values should be small
        assert any(p < 0.05 for p in result["p_value"].to_list())
