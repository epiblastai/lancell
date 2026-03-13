from pathlib import Path

import anndata as ad
import numpy as np
import obstore
import pyarrow as pa
import pytest
import scipy.sparse as sp

from lancell.atlas import RaggedAtlas, align_obs_to_schema
from lancell.group_specs import (
    ArraySpec,
    DTypeKind,
    FeatureAxisMode,
    PointerKind,
    SubgroupSpec,
    ZarrGroupSpec,
    register_spec,
    registered_feature_spaces,
)
from lancell.reconstruction import DenseReconstructor, SparseCSRReconstructor
from lancell.ingestion import add_from_anndata
from lancell.schema import (
    DatasetRecord,
    DenseZarrPointer,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)
from lancell.var_df import reindex_registry

UNIFORM_SPARSE_FS = "uniform_gene_expression_test"
UNIFORM_DENSE_FS = "uniform_protein_abundance_test"


def _register_uniform_specs() -> None:
    if UNIFORM_SPARSE_FS not in registered_feature_spaces():
        register_spec(
            ZarrGroupSpec(
                feature_space=UNIFORM_SPARSE_FS,
                pointer_kind=PointerKind.SPARSE,
                feature_axis_mode=FeatureAxisMode.UNIFORM,
                required_arrays=[
                    ArraySpec(
                        array_name="indices",
                        ndim=1,
                        dtype_kind=DTypeKind.UNSIGNED_INTEGER,
                    ),
                ],
                required_subgroups=[
                    SubgroupSpec(
                        subgroup_name="layers",
                        uniform_shape=True,
                        match_shape_of="indices",
                    ),
                ],
                required_layers=["counts"],
                allowed_layers=["counts"],
                reconstructor=SparseCSRReconstructor(),
            )
        )

    if UNIFORM_DENSE_FS not in registered_feature_spaces():
        register_spec(
            ZarrGroupSpec(
                feature_space=UNIFORM_DENSE_FS,
                pointer_kind=PointerKind.DENSE,
                feature_axis_mode=FeatureAxisMode.UNIFORM,
                required_subgroups=[
                    SubgroupSpec(
                        subgroup_name="layers",
                        uniform_shape=True,
                    ),
                ],
                required_layers=["counts"],
                allowed_layers=["counts"],
                reconstructor=DenseReconstructor(),
            )
        )


_register_uniform_specs()


class GeneFeatureSchema(FeatureBaseSchema):
    gene_name: str


class ProteinFeatureSchema(FeatureBaseSchema):
    protein_name: str


class UniformSparseCellSchema(LancellBaseSchema):
    uniform_gene_expression_test: SparseZarrPointer | None = None
    tissue: str | None = None


class UniformDenseCellSchema(LancellBaseSchema):
    uniform_protein_abundance_test: DenseZarrPointer | None = None
    tissue: str | None = None


class MixedCellSchema(LancellBaseSchema):
    gene_expression: SparseZarrPointer | None = None
    uniform_protein_abundance_test: DenseZarrPointer | None = None
    tissue: str | None = None


def _make_uniform_sparse_adata(
    n_obs: int,
    feature_uids: list[str],
    rng: np.random.Generator,
) -> ad.AnnData:
    n_vars = len(feature_uids)
    X = sp.random(
        n_obs,
        n_vars,
        density=0.35,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    X.data[:] = rng.integers(1, 10, size=X.nnz).astype(np.float32)
    obs = {"tissue": [f"tissue_{i % 3}" for i in range(n_obs)]}
    var = {"global_feature_uid": feature_uids}
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_uniform_dense_adata(
    n_obs: int,
    feature_uids: list[str],
    rng: np.random.Generator,
) -> ad.AnnData:
    X = rng.normal(size=(n_obs, len(feature_uids))).astype(np.float32)
    obs = {"tissue": [f"tissue_{i % 2}" for i in range(n_obs)]}
    var = {"global_feature_uid": feature_uids}
    return ad.AnnData(X=X, obs=obs, var=var)


def _create_atlas(
    tmp_path,
    *,
    cell_schema: type[LancellBaseSchema],
    registry_schemas: dict[str, type[FeatureBaseSchema]],
) -> RaggedAtlas:
    store = obstore.store.LocalStore(prefix=str(tmp_path))
    return RaggedAtlas.create(
        db_uri=str(tmp_path / "atlas.lancedb"),
        cell_table_name="cells",
        cell_schema=cell_schema,
        store=store,
        registry_schemas=registry_schemas,
        dataset_table_name="_datasets",
        dataset_schema=DatasetRecord,
    )


def test_uniform_sparse_round_trip(tmp_path):
    atlas = _create_atlas(
        tmp_path,
        cell_schema=UniformSparseCellSchema,
        registry_schemas={UNIFORM_SPARSE_FS: GeneFeatureSchema},
    )
    feature_uids = [f"gene_{i}" for i in range(6)]
    atlas.register_features(
        UNIFORM_SPARSE_FS,
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(feature_uids)],
    )
    reindex_registry(atlas._registry_tables[UNIFORM_SPARSE_FS])

    rng = np.random.default_rng(7)
    adata = align_obs_to_schema(
        _make_uniform_sparse_adata(10, feature_uids, rng),
        UniformSparseCellSchema,
    )
    add_from_anndata(
        atlas,
        adata,
        feature_space=UNIFORM_SPARSE_FS,
        zarr_group="ds/uniform_sparse",
        layer_name="counts",
    )

    out = atlas.query().feature_spaces(UNIFORM_SPARSE_FS).to_anndata()
    np.testing.assert_allclose(out.X.toarray(), adata.X.toarray())
    assert list(out.var.index) == feature_uids


def test_uniform_dense_round_trip(tmp_path):
    atlas = _create_atlas(
        tmp_path,
        cell_schema=UniformDenseCellSchema,
        registry_schemas={UNIFORM_DENSE_FS: ProteinFeatureSchema},
    )
    feature_uids = [f"protein_{i}" for i in range(4)]
    atlas.register_features(
        UNIFORM_DENSE_FS,
        [ProteinFeatureSchema(uid=uid, protein_name=f"P{i}") for i, uid in enumerate(feature_uids)],
    )
    reindex_registry(atlas._registry_tables[UNIFORM_DENSE_FS])

    rng = np.random.default_rng(11)
    adata = align_obs_to_schema(
        _make_uniform_dense_adata(8, feature_uids, rng),
        UniformDenseCellSchema,
    )
    add_from_anndata(
        atlas,
        adata,
        feature_space=UNIFORM_DENSE_FS,
        zarr_group="ds/uniform_dense",
        layer_name="counts",
    )

    out = atlas.query().feature_spaces(UNIFORM_DENSE_FS).to_anndata()
    np.testing.assert_allclose(out.X, adata.X)
    assert list(out.var.index) == feature_uids


def test_uniform_sparse_dataloader(tmp_path):
    atlas = _create_atlas(
        tmp_path,
        cell_schema=UniformSparseCellSchema,
        registry_schemas={UNIFORM_SPARSE_FS: GeneFeatureSchema},
    )
    feature_uids = [f"gene_{i}" for i in range(5)]
    atlas.register_features(
        UNIFORM_SPARSE_FS,
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(feature_uids)],
    )
    reindex_registry(atlas._registry_tables[UNIFORM_SPARSE_FS])

    rng = np.random.default_rng(19)
    adata = align_obs_to_schema(
        _make_uniform_sparse_adata(6, feature_uids, rng),
        UniformSparseCellSchema,
    )
    add_from_anndata(
        atlas,
        adata,
        feature_space=UNIFORM_SPARSE_FS,
        zarr_group="ds/uniform_sparse",
        layer_name="counts",
    )

    ds = atlas.query().feature_spaces(UNIFORM_SPARSE_FS).to_cell_dataset(
        feature_space=UNIFORM_SPARSE_FS,
        batch_size=10,
        shuffle=False,
        metadata_columns=["uid"],
    )
    batch = next(iter(ds))
    dense = np.zeros((len(batch.offsets) - 1, ds.n_features), dtype=np.float32)
    for i in range(len(batch.offsets) - 1):
        start, end = batch.offsets[i], batch.offsets[i + 1]
        dense[i, batch.indices[start:end]] = batch.values[start:end]

    np.testing.assert_allclose(dense, adata.X.toarray())


def test_uniform_ingest_requires_registry_order(tmp_path):
    atlas = _create_atlas(
        tmp_path,
        cell_schema=UniformSparseCellSchema,
        registry_schemas={UNIFORM_SPARSE_FS: GeneFeatureSchema},
    )
    feature_uids = [f"gene_{i}" for i in range(5)]
    atlas.register_features(
        UNIFORM_SPARSE_FS,
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(feature_uids)],
    )
    reindex_registry(atlas._registry_tables[UNIFORM_SPARSE_FS])

    rng = np.random.default_rng(23)
    shuffled = feature_uids[::-1]
    adata = align_obs_to_schema(
        _make_uniform_sparse_adata(4, shuffled, rng),
        UniformSparseCellSchema,
    )

    with pytest.raises(ValueError, match="exactly match registry order"):
        add_from_anndata(
            atlas,
            adata,
            feature_space=UNIFORM_SPARSE_FS,
            zarr_group="ds/bad_order",
            layer_name="counts",
        )

    assert atlas.list_datasets().is_empty()


def test_uniform_dense_validation_catches_width_mismatch(tmp_path):
    atlas = _create_atlas(
        tmp_path,
        cell_schema=UniformDenseCellSchema,
        registry_schemas={UNIFORM_DENSE_FS: ProteinFeatureSchema},
    )
    feature_uids = [f"protein_{i}" for i in range(4)]
    atlas.register_features(
        UNIFORM_DENSE_FS,
        [ProteinFeatureSchema(uid=uid, protein_name=f"P{i}") for i, uid in enumerate(feature_uids)],
    )
    reindex_registry(atlas._registry_tables[UNIFORM_DENSE_FS])

    group = atlas._root.create_group("ds/bad_uniform_dense")
    layers = group.create_group("layers")
    layers.create_array(
        "counts",
        data=np.ones((2, 3), dtype=np.float32),
        chunks=(2, 3),
        shards=(2, 3),
    )

    dataset_record = DatasetRecord(
        zarr_group="ds/bad_uniform_dense",
        feature_space=UNIFORM_DENSE_FS,
        n_cells=2,
    )
    atlas._dataset_table.add(
        pa.Table.from_pylist(
            [dataset_record.model_dump()],
            schema=DatasetRecord.to_arrow_schema(),
        )
    )

    records = [
        UniformDenseCellSchema(
            dataset_uid=dataset_record.uid,
            tissue="tissue_0",
            uniform_protein_abundance_test=DenseZarrPointer(
                feature_space=UNIFORM_DENSE_FS,
                zarr_group="ds/bad_uniform_dense",
                position=i,
            ),
        )
        for i in range(2)
    ]
    atlas.cell_table.add(
        pa.Table.from_pylist(
            [record.model_dump() for record in records],
            schema=UniformDenseCellSchema.to_arrow_schema(),
        )
    )

    errors = atlas.validate()
    assert any("has width 3" in error for error in errors)


def test_mixed_ragged_and_uniform_feature_spaces(tmp_path):
    atlas = _create_atlas(
        tmp_path,
        cell_schema=MixedCellSchema,
        registry_schemas={
            "gene_expression": GeneFeatureSchema,
            UNIFORM_DENSE_FS: ProteinFeatureSchema,
        },
    )
    gene_uids = [f"gene_{i}" for i in range(6)]
    protein_uids = [f"protein_{i}" for i in range(3)]
    atlas.register_features(
        "gene_expression",
        [GeneFeatureSchema(uid=uid, gene_name=f"GENE{i}") for i, uid in enumerate(gene_uids)],
    )
    atlas.register_features(
        UNIFORM_DENSE_FS,
        [ProteinFeatureSchema(uid=uid, protein_name=f"P{i}") for i, uid in enumerate(protein_uids)],
    )
    reindex_registry(atlas._registry_tables["gene_expression"])
    reindex_registry(atlas._registry_tables[UNIFORM_DENSE_FS])

    rng = np.random.default_rng(29)
    ragged = ad.AnnData(
        X=sp.random(5, 4, density=0.4, format="csr", dtype=np.float32, random_state=rng),
        obs={"tissue": [f"tissue_{i}" for i in range(5)]},
        var={"global_feature_uid": gene_uids[:4]},
    )
    ragged = align_obs_to_schema(ragged, MixedCellSchema)
    add_from_anndata(
        atlas,
        ragged,
        feature_space="gene_expression",
        zarr_group="ds/ragged_gene",
        layer_name="counts",
    )

    uniform = align_obs_to_schema(
        _make_uniform_dense_adata(4, protein_uids, rng),
        MixedCellSchema,
    )
    add_from_anndata(
        atlas,
        uniform,
        feature_space=UNIFORM_DENSE_FS,
        zarr_group="ds/uniform_protein",
        layer_name="counts",
    )

    gene_adata = atlas.query().feature_spaces("gene_expression").to_anndata()
    protein_adata = atlas.query().feature_spaces(UNIFORM_DENSE_FS).to_anndata()

    assert gene_adata.n_obs == 5
    assert gene_adata.n_vars == 4
    assert protein_adata.n_obs == 4
    assert protein_adata.n_vars == 3
