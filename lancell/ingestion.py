"""Reference ingestion functions for writing AnnData into a RaggedAtlas.

These are extracted from the original ``RaggedAtlas`` write path and serve as a
reference implementation.  Downstream projects can write their own ingestion
that calls the lower-level ``var_df`` helpers directly.
"""

import anndata as ad
import numpy as np
import polars as pl
import pyarrow as pa
import scipy.sparse as sp

from lancell.atlas import (
    PointerFieldInfo,
    RaggedAtlas,
    _schema_obs_fields,
    validate_obs_columns,
)
from lancell.group_specs import PointerKind, get_spec
from lancell.schema import (
    DatasetRecord,
    DenseZarrPointer,
    SparseZarrPointer,
)
from lancell.var_df import build_remap, write_remap, write_var_df


def add_from_anndata(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    *,
    feature_space: str,
    zarr_group: str | None = None,
    layer_name: str | None,
    chunk_size: int = 4096,
    shard_size: int = 65536,
    dataset_record: DatasetRecord | None = None,
    use_bitpacking: bool = False,
) -> int:
    """Ingest an AnnData into the atlas.

    Writes zarr arrays, var_df sidecar, remap, and inserts cell records
    into the cell table. Features must already be registered via
    :meth:`RaggedAtlas.register_features`, and ``adata.var`` must contain a
    ``global_feature_uid`` column.

    Parameters
    ----------
    atlas:
        The atlas to ingest into.
    adata:
        The AnnData to ingest.
    feature_space:
        Which feature space this data belongs to.
    zarr_group:
        Zarr group path (relative to atlas store) for this ingestion.
        If ``None`` (default), a UUID-based name is generated
        automatically, which guarantees uniqueness across concurrent
        writers.
    layer_name:
        Required for feature spaces with allowed_layers — the layer to
        write (e.g. ``"counts"``). Unused for feature spaces
        without layers (e.g. image_tiles), in which case set to None.
    chunk_size:
        Zarr chunk size for 1D arrays.
    shard_size:
        Zarr shard size for 1D arrays.
    dataset_record:
        Optional pre-built dataset record. If ``None``, a default
        :class:`DatasetRecord` is created. Pass a subclassed record
        for richer dataset metadata.
    use_bitpacking:
        If True, use BP-128 bitpacking instead of zstd for integer
        arrays (indices and count layers). Float arrays still use zstd.

    Returns
    -------
    int
        Number of cells ingested.
    """
    spec = get_spec(feature_space)

    if spec.allowed_layers and layer_name is None:
        raise ValueError(
            f"layer_name is required for feature space '{feature_space}'. "
            f"Allowed values: {spec.allowed_layers}"
        )
    if layer_name is not None and spec.allowed_layers and layer_name not in spec.allowed_layers:
        raise ValueError(
            f"layer_name '{layer_name}' is not allowed for feature space "
            f"'{feature_space}'. Allowed: {spec.allowed_layers}"
        )

    # Pre-flight: validate obs columns match schema before any writes
    obs_errors = validate_obs_columns(adata.obs, atlas._cell_schema)
    if obs_errors:
        raise ValueError(f"obs columns do not match cell schema: {obs_errors}")

    # Find the pointer field for this feature space
    pointer_field: PointerFieldInfo | None = None
    for pf in atlas._pointer_fields.values():
        if pf.feature_space == feature_space:
            pointer_field = pf
            break
    if pointer_field is None:
        raise ValueError(
            f"Schema {atlas._cell_schema.__name__} has no pointer field "
            f"for feature space '{feature_space}'"
        )

    n_cells = adata.n_obs

    # Create dataset record (FK for cells)
    if dataset_record is None:
        dataset_record = DatasetRecord(
            zarr_group="",  # placeholder, overwritten below
            feature_space=feature_space,
            n_cells=n_cells,
        )

    # Default zarr_group to dataset_uid to prevent collisions between
    # concurrent writers.
    if zarr_group is None:
        zarr_group = dataset_record.uid
    dataset_record.zarr_group = zarr_group
    dataset_arrow = pa.Table.from_pylist(
        [dataset_record.model_dump()],
        schema=type(dataset_record).to_arrow_schema(),
    )
    atlas._dataset_table.add(dataset_arrow)

    # Write zarr arrays
    if spec.pointer_kind is PointerKind.SPARSE:
        starts, ends = write_sparse_zarr(
            atlas,
            adata,
            zarr_group,
            layer_name,
            chunk_size,
            shard_size,
            use_bitpacking=use_bitpacking,
        )
    else:
        write_dense_zarr(atlas, adata, zarr_group, layer_name, chunk_size, shard_size)

    # Write var_df sidecar
    if spec.has_var_df:
        write_var_sidecar(atlas, adata, feature_space, zarr_group)

    # Build cell records from obs columns
    obs_field_names = list(_schema_obs_fields(atlas._cell_schema).keys())
    records = []
    for i in range(n_cells):
        if spec.pointer_kind is PointerKind.SPARSE:
            pointer = SparseZarrPointer(
                feature_space=feature_space,
                zarr_group=zarr_group,
                start=int(starts[i]),
                end=int(ends[i]),
                zarr_row=i,
            )
        else:
            pointer = DenseZarrPointer(
                feature_space=feature_space,
                zarr_group=zarr_group,
                position=i,
            )

        extra = {col: adata.obs.iloc[i][col] for col in obs_field_names if col in adata.obs.columns}
        record_kwargs = {
            pointer_field.field_name: pointer,
            "dataset_uid": dataset_record.uid,
            **extra,
        }
        records.append(atlas._cell_schema(**record_kwargs))

    arrow_schema = atlas._cell_schema.to_arrow_schema()
    arrow_table = pa.Table.from_pylist([r.model_dump() for r in records], schema=arrow_schema)
    atlas.cell_table.add(arrow_table)
    return n_cells


def write_sparse_zarr(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    zarr_group: str,
    layer_name: str,
    chunk_size: int,
    shard_size: int,
    use_bitpacking: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Write sparse (CSR) data to zarr arrays. Returns (starts, ends).

    Parameters
    ----------
    use_bitpacking
        If True, use BP-128 bitpacking for indices (with delta) and
        integer count layers instead of the default zstd codec.
    """
    csr = sp.csr_matrix(adata.X)
    flat_indices = csr.indices.astype(np.uint32)
    flat_values = csr.data

    starts = csr.indptr[:-1].astype(np.int64)
    ends = csr.indptr[1:].astype(np.int64)

    group = atlas._root.create_group(zarr_group)

    indices_kwargs: dict = {}
    layer_kwargs: dict = {}
    if use_bitpacking:
        from lancell.codecs.bitpacking import BitpackingCodec

        indices_kwargs["compressors"] = BitpackingCodec(transform="delta")
        # Only use bitpacking for integer layers
        if np.issubdtype(flat_values.dtype, np.integer):
            layer_kwargs["compressors"] = BitpackingCodec(transform="none")

    csr_group = group.create_group("csr")
    csr_group.create_array(
        "indices",
        data=flat_indices,
        chunks=(chunk_size,),
        shards=(shard_size,),
        **indices_kwargs,
    )

    layers = csr_group.create_group("layers")
    layers.create_array(
        layer_name,
        data=flat_values,
        chunks=(chunk_size,),
        shards=(shard_size,),
        **layer_kwargs,
    )

    return starts, ends


def write_dense_zarr(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    zarr_group: str,
    layer_name: str | None,
    chunk_size: int,
    shard_size: int,
) -> None:
    """Write dense data to a 2D zarr array."""
    data = np.asarray(adata.X, dtype=np.float32)

    group = atlas._root.create_group(zarr_group)

    n_cells, n_features = data.shape

    if layer_name is not None:
        layers_group = group.create_group("layers")
        layers_group.create_array(
            layer_name,
            data=data,
            chunks=(chunk_size, n_features),
            shards=(shard_size, n_features),
        )
    else:
        group.create_array(
            "data",
            data=data,
            chunks=(chunk_size, n_features),
            shards=(shard_size, n_features),
        )


def write_var_sidecar(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    feature_space: str,
    zarr_group: str,
) -> None:
    """Write var.parquet and version-gated remap.parquet for a dataset.

    Requires ``global_feature_uid`` in ``adata.var`` and features to
    already be registered via :meth:`RaggedAtlas.register_features`.  The remap
    is tagged with the current registry table version so readers can detect
    staleness.
    """
    var_df = pl.from_pandas(adata.var.reset_index())
    if "global_feature_uid" not in var_df.columns:
        raise ValueError(
            "adata.var must have a 'global_feature_uid' column. "
            "Set it before calling add_from_anndata()."
        )

    write_var_df(atlas._store, zarr_group, var_df)

    if feature_space in atlas._registry_tables:
        registry_table = atlas._registry_tables[feature_space]
        remap = build_remap(var_df, registry_table)
        group = atlas._root[zarr_group]
        write_remap(
            atlas._store,
            group,
            remap,
            registry_version=registry_table.version,
        )
